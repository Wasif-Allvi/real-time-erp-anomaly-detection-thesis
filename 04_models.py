# ============================================================
# 04_models.py
# Thesis: Deep Sequential Models for ERP Anomaly Detection
# Mustafa Wasif Allvi — University of Potsdam, M.Sc. Data Science
# ============================================================
# Three neural architectures (thesis Section 5.3):
#   - LSTMModel        (Section 5.3.1)
#   - GRUModel         (Section 5.3.2)
#   - TransformerModel (Section 5.3.3)
#
# Each model returns TWO outputs:
#   1. logits        : (B, num_classes) — next-activity prediction
#   2. reconstruction: (B, N_NUMERICAL) — numerical feature reconstruction
#
# Reconstruction head fix:
#   Output size is N_NUMERICAL (7), not input_dim (14)
#   Sigmoid bounds output to [0,1] matching min-max scaled inputs
#   This prevents MSE explosion from unbounded categorical integers
#
# Composite anomaly score (07_scoring.py):
#   score = alpha * confidence + beta * recon_error
#   confidence  = 1 - max(softmax(logits))
#   recon_error = weighted MSE(recon, numerical_features)
# ============================================================

import math
import torch
import torch.nn as nn

from config import (
    LSTM_CONFIG,
    GRU_CONFIG,
    TRANSFORMER_CONFIG,
    RECON_HIDDEN_DIM,
    N_NUMERICAL,
    DEVICE,
)


# ============================================================
# Reconstruction head — outputs N_NUMERICAL values in [0,1]
# ============================================================

class ReconstructionHead(nn.Module):
    """
    MLP reconstruction head.
    Maps hidden representation -> reconstructed numerical features.

    Input:  (B, hidden_dim)
    Output: (B, N_NUMERICAL=7) — all values in [0,1] via Sigmoid

    Sigmoid ensures output matches min-max scaled numerical inputs
    exactly, keeping MSE bounded between 0 and 1 per feature.
    """
    def __init__(self, hidden_dim: int,
                 recon_hidden: int = RECON_HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, recon_hidden),
            nn.ReLU(),
            nn.Linear(recon_hidden, N_NUMERICAL),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================
# LSTM Model (Section 5.3.1)
# ============================================================

class LSTMModel(nn.Module):
    """
    2-layer LSTM with reconstruction head.

    Forward returns:
        logits        : (B, num_classes)
        reconstruction: (B, N_NUMERICAL=7) — bounded to [0,1]
    """
    def __init__(self, input_dim: int, hidden_dim: int,
                 num_layers: int, num_classes: int,
                 dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size  = input_dim,
            hidden_size = hidden_dim,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
        )
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.recon_head = ReconstructionHead(hidden_dim)

    def forward(self, x: torch.Tensor):
        lstm_out, _ = self.lstm(x)
        last        = lstm_out[:, -1, :]
        last        = self.dropout(last)
        logits      = self.classifier(last)
        recon       = self.recon_head(last)
        return logits, recon


# ============================================================
# GRU Model (Section 5.3.2)
# ============================================================

class GRUModel(nn.Module):
    """
    2-layer GRU with reconstruction head.

    Forward returns:
        logits        : (B, num_classes)
        reconstruction: (B, N_NUMERICAL=7) — bounded to [0,1]
    """
    def __init__(self, input_dim: int, hidden_dim: int,
                 num_layers: int, num_classes: int,
                 dropout: float = 0.3):
        super().__init__()
        self.gru = nn.GRU(
            input_size  = input_dim,
            hidden_size = hidden_dim,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
        )
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.recon_head = ReconstructionHead(hidden_dim)

    def forward(self, x: torch.Tensor):
        gru_out, _ = self.gru(x)
        last       = gru_out[:, -1, :]
        last       = self.dropout(last)
        logits     = self.classifier(last)
        recon      = self.recon_head(last)
        return logits, recon


# ============================================================
# Positional Encoding for Transformer (Section 5.3.3)
# ============================================================

class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding from Vaswani et al. (2017).
    Adds position information since self-attention has no inherent order.
    """
    def __init__(self, d_model: int, max_len: int = 100,
                 dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe       = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ============================================================
# Transformer Model (Section 5.3.3)
# ============================================================

class TransformerModel(nn.Module):
    """
    Transformer encoder with reconstruction head.

    Reduces sequential ops from O(n) to O(1) per thesis Section 5.3.3.
    O(n^2) parallel attention cost negligible for n <= 26.

    Forward returns:
        logits        : (B, num_classes)
        reconstruction: (B, N_NUMERICAL=7) — bounded to [0,1]
    """
    def __init__(self, input_dim: int, d_model: int,
                 nhead: int, num_layers: int,
                 dim_feedforward: int, num_classes: int,
                 dropout: float = 0.1):
        super().__init__()
        self.input_proj   = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(
            d_model, max_len=100, dropout=dropout
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = d_model,
            nhead           = nhead,
            dim_feedforward = dim_feedforward,
            dropout         = dropout,
            batch_first     = True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)
        self.recon_head = ReconstructionHead(d_model)

    def forward(self, x: torch.Tensor):
        x      = self.input_proj(x)
        x      = self.pos_encoding(x)
        x      = self.transformer(x)
        last   = x[:, -1, :]
        last   = self.dropout(last)
        logits = self.classifier(last)
        recon  = self.recon_head(last)
        return logits, recon


# ============================================================
# Model factory
# ============================================================

def build_model(model_name: str, num_classes: int) -> nn.Module:
    """
    Instantiate a model by name using config hyperparameters.

    Parameters
    ----------
    model_name  : "lstm", "gru", or "transformer"
    num_classes : number of activity classes

    Returns
    -------
    nn.Module moved to DEVICE
    """
    name = model_name.lower()

    if name == "lstm":
        model = LSTMModel(
            input_dim   = LSTM_CONFIG["input_dim"],
            hidden_dim  = LSTM_CONFIG["hidden_dim"],
            num_layers  = LSTM_CONFIG["num_layers"],
            num_classes = num_classes,
            dropout     = LSTM_CONFIG["dropout"],
        )
    elif name == "gru":
        model = GRUModel(
            input_dim   = GRU_CONFIG["input_dim"],
            hidden_dim  = GRU_CONFIG["hidden_dim"],
            num_layers  = GRU_CONFIG["num_layers"],
            num_classes = num_classes,
            dropout     = GRU_CONFIG["dropout"],
        )
    elif name == "transformer":
        model = TransformerModel(
            input_dim       = TRANSFORMER_CONFIG["input_dim"],
            d_model         = TRANSFORMER_CONFIG["d_model"],
            nhead           = TRANSFORMER_CONFIG["nhead"],
            num_layers      = TRANSFORMER_CONFIG["num_layers"],
            dim_feedforward = TRANSFORMER_CONFIG["dim_feedforward"],
            num_classes     = num_classes,
            dropout         = TRANSFORMER_CONFIG["dropout"],
        )
    else:
        raise ValueError(
            f"Unknown model: '{model_name}'. "
            "Choose from: lstm, gru, transformer"
        )

    model = model.to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Built {model_name.upper()}: {total_params:,} parameters")
    return model


# ============================================================
# Standalone verification
# ============================================================

if __name__ == "__main__":
    import joblib
    import os
    from config import PREPROCESSED_DIR, MODEL_INPUT_DIM, N_NUMERICAL

    print("=" * 60)
    print("04_models.py — Architecture Verification")
    print("=" * 60)

    encoders    = joblib.load(
        os.path.join(PREPROCESSED_DIR, "label_encoders.joblib")
    )
    num_classes = len(encoders["concept:name"].classes_)

    print(f"\n  num_classes  : {num_classes}")
    print(f"  input_dim    : {MODEL_INPUT_DIM}")
    print(f"  n_numerical  : {N_NUMERICAL}")
    print(f"  device       : {DEVICE}")

    print("\nBuilding models...")
    lstm        = build_model("lstm",        num_classes)
    gru         = build_model("gru",         num_classes)
    transformer = build_model("transformer", num_classes)

    print("\nForward pass test (batch=4, seq=26, features=14)...")
    dummy = torch.zeros(4, 26, MODEL_INPUT_DIM).to(DEVICE)

    for name, model in [("LSTM", lstm),
                         ("GRU",  gru),
                         ("Transformer", transformer)]:
        model.eval()
        with torch.no_grad():
            logits, recon = model(dummy)
        print(f"  {name}:")
        print(f"    logits shape : {logits.shape}  "
              f"(expected [4, {num_classes}])")
        print(f"    recon shape  : {recon.shape}  "
              f"(expected [4, {N_NUMERICAL}])")
        print(f"    recon range  : [{recon.min():.4f}, {recon.max():.4f}]  "
              f"(expected [0,1] from Sigmoid)")
        assert logits.shape == (4, num_classes)
        assert recon.shape  == (4, N_NUMERICAL)
        assert recon.min()  >= 0.0 and recon.max() <= 1.0
        print(f"    All checks passed")

    print("\n  All three models verified.")
    print("=" * 60)