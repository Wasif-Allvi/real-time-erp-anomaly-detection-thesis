# ============================================================
# config.py
# Thesis: Deep Sequential Models for ERP Anomaly Detection
# Mustafa Wasif Allvi — University of Potsdam, M.Sc. Data Science
# ============================================================

import os
import torch

# ------------------------------------------------------------
# 1. PATHS
# ------------------------------------------------------------

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "..", "dataset")

XES_DOMESTIC = os.path.join(
    DATA_DIR,
    "BPI Challenge 2020_ Domestic Declarations_1_all",
    "DomesticDeclarations.xes",
    "DomesticDeclarations.xes"
)
XES_INTERNATIONAL = os.path.join(
    DATA_DIR,
    "BPI Challenge 2020_ International Declarations_1_all",
    "InternationalDeclarations.xes",
    "InternationalDeclarations.xes"
)

RESULTS_DIR      = os.path.join(ROOT_DIR, "results")
MODELS_DIR       = os.path.join(RESULTS_DIR, "models")
METRICS_DIR      = os.path.join(RESULTS_DIR, "metrics")
PREPROCESSED_DIR = os.path.join(RESULTS_DIR, "preprocessed")

SCALER_PATH      = os.path.join(PREPROCESSED_DIR, "num_scaler.joblib")
ENCODER_PATH     = os.path.join(PREPROCESSED_DIR, "label_encoders.joblib")
PROCESSED_CSV    = os.path.join(PREPROCESSED_DIR, "events_processed.csv")
SPLIT_IDS_PATH   = os.path.join(PREPROCESSED_DIR, "split_case_ids.joblib")
RAW_COMBINED_CSV = os.path.join(PREPROCESSED_DIR, "events_raw_combined.csv")

# ------------------------------------------------------------
# 2. REPRODUCIBILITY
# ------------------------------------------------------------

RANDOM_SEED = 42

# ------------------------------------------------------------
# 3. DATASET / FEATURE SCHEMA
# ------------------------------------------------------------

ACTIVITY_COL  = "concept:name"
CASE_ID_COL   = "case:concept:name"
TIMESTAMP_COL = "time:timestamp"
JOIN_KEY_COL  = "case:Permit ID"
SOURCE_COL    = "dataset_source"

CATEGORICAL_FEATURES = [
    "concept:name",
    "org:resource",
    "org:role",
    "case:Permit OrganizationalEntity",
    "case:Permit ProjectNumber",
    "case:Permit BudgetNumber",
    "case:BudgetNumber",
]

NUMERICAL_FEATURES = [
    "case:Amount",
    "case:RequestedAmount",
    "case:AdjustedAmount",
    "case:OriginalAmount",
    "case:Permit RequestedBudget",
    "delta_t",
    "cum_dur",
]

LOG_TRANSFORM_FEATURES = ["delta_t", "cum_dur"]

CATEGORICAL_FILL_TOKEN = "UNKNOWN"
NUMERICAL_FILL_VALUE   = 0.0

# Feature layout in the 14-dimensional input vector:
#   Indices 0-6  : categorical encoded integers (NOT bounded)
#   Indices 7-13 : numerical min-max scaled to [0,1]
N_CATEGORICAL    = 7
N_NUMERICAL      = 7
MODEL_INPUT_DIM  = 14
MIN_TRACE_LENGTH = 2

# ------------------------------------------------------------
# 4. SEQUENCE MODELING
# ------------------------------------------------------------

MAX_PREFIX_LENGTH = 26

# ------------------------------------------------------------
# 5. TRAIN / VAL / TEST SPLIT
# ------------------------------------------------------------

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.10
TEST_RATIO  = 0.20

# ------------------------------------------------------------
# 6. TRAINING HYPERPARAMETERS
# ------------------------------------------------------------

BATCH_SIZE    = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY  = 1e-5
NUM_RUNS      = 3

# Learning rate scheduler (ReduceLROnPlateau)
# Monitors validation loss — reduces LR when val loss stops improving
USE_LR_SCHEDULER = True
LR_PATIENCE      = 5
LR_FACTOR        = 0.5
LR_MIN           = 1e-5

# ------------------------------------------------------------
# 7. MODEL ARCHITECTURES
# ------------------------------------------------------------

LSTM_CONFIG = {
    "input_dim"  : MODEL_INPUT_DIM,
    "hidden_dim" : 128,
    "num_layers" : 2,
    "dropout"    : 0.3,
}

GRU_CONFIG = {
    "input_dim"  : MODEL_INPUT_DIM,
    "hidden_dim" : 128,
    "num_layers" : 2,
    "dropout"    : 0.3,
}

TRANSFORMER_CONFIG = {
    "input_dim"       : MODEL_INPUT_DIM,
    "d_model"         : 64,
    "nhead"           : 4,
    "num_layers"      : 2,
    "dim_feedforward" : 128,
    "dropout"         : 0.1,
}

# ------------------------------------------------------------
# 8. JOINT RECONSTRUCTION SCORING
# ------------------------------------------------------------
# Reconstruction targets NUMERICAL features only (indices 7-13)
# Sigmoid on output head bounds reconstruction to [0,1]
# Financial features weighted 2x to amplify anomaly signal

COMPOSITE_ALPHA  = 0.5
COMPOSITE_BETA   = 0.5
RECON_HIDDEN_DIM = 64
ALPHA_GRID       = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

LAMBDA_RECON = 0.5

# Financial feature indices within NUMERICAL slice (zero-based)
# Slice = features[7:14], financial at slice positions 0-4
FINANCIAL_INDICES_IN_NUMERICAL = [0, 1, 2, 3, 4]
FINANCIAL_RECON_WEIGHT         = 2.0

# ------------------------------------------------------------
# 9. ANOMALY INJECTION
# ------------------------------------------------------------

ANOMALY_FRACTION    = 0.10
ANOMALY_RANDOM_SEED = 2024

ANOMALY_TYPES = [
    "control_flow",
    "data",
    "resource",
    "temporal",
    "cross_entity",
]

ANOMALY_PARAMS = {
    "data"         : {"amount_multiplier": 10.0},
    "temporal"     : {"gap_multiplier": 50.0},
    "cross_entity" : {"amount_multiplier": 5.0},
}

# ------------------------------------------------------------
# 10. EVALUATION
# ------------------------------------------------------------

LATENCY_WARMUP_RUNS   = 10
LATENCY_MEASURE_RUNS  = 100
THROUGHPUT_BATCH_SIZE = 64
THROUGHPUT_BATCHES    = 50

# ------------------------------------------------------------
# 11. DEVICE
# ------------------------------------------------------------

DEVICE = torch.device("cpu")

# ------------------------------------------------------------
# 12. LOGGING
# ------------------------------------------------------------

LOG_INTERVAL = 5

# ------------------------------------------------------------
# Helper
# ------------------------------------------------------------

def make_output_dirs():
    for d in [RESULTS_DIR, MODELS_DIR, METRICS_DIR, PREPROCESSED_DIR]:
        os.makedirs(d, exist_ok=True)