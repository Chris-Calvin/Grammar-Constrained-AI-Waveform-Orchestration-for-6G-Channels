"""
constants.py - Central configuration constants for the Cognitive Waveform Orchestration System.

Loads all hyperparameters from config.yaml and exposes them as module-level constants.
"""

import os
import yaml

# ---------------------------------------------------------------------------
# Load config.yaml
# ---------------------------------------------------------------------------
_CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
_CONFIG_PATH = os.path.join(_CONFIG_DIR, "config.yaml")

with open(_CONFIG_PATH, "r") as _f:
    CONFIG = yaml.safe_load(_f)

# ---------------------------------------------------------------------------
# Random Seed
# ---------------------------------------------------------------------------
RANDOM_SEED = CONFIG["random_seed"]

# ---------------------------------------------------------------------------
# Vocabulary Sizes
# ---------------------------------------------------------------------------
VOCAB_SIZES = CONFIG["vocab_sizes"]

# ---------------------------------------------------------------------------
# Transformer Architecture
# ---------------------------------------------------------------------------
N_LAYERS = CONFIG["transformer"]["n_layers"]
N_HEADS = CONFIG["transformer"]["n_heads"]
EMBED_DIM = CONFIG["transformer"]["embed_dim"]
FF_DIM = CONFIG["transformer"]["ff_dim"]
DROPOUT = CONFIG["transformer"]["dropout"]
MAX_SEQ_LEN = CONFIG["transformer"]["max_seq_len"]
ACTIVATION = CONFIG["transformer"]["activation"]

TRANSFORMER_CONFIG = CONFIG["transformer"]

# ---------------------------------------------------------------------------
# Training Parameters
# ---------------------------------------------------------------------------
LEARNING_RATE = CONFIG["training"]["learning_rate"]
BATCH_SIZE = CONFIG["training"]["batch_size"]
EPOCHS = CONFIG["training"]["epochs"]
PATIENCE = CONFIG["training"]["patience"]
WEIGHT_DECAY = CONFIG["training"]["weight_decay"]
SCHEDULER = CONFIG["training"]["scheduler"]
WARMUP_STEPS = CONFIG["training"]["warmup_steps"]
GRADIENT_CLIP = CONFIG["training"]["gradient_clip"]

TRAINING_CONFIG = CONFIG["training"]

# ---------------------------------------------------------------------------
# Dataset Parameters
# ---------------------------------------------------------------------------
TOTAL_SAMPLES = CONFIG["dataset"]["total_samples"]
TRAIN_SPLIT = CONFIG["dataset"]["train_split"]
VAL_SPLIT = CONFIG["dataset"]["val_split"]
TEST_SPLIT = CONFIG["dataset"]["test_split"]
BOUNDARY_SAMPLES = CONFIG["dataset"]["boundary_samples"]

DATASET_CONFIG = CONFIG["dataset"]

# ---------------------------------------------------------------------------
# Waveform Candidates
# ---------------------------------------------------------------------------
WAVEFORM_CANDIDATES = CONFIG["waveform_candidates"]
NUM_WAVEFORMS = len(WAVEFORM_CANDIDATES)

# ---------------------------------------------------------------------------
# Frequency Bands
# ---------------------------------------------------------------------------
FREQUENCY_BANDS = CONFIG["frequency_bands"]

# ---------------------------------------------------------------------------
# CDL Channel Model Parameters
# ---------------------------------------------------------------------------
CDL_MODELS = CONFIG["cdl_model"]["models"]
CDL_CONFIG = CONFIG["cdl_model"]

# ---------------------------------------------------------------------------
# THz Window Definitions
# ---------------------------------------------------------------------------
THZ_WINDOWS = CONFIG["thz_windows"]
THZ_WINDOW1 = CONFIG["thz_windows"]["window1"]
THZ_WINDOW2 = CONFIG["thz_windows"]["window2"]
THZ_WINDOW3 = CONFIG["thz_windows"]["window3"]

# ---------------------------------------------------------------------------
# Project Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(_CONFIG_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
RESULTS_DIR = os.path.join(DATA_DIR, "results")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
