import numpy as np
import os
BEST_INITIAL_COMBINATIONS = { # Per the paper, page 8
    "convergence": {"alpha": 0.09, "beta": 1.10, "gamma": 0.2},
    "overall": {"alpha": 1.05, "beta": 1.35, "gamma": 1.25}
}
PROJECT_NAME = "deep-agla-test-local"
DATA_PATH = "/gpfs0/bgu-benshimo/users/guyperet/DeepAGLA/data/audio_gold_auto.npy"
GOLD_SHORT_DATA_PATH = "/gpfs0/bgu-benshimo/users/guyperet/DeepAGLA/data/audio_gold.npy"
if os.path.exists(DATA_PATH):
    EXAMPLE_AUDIO = np.load(DATA_PATH)[0]  # Load a single example for testing

SAMPLE_RATE = 22050 # Half of 44.1 kHz, as loaded by librosa
N_FFT = 512
HOP = N_FFT // 4
WIN_LEN = N_FFT

VAL_SPLIT = 0.05
TEST_SPLIT = 0.05
N_LAYERS = 64
LEARNING_RATE = 0.008
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 8
NUM_WORKERS = 8
NUM_EPOCHS = 200