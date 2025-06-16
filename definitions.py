import torch

BEST_INITIAL_COMBINATIONS = { # Per the paper, page 8
    "convergence": {"alpha": 0.09, "beta": 1.10, "gamma": 0.2},
    "overall": {"alpha": 1.05, "beta": 1.35, "gamma": 1.25}
}

SAMPLE_RATE = 44100
N_FFT = 2048
HOP = N_FFT // 8
N_LIN = 1 + N_FFT // 2

TRAIN_RATIO = 0.85
VAL_RATIO = 0.1

N_LAYERS = 8
LEARNING_RATE = 2e-3
WEIGHT_DECAY = 1e-6
BATCH_SIZE = 8
NUM_WORKERS = 8
NUM_EPOCHS = 100