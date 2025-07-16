import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, Any, Tuple

# Set multiprocessing start method to avoid DataLoader conflicts
import torch.multiprocessing as mp
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set

from ax.service.managed_loop import optimize
from train_new import main as train_once

# ---------------------------------------------------------------------------
# 1. Search space definition
# ---------------------------------------------------------------------------
PARAMS = [
    {"name": "lr", "type": "range", "bounds": [1e-4, 1e-2], "value_type": "float", "log_scale": True},
    {"name": "batch_size", "type": "choice", "values": [4, 8, 16], "value_type": "int", "is_ordered": True, "sort_values": True},
    {"name": "weight_decay", "type": "range", "bounds": [1e-5, 1e-2], "value_type": "float", "log_scale": True}
]

# ---------------------------------------------------------------------------
# 2. Single‑trial wrapper
# ---------------------------------------------------------------------------

def run_one_trial_internal(parameterization, run_name=None) -> Dict[str, float]:
    """Run a single optimization trial with the given parameters."""
    try:
        # Create args for the trial
        tag = f"trial_lr{parameterization['lr']}_bs{parameterization['batch_size']}_wd{parameterization['weight_decay']}"
        out_fold = os.path.join("/gpfs0/bgu-benshimo/users/guyperet/DeepAGLA/ax_trials", tag)
        os.makedirs(os.path.join(out_fold, "checkpoint"), exist_ok=True)
        
        print(f"Running trial: {tag}")
        print(f"Parameters: {parameterization}")
        
        # Run training
        ssnr_best = train_once(hparams=parameterization, run_name=f"ax_trial_{tag}")
        
        print(f"Trial {tag} completed with SSNR: {ssnr_best:.3f}")
        return {"ssnr_best": ssnr_best} # Maximize!
        
    except Exception as e:
        print(f"Trial failed with parameters {parameterization}: {e}")
        return {"ssnr_best": float('-inf')}  # Return worst score for failed trials
    


# ---------------------------------------------------------------------------
# 3. Ax optimisation loop
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--run_name", type=str, default=None,
                        help="WandB run name (optional)")
    args = parser.parse_args(argv)

    # Wrap Ax evaluation to inject run_name without signature conflicts
    def evaluation_function(parameterization, weight):
        return run_one_trial_internal(parameterization, run_name=None)

    best_params, best_vals, *_ = optimize(
        parameters=PARAMS,
        total_trials=args.trials,
        objective_name = "ssnr_best",  # Fixed: was "snr_best" but returns "ssnr_best"
        random_seed=42,
        evaluation_function=evaluation_function
        )
    

    print("\n===  Ax optimisation finished  ===")
    print("Best parameters:")
    for k, v in best_params.items():
        print(f"  {k}: {v:.4f}")
    print(f"Best: {best_vals}")


if __name__ == "__main__":
    main()