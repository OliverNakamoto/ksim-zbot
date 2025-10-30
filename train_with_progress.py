#!/usr/bin/env python
"""Training script with compilation progress logging."""

import os
import sys

# Enable JAX compilation logging BEFORE importing JAX
os.environ["JAX_LOG_COMPILES"] = "1"
os.environ["JAX_LOGGING_LEVEL"] = "INFO"

# Add a compilation progress callback
import jax
jax.config.update("jax_log_compiles", True)

print("=" * 80)
print("🚀 Starting training with compilation progress logging enabled")
print("=" * 80)
print()
print("📊 You will see messages like:")
print("   - 'Compiling <function_name>...' when JAX starts compiling")
print("   - Compilation logs from XLA optimizer")
print()
print("=" * 80)
print()

# Now import and run the actual training
from train import ZbotWalkingTask, ZbotWalkingTaskConfig

if __name__ == "__main__":
    ZbotWalkingTask.launch(
        ZbotWalkingTaskConfig(
            # Training parameters.
            num_envs=64,
            batch_size=8,
            learning_rate=1e-3,
            num_passes=4,
            # epochs_per_log_step=1,
            rollout_length_seconds=8.0,
            # Simulation parameters.
            dt=0.001,
            ctrl_dt=0.02,
            iterations=8,
            ls_iterations=8,
            # Checkpointing parameters.
            save_every_n_seconds=60,
            valid_every_n_steps=5,
            # render_full_every_n_seconds=10,
            render_azimuth=145.0,
            action_latency_range=(0.003, 0.10),
        ),
    )
