"""Debug script to inspect network predictions for the right knee joint."""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle

from train_standstill import ZbotWalkingTask, ZbotWalkingTaskConfig, JOINT_BIASES

def load_checkpoint(ckpt_path: str):
    """Load model checkpoint."""
    print(f"Loading checkpoint from: {ckpt_path}")
    with open(ckpt_path, "rb") as f:
        checkpoint = pickle.load(f)
    return checkpoint

def analyze_network_predictions(ckpt_path: str, num_samples: int = 10000):
    """Analyze what the network predicts for right knee joint."""

    # Create task and load checkpoint
    config = ZbotWalkingTaskConfig(
        num_envs=1,
        batch_size=1,
    )
    task = ZbotWalkingTask(config)

    # Load checkpoint
    checkpoint = load_checkpoint(ckpt_path)
    model = checkpoint["model"]

    # Get physics model
    physics_model = task.get_physics_model()

    # Create a dummy initial state
    key = jax.random.PRNGKey(0)
    initial_state = task.get_initial_physics_state(physics_model, key)

    # Get observations for a standing pose
    key, obs_key = jax.random.split(key)
    obs_input = task.get_observation_input(
        physics_model=physics_model,
        physics_state=initial_state,
        command={cmd.get_name(): cmd.initial_command(initial_state.data, jnp.array(0.0), obs_key)
                 for cmd in task.get_commands(physics_model)},
        obs_carry=task.get_initial_obs_carry(obs_key),
    )
    observations = {k: v.observe(obs_input, jnp.array(0.0), obs_key) for k, v in task.observations.items()}

    # Get commands (zero commands for standing)
    commands = {cmd.get_name(): cmd.initial_command(initial_state.data, jnp.array(0.0), obs_key)
                for cmd in task.get_commands(physics_model)}

    # Get initial model carry
    model_carry = task.get_initial_model_carry(key)
    actor_carry = model_carry[0]

    # Run actor to get distribution
    key, actor_key = jax.random.split(key)
    action_dist, _ = task.run_actor(
        model=model.actor,
        observations=observations,
        commands=commands,
        carry=actor_carry,
        rng=actor_key,
    )

    # Extract information about the right knee (index 3)
    RIGHT_KNEE_IDX = 3
    right_knee_bias = JOINT_BIASES[RIGHT_KNEE_IDX][1]
    right_knee_name = JOINT_BIASES[RIGHT_KNEE_IDX][0]
    right_knee_range = (-4.712389, -1.570796)  # From the code

    print("\n" + "=" * 80)
    print(f"ANALYZING: {right_knee_name} (index {RIGHT_KNEE_IDX})")
    print("=" * 80)
    print(f"Joint bias: {right_knee_bias:.4f}")
    print(f"Joint range: [{right_knee_range[0]:.4f}, {right_knee_range[1]:.4f}]")

    # Access the mixture distribution internals
    # action_dist is a MixtureOfGaussians with:
    # - means_nm: shape (NUM_JOINTS, num_mixtures)
    # - stds_nm: shape (NUM_JOINTS, num_mixtures)
    # - logits_nm: shape (NUM_JOINTS, num_mixtures)

    means_for_joint = action_dist.components_distribution.loc[RIGHT_KNEE_IDX, :]  # shape (num_mixtures,)
    stds_for_joint = action_dist.components_distribution.scale[RIGHT_KNEE_IDX, :]  # shape (num_mixtures,)
    logits_for_joint = action_dist.mixture_distribution.logits[RIGHT_KNEE_IDX, :]  # shape (num_mixtures,)

    # Compute mixture weights
    weights = jax.nn.softmax(logits_for_joint)

    print(f"\n{'Mixture':<10} {'Weight':<12} {'Mean (final)':<15} {'Raw Mean':<15} {'Std':<10}")
    print("-" * 80)
    for i in range(len(weights)):
        raw_mean = means_for_joint[i] - right_knee_bias
        print(f"{i:<10} {float(weights[i]):<12.4f} {float(means_for_joint[i]):<15.4f} "
              f"{float(raw_mean):<15.4f} {float(stds_for_joint[i]):<10.4f}")

    # Find dominant mixture
    dominant_idx = int(jnp.argmax(weights))
    print(f"\nDominant mixture: {dominant_idx} (weight: {float(weights[dominant_idx]):.4f})")
    print(f"Dominant mixture mean: {float(means_for_joint[dominant_idx]):.4f}")

    # Sample many actions
    print(f"\n{'=' * 80}")
    print(f"SAMPLING {num_samples} ACTIONS")
    print("=" * 80)

    # Sample stochastically
    keys = jax.random.split(key, num_samples)
    stochastic_samples = jax.vmap(lambda k: action_dist.sample(seed=k)[RIGHT_KNEE_IDX])(keys)
    stochastic_samples = np.array(stochastic_samples)

    # Get deterministic (mode)
    deterministic_action = action_dist.mode()[RIGHT_KNEE_IDX]

    print(f"\nDeterministic action (mode): {float(deterministic_action):.6f}")
    print(f"\nStochastic samples statistics:")
    print(f"  Mean:   {np.mean(stochastic_samples):.6f}")
    print(f"  Median: {np.median(stochastic_samples):.6f}")
    print(f"  Std:    {np.std(stochastic_samples):.6f}")
    print(f"  Min:    {np.min(stochastic_samples):.6f}")
    print(f"  Max:    {np.max(stochastic_samples):.6f}")

    # Check for exact values (bitwise equality)
    unique_values, counts = np.unique(stochastic_samples, return_counts=True)
    print(f"\nNumber of unique sampled values: {len(unique_values)}")

    # Check if there's clamping to limits
    at_lower_limit = np.sum(np.isclose(stochastic_samples, right_knee_range[0], atol=1e-9))
    at_upper_limit = np.sum(np.isclose(stochastic_samples, right_knee_range[1], atol=1e-9))

    print(f"\nSamples at lower limit ({right_knee_range[0]:.4f}): {at_lower_limit} ({100*at_lower_limit/num_samples:.2f}%)")
    print(f"Samples at upper limit ({right_knee_range[1]:.4f}): {at_upper_limit} ({100*at_upper_limit/num_samples:.2f}%)")

    if at_lower_limit > num_samples * 0.01 or at_upper_limit > num_samples * 0.01:
        print("\n⚠️  WARNING: >1% of samples are exactly at joint limits!")
        print("   This suggests the actions are being clipped somewhere.")

    # Check for bitwise equality
    if len(unique_values) < num_samples * 0.5:
        print(f"\n⚠️  WARNING: Only {len(unique_values)} unique values out of {num_samples} samples!")
        print("   Top 10 most common values:")
        top_indices = np.argsort(-counts)[:10]
        for idx in top_indices:
            val = unique_values[idx]
            cnt = counts[idx]
            print(f"   Value: {val:.6f}, Count: {cnt} ({100*cnt/num_samples:.2f}%)")

    # Plot histogram
    print("\n" + "=" * 80)
    print("GENERATING PLOTS")
    print("=" * 80)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Histogram of samples
    ax = axes[0, 0]
    ax.hist(stochastic_samples, bins=100, alpha=0.7, edgecolor='black')
    ax.axvline(float(deterministic_action), color='r', linestyle='--', linewidth=2, label='Deterministic (mode)')
    ax.axvline(right_knee_range[0], color='orange', linestyle=':', linewidth=2, label='Lower limit')
    ax.axvline(right_knee_range[1], color='purple', linestyle=':', linewidth=2, label='Upper limit')
    ax.set_xlabel('Action Value (radians)')
    ax.set_ylabel('Frequency')
    ax.set_title(f'{right_knee_name} - Sampled Actions Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Mixture components
    ax = axes[0, 1]
    x_range = np.linspace(right_knee_range[0] - 0.5, right_knee_range[1] + 0.5, 1000)
    for i in range(len(weights)):
        if weights[i] > 0.01:  # Only plot significant mixtures
            mean = float(means_for_joint[i])
            std = float(stds_for_joint[i])
            weight = float(weights[i])
            pdf = weight * np.exp(-0.5 * ((x_range - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))
            label = f'Mix {i} (w={weight:.2f})'
            linestyle = '--' if i == dominant_idx else '-'
            linewidth = 3 if i == dominant_idx else 1.5
            ax.plot(x_range, pdf, label=label, linestyle=linestyle, linewidth=linewidth)
    ax.axvline(right_knee_range[0], color='orange', linestyle=':', linewidth=2, alpha=0.5)
    ax.axvline(right_knee_range[1], color='purple', linestyle=':', linewidth=2, alpha=0.5)
    ax.set_xlabel('Action Value (radians)')
    ax.set_ylabel('Probability Density')
    ax.set_title(f'{right_knee_name} - Mixture Components (dominant in bold)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Cumulative distribution
    ax = axes[1, 0]
    sorted_samples = np.sort(stochastic_samples)
    ax.plot(sorted_samples, np.arange(len(sorted_samples)) / len(sorted_samples), linewidth=2)
    ax.axvline(float(deterministic_action), color='r', linestyle='--', linewidth=2, label='Deterministic')
    ax.axvline(right_knee_range[0], color='orange', linestyle=':', linewidth=2, label='Lower limit')
    ax.axvline(right_knee_range[1], color='purple', linestyle=':', linewidth=2, label='Upper limit')
    ax.set_xlabel('Action Value (radians)')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title(f'{right_knee_name} - CDF of Sampled Actions')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Mixture weights
    ax = axes[1, 1]
    mixture_indices = np.arange(len(weights))
    bars = ax.bar(mixture_indices, weights, alpha=0.7, edgecolor='black')
    bars[dominant_idx].set_color('red')
    ax.set_xlabel('Mixture Index')
    ax.set_ylabel('Weight')
    ax.set_title(f'{right_knee_name} - Mixture Weights (dominant in red)')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path = Path("debug_right_knee_predictions.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")

    # Check if actions would be clipped
    print("\n" + "=" * 80)
    print("CHECKING FOR ACTION CLIPPING")
    print("=" * 80)

    samples_below_limit = np.sum(stochastic_samples < right_knee_range[0])
    samples_above_limit = np.sum(stochastic_samples > right_knee_range[1])
    samples_in_range = np.sum((stochastic_samples >= right_knee_range[0]) &
                               (stochastic_samples <= right_knee_range[1]))

    print(f"Samples below lower limit: {samples_below_limit} ({100*samples_below_limit/num_samples:.2f}%)")
    print(f"Samples above upper limit: {samples_above_limit} ({100*samples_above_limit/num_samples:.2f}%)")
    print(f"Samples within valid range: {samples_in_range} ({100*samples_in_range/num_samples:.2f}%)")

    if samples_below_limit > 0 or samples_above_limit > 0:
        print(f"\n⚠️  WARNING: {samples_below_limit + samples_above_limit} samples ({100*(samples_below_limit + samples_above_limit)/num_samples:.2f}%) are outside valid range!")
        print("   These actions would need to be clipped or would hit joint limits in simulation.")
        print("   This means the network is predicting invalid actions.")
    else:
        print("\n✅ All sampled actions are within valid joint range.")
        print("   The network has learned to respect joint limits.")

    return {
        'means': means_for_joint,
        'stds': stds_for_joint,
        'weights': weights,
        'samples': stochastic_samples,
        'deterministic': deterministic_action,
    }

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python debug_network_predictions.py <checkpoint_path> [num_samples]")
        print("Example: python debug_network_predictions.py zbot_walking_task/run_123/checkpoints/ckpt.bin 10000")
        sys.exit(1)

    ckpt_path = sys.argv[1]
    num_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 10000

    results = analyze_network_predictions(ckpt_path, num_samples)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nTo visualize in the viewer:")
    print(f"  Deterministic: python train_standstill.py run_mode=view load_from_ckpt_path={ckpt_path} viewer_argmax_action=True")
    print(f"  Stochastic:    python train_standstill.py run_mode=view load_from_ckpt_path={ckpt_path} viewer_argmax_action=False")
