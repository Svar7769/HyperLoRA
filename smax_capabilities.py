"""
Extract and normalize SMAX unit capabilities for use as context features.
Instead of one-hot encoding, we use actual unit stats (health, attack, speed, etc.)
which provides more meaningful information for the hypernetwork.
"""

import jax
import jax.numpy as jnp


def get_unit_capabilities(env, unit_types):
    """
    Extract normalized capability features for each unit based on their type.

    Args:
        env: SMAX or HeuristicEnemySMAX environment instance
        unit_types: Array of unit type indices [num_units]

    Returns:
        capability_features: Array of shape [num_units, feature_dim] containing:
            - health: Maximum health points (normalized)
            - attack: Attack damage (normalized)
            - attack_range: Attack range (normalized)
            - velocity: Movement speed (normalized)
            - sight_range: Vision range (normalized)
            - radius: Unit collision radius (normalized)
            - weapon_cooldown: Time between attacks (normalized, inverted so higher = faster)
    """
    # Get the wrapped SMAX environment
    smax_env = env._env if hasattr(env, "_env") else env

    # Extract unit type stats from environment
    health = smax_env.unit_type_health[unit_types]
    attack = smax_env.unit_type_attacks[unit_types]
    attack_range = smax_env.unit_type_attack_ranges[unit_types]
    velocity = smax_env.unit_type_velocities[unit_types]
    sight_range = smax_env.unit_type_sight_ranges[unit_types]
    radius = smax_env.unit_type_radiuses[unit_types]
    weapon_cooldown = smax_env.unit_type_weapon_cooldowns[unit_types]

    # Normalize features to [0, 1] range based on max values across all unit types
    # This ensures features are on similar scales for the neural network
    health_norm = health / jnp.max(smax_env.unit_type_health)
    attack_norm = attack / jnp.max(smax_env.unit_type_attacks)
    attack_range_norm = attack_range / jnp.max(smax_env.unit_type_attack_ranges)
    velocity_norm = velocity / jnp.max(smax_env.unit_type_velocities)
    sight_range_norm = sight_range / jnp.max(smax_env.unit_type_sight_ranges)
    radius_norm = radius / jnp.max(smax_env.unit_type_radiuses)

    # Invert weapon cooldown so higher value = faster attack (more useful for learning)
    # Then normalize
    max_cooldown = jnp.max(smax_env.unit_type_weapon_cooldowns)
    attack_speed_norm = (max_cooldown - weapon_cooldown) / max_cooldown

    # Stack all features into a single capability vector [num_units, 7]
    capability_features = jnp.stack(
        [
            health_norm,
            attack_norm,
            attack_range_norm,
            velocity_norm,
            sight_range_norm,
            radius_norm,
            attack_speed_norm,
        ],
        axis=1,
    )

    return capability_features


def get_capability_dim():
    """Return the dimensionality of capability features."""
    return (
        7  # health, attack, attack_range, velocity, sight_range, radius, attack_speed
    )


def get_capability_names():
    """Return names of capability features for logging/debugging."""
    return [
        "health",
        "attack",
        "attack_range",
        "velocity",
        "sight_range",
        "radius",
        "attack_speed",
    ]
