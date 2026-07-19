# Wind Flocking Implementation Summary

## Overview

Successfully implemented a wind flocking scenario with position-based actions, hypernetwork-driven diversity control, and SND-based formation experiments.

## What Was Implemented

### 1. **New Scenario: `wind_flocking_position.py`**

**Key Features:**
- **Position-based actions**: Agents output normalized positions [-1, 1] that are converted to forces using a proportional controller
- **Configurable agent radii**: Supports N agents with different radii (default: 2 agents with radii [0.05, 0.03])
- **Wind shielding mechanism**: Larger agents can shield smaller ones when positioned upstream

**Reward Components:**
- **Energy reward** (`-distance_moved`): Penalizes movement, encourages staying in place
- **Wind reward** (`-wind_exposure`): Penalizes wind exposure after shielding
- **Formation shaping reward**: Rewards agents for aligning with wind direction (small agents downstream of big agent)
  - Uses cosine similarity between (small_pos - big_pos) and wind_direction
  - Perfect alignment (0°): reward = +1
  - Perpendicular (90°): reward = 0
  - Opposite (180°): reward = -1

**Position Controller:**
```python
desired_velocity = k_p * (target_position - current_position)
# Clip to max_speed and apply to agent
```

**Observation Space:**
For 2 agents: `pos(2) + vel(2) + rel_pos_to_other(2) + wind(2) = 8 dims`
For N agents: `pos(2) + vel(2) + rel_pos_to_others(2*(N-1)) + wind(2)`

### 2. **Configuration File: `config_wind_flocking.yaml`**

**Key Parameters:**
```yaml
env:
  scenario_name: "wind_flocking_position"
  num_agents: 2
  wind: 2.0  # Wind magnitude
  agent_radii: [0.05, 0.03]  # Big agent shields small agent
  
  # Reward weights
  energy_reward_weight: 1.0
  wind_reward_weight: 1.0
  formation_shaping_weight: 0.5
  
  # Position controller
  position_gain: 2.0  # Proportional gain
  max_speed: 0.5  # Max velocity
  position_range: 5.0  # Action bounds: [-5, 5]

model:
  use_agent_position_context: true  # Feed initial positions to hypernetwork
  agent_position_dim: 2  # x, y coordinates
  
  use_target_snd_context: true  # Feed target SND to hypernetwork
  target_snd_dim: 1  # Scalar target SND value

training:
  use_diversity_control: true
  target_snd: 0.5  # Target diversity level

evaluation:
  eval_snds: [0.3, 0.5, 0.7, 1.0, 1.5]  # Test different formations
```

### 3. **Hypernetwork Integration**

**Inputs to Hypernetwork:**
1. **Initial agent positions** (relative to center of mass):
   - Ensures each agent gets different inputs → non-zero adapter SND
   - Shape: `(num_envs, num_agents, 2)`
   - Computed at episode start and passed to hypernetwork

2. **Target SND value**:
   - Controls desired behavioral diversity
   - Shape: `(num_envs, num_agents, 1)`
   - Low SND → tight formation, High SND → spread out

**Updated `train.py`:**
- Modified `extract_agent_positions()` to:
  - Support both `dispersion_vmas` and `wind_flocking_position`
  - Compute RELATIVE positions (relative to center of mass)
  - Return shape: `(num_envs, num_agents, 2)`

### 4. **Environment Setup: `env_setup.py`**

Added handler for `wind_flocking_position` scenario that:
- Dynamically loads the custom scenario
- Passes all configuration parameters
- Supports variable number of agents and radii

### 5. **Test Script: `test_wind_flocking.py`**

**Verifies:**
- ✓ Environment loads correctly
- ✓ Observations have expected shape
- ✓ Position controller works
- ✓ Rewards computed correctly (energy, wind, formation)
- ✓ Agent positions extracted correctly (relative to center of mass)
- ✓ Supports 2, 3, and 4 agents

**Test Results:**
```
All tests passed! ✓
- 2 agents: obs_dim=8
- 3 agents: obs_dim=10
- 4 agents: obs_dim=12
```

## How It Works

### Training Flow

1. **Episode Start**:
   - Agents spawn in a line perpendicular to wind
   - Initial positions extracted and converted to relative positions
   - Target SND value sampled/specified
   - Hypernetwork generates adapters based on (initial_positions, target_SND)

2. **During Episode**:
   - Agents output position commands (normalized actions)
   - Position controller converts to velocities/forces
   - Agents experience wind (modified by shielding)
   - Rewards:
     - Stay in place (low energy) vs. move to shield position (low wind)
     - Alignment bonus for downstream positioning

3. **Learning**:
   - Policy learns to balance energy cost vs. wind reduction
   - Hypernetwork learns to generate adapters that produce different formations based on target SND

### Evaluation Flow

To test different formations with the same trained policy:

```python
for target_snd in [0.3, 0.5, 0.7, 1.0, 1.5]:
    # Query hypernetwork with different SND values
    adapters = hypernetwork(initial_positions, target_snd=target_snd)
    
    #Run episode with those adapters
    # Observe formation behavior
```

**Expected Behavior:**
- Low SND (0.3): Tight formation, agents clustered for maximum shielding
- Medium SND (0.5-0.7): Balanced formation
- High SND (1.5): Spread out, diverse behaviors

## Key Design Decisions

### 1. **Position-Based Actions (Option B)**
- Direct force application: `force = k_p * (target_pos - current_pos)`
- Simpler and more direct than velocity controller approach
- Bounded by `position_range` and `max_speed`

### 2. **Angle-Based Shaping Reward (Not Distance-Based)**
- Rewards alignment with wind direction
- No fixed desired distance between agents
- Allows agents to find optimal spacing naturally

### 3. **One Big Agent Shields All**
- Largest agent (by radius) shields all smaller agents
- Shielding effectiveness based on alignment angle
- Scales to N agents with varying radii

### 4. **Relative Positions for Hypernetwork**
- Positions relative to center of mass
- Translation invariant
- Ensures different inputs per agent → non-zero adapter SND
- Better generalization than absolute positions

### 5. **Reward Balance**
```python
total_reward = (
    energy_rew * energy_weight +        # Stay in place
    wind_rew * wind_weight +            # Reduce wind exposure
    formation_rew * formation_weight    # Align with wind
)
```
Default weights: all 1.0 except formation_shaping = 0.5

## Usage

### Training
```bash
python train.py --config config_wind_flocking.yaml
```

### Evaluation with Different SND Values
```bash
python evaluate.py \
  --checkpoint checkpoints/wind_flocking_... \
  --target-snd 0.3  # Test tight formation

python evaluate.py \
  --checkpoint checkpoints/wind_flocking_... \
  --target-snd 1.5  # Test spread out formation
```

### Testing
```bash
python test_wind_flocking.py
```

## Scaling to More Agents

To use 3 agents:

```yaml
env:
  num_agents: 3
  agent_radii: [0.06, 0.04, 0.03]  # Largest shields the rest
```

To use 4 agents:

```yaml
env:
  num_agents: 4
  agent_radii: [0.06, 0.05, 0.04, 0.03]
```

The implementation automatically:
- Adjusts observation dimensions
- Extracts N agent positions
- Computes shielding for all agents
- Scales rewards appropriately

## Files Created/Modified

### Created:
1. `wind_flocking_position.py` - New scenario implementation
2. `config_wind_flocking.yaml` - Configuration file
3. `test_wind_flocking.py` - Test script

### Modified:
1. `env_setup.py` - Added handler for wind_flocking_position
2. `train.py` - Updated `extract_agent_positions()` to support wind flocking and compute relative positions

## Next Steps

1. **Train the model**:
   ```bash
   python train.py --config config_wind_flocking.yaml --wandb
   ```

2. **Monitor training**:
   - Watch for emergent formations
   - Check if energy/wind rewards balance properly
   - Verify SND control is working

3. **Evaluate with different SNDs**:
   - Test if low SND → tight formation
   - Test if high SND → spread out formation
   - Visualize with GIF generation

4. **Tune if needed**:
   - Adjust reward weights if agents don't form good formations
   - Adjust `formation_shaping_weight` if alignment is too weak/strong
   - Adjust `position_gain` if position tracking is unstable

## Potential Issues & Solutions

### Issue: Agents just stay still
**Cause**: Energy penalty too high relative to wind penalty
**Solution**: Increase `wind_reward_weight` or decrease `energy_reward_weight`

### Issue: Agents don't align properly
**Cause**: Formation shaping reward too weak
**Solution**: Increase `formation_shaping_weight` (try 1.0 or 2.0)

### Issue: Position controller unstable
**Cause**: Proportional gain too high
**Solution**: Decrease `position_gain` (try 1.0 instead of 2.0)

### Issue: Adapter SND is zero
**Cause**: All agents receiving same hypernetwork input
**Solution**: Verified - positions are relative to center of mass, so each agent gets different input ✓

## Summary

The implementation is complete and tested! You now have:
- ✓ Position-based action space
- ✓ Configurable agent radii (scalable to N agents)
- ✓ Energy + wind + formation rewards
- ✓ Angle-based shaping for alignment
- ✓ Hypernetwork integration with initial positions + target SND
- ✓ SND-based diversity control
- ✓ Full configuration file
- ✓ Test script verification

Ready to train and experiment with different formation behaviors! 🚀
