# Training Run Report: fullstate.md
========================================

## 1. Hyperparameters
```yaml
agent:
  cnn:
    conv_channels: 16
    hidden_size: 128
    input_shape:
    - 4
    - 4
    - 4
  epsilon_decay: 2000
  epsilon_end: 0.01
  epsilon_start: 1.0
  gamma: 0.99
  learning_rate: 0.0005
  mlp:
    hidden_size: 128
  model_type: MLP
  target_update_freq: 10
env:
  agent: 1
  goal: 0.7
  hole: 0
  ice: 0
  is_slippery: false
  map_size: 4
memory:
  batch_size: 64
  buffer_size: 10000
reward:
  goal: 1.0
  hole: 0
  ice: -0.05
  wall: -0.1
testing:
  num_episodes: 10000
  speed: 10000000
training:
  log_interval: 100
  max_steps_per_episode: 200
  num_episodes: 4000
  save_interval: 500
  speed: 10000000
```

