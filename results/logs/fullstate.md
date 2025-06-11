# Training Run Report: fullstate.md
========================================

## 1. Hyperparameters
```yaml
agent:
  epsilon_decay: 0.995
  epsilon_end: 0.01
  epsilon_start: 1.0
  gamma: 0.99
  hidden_size: 128
  learning_rate: 0.0005
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
  goal: 5.0
  hole: -1.0
  ice: 0
  wall: -0.3
testing:
  num_episodes: 10
  speed: 10000000
training:
  log_interval: 100
  max_steps_per_episode: 30
  num_episodes: 5000
  save_interval: 500
  speed: 10000000
```

