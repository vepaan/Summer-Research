# Training Run Report: onehot.md
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
  is_slippery: false
  map_size: 4
memory:
  batch_size: 64
  buffer_size: 10000
reward: null
training:
  log_interval: 10
  max_steps_per_episode: 5
  num_episodes: 100
  save_interval: 500
  speed: 10000000
```

