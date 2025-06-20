# Training Run Report: ddqn_cnn_slippery.md
========================================

## 1. Hyperparameters
```yaml
agent:
  clip_epsilon: 0.2
  cnn:
    conv_channels: 16
    hidden_size: 128
    input_shape:
    - 4
    - 4
    - 4
  entropy_coeff: 0.01
  epsilon_decay: 2000
  epsilon_end: 0.01
  epsilon_start: 1.0
  gae_lambda: 0.95
  gamma_ddqn: 0.99
  gamma_ppo: 0.99
  learning_rate_ddqn: 0.0005
  learning_rate_ppo: 0.0003
  mlp:
    hidden_size: 128
  model_type: CNN
  ppo_epochs: 4
  rl_type: DDQN
  target_update_freq: 10
env:
  agent: 1
  goal: 0.7
  hole: 0
  ice: 0
  is_slippery: true
  map_size: 4
memory:
  batch_size: 64
  buffer_size: 10000
reward:
  goal: 1.0
  hole: -0.5
  ice: -0.05
  wall: -0.1
testing:
  num_episodes: 10000
  speed: 10000000
training:
  log_interval: 100
  max_steps_per_episode: 32
  num_episodes: 4000
  save_interval: 500
  speed: 10000000
```

