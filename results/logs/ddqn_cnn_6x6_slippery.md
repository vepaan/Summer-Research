# Training Run Report: ddqn_cnn_6x6_slippery.md
========================================

## 1. Hyperparameters
```yaml
agent:
  clip_epsilon: 0.2
  cnn:
    conv_channels: 32
    hidden_size: 1024
    input_shape:
    - 4
    - 6
    - 6
  entropy_coeff: 0.005
  epsilon_decay: 3000
  epsilon_end: 0.01
  epsilon_start: 1.0
  gae_lambda: 0.95
  gamma_ddqn: 0.85
  gamma_ppo: 0.99
  learning_rate_ddqn: 0.0005
  learning_rate_ppo: 0.0003
  mlp:
    hidden_size: 128
  model_type: CNN
  ppo_epochs: 6
  rl_type: DDQN
  target_update_freq: 10
env:
  agent: 1
  goal: 0.7
  hole: 0
  ice: 0
  is_slippery: true
  map_size: 6
memory:
  batch_size: 128
  buffer_size: 50000
reward:
  goal: 4.0
  hole: -3.0
  ice: -0.05
  wall: -0.1
testing:
  num_episodes: 4000
  speed: 10000000
training:
  log_interval: 100
  max_steps_per_episode: 300
  num_episodes: 5000
  save_interval: 500
  speed: 10000000
```

