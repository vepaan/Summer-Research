import os
import yaml


def Log(log_dir: str, log_name: str, config: dict):
    log_name = log_name
    log_dir = os.path.join(log_dir, log_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    report_path = os.path.join(log_dir, log_name)

    with open(report_path, 'w') as f:
        f.write(f"# Training Run Report: {log_name}\n")
        f.write("="*40 + "\n\n")

    if config:
        log_hyperparams(report_path, config)


def log_hyperparams(report_path: str, config: dict):
    with open(report_path, 'a') as f:
        f.write("## 1. Hyperparameters\n")
        f.write("```yaml\n")
        yaml.dump(config, f, default_flow_style=False)
        f.write("```\n\n")