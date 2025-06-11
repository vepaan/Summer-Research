import os
import yaml

class TrainingLogger:

    def __init__(self, log_dir: str, log_name: str, config: dict):
        self.log_name = log_name
        self.log_dir = os.path.join(log_dir, log_name)
        os.makedirs(self.log_dir, exist_ok=True)

        self.report_path = os.path.join(self.log_dir, f"report_{self.log_name}.md")

        with open(self.report_path, 'w') as f:
            f.write(f"# Training Run Report: {self.run_name}\n")
            f.write("="*40 + "\n\n")

        if config:
            self.log_hyperparams(config)

    
    def log_hyperparams(self, config: dict):
        with open(self.report_path, 'a') as f:
            f.write("## 1. Hyperparameters\n")
            f.write("```yaml\n")
            yaml.dump(config, f, default_flow_style=False)
            f.write("```\n\n")