import torch.nn as nn
import torch.nn.functional as F
import torch
import os

class DQN(nn.Module):

    def __init__(self, input_size: int, hidden_size: int, output_size: int, device: str = 'cpu'):
        super(DQN, self).__init__()

        self.device = torch.device(device)

        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, output_size)
        
        self._initialize_weights()
        self.to(self.device)


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.device != self.device:
            x = x.to(self.device)

        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
    

    def save(self, file_name: str ='model.pth', folder_path: str = "../../results/models/"):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        file_name = os.path.join(folder_path, file_name)
        torch.save(self.state_dict(), file_name)


    def load(self, file_path: str):
        self.load_state_dict(torch.load(file_path, map_location=self.device))
        self.eval()


