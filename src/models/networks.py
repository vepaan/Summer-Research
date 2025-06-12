import torch.nn as nn
import torch.nn.functional as F
import torch
import os

class BaseNet(nn.Module):

    def __init__(self, device='cpu'):
        super(BaseNet, self).__init__()
        self.device = torch.device(device)
        self.to(self.device)


    def save(self, file_name: str, folder_path: str):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        full_path = os.path.join(folder_path, file_name)
        torch.save(self.state_dict(), full_path)


    def load(self, file_path: str):
        self.load_state_dict(torch.load(file_path, map_location=self.device))
        self.eval()


class MLP(BaseNet):

    def __init__(self, input_size: int, hidden_size: int, output_size: int, device: str = 'cpu'):
        super().__init__(device)

        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, output_size)
        
        self._initialize_weights()


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


class CNN(BaseNet):

    def __init__(self, input_shape: int, conv_channels: int, hidden_size: int, output_size: int, device: str = 'cpu'):
        super().__init__(device)
        c, h, w = input_shape

        self.conv1 = nn.Conv2d(c, conv_channels, kernel_size=2, stride=1)
        self.conv2 = nn.Conv2d(conv_channels, conv_channels*2, kernel_size=2, stride=1)

        conv_out_h = h-2 #for 2 conv layers
        conv_out_w = w-2
        conv_out_size = conv_channels * 2 * conv_out_h * conv_out_w

        self.fc1 = nn.Linear(conv_out_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

        self._initialize_weights()


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.device != self.device:
            x = x.to(self.device)
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


