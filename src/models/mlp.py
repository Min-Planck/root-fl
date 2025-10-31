from torch import nn
import torch


class MLP_header(nn.Module): 
    def __init__(self, num_channels=1, im_size=(28, 28), hidden_size=200) -> None:
        super(MLP_header, self).__init__()
        self.num_channels = num_channels
        self.im_size = im_size
        self.layer1 = nn.Linear(num_channels * im_size[0] * im_size[1], hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, self.num_channels * self.im_size[0] * self.im_size[1])
        x = self.layer1(x)
        x = self.relu(x)

        x = self.layer2(x)
        x = self.relu(x)
        
        return x
    
class MLP(nn.Module): 
    def __init__(self, num_channels=1, im_size=(28, 28), hidden_size=200, output_size=10) -> None:
        super(MLP, self).__init__()
        self.encode = MLP_header(num_channels=num_channels, im_size=im_size, hidden_size=hidden_size)
        self.classification = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encode(x) 
        x = self.classification(x)
        return x