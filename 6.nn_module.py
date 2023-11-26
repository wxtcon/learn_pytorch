import torch
from torch import nn

class Damon(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        output = input + 1
        return output

damon = Damon()
x = torch.tensor(1.0)
output = damon(x)
print(output)
