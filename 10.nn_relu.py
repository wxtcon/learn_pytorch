import torch
from torch import nn
from torch.nn import ReLU

input = torch.tensor([[1, -0.5],
             [-0.2, 3]])

class Damon(nn.Module):
    def __init__(self):
        super(Damon, self).__init__()
        self.relu1 = ReLU()

    def forward(self, input):
        output = self.relu1(input)
        return output

damon = Damon()
output = damon(input)

print(output)



