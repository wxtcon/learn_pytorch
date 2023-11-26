import torch
import torchvision
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("./my_dataset",
                                       train=False,
                                       transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset, batch_size=64)

class Damon(nn.Module):
    def __init__(self):
        super(Damon, self).__init__()
        self.linear1 = Linear(196608,10)

    def forward(self, input):
        output = self.linear1(input)
        return output

t = Damon()
for data in dataloader:
    imgs, targets = data
    imgs = torch.flatten(imgs)
    output = t(imgs)
    print(output.shape)