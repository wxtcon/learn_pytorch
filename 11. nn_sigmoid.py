import torch
import torchvision
from torch import nn
from torch.nn import Sigmoid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root='./my_dataset',
                                       train=False,
                                       transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset, batch_size=64)

class Damon(nn.Module):
    def __init__(self):
        super(Damon, self).__init__()
        self.sigmoid1 = Sigmoid()

    def forward(self, input):
        output = self.sigmoid1(input)
        return output

damon = Damon()
writer = SummaryWriter('./logs')
step = 0
for data in dataloader:
    step = step + 1
    imgs, targets = data
    output = damon(imgs)
    writer.add_images('sigmoid', output, step)

writer.close()








