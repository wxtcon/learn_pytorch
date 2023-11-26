import torch
import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("./my_dataset",
                                       train=False,
                                       transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset, batch_size=64)

class Damon(nn.Module):
    def __init__(self):
        super(Damon, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        return x

damon = Damon()

writer = SummaryWriter('./logs')

step = 0
for data in dataloader:
    step = step + 1
    imgs, targets = data

    output = damon(imgs)
    # print(output.shape)
    writer.add_images('input', imgs, step)

    # output = torch.reshape(output, (-1, 3, 30, 30))
    # output = output.view(-1, 3, 30, 30)
    writer.add_images('output', output, step)