import torch
import torchvision
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root='./my_dataset',
                                       train=False,
                                       transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataload = DataLoader(dataset, batch_size=64)

# input = torch.tensor([[1, 2, 0, 3, 1],
#               [0, 1, 2, 3, 1],
#               [1, 2, 1, 0, 0],
#               [5, 2, 3, 1, 1],
#               [2, 1, 0, 1, 1]], dtype=torch.float32)
# input = torch.reshape(input, (-1, 1, 5, 5))

class Damon(nn.Module):
    def __init__(self):
        super(Damon, self).__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=False)

    def forward(self, input):
        output = self.maxpool1(input)
        return output

damon = Damon();

writer = SummaryWriter('logs')
step = 0
for data in dataload:
    step = step + 1
    imgs, targets = data
    output = damon(imgs)
    writer.add_images('maxpool', output, step)

writer.close()




