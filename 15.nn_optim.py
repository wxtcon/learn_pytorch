import torch
import torchvision
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("./my_dataset",
                                       train=False,
                                       transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset, batch_size=1)

class Damon(nn.Module):
    def __init__(self):
        super(Damon, self).__init__()
        self.model1 = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, input):
        output = self.model1(input)
        return output

damon = Damon() # instantiation
celoss = nn.CrossEntropyLoss() # Loss definition
optim = torch.optim.SGD(damon.parameters(), lr=0.01)

for epoch in range(20):
    run_loss = 0.0
    for data in dataloader:
        optim.zero_grad()

        imgs, targets = data
        output = damon(imgs)

        loss = celoss(output, targets)
        loss.backward()

        optim.step()
        run_loss = run_loss + loss
    print(run_loss)








