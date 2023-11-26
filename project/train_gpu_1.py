import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear
import time

train_dataset = torchvision.datasets.CIFAR10('../my_dataset', train=True,  transform=torchvision.transforms.ToTensor(), download=True)
test_dataset  = torchvision.datasets.CIFAR10('../my_dataset', train=False, transform=torchvision.transforms.ToTensor(), download=True)

train_dataloader = DataLoader(train_dataset, batch_size=64)
test_dataloader = DataLoader(test_dataset, batch_size=64)

class Damon(nn.Module):
    def __init__(self):
        super(Damon, self).__init__()
        self.model = nn.Sequential(
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
        output = self.model(input)
        return output

print("Train pic Num: {}".format(len(train_dataset)))
print("Test pic  Num: {}".format(len(test_dataset)))

epoch = 10
learning_rate = 1e-2
total_train_num = 0
total_test_num = 0

damon = Damon()
# damon = damon.cuda()

celoss = nn.CrossEntropyLoss()
# celoss = celoss.cuda()
optimizer = torch.optim.SGD(damon.parameters(), learning_rate)


writer = SummaryWriter('../logs')

now_time = time.time()
for i in range(10):
    print("-----------Epoch:{}---------------".format(i))
    for data in train_dataloader:
        imgs, targets = data
        # imgs = imgs.cuda()
        # targets = targets.cuda()
        output = damon(imgs)

        # calc Loss
        loss = celoss(output, targets)

        # optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_num = total_train_num + 1
        if total_train_num % 200 == 0:
            print("Train Num {}, Current Loss {}, time: {}".format(total_train_num, loss, time.time()-now_time))
            writer.add_scalar('train_loss', loss, total_train_num)

    total_test_loss = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            # imgs = imgs.cuda()
            # targets = targets.cuda()
            output = damon(imgs)

            loss = celoss(output, targets)
            total_test_loss = total_test_loss + loss
    total_test_num = total_test_num + 1
    writer.add_scalar('test_loss', total_test_loss, total_test_num)
    print("test loss{}".format(total_test_loss))






