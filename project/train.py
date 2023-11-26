import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader

from model import Damon

train_dataset = torchvision.datasets.CIFAR10('../my_dataset', train=True,  transform=torchvision.transforms.ToTensor(), download=True)
test_dataset  = torchvision.datasets.CIFAR10('../my_dataset', train=False, transform=torchvision.transforms.ToTensor(), download=True)

train_dataloader = DataLoader(train_dataset, batch_size=64)
test_dataloader = DataLoader(test_dataset, batch_size=64)

print("Train pic Num: {}".format(len(train_dataset)))
print("Test pic  Num: {}".format(len(test_dataset)))

epoch = 10
learning_rate = 1e-2
total_train_num = 0
total_test_num = 0

damon = Damon()
celoss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(damon.parameters(), learning_rate)




for i in range(10):
    print("-----------Epoch:{}---------------".format(i))
    for data in train_dataloader:
        imgs, targets = data
        output = damon(imgs)

        # calc Loss
        loss = celoss(output, targets)

        # optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_num = total_train_num + 1
        if total_train_num % 50 == 0:
            print("Train Num {}, Current Loss {}".format(total_train_num, loss))

    total_test_loss = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            output = damon(imgs)

            loss = celoss(output, targets)
            total_test_loss = total_test_loss + loss
    print("test loss{}".format(total_test_loss))






