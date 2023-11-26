import torchvision
from torch.utils.tensorboard import SummaryWriter

dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

train_set = torchvision.datasets.CIFAR10(root="./my_dataset",
                                         train=True,
                                         download=True,
                                         transform=dataset_transform)
test_set = torchvision.datasets.CIFAR10(root="./my_dataset",
                                        train=False,
                                        download=True,
                                        transform=dataset_transform)

# dataset_transforms = torchvision.transforms.Compose([])


# img.show()
writer = SummaryWriter('pic10')

print(test_set[0])

for i in range(10):
    img, target = test_set[i]
    writer.add_image('pic', img, i)

writer.close()



