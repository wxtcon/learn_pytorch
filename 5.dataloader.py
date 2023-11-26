import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

test_data = torchvision.datasets.CIFAR10("./my_dataset", train=False, transform=dataset_transform, download=True)
test_loader = DataLoader(dataset=test_data, batch_size=128, shuffle=True, num_workers=0, drop_last=False)

img, target = test_data[0]
print(img.shape)
print(target)

writer = SummaryWriter('dataloader')

step = 0
for data in test_loader:
    imgs, targets = data
    # print(imgs.shape)
    # print(targets)
    writer.add_images("dataloader", imgs, step)
    step = step + 1

writer.close()
