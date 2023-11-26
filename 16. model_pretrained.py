import torchvision
from torch import nn
# train_data = torchvision.datasets.ImageNet(root="./my_dataset",
#                                            split="train",
#                                            download=True,
#                                            transform=torchvision.transforms.ToTensor())


vgg16_raw = torchvision.models.vgg16()

vgg16_raw.add_module('my_linear', nn.Linear(1000, 10))
vgg16_raw.classifier.add_module("my_linear2", nn.Linear(100, 100))
vgg16_raw.classifier[6] = nn.Linear(4096, 10)


print(vgg16_raw)
