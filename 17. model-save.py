import torchvision
import torch

# raw method
vgg16 = torchvision.models.vgg16()

# save method1:
torch.save(vgg16, "vgg16_method1.pth")

# load method1:
model1 = torch.load("vgg16_method1.pth")
# -------------------------

# save method2:
torch.save(vgg16.state_dict(), 'vgg16_method2.pth')

# load method2:
model2 = torch.load("vgg16_method2.pth") # only load model parameter, not contains the construction of the model
vgg16.load_state_dict(torch.load("vgg16_method2.pth"))


