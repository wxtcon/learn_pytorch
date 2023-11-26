from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

# read the image
img_path = "dataset/Type1/train/ants/0013035.jpg"
img = Image.open(img_path)
print(img)
writer = SummaryWriter('logs')

# 1. ToTensor
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)

writer.add_image("Tensor_img", tensor_img)


# 2. ToPILImage
pass

# 3. Normalize
print(tensor_img[0][0][0])
trans_norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
img_norm = trans_norm(tensor_img)
print(img_norm[0][0][0])

writer.add_image('normalize', img_norm)

# 4. resize
print(img.size)
trans_resize = transforms.Resize((128, 128))
img_resize = trans_resize(img)
print(img.size)
img_resize = tensor_trans(img_resize)
writer.add_image("resize", img_resize)

# 5. compose
trans_resize2 = transforms.Resize(64)
trans_compose = transforms.Compose([trans_resize2, tensor_trans])
img_compose = trans_compose(img)

writer.add_image('compose1', img_compose)

# 6. RandomCrop
trans_random_crop = transforms.RandomCrop(256)
trans_compose_2 = transforms.Compose([trans_random_crop, tensor_trans])

for i in range(10):
    img_compose_2 = trans_compose_2(img)
    writer.add_image('compose2', img_compose_2, i)


writer.close()








