from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

writer = SummaryWriter("logs")

img_path = 'dataset/Type1/train/ants/6240329_72c01e663e.jpg'
img = Image.open(img_path)
img_array = np.array(img)

writer.add_image('test', img_array, 1, dataformats='HWC')


# for i in range(100):
#     writer.add_scalar('y = x', 2*i, i)


# writer.add_scalar()


writer.close()


