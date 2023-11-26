import torch
import torch.nn.functional as F

input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]])

kernel = torch.tensor([[1, 2, 1],
                       [0, 1, 0],
                       [2, 1, 0]])


input = torch.reshape(input, (1, 1, input.shape[0], input.shape[1]))
kernel = torch.reshape(kernel, (1, 1, kernel.shape[0], kernel.shape[1]))

output = F.conv2d(input, kernel, stride=1, padding=1)

print(output)


