import numpy as np
import torch
from torchvision import transforms


def im2col(x, kernel_shape):
    img_to_pad = transforms.Compose([
             transforms.ToPILImage(),
             transforms.Pad(padding=1, fill=0, padding_mode="constant"),
             transforms.ToTensor(),
             ])
    x = img_to_pad(x)
    x = x[0]
    rows = []
    for row in range(x.shape[0]-2):
        for col in range(x.shape[1]-2):
            window = x[row:row+kernel_shape, col:col+kernel_shape]
            rows.append(window.flatten())
    rows = torch.stack(rows)
    result = torch.tensor(rows).t().long()
    return result*255