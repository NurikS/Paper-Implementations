import sys
sys.path.append('../src/helpers')
from im2col import im2col
import torch

def convolve2d(X, kernel):
    cols = im2col(X, 3)
    convolved = torch.matmul(kernel.flatten(), cols.long())
    shapes = convolved.shape[0]
    shape = torch.tensor(shapes,dtype=torch.float32)
    return convolved.view(int(torch.sqrt(shape)), int(torch.sqrt(shape)))