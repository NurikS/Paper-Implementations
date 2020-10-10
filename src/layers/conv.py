import sys
sys.path.append('../helpers')
from im2col import im2col
import numpy as np

def convolve2d(X, kernel):
    convolved = np.dot(kernel.flatten(), im2col(X, 3))
    shapes = convolved.shape[0]
    return convolved.reshape(int(np.sqrt(shapes)), int(np.sqrt(shapes)))