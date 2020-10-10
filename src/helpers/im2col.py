import numpy as np

def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 0)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value

def im2col(x, kernel_shape):
    x = np.pad(x,1,pad_with)
    rows = []
    for row in range(x.shape[0]-2):
        for col in range(x.shape[1]-2):
            window = x[row:row+kernel_shape, col:col+kernel_shape]
            rows.append(window.flatten())
    return np.transpose(np.array(rows))