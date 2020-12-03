import numpy as np


def uniform_kernel(ksize:int) -> np.ndarray:
    kernel = np.ones((ksize, ksize), dtype=np.float)
    return kernel / (ksize*ksize)

def triangular_kernel(ksize:int) -> np.ndarray:
    kernel = np.zeros((ksize, ksize), dtype=np.float)
    w = ksize // 2 + 1
    c = w - 1
    w_quad = w * w * w * w
    kernel[c][c] = w_quad
    for x in range(1, w):
        for y in range(1, x+1):
            v = (w-x) * (w-y)
            kernel[c+x][c+y] = kernel[c+x][c-y] = kernel[c-x][c+y] = kernel[c-x][c-y] \
                = kernel[c+y][c+x] = kernel[c+y][c-x] = kernel[c-y][c+x] = kernel[c-y][c-x] = v
    return kernel / w_quad