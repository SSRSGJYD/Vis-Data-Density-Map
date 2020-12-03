import cv2
import numpy as np
from scipy import ndimage, signal
from kernel import uniform_kernel, triangular_kernel


def kernel_density_map(input_map:np.ndarray, ksize:int, method:str) -> np.ndarray:
    '''
    Calculate density map using kernel functions.

    Args:
    input_map: number of points in each grid, shaped (C, H, W) where C is number of class.
    ksize: kernel size, must be odd number.
    method: kernel function used.

    Return:
    grid_maps: (C, H, W)
    '''

    C, H, W = input_map.shape
    if method == 'uniform':
        kernel = np.expand_dims(uniform_kernel(ksize), axis=0).repeat(C, axis=0)
        return signal.fftconvolve(input_map, kernel, mode='same')
    elif method == 'triangular':
        kernel = np.expand_dims(triangular_kernel(ksize), axis=0).repeat(C, axis=0)
        return signal.fftconvolve(input_map, kernel, mode='same')
    elif method == 'gaussian':
        output = ndimage.gaussian_filter1d(input_map, 1.0, mode='nearest', truncate=ksize//2, axis=1)
        output = ndimage.gaussian_filter1d(output, 1.0, mode='nearest', truncate=ksize//2, axis=2)
        return output


def density_map_KDE(points:np.ndarray, labels:np.ndarray, classes:int, height:int, width:int, ksize:int, method:str) -> np.ndarray:
    '''
    Generate density map for each class using simplified KDE.

    Args:
    points: (x,y) of all points, shaped (N, 2)
    labels: labels of all points, (N)
    classes: number of classes.
    height, width: resolution of output map.
    ksize: kernel size, must be odd number.
    method: kernel function used.

    Return:
    grid_maps: (C, H, W)
    '''
    grid_maps = np.zeros((classes, height, width), dtype=np.float)
    xmin = np.min(points[:, 0])
    xmax = np.max(points[:, 0])
    ymin = np.min(points[:, 1])
    ymax = np.max(points[:, 1])
    for i in range(classes):
        indices = (labels == i)
        grid_maps[i], xedges, yedges = np.histogram2d(points[indices, 0], points[indices, 1], range=np.array([[xmin, xmax], [ymin, ymax]]), bins=(width, height))
        grid_maps[i] /= len(indices)

    grid_maps = kernel_density_map(grid_maps, ksize, method)
    return grid_maps


def draw_density_map_interpolate(maps:np.ndarray, norm_class=True) -> np.ndarray:
    '''
    Draw density maps of different classes on an image. Interpolate colors in HSV color space.
    Should have more than one class.

    Args:
    maps: density maps shaped (C, H, W).
    norm_class: whether to normalize values of each class to [0,1]. Default True.

    Returns:
    image: a HSV image shaped (3, H, W)
    '''

    C, H, W = maps.shape
    image_HSV = np.zeros((3, H, W), dtype=np.float)

    if norm_class:
        cls_max = np.max(maps, axis=(1,2), keepdims=True)
        cls_min = np.min(maps, axis=(1,2), keepdims=True)
        maps = (maps - cls_min) / (cls_max - cls_min)

    # norm along all pixels
    pixel_sum = np.sum(maps, axis=0, keepdims=True) # (1, H, W)
    pixel_max = np.max(pixel_sum)
    maps /= pixel_max
    pixel_sum /= pixel_max # (1, H, W)

    # interpolate in Hue space
    H_palette = np.zeros((C, 1, 1))
    for i in range(C):
        H_palette[i, :, :] = 360 * i / C

    image_HSV[0, :, :] = np.sum(maps * H_palette / pixel_sum, axis=0) # (C, H, W)

    # Decide saturation by pixel_sum
    image_HSV[1, :, :] = pixel_sum * 255

    # V
    image_HSV[2, :, :] = 255

    image_HSV = image_HSV.astype(np.uint8)

    return image_HSV

if __name__ == "__main__":
    X = np.load('tsne/pca100_50_5000.npy')
    y = np.load('data/sampled_label.npy')

    for imgsize in [50,100,200]:
        for method in ['uniform','triangular', 'gaussian']:
            for ksize in [7,17,27]:
                for norm_class in [True, False]:
                    grid_maps = density_map_KDE(X, y, 10, imgsize, imgsize, ksize, method)
                    image_HSV = draw_density_map_interpolate(grid_maps, norm_class)
                    image_HSV = np.transpose(image_HSV, (1,2,0))
                    image_BGR = cv2.cvtColor(image_HSV, cv2.COLOR_HSV2BGR)
                    cv2.imwrite('./result/{},ksize={},normclass={}({}).png'.format(method, ksize, norm_class, imgsize), image_BGR)