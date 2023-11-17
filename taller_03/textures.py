from skimage.feature import graycomatrix, graycoprops
import numpy as np
import math
from itertools import product
from dask.distributed import print as dprint

def get_circular_kernel(diameter, type=None):

    mid = (diameter - 1) / 2
    distances = np.indices((diameter, diameter)) - np.array([mid, mid])[:, None, None]
    kernel = ((np.linalg.norm(distances, axis=0) - mid) <= 0).astype(int)
    
    if type is not None:
        if type == "mean":
            kernel = kernel/kernel.sum()

    return kernel

def square_pos_kernel(radius: int = 7):
    # return distance and angle
    r = range(-radius, radius + 1)
    col, row = np.meshgrid(r, r)
    return (col**2 + row**2)**.5, np.arctan(col/row)


def unique_polar(distance: np.array, angle: np.array):
    dis = np.unique(distance)
    ang = np.unique(angle)
    return dis[dis != 0], ang[~np.isnan(ang)]

def graycoprops_(array: np.array):
    texts = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]
    ans = np.empty((6, array.shape[-2], array.shape[-1]))
    # dprint(array.shape)
    # dprint(ans.shape)
    for i, v in enumerate(texts):
        g_ = graycoprops(array, v)
        # dprint("-------------")
        # dprint(g_.shape)
        ans[i, :, :] = g_
    
    return ans

def textures(glcm: np.array, summarize: str = "mean"):
    # glcm should be of 4D (nlevels, nleves, nº distances, nº angles)
    if len(glcm.shape) != 4:
        raise Exception(f"Matrix should be of 4 dimensions: {glcm.shape} detected.")

    n_dis = glcm.shape[2]
    n_ang = glcm.shape[3]
    
    # TODO: should be an array of 0s of float32 instead of floeat64?
    ans = np.empty((6, n_dis, n_ang))
    
    # for d in range(n_dis):
    #    for a in range(n_ang):
    #        ans[:, d, a] = graycoprops_(glcm[:, :, d, a])
    ans = graycoprops_(glcm)

    if summarize == "mean":
        # TODO: should the conversion to float32 be here? before? after?
        return ans.astype(np.float32).mean(axis=(1, 2))
    else:
        return ans
    

def glcm_texture(
    array: np.array,
    radius: int = 1,
    distances: list = range(-1, 2),
    angles: list = [0, math.pi / 2],
    nan_supression: int = 0,
    skip_nan: bool = True,
    **kwargs,
):
    # nan_supression: 0 nothing
    # nan_supression: 1 only 0,0
    # nan_supression: 2 full 0 row and full 0 column
    view = np.lib.stride_tricks.sliding_window_view(
        array, (radius * 2 + 1, radius * 2 + 1)
    )

    metrics = 6  # hardcoded !!! we only need 7 right now
    response = np.zeros([metrics] + list(view.shape[:2]))
    range_i, range_j = range(view.shape[0]), range(view.shape[1])

    for i, j in product(range_i, range_j):
        subarray = view[i, j, :, :]
        
        if array[i, j] == 0 and skip_nan:
            response[:, i, j] = np.zeros(metrics)
        else:
            glcm = graycomatrix(subarray, distances, angles, **kwargs)
            if nan_supression == 1:
                glcm[0, 0, :, :] = 0
            if nan_supression == 2:
                glcm[:, 0, :, :] = 0
                glcm[0, :, :, :] = 0

            response[:, i, j] = textures(glcm)
            
    if skip_nan or nan_supression > 0:
        response[response == 0] = np.nan

    return response