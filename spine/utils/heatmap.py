import numpy as np
import torch
import itertools


def _gaussian(x, sigma):
    return np.exp(-x / (2 * sigma ** 2))


#gt_coords: (N, 2) N: 5*3
def heatmap_2d_encoder(heatmap_size, gt_coords, gt_classes, num_classes, sigma, stride):
    if not isinstance(gt_coords, np.ndarray):
        gt_coords = gt_coords.detach().cpu().numpy()
    heatmap = np.zeros((num_classes, *heatmap_size))

    ii = np.arange(heatmap_size[0])
    jj = np.arange(heatmap_size[1])
    ii, jj = np.meshgrid(ii, jj, indexing='ij')
    gridmap = np.stack([ii+0.5, jj+0.5], axis=-1)
    gridmap *= stride

    for i, (gt_coord, gt_class) in enumerate(zip(gt_coords, gt_classes)):
        distance = np.square((gt_coord - gridmap)).sum(axis=-1)
        heatmap[i] = _gaussian(distance, sigma).reshape(heatmap_size)
        heatmap[i] = heatmap[i] / heatmap[i].sum()

    return heatmap

def heatmap_3d_encoder(heatmap_size, gt_coords, gt_classes, num_classes, sigma, stride):
    if not isinstance(gt_coords, np.ndarray):
        gt_coords = gt_coords.detach().cpu().numpy()
    heatmap = np.zeros((num_classes, *heatmap_size))

    ii = np.arange(heatmap_size[0])
    jj = np.arange(heatmap_size[1])
    kk = np.arange(heatmap_size[2])
    ii, jj, kk = np.meshgrid(ii, jj, kk, indexing='ij')
    gridmap = np.stack([ii+0.5, jj+0.5, kk+0.5], axis=-1)
    gridmap *= stride

    for i, (gt_coord, gt_class) in enumerate(zip(gt_coords, gt_classes)):
        distance = np.square((gt_coord - gridmap)).sum(axis=-1)
        heatmap[i] = _gaussian(distance, sigma).reshape(heatmap_size)
        heatmap[i] = heatmap[i] / heatmap[i].sum()

    return heatmap