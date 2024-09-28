import numpy as np
import torch
import itertools
from spine.spine_exam import SpineSeries, InstanceMeta

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

# TODO generate coordinates for centers of each voxel
def generate_world_mesh_coords(series: SpineSeries, stride):
    # x,y,z -> d,h,w
    image_orientation_patient = np.stack([ inst.orientation for inst in series.meta]) # (N, 6) -> concat (N,6)(N,3) -> (N,9) -> (N, 1, 1, 3, 3)
    image_orientation_patient = np.concatenate([image_orientation_patient, np.zeros((image_orientation_patient.shape[0],3))], axis=1)
    image_orientation_patient = image_orientation_patient.reshape(-1,1,1,3,3)[...,::-1, ::-1]

    image_position_patient = np.stack([ inst.position for inst in series.meta]).reshape(-1,1,1,1,3)[...,::-1]

    pixel_spacing = np.stack([ inst.pixel_spacing for inst in series.meta])
    # add one as third dimmension of pixel spacing
    pixel_spacing = np.concatenate([pixel_spacing, np.ones((pixel_spacing.shape[0],1))], axis=-1)[...,::-1]
    pixel_spacing = pixel_spacing.reshape(-1,1,1,1,3)
    d, h, w = series.volume.shape
    ii, jj, kk = np.meshgrid(
                         np.zeros(d, dtype=int),
                         np.arange(0, h, stride[0]),
                         np.arange(0, w, stride[1]),
                         indexing='ij')
    coords = np.stack([ii, jj, kk], axis=-1)

    coords = np.expand_dims(coords, axis=-2)
    coords = (image_position_patient+((coords*pixel_spacing)@image_orientation_patient)).squeeze()
    return coords

# TODO it is not resilent for coords outside the volume
def heatmap_3d_encoder(series: SpineSeries, stride, gt_coords, coords_mask, gt_classes, num_classes, sigma):
    mesh = generate_world_mesh_coords(series, stride)
    num_points = 25
    heatmap =  np.zeros((num_points*num_classes, *mesh.shape[:-1]))
    if gt_coords is None:
        return heatmap  
    gt_coords = gt_coords.reshape(-1,1,1,1,3)

    for i, (gt_coord, mask, gt_class) in enumerate(zip(gt_coords, coords_mask, gt_classes)):
        class_idx = gt_class # np.where(gt_class)[0]
        if class_idx<0 or not mask:
            continue
        class_heatmap = heatmap[3*i+class_idx]
        distance = np.square((gt_coord - mesh)).sum(axis=-1)
        class_heatmap = _gaussian(distance, sigma).reshape(mesh.shape[:-1])
        heatmap_max = class_heatmap.max()
        if heatmap_max > 0:
            class_heatmap /= heatmap_max
        heatmap[3*i+class_idx] = class_heatmap
    return heatmap


def heatmap_3d_simple_encoder(heatmap_size, gt_coords, gt_classes, num_classes, sigma, stride):
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