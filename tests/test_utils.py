from spine.utils.heatmap import heatmap_2d_encoder, heatmap_3d_encoder
import torch
import numpy as np

def test_heatmap_2d_encoder():
    heatmap_size = (4, 4)
    gt_coords = torch.tensor([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5]], dtype=torch.float32)
    gt_classes = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.int64)
    num_classes = 6
    sigma = 1
    stride = 1
    heatmap = heatmap_2d_encoder(heatmap_size, gt_coords, gt_classes, num_classes, sigma, stride)
    assert heatmap.shape == (num_classes, *heatmap_size)

#def test_heatmap_3d_encoder():
#    heatmap_size = (4, 4, 4)
#    gt_coords = torch.tensor([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5]], dtype=torch.float32)
#    gt_classes = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.int64)
#    num_classes = 6
#    sigma = 1
#    stride = np.array([1,1])
#    heatmap = heatmap_3d_encoder(series=, stride, gt_coords, gt_classes, sigma)
#    assert heatmap.shape == (num_classes, *heatmap_size)