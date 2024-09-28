from spine.utils.heatmap import generate_world_mesh_coords, heatmap_3d_encoder
from spine.spine_exam import SpineSeries, InstanceMeta
import numpy as np


instance_meta = [InstanceMeta(
    instance_number=1,
    rows=10,
    cols=10,
    position=np.array([0, 0, i]),
    orientation=np.array([1, 0, 0, 0, 1, 0]),
    normal=np.array([0, 0, 1]),
    projection=np.array([1, 0, 0]),
    pixel_spacing=np.array([1, 1]),
    spacing_between_slices=1,
    slice_thickness=1,
    scale=1.0
) for i in range(10)]

series = SpineSeries(123, 345, "test", np.random.rand(10, 20, 20), 
                        meta=instance_meta, scale=1.0)

def test_generate_world_mesh_coords():
    coords = generate_world_mesh_coords(series, stride=(1, 1, 1))
    assert coords.shape == (10, 20, 20, 3)
    print(coords[0, 0, 0, :])
    np.testing.assert_allclose(coords[0, 0, 0, :], np.array([0, 0, 0]))
    np.testing.assert_allclose(coords[0, 1, 0, :], np.array([0, 1, 0]))
    np.testing.assert_allclose(coords[0, 0, 1, :], np.array([0, 0, 1]))
    np.testing.assert_allclose(coords[1, 1, 1, :], np.array([1, 1, 1]))

def test_generate_world_mesh_coords2():

    # 45 degree rotation around x axis
    rot_mat = np.array([[1, 0, 0], [0, np.cos(np.pi/4), np.sin(np.pi/4)], [0, -np.sin(np.pi/4), np.cos(np.pi/4)]])
    x_orientation = np.array([1,0,0])
    y_orientation = np.array([0,1,0])@rot_mat


    instance_meta2 = [InstanceMeta(
        instance_number=1,
        rows=3,
        cols=3,
        position=np.array([0, 0, i]),
        orientation=np.concatenate([x_orientation, y_orientation]),
        normal=np.cross(x_orientation,y_orientation),
        projection=np.array([0, 0, i])+np.cross(x_orientation,y_orientation),
        pixel_spacing=np.array([2, 2]),
        spacing_between_slices=1,
        slice_thickness=1,
        scale=1.0
    ) for i in range(10)]

    series2 = SpineSeries(123, 345, "test", np.random.rand(10, 20, 20), 
                            meta=instance_meta2, scale=1.0)
    coords = generate_world_mesh_coords(series2, stride=(1, 1, 1))
    np.testing.assert_allclose(coords[0,0,0] , np.array([0, 0, 0]))
    np.testing.assert_allclose(coords[0,0,1] , np.array([0, 0, 2]))
    np.testing.assert_allclose(coords[0,1,0] , np.array([2*np.cos(np.pi/4), 2*np.cos(np.pi/4), 0]))

    

def test_generate_heatmap():
    return
    heatmaps = heatmap_3d_encoder(series,
                                  stride=(1, 1, 1),
                                  gt_coords=np.random.rand(5, 3),
                                  gt_classes=np.random.randint(0, 5, 5),
                                  sigma=1)



