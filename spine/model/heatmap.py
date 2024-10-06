import torch

def heatmap_to_coord(heatmap):
    num_image = len(heatmap)
    device = heatmap[0].device
    num_point, num_grade, _, H, W = heatmap[0].shape
    D = max([h.shape[2] for h in heatmap])

    # create coordinates grid.
    x = torch.linspace(0, W - 1, W, device=device)
    y = torch.linspace(0, H - 1, H, device=device)
    z = torch.linspace(0, D - 1, D, device=device)

    points = []
    for i in range(num_image):
        num_point, num_grade, D, H, W = heatmap[i].shape
        pos_x = x.reshape(1,1,1,1,W)
        pos_y = y.reshape(1,1,1,H,1)
        pos_z = z[:D].reshape(1,1,D,1,1)
        # px, py shape (num_point)
        py = torch.sum(pos_y * heatmap[i], dim=(1,2,3,4))
        px = torch.sum(pos_x * heatmap[i], dim=(1,2,3,4))
        pz = torch.sum(pos_z * heatmap[i], dim=(1,2,3,4))

        points.append(torch.stack([pz,py,px]).T)

    coords = torch.stack(points)
    return coords

def heatmap_to_grade(heatmap):
    num_image = len(heatmap)
    grade = []
    for i in range(num_image):
        num_point, num_grade, D, H, W = heatmap[i].shape
        g = torch.sum(heatmap[i], dim=(2,3,4))
        grade.append(g)
    grade = torch.stack(grade) # (num_image, num_point, num_grade)
    return grade