import numpy as np
from spine.spine_exam import InstanceMeta

def normalise_to_8bit(x, lower=0.1, upper=99.9): # 1, 99 #0.05, 99.5 #0, 100
    lower, upper = np.percentile(x, (lower, upper))
    x = np.clip(x, lower, upper)
    x = x - np.min(x)
    x = x / np.max(x)
    return (x * 255).astype(np.uint8)

def normalise_to_01(x, lower=0.1, upper=99.9): # 1, 99 #0.05, 99.5 #0, 100
    lower, upper = np.percentile(x, (lower, upper))
    x = np.clip(x, lower, upper)
    x -= np.min(x)
    x /= np.max(x)
    return x


def image_to_patient_coords_3d(x, y, 
               image_position_patient: np.ndarray,
               image_orientation_patient:np.ndarray,
               pixel_spacing:np.ndarray) -> np.ndarray:
    ax = np.array(image_orientation_patient[:3])
    ay = np.array(image_orientation_patient[3:])

    return image_position_patient+ax*x*pixel_spacing[0]+ay*y*pixel_spacing[1]



#back project 3D to 2d
def patient_coords_to_image_2d(coords:np.ndarray, instance_meta:InstanceMeta,
                               return_if_contains=False) -> np.ndarray:
    xx, yy, zz = coords
    sx, sy, sz = instance_meta.position
    o0, o1, o2, o3, o4, o5, = instance_meta.orientation
    delx, dely = instance_meta.pixel_spacing
    delz = instance_meta.slice_thickness

    ax = np.array([o0,o1,o2])
    ay = np.array([o3,o4,o5])
    az = np.cross(ax,ay)

    p = np.array([xx-sx,yy-sy,zz-sz])
    x = np.dot(ax, p)/delx
    y = np.dot(ay, p)/dely
    z = np.dot(az, p)/delz
    x = int(round(x))
    y = int(round(y))
    z = int(round(z))

    if return_if_contains:
        if x<0 or x>=instance_meta.cols or y<0 or y>=instance_meta.rows or z<-0.5 or z>0.5:
            return np.array([x,y,z]), False
        else:
            return np.array([x,y,z]), True

    return np.array([x,y,z]) 