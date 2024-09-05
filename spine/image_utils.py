import numpy as np
import cv2
from typing import Tuple

def resize_image_with_pad(image, target_size) -> Tuple[np.ndarray, float]:
    h, w = target_size
    h_1, w_1 = image.shape[:2]
    ratio_f = w / h
    ratio_1 = w_1 / h_1

    # check if the original and final aspect ratios are the same within a margin
    if round(ratio_1, 2) != round(ratio_f, 2):
        # padding to preserve aspect ratio
        hp = int(w_1/ratio_f - h_1)
        wp = int(ratio_f * h_1 - w_1)
        if hp > 0 and wp < 0:
            scale = w/w_1
            image = cv2.copyMakeBorder(image, 0, hp, 0, 0, cv2.BORDER_CONSTANT, value=0)
        elif hp < 0 and wp > 0:
            scale = h/h_1
            image = cv2.copyMakeBorder(image, 0, 0, 0, wp, cv2.BORDER_CONSTANT, value=0)
    else:
        scale = h/h_1

    return cv2.resize(image, (w,h), interpolation = cv2.INTER_LINEAR), scale

def crop_or_pad(image: np.ndarray, target_size):
    h, w = target_size
    h_1, w_1 = image.shape
    if w_1 > w and h_1 > h:
        return image[:h, :w]
    else:
        new_image = np.zeros((h, w), dtype=image.dtype)
        wm = min(w, w_1)
        hm = min(h, h_1)
        new_image[:hm, :wm] = image[:hm, :wm]
        return new_image

