import cv2
import torch
import numpy as np


def tensor2cv2image(x: torch.tensor) -> np.array:
    x = x.squeeze()
    x = x.permute(1, 2, 0)
    x = x.cpu().numpy()
    x *= 255
    x = np.uint8(x)
    return x


def image_array2tensor(image: np.array) -> torch.tensor:
    image = np.array(image, dtype=float)
    image /= 255
    image = torch.tensor(image, dtype=torch.float32)
    image = image.permute(2, 0, 1)
    image = image.unsqueeze(0)
    return image


def scale_bbox(bx1, by1, bx2, by2, ratio=3):
    cx = (bx1 + bx2) / 2
    cy = (by1 + by2) / 2
    len_x = cx - bx1
    len_y = cy - by1
    len_x /= ratio
    len_y /= ratio
    return int(cx - len_x), int(cy - len_y), int(cx + len_x), int(cy + len_y)


def get_size_of_bbox(bx1, by1, bx2, by2):
    return torch.Size([bx2 - bx1, by2 - by1])


def assert_bbox(bx1, by1, bx2, by2):
    if bx2 - bx1 <= 10:
        return False
    if by2 - by1 <= 10:
        return False
    return True
