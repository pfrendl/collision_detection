from typing import Tuple
import numpy as np
from quadtrees import QuadTree
import cv2


def draw_quadtree(img: np.ndarray, quadtree: QuadTree, zoom: float, principal_point: np.ndarray) -> None:
    if quadtree.cell_indices is None:
        for child in quadtree.children:
            draw_quadtree(img, child, zoom, principal_point)
    else:
        pt1 = tuple((zoom * quadtree.min_point + principal_point).astype(np.int))
        pt2 = tuple((zoom * quadtree.max_point + principal_point).astype(np.int))
        cv2.rectangle(img, pt1, pt2, (1.0, 1.0, 1.0), 1)


def draw(quadtree: QuadTree, positions: np.ndarray, sizes: np.ndarray, img_res: Tuple[int, int], zoom: float):
    img_res = np.array(img_res)
    principal_point = img_res / 2
    img = np.zeros((*img_res[::-1], 3), dtype=np.float32)
    draw_quadtree(img, quadtree, zoom, principal_point)
    for position, size in zip(positions, sizes):
        center = tuple((zoom * position + principal_point).astype(np.int))
        radius = int(zoom * size / 2)
        cv2.circle(img, center, radius, (1.0, 1.0, 1.0), -1)
    return img
