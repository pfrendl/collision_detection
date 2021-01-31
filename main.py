from typing import List, Tuple, Optional
import numpy as np
from dataclasses import dataclass
import cv2


@dataclass
class QuadTree:
    min_point: np.ndarray
    max_point: np.ndarray
    cell_indices: Optional[List[int]]
    children: Optional[Tuple["QuadTree", "QuadTree", "QuadTree", "QuadTree"]]


def quadtree_insert(
        quadtree: QuadTree,
        positions: np.ndarray,
        sizes: np.ndarray,
        idx: int,
        expand_threshold: int
) -> None:
    if quadtree.cell_indices is None:
        position = positions[idx]
        size = sizes[idx]
        half_size = size / 2
        x_min, y_min = position - half_size
        x_max, y_max = position + half_size
        center = (quadtree.min_point + quadtree.max_point) / 2
        if x_min < center[0]:
            if y_min < center[1]:
                quadtree_insert(quadtree.children[0], positions, sizes, idx, expand_threshold)
            if y_max >= center[1]:
                quadtree_insert(quadtree.children[1], positions, sizes, idx, expand_threshold)
        if x_max >= center[0]:
            if y_min < center[1]:
                quadtree_insert(quadtree.children[2], positions, sizes, idx, expand_threshold)
            if y_max >= center[1]:
                quadtree_insert(quadtree.children[3], positions, sizes, idx, expand_threshold)
    else:
        quadtree.cell_indices.append(idx)
        if len(quadtree.cell_indices) >= expand_threshold:
            children = []
            for i in range(2):
                for j in range(2):
                    half_size = (quadtree.max_point - quadtree.min_point) / 2
                    child = QuadTree(
                        np.array([quadtree.min_point[0] + i * half_size[0], quadtree.min_point[1] + j * half_size[1]]),
                        np.array([quadtree.min_point[0] + (i + 1) * half_size[0], quadtree.min_point[1] + (j + 1) * half_size[1]]),
                        [], None)
                    children.append(child)
            quadtree.children = tuple(children)
            cell_indices = quadtree.cell_indices
            quadtree.cell_indices = None
            for cell_idx in cell_indices:
                quadtree_insert(quadtree, positions, sizes, cell_idx, expand_threshold)


def create_quadtree(positions: np.ndarray, sizes: np.ndarray, expand_threshold: int) -> QuadTree:
    quadtree = QuadTree(
        np.array([-1, -1], dtype=np.float),
        np.array([1, 1], dtype=np.float),
        [], None)
    for idx in range(positions.shape[0]):
        quadtree_insert(quadtree, positions, sizes, idx, expand_threshold)
    return quadtree


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


if __name__ == "__main__":
    cell_count = 1000
    positions = np.random.normal(loc=0.0, scale=0.2, size=(cell_count, 2))
    # positions = np.random.uniform(low=-1, high=1, size=(cell_count, 2))
    sizes = np.random.uniform(low=0.001, high=0.01, size=(cell_count,))
    expand_threshold = 10
    quadtree = create_quadtree(positions, sizes, expand_threshold)

    img_res = (960, 780)
    zoom = 350.0
    img = draw(quadtree, positions, sizes, img_res, zoom)
    while True:
        cv2.imshow("image", img)
        cv2.waitKey(delay=1000)
