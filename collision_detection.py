from typing import List, Set, Tuple, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class QuadTree:
    min_point: np.ndarray
    max_point: np.ndarray
    cell_indices: Optional[List[int]]
    children: Optional[Tuple["QuadTree", "QuadTree", "QuadTree", "QuadTree"]]


def expand_quadtree(quadtree: QuadTree, positions: np.ndarray, sizes: np.ndarray, expand_threshold: int, min_half_size: float):
    if len(quadtree.cell_indices) >= expand_threshold:
        children = []
        half_size = (quadtree.max_point - quadtree.min_point) / 2
        if half_size[0] < min_half_size:
            return
        for i in range(2):
            for j in range(2):
                child = QuadTree(
                    np.array([
                        quadtree.min_point[0] + i * half_size[0],
                        quadtree.min_point[1] + j * half_size[1]]),
                    np.array([
                        quadtree.min_point[0] + (i + 1) * half_size[0],
                        quadtree.min_point[1] + (j + 1) * half_size[1]]),
                    [], None)
                children.append(child)
        quadtree.children = tuple(children)
        cell_indices = quadtree.cell_indices
        quadtree.cell_indices = None

        center = children[0].max_point
        for idx in cell_indices:
            position = positions[idx]
            half_size = sizes[idx] / 2
            x_min, y_min = position - half_size
            x_max, y_max = position + half_size
            if x_min < center[0]:
                if y_min < center[1]:
                    children[0].cell_indices.append(idx)
                if y_max >= center[1]:
                    children[1].cell_indices.append(idx)
            if x_max >= center[0]:
                if y_min < center[1]:
                    children[2].cell_indices.append(idx)
                if y_max >= center[1]:
                    children[3].cell_indices.append(idx)

        for child in children:
            expand_quadtree(child, positions, sizes, expand_threshold, min_half_size)


def create_quadtree(positions: np.ndarray, sizes: np.ndarray, expand_threshold: int, min_half_size: float) -> QuadTree:
    quadtree = QuadTree(
        np.array([-1, -1], dtype=np.float),
        np.array([1, 1], dtype=np.float),
        list(range(positions.shape[0])), None)
    expand_quadtree(quadtree, positions, sizes, expand_threshold, min_half_size)
    return quadtree


def narrow_phase(quadtree: QuadTree, positions: np.ndarray, sizes: np.ndarray) -> Set[Tuple[int, int]]:
    if quadtree.cell_indices is None:
        collision_sets = [narrow_phase(child, positions, sizes) for child in quadtree.children]
        collision_set = set.union(*collision_sets)
    else:
        collision_set = set()
        for i in range(len(quadtree.cell_indices) - 1):
            idx_i = quadtree.cell_indices[i]
            position_i = positions[idx_i]
            size_i = sizes[idx_i]
            for j in range(i + 1, len(quadtree.cell_indices)):
                idx_j = quadtree.cell_indices[j]
                touch_distance = (size_i + sizes[idx_j]) / 2
                delta = position_i - positions[idx_j]
                distance = np.linalg.norm(delta)
                if distance < touch_distance:
                    collision_set.add((idx_i, idx_j))
    return collision_set
