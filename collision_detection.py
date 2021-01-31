from typing import Set, Tuple, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class QuadTree:
    min_point: np.ndarray
    max_point: np.ndarray
    cell_indices: Optional[np.ndarray]
    children: Optional[Tuple["QuadTree", "QuadTree", "QuadTree", "QuadTree"]]


def expand_quadtree(
        quadtree: QuadTree,
        positions: np.ndarray,
        radii: np.ndarray,
        expand_threshold: int,
        min_half_size: float
) -> None:
    if len(quadtree.cell_indices) >= expand_threshold:
        half_size = (quadtree.max_point - quadtree.min_point) / 2
        if half_size[0] < min_half_size:
            return

        cell_indices = quadtree.cell_indices
        quadtree.cell_indices = None

        center = quadtree.min_point + half_size
        idx_positions = positions[cell_indices]
        idx_radii = radii[cell_indices, None]
        idx_mins = idx_positions - idx_radii
        idx_maxes = idx_positions + idx_radii
        min_mask = idx_mins < center
        max_mask = idx_maxes >= center
        xy_mask = np.stack([min_mask, max_mask], axis=2)
        cases = xy_mask[:, 0, :, None] @ xy_mask[:, 1, None, :]

        children = []
        for i in range(2):
            for j in range(2):
                child = QuadTree(
                    np.array([
                        x0 := quadtree.min_point[0] + i * half_size[0],
                        y0 := quadtree.min_point[1] + j * half_size[1]]),
                    np.array([x0 + half_size[0], y0 + half_size[1]]),
                    cell_indices[cases[:, i, j]], None)
                expand_quadtree(child, positions, radii, expand_threshold, min_half_size)
                children.append(child)
        quadtree.children = tuple(children)


def create_quadtree(
        positions: np.ndarray,
        radii: np.ndarray,
        expand_threshold: int,
        min_half_size: float
) -> QuadTree:
    quadtree = QuadTree(
        np.array([-1, -1], dtype=np.float),
        np.array([1, 1], dtype=np.float),
        np.arange(positions.shape[0]), None)
    expand_quadtree(quadtree, positions, radii, expand_threshold, min_half_size)
    return quadtree


def narrow_phase(quadtree: QuadTree, positions: np.ndarray, radii: np.ndarray) -> Set[Tuple[int, int]]:
    if quadtree.cell_indices is None:
        collision_sets = [narrow_phase(child, positions, radii) for child in quadtree.children]
        collision_set = set.union(*collision_sets)
    else:
        collision_set = set()
        for i in range(len(quadtree.cell_indices) - 1):
            idx_i = quadtree.cell_indices[i]
            position_i = positions[idx_i]
            radius_i = radii[idx_i]
            for j in range(i + 1, len(quadtree.cell_indices)):
                idx_j = quadtree.cell_indices[j]
                touch_distance = radius_i + radii[idx_j]
                delta = position_i - positions[idx_j]
                distance = np.linalg.norm(delta)
                if distance < touch_distance:
                    collision_set.add((idx_i, idx_j))
    return collision_set
