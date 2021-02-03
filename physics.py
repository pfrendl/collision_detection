from typing import Set, Tuple
import numpy as np


def apply_forces(
        positions: np.ndarray,
        velocities: np.ndarray,
        radii: np.ndarray,
        cell_firmness: float,
        map_boundary_firmness: float,
        map_radius: float,
        collision_set: Set[Tuple[int, int]]
) -> np.ndarray:
    forces = np.zeros_like(velocities)
    if collision_set:
        left, right = zip(*collision_set)
        left = list(left)
        right = list(right)
        position_deltas = positions[left] - positions[right]
        distances = np.linalg.norm(position_deltas, axis=1, keepdims=True)
        touch_distances = radii[left] + radii[right]
        collision_depths = np.clip(touch_distances[:, None] - distances, a_min=0, a_max=None)
        collision_force_lengths = cell_firmness * collision_depths
        collision_force_directions = position_deltas / np.where(distances > 0, distances, 1)
        collision_forces = collision_force_lengths * collision_force_directions
        np.add.at(forces, left, collision_forces)
        np.add.at(forces, right, -collision_forces)

    position_lengths = np.linalg.norm(positions, axis=1, keepdims=True)
    distances_to_map_edge = np.clip(position_lengths - map_radius, a_min=0, a_max=None)
    collision_force_lengths = (cell_firmness + map_boundary_firmness) / 2 * distances_to_map_edge
    collision_force_directions = positions / np.where(position_lengths > 0, position_lengths, 1)
    collision_forces = collision_force_lengths * collision_force_directions
    forces -= collision_forces

    return forces
