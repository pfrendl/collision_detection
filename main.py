import numpy as np
import cv2
import time
from collision_detection import create_quadtree, narrow_phase
from visualization import draw


if __name__ == "__main__":
    cell_count = 1000
    positions = np.random.normal(loc=0.0, scale=0.2, size=(cell_count, 2))
    velocities = np.zeros_like(positions)
    radii = np.random.uniform(low=0.0005, high=0.01, size=(cell_count,))
    expand_threshold = 10

    velocity_dampening = 0.2

    cell_firmness = 100
    map_boundary_firmness = 100
    map_radius = 1.0

    img_res = (1440, 900)
    zoom = 400.0

    last_simulation = time.perf_counter()
    last_draw = last_simulation
    while True:
        quadtree = create_quadtree(positions, radii, expand_threshold, 0.01)
        collision_set = narrow_phase(quadtree, positions, radii)

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
            collision_force_directions = position_deltas / np.where(position_deltas > 0, distances, 1)
            collision_forces = collision_force_lengths * collision_force_directions
            for collision_idx, (idx_i, idx_j) in enumerate(collision_set):
                collision_force = collision_forces[collision_idx]
                forces[idx_i] += collision_force
                forces[idx_j] -= collision_force

        position_lengths = np.linalg.norm(positions, axis=1, keepdims=True)
        distances_to_map_edge = np.clip(position_lengths - map_radius, a_min=0, a_max=None)
        collision_force_lengths = (cell_firmness + map_boundary_firmness) / 2 * distances_to_map_edge
        collision_force_directions = positions / np.where(position_lengths > 0, position_lengths, 1)
        collision_forces = collision_force_lengths * collision_force_directions
        forces -= collision_forces

        current_time = time.perf_counter()
        delta_time = current_time - last_simulation
        last_simulation = current_time

        velocities = (1 - delta_time * velocity_dampening) * velocities + delta_time * forces
        positions += delta_time * velocities

        if current_time - last_draw > 0.016:
            last_draw = current_time
            img = draw(quadtree, collision_set, positions, radii, map_radius, img_res, zoom)
            cv2.imshow("image", img)
            cv2.waitKey(delay=1)
