import numpy as np
import cv2
import time
from collision_detection import create_quadtree, narrow_phase
from physics import apply_forces
from visualization import draw


if __name__ == "__main__":
    cell_count = 1000
    positions = np.random.normal(loc=0.0, scale=0.2, size=(cell_count, 2))
    velocities = np.zeros_like(positions)
    radii = np.random.uniform(low=0.003, high=0.01, size=(cell_count,))
    masses = radii ** 2 * np.pi

    expand_threshold = 50

    velocity_dampening = 0.75

    cell_firmness = 0.05
    map_boundary_firmness = 0.05
    map_radius = 1.0

    img_res = (1440, 900)
    zoom = 400.0

    last_simulation = time.perf_counter()
    last_draw = last_simulation
    while True:
        radii_ = radii[:, None]
        bounding_boxes = np.stack([positions - radii_, positions + radii_], axis=1)
        quadtree = create_quadtree(bounding_boxes, expand_threshold, 0.01)
        collision_set = narrow_phase(quadtree, positions, radii)

        forces = apply_forces(
            positions, velocities, radii, cell_firmness, map_boundary_firmness, map_radius, collision_set)

        current_time = time.perf_counter()
        delta_time = current_time - last_simulation
        last_simulation = current_time

        velocities = (1 - delta_time * velocity_dampening) * velocities + delta_time * forces / masses[:, None]
        positions += delta_time * velocities

        if last_simulation - last_draw > 0.016:
            last_draw = last_simulation
            img = draw(quadtree, collision_set, positions, radii, map_radius, img_res, zoom)
            cv2.imshow("image", img)
            cv2.waitKey(delay=1)
