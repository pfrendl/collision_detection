import time

import cv2
import numpy as np

from collision_detection import narrow_phase, sweep_and_prune
from physics import apply_forces
from visualization import draw


if __name__ == "__main__":
    cell_count = 1000
    positions = np.random.normal(loc=0.0, scale=0.2, size=(cell_count, 2))
    velocities = np.zeros_like(positions)
    radii = np.random.uniform(low=0.003, high=0.01, size=(cell_count,))
    masses = radii ** 2 * np.pi

    velocity_dampening = 0.75

    cell_firmness = 10.0
    map_boundary_firmness = 10.0
    map_radius = 1.0

    img_res = (1440, 900)
    zoom = 400.0

    last_simulation = time.perf_counter()
    last_draw = last_simulation
    while True:
        radii_ = radii[:, None]
        bounding_boxes = np.stack([positions - radii_, positions + radii_], axis=1)
        collision_set = sweep_and_prune(bounding_boxes)
        collision_set_np = narrow_phase(collision_set, positions, radii)

        forces = apply_forces(
            positions, velocities, radii, cell_firmness, map_boundary_firmness, map_radius, collision_set_np
        )

        current_time = time.perf_counter()
        delta_time = current_time - last_simulation
        last_simulation = current_time

        velocities = (1 - delta_time * velocity_dampening) * velocities + delta_time * forces / masses[:, None]
        positions += delta_time * velocities

        if last_simulation - last_draw > 0.016:
            last_draw = last_simulation
            img = draw(collision_set_np, positions, radii, map_radius, img_res, zoom)
            cv2.imshow("image", img)
            cv2.waitKey(delay=1)
