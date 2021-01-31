import numpy as np
import cv2
from collision_detection import create_quadtree, narrow_phase
from visualization import draw


if __name__ == "__main__":
    cell_count = 1000
    positions = np.random.normal(loc=0.0, scale=0.2, size=(cell_count, 2))
    velocities = np.zeros_like(positions)
    sizes = np.random.uniform(low=0.001, high=0.01, size=(cell_count,))
    expand_threshold = 10

    quadtree = create_quadtree(positions, sizes, expand_threshold, 0.01)
    collision_set = narrow_phase(quadtree, positions, sizes)

    img_res = (1440, 900)
    zoom = 400.0
    while True:
        # simulation step here

        img = draw(quadtree, collision_set, positions, sizes, img_res, zoom)
        cv2.imshow("image", img)
        cv2.waitKey(delay=1000)
