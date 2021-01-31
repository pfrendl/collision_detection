import numpy as np
import cv2
from quadtrees import create_quadtree
from visualization import draw


if __name__ == "__main__":
    cell_count = 1000
    positions = np.random.normal(loc=0.0, scale=0.2, size=(cell_count, 2))
    sizes = np.random.uniform(low=0.001, high=0.01, size=(cell_count,))
    expand_threshold = 10
    quadtree = create_quadtree(positions, sizes, expand_threshold, 0.01)

    img_res = (960, 780)
    zoom = 350.0
    img = draw(quadtree, positions, sizes, img_res, zoom)
    while True:
        cv2.imshow("image", img)
        cv2.waitKey(delay=1000)
