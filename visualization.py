from typing import Tuple
import numpy as np
import cv2


def draw_collisions(
        img: np.ndarray,
        collision_set: np.ndarray,
        positions: np.ndarray,
        radii: np.ndarray,
        zoom: float,
        principal_point: np.ndarray
) -> None:
    for collision in collision_set:
        for idx in collision:
            center = tuple((zoom * positions[idx] + principal_point).astype(np.int))
            radius = int(zoom * radii[idx])
            cv2.circle(img, center, radius, (0, 0, 255), -1, cv2.LINE_AA)


def draw_cells(
        img: np.ndarray,
        positions: np.ndarray,
        radii: np.ndarray,
        zoom: float,
        principal_point: np.ndarray
) -> None:
    for position, radius in zip(positions, radii):
        center = tuple((zoom * position + principal_point).astype(np.int))
        radius = int(zoom * radius)
        cv2.circle(img, center, radius, (0.0, 0.0, 0.0), -1, cv2.LINE_AA)


def draw(
        collision_set: np.ndarray,
        positions: np.ndarray,
        radii: np.ndarray,
        map_radius: float,
        img_res: Tuple[int, int],
        zoom: float
) -> np.ndarray:
    img_res = np.array(img_res)
    principal_point = img_res / 2
    img = np.full((*img_res[::-1], 3), 77, dtype=np.uint8)
    cv2.circle(img, tuple(principal_point.astype(np.int)), int(zoom * map_radius), (255, 255, 255), -1, cv2.LINE_AA)
    draw_cells(img, positions, radii, zoom, principal_point)
    draw_collisions(img, collision_set, positions, radii, zoom, principal_point)
    return img
