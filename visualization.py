import cv2
import numpy as np


def draw_collisions(
    img: np.ndarray,
    collision_set: np.ndarray,
    positions: np.ndarray,
    radii: np.ndarray,
) -> None:
    for collision in collision_set:
        for idx in collision:
            cv2.circle(img, tuple(positions[idx]), radii[idx], (0, 0, 255), -1, cv2.LINE_AA)


def draw_cells(img: np.ndarray, positions: np.ndarray, radii: np.ndarray) -> None:
    for position, radius in zip(positions, radii):
        cv2.circle(img, tuple(position), radius, (0.0, 0.0, 0.0), -1, cv2.LINE_AA)


def draw(
    collision_set: np.ndarray,
    positions: np.ndarray,
    radii: np.ndarray,
    map_radius: float,
    img_res: tuple[int, int],
    zoom: float,
) -> np.ndarray:

    img_res_np = np.array(img_res)
    principal_point = img_res_np / 2
    img = np.full((*img_res_np[::-1], 3), 77, dtype=np.uint8)

    cv2.circle(img, tuple(principal_point.astype(int)), int(zoom * map_radius), (255, 255, 255), -1, cv2.LINE_AA)

    positions = (zoom * positions + principal_point).astype(int)
    radii = (zoom * radii).astype(int)
    draw_cells(img, positions, radii)
    draw_collisions(img, collision_set, positions, radii)

    return img
