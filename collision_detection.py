import itertools

import numpy as np


def inter_axis(entries):
    open_entries = set()
    inter = set()
    for entry in entries:
        if entry[1] < 0:
            id = -entry[1] - 1
            for open_entry in open_entries:
                inter.add((open_entry, id) if open_entry < id else (id, open_entry))
            open_entries.add(id)
        else:
            open_entries.remove(entry[1] - 1)
    return inter


def sweep_and_prune(bounding_boxes: np.ndarray) -> set[tuple[int, int]]:
    x_mins = bounding_boxes[:, 0, 0].tolist()
    y_mins = bounding_boxes[:, 0, 1].tolist()
    x_maxes = bounding_boxes[:, 1, 0].tolist()
    y_maxes = bounding_boxes[:, 1, 1].tolist()
    idxs = np.arange(start=1, stop=bounding_boxes.shape[0] + 1)
    nidxs = -idxs
    idxs = idxs.tolist()
    nidxs = nidxs.tolist()
    x_mins = zip(x_mins, nidxs)
    y_mins = zip(y_mins, nidxs)
    x_maxes = zip(x_maxes, idxs)
    y_maxes = zip(y_maxes, idxs)

    x_entries = sorted(itertools.chain(x_mins, x_maxes), key=lambda x: x[0])
    y_entries = sorted(itertools.chain(y_mins, y_maxes), key=lambda x: x[0])

    x_inter = inter_axis(x_entries)
    y_inter = inter_axis(y_entries)
    collision_set = set.intersection(x_inter, y_inter)

    return collision_set


def narrow_phase(collision_set: set[tuple[int, int]], positions: np.ndarray, radii: np.ndarray) -> np.ndarray:
    collision_set_np = np.array(list(collision_set), dtype=np.int64).reshape((-1, 2))
    left, right = collision_set_np[:, 0], collision_set_np[:, 1]
    position_deltas = positions[left] - positions[right]
    distances = np.linalg.norm(position_deltas, axis=1)
    touch_distances = radii[left] + radii[right]
    collision_depths = np.clip(touch_distances - distances, a_min=0, a_max=None)
    collision_set_np = collision_set_np[collision_depths > 0]
    return collision_set_np
