import numpy as np

def prob_map_from_coord(coord, grid_shape=(64, 64), sigma=1.0, normalize=True):
    """
    Create a 2D probability map (Gaussian) centered at 'coord' on a grid.

    coord: (row, col) in grid coordinates (0-based).
    grid_shape: (H, W) output map size.
    sigma: standard deviation of the Gaussian (same in both axes).
    normalize: if True, sum of all values equals 1.
    """
    if not (isinstance(coord, (list, tuple, np.ndarray)) and len(coord) == 2):
        raise ValueError("coord must be a sequence of (row, col).")

    H, W = int(grid_shape[0]), int(grid_shape[1])
    cy, cx = float(coord[0]), float(coord[1])

    y = np.arange(H)
    x = np.arange(W)
    X, Y = np.meshgrid(x, y)

    # Gaussian centered at (cx, cy)
    gauss = np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2.0 * (sigma ** 2)))

    if normalize:
        s = gauss.sum()
        if s > 0:
            gauss = gauss / s

    return gauss