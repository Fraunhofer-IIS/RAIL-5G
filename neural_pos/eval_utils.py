import numpy as np
from sklearn.manifold import trustworthiness
import torch
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix

def plot_cdf(
    distances,
    max_points: int = 1000,
    percentiles: list[int] = [50, 90, 95],
    max_x: float | None = None,
    figsize = None
) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize = figsize)

    # --- data -------------------------------------------------------------
    dist_t = torch.tensor(distances)
    sorted_dist = torch.sort(dist_t).values
    cdf = torch.arange(1, len(sorted_dist) + 1) / len(sorted_dist)

    if len(sorted_dist) > max_points:
        idx = torch.linspace(0, len(sorted_dist) - 1, steps=max_points).long()
        sorted_dist, cdf = sorted_dist[idx], cdf[idx]

    ax.plot(sorted_dist.numpy(), cdf.numpy(), label="CDF")

    # --- CE lines ---------------------------------------------------------
    for i, ce in enumerate(percentiles):
        ce_val = torch.quantile(dist_t, ce / 100).item()

        # vertical then horizontal “L”
        ax.plot([ce_val, ce_val], [0, ce / 100], ls="--", color=plt.cm.tab10.colors[i])
        ax.plot([0, ce_val],     [ce / 100, ce / 100], ls="--", color=plt.cm.tab10.colors[i],
                label=f"CE{ce} = {ce_val:.2f}")

    # --- cosmetics --------------------------------------------------------
    ax.set_xlabel("Distance [m]")
    ax.set_ylabel("CDF")
    ax.set_ylim(0, 1)
    ax.set_xlim(left=0)
    if max_x is not None:
        ax.set_xlim(0, max_x)

    ax.legend(loc="lower right")
    ax.grid(True, ls=":", alpha=0.5)
    return fig, ax

def plot_trajectory(prediction, ground_truth, figsize = None):
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(ground_truth[:,0], ground_truth[:,1], label="Ground Truth", linewidth=0.8, color=plt.cm.tab10.colors[0])
    ax.scatter(prediction[:,0], prediction[:,1], s=10, label="Prediction", color=plt.cm.tab10.colors[1])
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.axis("equal")
    ax.legend()
    plt.tight_layout()
    return fig, ax

def plot_hexbin(ground_truth, error, figsize=None, vmin = None, vmax = None):
    fig, ax = plt.subplots(figsize=figsize)
    hb = ax.hexbin(ground_truth[:,0], ground_truth[:,1], error, vmin=vmin, vmax=vmax)
    plt.colorbar(hb, ax=ax, label="Error [m]")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.axis("equal")
    fig.legend()
    fig.tight_layout()
    return fig, ax

def plot_colorized(positions, groundtruth_positions, title = None, show = False, alpha = 1.0):
     
    # Generate RGB colors for datapoints
    center_point = np.zeros(2, dtype = np.float32)
    center_point[0] = 0.5 * (np.min(groundtruth_positions[:, 0], axis = 0) + np.max(groundtruth_positions[:, 0], axis = 0))
    center_point[1] = 0.5 * (np.min(groundtruth_positions[:, 1], axis = 0) + np.max(groundtruth_positions[:, 1], axis = 0))
    NormalizeData = lambda in_data : (in_data - np.min(in_data)) / (np.max(in_data) - np.min(in_data))
    rgb_values = np.zeros((groundtruth_positions.shape[0], 3))
    rgb_values[:, 0] = 1 - 0.9 * NormalizeData(groundtruth_positions[:, 0])
    rgb_values[:, 1] = 0.8 * NormalizeData(np.square(np.linalg.norm(groundtruth_positions - center_point, axis=1)))
    rgb_values[:, 2] = 0.9 * NormalizeData(groundtruth_positions[:, 1])

    # Plot datapoints
    figure = plt.figure(figsize=(6, 6))
    if title is not None:
        plt.title(title, fontsize=16)
    plt.scatter(positions[:, 0], positions[:, 1], c = rgb_values, alpha = alpha, s = 10, linewidths = 0)
    plt.xlabel("x coordinate")
    plt.ylabel("y coordinate")
    if show:
        plt.show()
    return figure


def affine_transform_channel_chart(groundtruth_pos, channel_chart_pos):
    pad = lambda x: np.hstack([x, np.ones((x.shape[0], 1))])
    unpad = lambda x: x[:,:-1]
    A, res, rank, s = np.linalg.lstsq(pad(channel_chart_pos), pad(groundtruth_pos), rcond = None)
    transform = lambda x: unpad(np.dot(pad(x), A))
    return transform(channel_chart_pos)

def continuity(*args, **kwargs):
	args = list(args)
	args[0], args[1] = args[1], args[0]
	return trustworthiness(*args, **kwargs)

def kruskal_stress(X, X_embedded):
	dist_X = distance_matrix(X, X)
	dist_X_embedded = distance_matrix(X_embedded, X_embedded)
	beta = np.divide(np.sum(dist_X * dist_X_embedded), np.sum(dist_X_embedded * dist_X_embedded))

	return np.sqrt(np.divide(np.sum(np.square((dist_X - beta * dist_X_embedded))), np.sum(dist_X * dist_X)))