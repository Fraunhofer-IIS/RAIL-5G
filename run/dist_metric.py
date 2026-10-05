import json
from pathlib import Path

import hydra
from matplotlib import pyplot as plt
import numpy as np
from omegaconf import OmegaConf
from scipy.sparse.csgraph import connected_components

from neural_pos.cc_metrics import CCMetrics
from neural_pos.cc_metrics.utils import calculate_geodesic_distances_condensed, plot_dissimilarity_over_euclidean_distance_condensed
from neural_pos.datasets.dist_metric_ds import DistMetricDataset
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
import pandas as pd

@hydra.main(version_base=None, config_path="../configs/dist_metric", config_name="default")
def calc_dist(cfg):
    in_ds = DistMetricDataset(**cfg["dataset_config"])

    cirs = in_ds.csi
    ts = in_ds.timestamps
    ref = in_ds.ref

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cc_metric = CCMetrics[cfg["metric"]["metric_name"]](**cfg["metric"]["metric_params"])

    cc_dist = cc_metric(cirs, timestamps=ts)

    import gc; gc.collect

    is_nan = np.isnan(cc_dist).sum()
    is_inf = np.isinf(cc_dist).sum()
    print(f"NaNs: ({is_nan} / {len(cc_dist)} ({is_nan/len(cc_dist):.2f}))")
    print(f"Infs: ({is_inf} / {len(cc_dist)} ({is_inf/len(cc_dist):.2f}))")
    cc_dist = np.nan_to_num(cc_dist, nan=1e10)

    if cfg["metric"]["geodesic"] is True:
        n_neighbors = cfg["metric"]["geodesic_neighbors"]

        nbrs_alg = NearestNeighbors(n_neighbors = n_neighbors, metric="precomputed", n_jobs = -1)
        nbrs = nbrs_alg.fit(squareform(cc_dist))
        nbg = kneighbors_graph(nbrs, n_neighbors, metric = "precomputed", mode="distance")

        n_components = connected_components(nbg, directed=False, return_labels=False)
        if n_components != 1:
            print(f"[WARN] Geodesic graph has {n_components} disconnected subgraphs")

        geodesic_matrix = calculate_geodesic_distances_condensed(nbg)
        is_inf_g = np.isinf(geodesic_matrix).sum()
        print(f"Infs Geodesic: ({is_inf_g} / {len(geodesic_matrix)} ({is_inf_g/len(geodesic_matrix):.2f}))")

    else:
        geodesic_matrix = None

    plot_sample_fraction = 0.2

    if ref is not None:
        metric_name = cfg["metric"]["metric_name"]
        real_distances = pdist(ref, metric='euclidean')

        fig, ax1 = plt.subplots(figsize=(8, 5))

        plot_dissimilarity_over_euclidean_distance_condensed(cc_dist, real_distances, label=metric_name, ax=ax1, color='C0', sample_fraction=plot_sample_fraction)
        ax1.set_ylabel(f'{metric_name} Dissimilarity', color='C0')
        ax1.tick_params(axis='y', labelcolor='C0')
        if geodesic_matrix is not None:
            ax2 = ax1.twinx()
            plot_dissimilarity_over_euclidean_distance_condensed(geodesic_matrix, real_distances, label=f"Geodesic {metric_name}", ax=ax2, color='C1', sample_fraction=plot_sample_fraction)
            ax2.set_ylabel(f"Geodesic {metric_name} Dissimilarity", color='C1')
            ax2.tick_params(axis='y', labelcolor='C1')

        plt.tight_layout()
        plt.savefig(output_dir / "metric_vs_real.pdf")


        plt.figure(figsize=(8, 5))
        plt.xlabel("True Distance [m]")
        plt.ylabel("Distance Mean [m]")
        plt.hist2d(real_distances.flatten(), cc_dist.flatten(), bins = 100)
        plt.savefig(output_dir / "metric_vs_real_hist.pdf")

        if geodesic_matrix is not None:
            plt.figure(figsize=(8, 5))
            plt.xlabel("True Distance [m]")
            plt.ylabel("Geodesic Distance Mean [m]")
            plt.hist2d(real_distances.flatten(), geodesic_matrix.flatten(), bins = 100)
            plt.savefig(output_dir / "geodesic_metric_vs_real_hist.pdf")
    else:
        print("[WARN] No reference available, skipping eval...")

    if not cfg["eval_only"]:
        output_dists = geodesic_matrix if geodesic_matrix is not None else cc_dist
        gen_params = json.dumps(OmegaConf.to_container(cfg, resolve=True))  # Ensure everything is serializable by h5py
        in_ds.write(output_dir / "cc.h5", output_dists, gen_params = gen_params)
    

if __name__ == "__main__":
    calc_dist()