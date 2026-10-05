# %%
import numpy as np
from scipy.constants import c

import neural_pos.datasets.fp_ds as ds
from neural_pos.tdoa import tdoa_batch


# %%

if __name__ == "__main__":

    ds.FP_DATASET_NAME = "FP"

    dataset = ds.FingerprintingDataset(
        file_path="/Users/pirkl/Code/datasets/Bischheim/final/test.h5",
        tdoa_correction={"kernel_size": 500}
    )

    # Define antenna positions (example: 4 antennas in 2D)

    antenna_positions = np.array([
        [15690, 11341, 3875],
        [35004, 9973, 6212],
        [57143, 11853, 5627],
        [89658, 9890, 5672],
        [60456, 22993, 6505],
        [62589, 33680, 6880],
        [35558, 36878, 7399],
        [15846, 32012, 3826]
    ]) * 1e-3

    # Fixed height (for 2D localization)

    source_height = 0.86  # meters
    alpha = 0.05
    sigma = None

    # %%

    # %%

    toas = dataset.toa

    positions, inliers_batch, success = tdoa_batch(
        antenna_positions,
        toas,         # (T, 8) array
        fixed_dim=2,
        fixed_value=0.86,
        speed_of_signal=c,
        inlier_threshold=6
    )
    # %%

    ref = dataset.reference

    diffs = positions[:, :2] - dataset.reference
    errors = np.linalg.norm(np.abs(diffs), axis=1)

    # %%

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4))

    vmax = 8

    hb = ax.hexbin(ref[:,0], ref[:,1], C=errors, vmax=vmax)
    plt.colorbar(hb)



    # %%

    plt.scatter(positions[:,0], positions[:, 1], s=10)
    plt.xlim(20,60)
    plt.ylim(10, 30)
    # %%

    fig, ax = plt.subplots(figsize=(10, 4), sharex=True)

    ax.ecdf(errors)
    plt.xlim(0, 20)

    print("Mean:", errors.mean())

    for precentile in [50, 75, 90, 95]:
        print(f"CE{precentile}:", np.percentile(errors, precentile))
    # %%
