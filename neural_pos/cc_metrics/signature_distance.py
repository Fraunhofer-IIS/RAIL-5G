import numpy as np
from scipy.spatial.distance import pdist
from tqdm import tqdm


class SignatureDistance:
    """Computes pairwise signature distance (eq. 11 & 12) with missing antenna support."""

    def __init__(self, antenna_aggregation="sum"):
        from neural_pos.cc_metrics import Aggregations
        self.antenna_aggregation = Aggregations[antenna_aggregation]

    def __call__(self, x, timestamps=None):
        """
        Parameters
        ----------
        x : np.ndarray, shape (N, 2, A, T)
            N samples, 2 = real/imag, A antennas, T delay bins.
            Missing data indicated by NaN values.
        timestamps : ignored

        Returns
        -------
        dist : np.ndarray, shape (N*(N-1)//2,)
            Condensed pairwise distance matrix.
        """
        N, _, A, T = x.shape

        # Power per tap: (N, A, T)
        power = x[:, 0, :, :] ** 2 + x[:, 1, :, :] ** 2

        # Energy normalization per (sample, antenna)
        total_energy = np.nansum(power, axis=-1, keepdims=True)
        total_energy = np.where(total_energy == 0, 1.0, total_energy)
        p = power / total_energy  # (N, A, T)

        # Uniform tap timing
        t = np.arange(1, T + 1) / T
        t_N = 1.0
        t_prev = np.concatenate(([0.0], t[:-1]))

        # Eq (12): weight per tap, shape (T,)
        w = t_N ** 2 + 2 * t ** 2 + 2 * t_prev ** 2 + 2 * t * t_prev - 3 * t_N * (t + t_prev)

        # s_m^(i) = (1/12) * sum_n p_n * w_n  -> (N, A)
        # Use nan-aware: NaN taps contribute NaN -> antenna becomes NaN
        s = (1.0 / 12.0) * np.einsum('nat,t->na', p, w)

        # Per-antenna L1 distance, aggregated with nan-aware function
        condensed_length = N * (N - 1) // 2
        distances_per_a = np.zeros((A, condensed_length))

        for a_idx in tqdm(range(A), desc="Calculating signature distances per antenna"):
            s_slice = s[:, a_idx]  # (N,)
            distances_per_a[a_idx] = pdist(s_slice[:, None], metric='cityblock')

        condensed = self.antenna_aggregation(distances_per_a, axis=0)

        return condensed