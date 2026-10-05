import abc
import hashlib
import json
from pathlib import Path
import pickle
import random

import h5py
import numpy as np
from omegaconf import OmegaConf
import torch
from tqdm import tqdm
import sklearn
import scipy
import multiprocessing as mp

from neural_pos.datasets.siamese_cc_ds import CC_DIST_DS_NAME

CACHE_DIR = Path(".gdm_cache")

# Adapted from https://github.com/Jeija/Geodesic-Uncertainty-Loss-ChannelCharting/blob/main/ChannelChartingCore.py


def _shortest_path_worker(todo_queue, output_queue, nbg, target_nodes):
    while True:
        index = todo_queue.get()

        if index == -1:
            output_queue.put((-1, None))
            break

        d, predecessors = scipy.sparse.csgraph.dijkstra(
            nbg, directed=False, indices=target_nodes[index], return_predecessors=True
        )
        predecessors[predecessors == -9999] = -1
        del d

        output_queue.put((index, predecessors))


def _path_hops_worker(todo_queue, output_queue, predecessor_matrix):
    while True:
        i = todo_queue.get()

        if i is None:
            output_queue.put(None)
            break

        hops = 0
        current = np.arange(predecessor_matrix.shape[1], dtype=np.int32)
        active = (current != -1)
        while np.any(active):
            current[active] = predecessor_matrix[i, current[active]]
            active = (current != -1)
            hops += 1

        output_queue.put(hops)


def find_shortest_paths(pairwise_dissimilarity_matrix, target_nodes=None, n_neighbors=20, max_processes=10):
    nbrs_alg = sklearn.neighbors.NearestNeighbors(n_neighbors=n_neighbors, metric="precomputed", n_jobs=-1)
    nbrs = nbrs_alg.fit(pairwise_dissimilarity_matrix)
    nbg = sklearn.neighbors.kneighbors_graph(nbrs, n_neighbors, metric="precomputed", mode="distance")

    if target_nodes is None:
        target_nodes = np.arange(nbg.shape[0], dtype=np.int32)

    geodesic_predecessor_matrix = np.zeros((target_nodes.shape[0], nbg.shape[1]), dtype=np.int32)

    with tqdm(total=len(target_nodes) * nbg.shape[0], desc="Computing Shortest Paths") as pbar:
        todo_queue = mp.Queue()
        output_queue = mp.Queue()

        for i in tqdm(range(len(target_nodes)), desc="Preparing Dijkstra Inputs"):
            todo_queue.put(i)

        process_count = min(max_processes, mp.cpu_count())

        for i in tqdm(range(process_count), desc="Starting Processes"):
            todo_queue.put(-1)
            p = mp.Process(target=_shortest_path_worker, args=(todo_queue, output_queue, nbg, target_nodes))
            p.start()

        finished_processes = 0
        while finished_processes != process_count:
            i, p = output_queue.get()

            if i == -1:
                finished_processes += 1
            else:
                geodesic_predecessor_matrix[i, :] = p
                pbar.update(len(p))

    del nbg, nbrs, nbrs_alg

    return geodesic_predecessor_matrix


def find_path_with_most_hops_old(predecessor_matrix):
    most_hops = 0

    with tqdm(total=predecessor_matrix.shape[0], desc="Computing longest paths") as pbar:
        todo_queue = mp.Queue()
        output_queue = mp.Queue()

        for i in tqdm(range(predecessor_matrix.shape[0]), desc="Preparing tasks"):
            todo_queue.put(i)

        for i in tqdm(range(mp.cpu_count()), desc="Starting processes"):
            todo_queue.put(None)
            p = mp.Process(target=_path_hops_worker, args=(todo_queue, output_queue, predecessor_matrix))
            p.start()

        finished_processes = 0
        while finished_processes != mp.cpu_count():
            hops = output_queue.get()

            if hops is None:
                finished_processes += 1
            else:
                if hops > most_hops:
                    most_hops = hops
                pbar.update(1)

    return most_hops


def find_path_with_most_hops(predecessor_matrix):
    most_hops = 0
    for i in tqdm(range(predecessor_matrix.shape[0]), desc="Computing longest paths"):
        hops = 0
        current = np.arange(predecessor_matrix.shape[1], dtype=np.int32)
        active = current != -1
        while np.any(active):
            current[active] = predecessor_matrix[i, current[active]]
            active = current != -1
            hops += 1
        most_hops = max(most_hops, hops)
    return most_hops


def contract_path(predecessors, dissimilarity_choices, metric_to_contract):
    contractable = np.full(predecessors.shape, True)

    while contractable.sum() > 0:
        # Get choice of dissimilarity metric from current node to predecessor
        # and from predecessor to predecessor of predecessor.
        current_choice = dissimilarity_choices
        predecessors_choice = np.take_along_axis(dissimilarity_choices, predecessors, 1)

        # Check which path sections are contractable and perform contraction
        predecessors_of_predecessors = np.take_along_axis(predecessors, predecessors, 1)
        contractable = np.logical_and(current_choice == metric_to_contract, predecessors_choice == metric_to_contract)
        contractable = np.logical_and(contractable, predecessors != -1)
        contractable = np.logical_and(contractable, predecessors_of_predecessors != -1)

        print(f"{contractable.sum()} path sections remain to be contracted")
        predecessors[contractable] = predecessors_of_predecessors[contractable]


class GaussianDissimilarityModel:
    def __init__(self, metrics, enable_path_contraction=True):
        self.metrics = metrics

        self.datapoint_count = metrics[0].get_datapoint_count()
        for metric in metrics[1:]:
            assert (metric.get_datapoint_count() == self.datapoint_count)

        self.enable_path_contraction = enable_path_contraction

    def generate_short_paths(self, total_path_count=40000, realization_count=8, variance_scale=0.1, rng=None):
        """
        Args:
            rng: np.random.Generator used for ALL randomness in this method (and everything
                it calls). Must be explicitly provided for reproducible, cacheable results.
                Falls back to a fresh (non-reproducible) Generator if not provided.
        """
        if rng is None:
            rng = np.random.default_rng()

        assert (total_path_count % realization_count == 0)
        paths_per_realization = total_path_count // realization_count

        self.predecessor_matrix = np.zeros((total_path_count, self.datapoint_count), dtype=np.int32)
        self.target_nodes = rng.integers(self.datapoint_count, size=total_path_count)
        self.dissimilarity_matrix_choices = np.zeros((total_path_count, self.datapoint_count), dtype=np.int8)

        # Does not return anything, but does the processing...
        # Rounds determines how many times realizations should be drawn randomly
        for realization_index in tqdm(range(realization_count)):
            first_path_index = realization_index * paths_per_realization
            last_path_index = (realization_index + 1) * paths_per_realization

            print("Generating dissimilarity realizations...")
            dissimilarity_metrics_count = len(self.metrics)
            realizations = np.zeros((self.datapoint_count, self.datapoint_count, dissimilarity_metrics_count))
            for i, metric in enumerate(tqdm(self.metrics)):
                metric.get_realization(realizations[:, :, i], variance_scale, rng)

            # For every datapoint pair, select smallest dissimilarity realization
            print("Choosing smallest dissimilarity realization pair-wise...")
            dissimilarity_matrix_choice = np.argmin(realizations, axis=-1, keepdims=True)
            pairwise_dissimilarity_matrix = np.take_along_axis(realizations, dissimilarity_matrix_choice, axis=-1)[:, :, 0]

            # Run shortest path algorithm
            # dissimilarity_matrix_choices stores which type of dissimilarity (velocity model, adp model, ...) was used to go from datapoint x along
            # the path towards the target datapoint to the next hop.
            # It has shape (total_path_count, self.datapoint_count), so the first axis determines the path we are on (and hence also the target datapoint) and the
            # second axis determines the datapoint (node) from which the current hop starts.
            print("Running shortest path algorithm...")
            current_target_nodes = self.target_nodes[first_path_index:last_path_index]
            predecessors = find_shortest_paths(pairwise_dissimilarity_matrix, current_target_nodes)

            assert (np.all(np.sum(np.where(predecessors == -1, 1, 0), axis=1) == 1))

            self.predecessor_matrix[first_path_index:last_path_index] = predecessors
            self.dissimilarity_matrix_choices[first_path_index:last_path_index] = dissimilarity_matrix_choice[np.arange(self.datapoint_count)[np.newaxis, :], predecessors][..., 0]

            del pairwise_dissimilarity_matrix
            del dissimilarity_matrix_choice
            del realizations

        # Optional step for faster training: Contract predecessor matrix
        # Some dissimilarity metrics may be "contractable", which means that path A->B->C and path A->C have the same
        # mean, variance dissimilarity if all hops "->" refer to the same dissimilarity.
        # In that case, we can shorten the path by replacing the predecessor of C (which is B) with A.
        # We can detect this from the predecessor matrix by checking if an entry has the same dissimilarity type as its predecessor.
        # This algorithm has log(N) complexity, where N is the length of the longest path for the same dissimilarity type.
        if self.enable_path_contraction:
            for metric_type, metric in enumerate(self.metrics):
                if metric.is_contractable():
                    print(f"Contracting paths for metric {metric.__class__.__name__}")
                    contract_path(self.predecessor_matrix, self.dissimilarity_matrix_choices, metric_type)

        # Determine new longest path after contraction
        print("Determining longest short path...")
        self.longest_shortest_path = find_path_with_most_hops(self.predecessor_matrix)
        print(f"Longest short path has {self.longest_shortest_path} hops")

    def get_longest_shortest_path(self):
        return self.longest_shortest_path

    def get_random_short_paths(self, path_count, max_pathhops, hop_skip_limit=None, rng=None):
        """
        Args:
            rng: np.random.Generator used for ALL randomness in this method. Must be
                explicitly provided for reproducible, cacheable results. Falls back to a
                fresh (non-reproducible) Generator if not provided.

        returns (path_targets, path_sources, paths, path_hops, path_means, path_variances)
        where paths is of shape (path_count, maximum path length) and all others are of shape path_count
        """
        if rng is None:
            rng = np.random.default_rng()

        # Target and source indices to cached predecessor matrix
        # Source indices are also datapoint indices, but target indices must be translated to datapoint indices
        # using self.target_nodes[path_target_indices]
        path_target_indices = rng.integers(self.predecessor_matrix.shape[0], size=path_count)
        path_source_indices = rng.integers(self.predecessor_matrix.shape[1], size=path_count)

        # Prevent pairs where both indices refer to the same datapoint
        path_source_indices[path_source_indices == self.target_nodes[path_target_indices]] = (path_source_indices[path_source_indices == self.target_nodes[path_target_indices]] + 1) % self.predecessor_matrix.shape[1]

        current = np.copy(path_source_indices)
        paths = np.zeros((len(current), self.longest_shortest_path), dtype=np.int32)
        path_hops = np.zeros(len(current), dtype=np.int32)

        for i in range(self.longest_shortest_path):
            paths[:, i] = current
            previous = current
            active = (current != self.target_nodes[path_target_indices])

            current[active] = self.predecessor_matrix[path_target_indices[active], current[active]]
            path_hops[np.logical_and(active, current == self.target_nodes[path_target_indices])] = i + 1

        # Compute mean total dissimilarity as well as uncertainty about it (variance) along paths
        # Use provided models to compute means / variances for individual dissimilarity types
        # Assume that dissimilarity models are independent, i.e., variances are just added up
        dissim_choice = np.take_along_axis(self.dissimilarity_matrix_choices[path_target_indices], paths[:, :-1], 1)

        # Assume that p(d) from different models are entirely uncorrelated
        total_dissimilarity_means = np.zeros(len(paths))
        total_dissimilarity_variances = np.zeros(len(paths))

        for metric_type, metric in enumerate(self.metrics):
            means, variances = metric.mean_variance_along_path(paths, dissim_choice == metric_type)
            total_dissimilarity_means += means
            total_dissimilarity_variances += variances

        # Subsample paths such that there are no more than subsampled_pathhops hops
        if hop_skip_limit is not None:
            for i in range(len(paths)):
                l = int(min(max(total_dissimilarity_means[i] / hop_skip_limit[i], 1), path_hops[i], max_pathhops))
                paths[i, :l + 1] = paths[i, np.linspace(0, path_hops[i], l + 1, dtype=np.int32)]
                paths[i, l + 1:] = paths[i, -1]
                path_hops[i] = l

        return paths, path_hops, total_dissimilarity_means, total_dissimilarity_variances


class GaussianDissimilarityMetric(abc.ABC):
    """
    Class that models an uncertain dissimilarity metric, where each dissimilarity is normally distributed.
    """

    @abc.abstractmethod
    def get_realization(self, output_matrix, variance_scale, rng):
        """
        Retrieve a relization of the dissimilarity matrix of shape `(datapoint_count, datapoint_count)`.
        Entries with value `np.inf` mean that there is no known path according to this dissimilarity metric.
        The scale of the output must be consistent across all dissimilarity metrics.
        The convention is to provide the dissimilarities in meters.
        To speed up this processing step, the result is not provided as a return value,
        but is written to the pre-allocated buffer provided as a parameter.

        :param output_matrix: The realization of the dissimilarity matrix, NumPy array.
        :param variance_scale: Scaling factor for the variance. May want to scale down variance if number of realizations is small to achieve result close to mean.
        :param rng: np.random.Generator to use for all randomness. Must be used instead of the
            global np.random state to keep cached preprocessing results reproducible.
        """
        pass

    @abc.abstractmethod
    def mean_variance_along_path(self, paths, mask):
        """
        Assuming the transmitter moves along the provided path sections of shape `(path count, maximum path length)`,
        but only along those sections where corresponding mask of same shape is True, calculate the mean and variance
        of the distribution of the summed up dissimilarities, which are assumed to be normally distributed.

        This function can model correlations between individual dissimilarities of the same type as desired.

        Different dissimilarity types are always assumed to be uncorrelated by the current model.

        The implementation with constant-hop count paths and mask allows for better vecorization as opposed to an approach
        where the paths are of variable hop count.

        :param paths: Datapoint indices of path sections, NumPy array of shape `(path count, maximum path length)`
        :param mask: Boolean NumPy Array that indicates whether path secction should be included in the sum or ignored
        :return:
            - mean_sum - Mean of sum of dissimilarities
            - variance_sum - Variance of sum of dissimilarities
        """
        pass

    @abc.abstractmethod
    def get_datapoint_count(self):
        """
        Query the number of datapoints for which this metric provides dissimilarities.
        """
        pass

    @abc.abstractmethod
    def is_contractable(self):
        """
        Query whether this type of dissimilarity metric can be contracted.
        Path contraction eliminates hops A->B->C where both hops take the same dissimilarity metric.
        This requires an implementation the returns the same mean / variance for paths A->B->C and A->C,
        which is only possible in special cases.
        """
        pass


class LinearGaussianDissimilarityMetric(GaussianDissimilarityMetric):
    def __init__(self, dissimilarity_matrix, slope, bias, variance_factor, saturation_threshold=None):
        self.dissimilarity_matrix = dissimilarity_matrix
        self.slope = slope
        self.bias = bias
        self.variance_factor = variance_factor
        self.saturation_threshold = saturation_threshold

    def _mean(self):
        mean = self.dissimilarity_matrix * self.slope + self.bias
        mean = np.where(self.dissimilarity_matrix == 0, 0, mean)
        if self.saturation_threshold is not None:
            mean = np.where(self.dissimilarity_matrix > self.saturation_threshold, np.inf, mean)
        return mean

    def _variance(self):
        mean = self._mean()
        variance = np.where(mean == 0, 0, np.where(mean == np.inf, np.inf, self.variance_factor * mean))
        return variance

    def get_realization(self, output_matrix, variance_scale, rng):
        mean = self._mean()
        variance = self._variance()
        finite_distances = np.triu(mean != np.inf)
        random_numbers = np.abs(
            rng.normal(
                mean[finite_distances],
                np.sqrt(
                    np.maximum(
                        variance[finite_distances] * variance_scale,
                        0
                    )
                )
            )
        )
        output_matrix.fill(np.inf)
        output_matrix[finite_distances] = random_numbers
        np.transpose(output_matrix)[finite_distances] = random_numbers

    def mean_variance_along_path(self, paths, mask):
        mean = self._mean()[paths[:, :-1], paths[:, 1:]]
        variance = self._variance()[paths[:, :-1], paths[:, 1:]]

        mean_sum = np.sum(np.where(mask, mean, 0), axis=1)
        variance_sum = np.sum(np.where(mask, variance, 0), axis=1)

        return mean_sum, variance_sum

    def get_datapoint_count(self):
        return self.dissimilarity_matrix.shape[0]

    def is_contractable(self):
        return False


class SimpleGaussianProcess:
    """
    Models a stationary Gaussian stochastic process that is either perfectly uncorrelated
    or perfectly correlated.
    """

    def __init__(self, process_mean, process_variance, perfectly_correlated, realization_count=None):
        # Here, we only model the extreme cases of the Gaussian process being perfectly correlated
        # (then perfectly_correlated = True) or perfectly uncorrelated (then perfectly_correlated = False)
        self.process_mean = process_mean
        self.process_variance = process_variance
        self.perfectly_correlated = perfectly_correlated
        self.realization_count = realization_count

    # Compute the distribution of a random variable that is the sum of the integration of a gaussian process
    # over multiple intervals from t_a to t_b. Needs to take into account that the integrals are correlated.
    def get_sum_of_interval_integrals_mean_variance(self, t_a, t_b, mask):
        # t_a and t_b have shape (:, interval_count)
        total_delta_t = np.sum(np.where(mask, np.abs(t_b - t_a), 0), axis=1)
        integrated_mean = self.process_mean * total_delta_t

        if self.perfectly_correlated:
            integrated_variance = total_delta_t**2 * self.process_variance
        else:
            integrated_variance = total_delta_t * self.process_variance

        return integrated_mean, integrated_variance

    def get_realization(self, t, variance_scale, rng):
        if self.perfectly_correlated:
            # Constrained random sampling: Make sure realization is within 1 standard deviation of mean in perfectly correlated case,
            # otherwise can get unlucky with result due to limited realization count
            return rng.normal(self.process_mean, np.sqrt(self.process_variance * variance_scale)) * np.ones_like(t)[np.newaxis, :]
        else:
            return np.abs(rng.normal(self.process_mean, np.sqrt(self.process_variance * variance_scale), size=(self.realization_count, len(t))))


class VelocityDissimilarityMetric(GaussianDissimilarityMetric):
    def __init__(self, velocity_mean, velocity_variance, perfectly_correlated, timestamps):
        # Model absolute value of velocity (speed) as Gaussian process
        self.velocity_model = SimpleGaussianProcess(velocity_mean, velocity_variance, perfectly_correlated)
        self.timestamps = timestamps

    def get_realization(self, output_matrix, variance_scale, rng):
        velocities = self.velocity_model.get_realization(self.timestamps[:-1], variance_scale, rng)
        displacements = np.concatenate([[0], np.cumsum(velocities * np.diff(self.timestamps))])
        output_matrix[:] = np.abs(displacements[np.newaxis, :] - displacements[:, np.newaxis])

        # Numerical trick that makes shortest path algorithm "skip" unnecessary intermediary hops:
        # Add a tiny additional cost to each hop. This way, e.g. path A->C is cheaper than path A->B->C
        # Since this reduces the overall path length (and hence also the length of the longest shortest path),
        # this makes path generation much faster later on.
        # (Since CC runs shortest path algorithm on neighborhood graph, not all intermediary nodes will be skipped.
        # This is only achieved later through path contraction.)
        path_hop_cost = np.ones_like(output_matrix) * 1e-5
        np.fill_diagonal(path_hop_cost, 0)
        output_matrix[:] += path_hop_cost

    def mean_variance_along_path(self, paths, mask):
        t_a = self.timestamps[paths[:, :-1]]
        t_b = self.timestamps[paths[:, 1:]]

        return self.velocity_model.get_sum_of_interval_integrals_mean_variance(t_a, t_b, mask)

    def is_contractable(self):
        return True

    def get_datapoint_count(self):
        return self.timestamps.shape[0]


def _batch_worker(
    todo_queue: mp.Queue,
    output_queue: mp.Queue,
    GDM: GaussianDissimilarityModel,
    training_batches: int,
    min_batch_size: int,
    max_batch_size: int,
    max_hoplength: float,
    min_hoplength: float,
    randomize_pathhops: bool,
    max_pathhops: int,
    seed: int,
):
    while True:
        batch_idx = todo_queue.get()
        if batch_idx == -1:
            output_queue.put((-1, None))
            break

        # Deterministic, worker-scheduling-independent RNG: depends only on (seed, batch_idx),
        # not on which worker process happened to pick up this batch or in what order.
        rng = np.random.default_rng((seed, batch_idx))

        batch_size = int(np.round(
            batch_idx / training_batches * (max_batch_size - min_batch_size) + min_batch_size
        ))
        batch_size = int(np.round(batch_size / 200) * 200)

        pathhops_limit = (
            (batch_idx / training_batches) ** 0.15
            * (min_hoplength - max_hoplength)
            + max_hoplength

        )

        if randomize_pathhops:
            pathhops_maxlength = rng.uniform(pathhops_limit, max_hoplength, size=batch_size)
        else:
            pathhops_maxlength = np.full(batch_size, pathhops_limit, dtype=np.float32)

        paths, path_hops, means, variances = GDM.get_random_short_paths(
            batch_size, max_pathhops=max_pathhops, hop_skip_limit=pathhops_maxlength, rng=rng
        )
        paths = paths[:, : max_pathhops + 1]

        output_queue.put((batch_idx, {
            "paths":     paths.astype(np.int64),
            "path_hops": path_hops.astype(np.int64),
            "means":     means.astype(np.float32),
            "variances": variances.astype(np.float32),
        }))


def _make_cache_key(file_path: str, gdm_params: dict, training_params: dict, seed: int) -> str:
    payload = {
        "file_path": str(Path(file_path).resolve()),
        "gdm_params": OmegaConf.to_container(gdm_params),
        "training_params": OmegaConf.to_container(training_params),
        "seed": seed,
    }
    serialized = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(serialized.encode()).hexdigest()


class GDMDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        file_path: str,
        base_dataset_type: str,
        base_dataset_params: dict = {},
        gdm_params: dict = {},
        training_params: dict = {},
        no_cache: bool = False,
    ):
        from . import Datasets

        base_dataset = Datasets[base_dataset_type](file_path, **base_dataset_params)
        self.timestamps: np.ndarray = base_dataset.timestamps
        self.valid_mask: np.ndarray = base_dataset.valid_mask
        self.reference: np.ndarray = base_dataset.reference
        self.dtype = base_dataset.dtype

        # Pre-apply augmentations once
        if len(base_dataset.augmentations) > 0:
            print(f"[WARN] Augmentations are preapplied to {type(self)}")
            csi_augmented = []
            for i in tqdm(range(len(base_dataset.csi)), "Applying augmentaion"):
                item = {"inputs": base_dataset.csi[i].astype(self.dtype)}
                if self.valid_mask is not None:
                    item["valid_mask"] = self.valid_mask[i]
                for aug in base_dataset.augmentations:
                    item = aug(item)
                csi_augmented.append(item["inputs"])
            self.csi = np.stack(csi_augmented, axis=0)
        else:
            self.csi = base_dataset.csi.astype(self.dtype)

        # Reproducible preprocessing seed, derived from whatever seed PyTorch/Lightning's
        # seed_everything() already established for this run. Everything that gets cached
        # to disk below is a deterministic function of this seed (and the other params),
        # independent of global RNG state / call order elsewhere in the program. This
        # guarantees identical results whether loaded from cache or recomputed from scratch.
        seed = torch.initial_seed() % (2**32)

        # --- Cache setup ---
        cache_key = _make_cache_key(file_path, gdm_params, training_params, seed)
        cache_file = CACHE_DIR / f"{cache_key}.pkl"

        if not no_cache and cache_file.exists():
            print(f"[GDMDataset] Loading cached path_batches from {cache_file}")
            with cache_file.open("rb") as fh:
                self.path_batches = pickle.load(fh)
            return

        # --- Build GDM ---
        with h5py.File(file_path) as f:
            distances_condensed = f[CC_DIST_DS_NAME][:]
        dist_matrix = scipy.spatial.distance.squareform(distances_condensed)

        gp = gdm_params
        dist_metric = LinearGaussianDissimilarityMetric(
            dist_matrix,
            bias=gp["bias"],
            slope=gp["slope"],
            variance_factor=gp["var_factor"],
            saturation_threshold=gp["saturation_threshold"]
        )
        velocity_metric = VelocityDissimilarityMetric(
            gp["velocity_mean"],
            gp["velocity_variance"],
            True,
            self.timestamps,
        )
        gdm = GaussianDissimilarityModel(
            [dist_metric, velocity_metric],
            enable_path_contraction=gp["enable_path_contraction"],
        )

        # Dedicated, explicitly-seeded generator for the (single-process) path generation step.
        gen_rng = np.random.default_rng(seed)
        gdm.generate_short_paths(
            total_path_count=gp["total_path_count"],
            realization_count=gp["realization_count"],
            variance_scale=gp["variance_scale"],
            rng=gen_rng,
        )

        self.path_batches = self._precompute_paths(gdm, training_params, seed)

        # --- Persist to cache ---
        if not no_cache:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            with cache_file.open("wb") as fh:
                pickle.dump(self.path_batches, fh)
            print(f"[GDMDataset] Cached path_batches to {cache_file}")

    # ------------------------------------------------------------------
    def _precompute_paths(self, gdm, tp: dict, seed: int) -> list[dict]:
        training_batches = tp["training_batches"]
        min_batch_size   = tp["min_batch_size"]
        max_batch_size   = tp["max_batch_size"]
        max_hoplength    = tp["max_hoplength"]
        min_hoplength    = tp["min_hoplength"]
        randomize        = tp["randomize_pathhops"]
        max_pathhops     = tp["max_pathhops"]
        nw               = tp["path_worker_count"]

        todo_q: mp.Queue = mp.Queue()
        out_q:  mp.Queue = mp.Queue()

        for i in range(1, training_batches + 1):
            todo_q.put(i)
        for _ in range(nw):
            todo_q.put(-1)

        for _ in range(nw):
            p = mp.Process(
                target=_batch_worker,
                args=(
                    todo_q, out_q, gdm, training_batches,
                    min_batch_size, max_batch_size,
                    max_hoplength, min_hoplength,
                    randomize, max_pathhops,
                    seed,
                ),
                daemon=True,
            )
            p.start()

        batches_dict = {}
        finished = 0
        with tqdm(total=training_batches, desc="Pre-computing training paths") as pbar:
            while finished < nw:
                idx, data = out_q.get()
                if idx == -1:
                    finished += 1
                else:
                    batches_dict[idx] = data
                    pbar.update(1)

        return [batches_dict[i] for i in range(1, training_batches + 1)]

    def get_data_shape(self):
        return self.csi.shape[1:]

    def __len__(self) -> int:
        return len(self.path_batches)

    def __getitem__(self, index: int) -> dict:
        batch = self.path_batches[index]

        ret = {
            "inputs":      self.csi.astype(self.dtype),         # (N, *csi_shape)
            "timestamps":  self.timestamps.astype(self.dtype),  # (N,)
            "paths":       batch["paths"],                      # (B, max_pathhops+1)
            "path_hops":   batch["path_hops"],                  # (B,)
            "means":       batch["means"],                      # (B,)
            "variances":   batch["variances"],                  # (B,)
        }

        if self.valid_mask is not None:
            ret["valid_mask"] = self.valid_mask      # (N, *mask_shape)

        return ret