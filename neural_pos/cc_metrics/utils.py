from concurrent.futures import ThreadPoolExecutor, as_completed

from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
from scipy.sparse.csgraph import dijkstra
import multiprocessing
from tqdm import tqdm

def calculate_geodesic_distances_condensed(nbg, n_workers=None):
    """
    Calculate geodesic distances in condensed form (upper triangle only).
    
    Parameters
    ----------
    nbg : sparse matrix
        k-nearest neighbors graph
    n_workers : int, optional
        Number of worker processes (default: cpu_count)
    
    Returns
    -------
    CondensedMatrix
        Geodesic distance matrix in condensed form
    """
    if n_workers is None:
        n_workers = multiprocessing.cpu_count()
    
    L = nbg.shape[0]
    condensed_length = L * (L - 1) // 2
    
    # Use shared memory array for results
    condensed_distances = np.zeros(condensed_length, dtype=np.float32)
    
    # Setup queues
    todo_queue = multiprocessing.Queue()
    output_queue = multiprocessing.Queue()
    
    # Queue all source indices
    for i in range(L):
        todo_queue.put(i)
    
    # Add termination signals
    for _ in range(n_workers):
        todo_queue.put(-1)
    
    # Start worker processes
    processes = []
    for _ in range(n_workers):
        p = multiprocessing.Process(
            target=shortest_path_worker_condensed,
            args=(nbg, L, todo_queue, output_queue)
        )
        p.start()
        processes.append(p)
    
    # Collect results
    finished_processes = 0
    with tqdm(total=L, desc="Calculating geodesic distance") as pbar:
        while finished_processes < n_workers:
            i, distances_upper = output_queue.get()
            
            if i == -1:
                finished_processes += 1
            else:
                # Store upper triangle distances in condensed array
                # For source i, store distances to all j > i
                start_idx = L * i - (i * (i + 1)) // 2
                end_idx = start_idx + (L - i - 1)
                condensed_distances[start_idx:end_idx] = distances_upper
                pbar.update(1)
    
    # Wait for all processes to finish
    for p in processes:
        p.join()
    
    return condensed_distances


def calculate_geodesic_distances_squared(nbg, n_workers=None):
    dissimilarity_matrix_geodesic = np.zeros((nbg.shape[0], nbg.shape[1]), dtype = np.float32)

    with tqdm(total = nbg.shape[0]**2, desc="Calculating geodesic distance") as pbar:
        todo_queue = multiprocessing.Queue()
        output_queue = multiprocessing.Queue()

        for i in range(nbg.shape[0]):
            todo_queue.put(i)
        
        for i in range(multiprocessing.cpu_count()):
            todo_queue.put(-1)
            p = multiprocessing.Process(target = shortest_path_worker, args = (nbg, todo_queue, output_queue))
            p.start()

        print("Start procesing...")
        finished_processes = 0
        while finished_processes != multiprocessing.cpu_count():
            i, d = output_queue.get()

            if i == -1:
                finished_processes = finished_processes + 1
            else:
                dissimilarity_matrix_geodesic[i,:] = d
                pbar.update(len(d))
    return dissimilarity_matrix_geodesic


def shortest_path_worker(nbg, todo_queue, output_queue):
		while True:
			index = todo_queue.get()

			if index == -1:
				output_queue.put((-1, None))
				break

			d = dijkstra(nbg, directed=False, indices=index)
			output_queue.put((index, d))

def shortest_path_worker_condensed(nbg, L, todo_queue, output_queue):
    """
    Worker process that computes distances and returns only upper triangle.
    
    Parameters
    ----------
    nbg : sparse matrix
        k-nearest neighbors graph
    L : int
        Number of nodes
    todo_queue : Queue
        Queue of source indices to process
    output_queue : Queue
        Queue for returning results
    """
    while True:
        index = todo_queue.get()
        
        if index == -1:
            output_queue.put((-1, None))
            break
        
        # Compute distances from source index to all nodes
        d = dijkstra(nbg, directed=False, indices=index)
        
        # Extract only upper triangle (distances to nodes j > index)
        distances_upper = d[index + 1:]
        
        output_queue.put((index, distances_upper))


def plot_dissimilarity_over_euclidean_distance(dissimilarity_matrix, distance_matrix, label=None, nth_reduction=10, ax=None, color="C0"):
    if ax is None:
        ax = plt.gca()
    
    dissimilarities_flat = dissimilarity_matrix[::nth_reduction, ::nth_reduction].flatten()
    distances_flat = distance_matrix[::nth_reduction, ::nth_reduction].flatten()

    max_distance = np.max(distances_flat)
    bins = np.linspace(0, max_distance, 200)
    bin_indices = np.digitize(distances_flat, bins)

    bin_medians = np.zeros(len(bins) - 1)
    bin_25_perc = np.zeros(len(bins) - 1)
    bin_75_perc = np.zeros(len(bins) - 1)
    
    for i in range(1, len(bins)):
        bin_values = dissimilarities_flat[bin_indices == i]
        
        # Überprüfe, ob das Bin Werte enthält
        if len(bin_values) > 0:
            bin_25_perc[i - 1], bin_medians[i - 1], bin_75_perc[i - 1] = np.percentile(bin_values, [25, 50, 75])
        else:
            # Setze NaN für leere Bins (oder 0, je nach Anwendung)
            bin_25_perc[i - 1] = np.nan
            bin_medians[i - 1] = np.nan
            bin_75_perc[i - 1] = np.nan

    ax.plot(bins[:-1], bin_medians, label=label, color=color)
    ax.fill_between(bins[:-1], bin_25_perc, bin_75_perc, alpha=0.5, color=color)


def plot_dissimilarity_over_euclidean_distance_condensed(dissimilarity_condensed, distance_condensed, 
                                                label=None, sample_fraction=1.0, ax=None, color="C0"):
    """
    Plot dissimilarity vs euclidean distance using condensed distance matrices.
    
    Parameters:
    -----------
    dissimilarity_condensed : array-like
        Condensed dissimilarity matrix (from scipy.spatial.distance.pdist or squareform)
    distance_condensed : array-like
        Condensed distance matrix
    label : str, optional
        Label for the plot
    sample_fraction : float, optional
        Fraction of data points to sample (0 < sample_fraction <= 1.0)
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on. If None, uses current axes.
    
    Returns:
    --------
    ax : matplotlib.axes.Axes
        The axes object used for plotting
    """
    if ax is None:
        ax = plt.gca()
    
    # Sample data if needed
    if sample_fraction < 1.0:
        n_samples = int(len(dissimilarity_condensed) * sample_fraction)
        sample_indices = np.random.choice(len(dissimilarity_condensed), 
                                         size=n_samples, 
                                         replace=False)
        dissimilarities_flat = dissimilarity_condensed[sample_indices]
        distances_flat = distance_condensed[sample_indices]
    else:
        dissimilarities_flat = dissimilarity_condensed
        distances_flat = distance_condensed
    
    max_distance = np.max(distances_flat)
    bins = np.linspace(0, max_distance, 200)
    bin_indices = np.digitize(distances_flat, bins)
    
    bin_medians = np.zeros(len(bins) - 1)
    bin_25_perc = np.zeros(len(bins) - 1)
    bin_75_perc = np.zeros(len(bins) - 1)
    
    for i in range(1, len(bins)):
        bin_values = dissimilarities_flat[bin_indices == i]
        if len(bin_values) > 0:
            bin_25_perc[i - 1], bin_medians[i - 1], bin_75_perc[i - 1] = np.percentile(bin_values, [25, 50, 75])
    
    ax.plot(bins[:-1], bin_medians, label=label, color=color)
    ax.fill_between(bins[:-1], bin_25_perc, bin_75_perc, alpha=0.5, color=color)
    
    # Add axis labels
    ax.set_xlabel('Euclidean Distance')
    ax.set_ylabel('Dissimilarity')
    
    return ax