import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import pdist



class GeneralDistMetric:

    def __init__(self, metric='cosine', single_antenna=True, antenna_aggregation="sum"):
        from neural_pos.cc_metrics import Aggregations

        self.metric = metric
        self.single_antenna = single_antenna
        aggregation = Aggregations[antenna_aggregation]
        self.antenna_aggregation = aggregation

    def compute_distance(self, X):
        M = np.linalg.norm(X, axis=1)  # Shape (L, A, T)
        L, A, T = M.shape

        if self.single_antenna:
            condensed_length = L * (L - 1) // 2
            distances_per_a = np.zeros((A, condensed_length))

            for a_idx in tqdm(range(A), desc="Calculating distances per a"):
                M_slice = M[:, a_idx, :]
                distances_per_a[a_idx] = pdist(M_slice, metric=self.metric)

            condensed = self.antenna_aggregation(distances_per_a, axis=0)
            del distances_per_a
            return condensed
        else:
            return pdist(M.reshape(L, A*T), metric=self.metric)

    def __call__(self, x, timestamps=None):
        return self.compute_distance(x)
    
class CosineSimMetric(GeneralDistMetric):

    def __init__(self, single_antenna=True, antenna_aggregation="sum"):
        super().__init__(metric='cosine', single_antenna=single_antenna, antenna_aggregation=antenna_aggregation)


class CIRAMetric(GeneralDistMetric):

    def __init__(self, single_antenna=True, antenna_aggregation="sum"):
        super().__init__(metric='cityblock', single_antenna=single_antenna, antenna_aggregation=antenna_aggregation)