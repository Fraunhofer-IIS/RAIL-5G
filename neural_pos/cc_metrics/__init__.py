import numpy as np

from neural_pos.cc_metrics.general_dist import CIRAMetric, CosineSimMetric, GeneralDistMetric
from neural_pos.cc_metrics.signature_distance import SignatureDistance

Aggregations = {
    "sum": np.nansum,
    "mean": np.nanmean,
    "max": np.nanmax,
    "median": np.nanmedian
}

CCMetrics = {
    "CIRA": CIRAMetric,
    "cosine": CosineSimMetric,
    "general": GeneralDistMetric,
    "signature": SignatureDistance
}