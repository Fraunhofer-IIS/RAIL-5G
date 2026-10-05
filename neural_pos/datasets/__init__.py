
from neural_pos.datasets.fp_ds import FingerprintingDataset
from neural_pos.datasets.gaussian_dissimilarity_ds import GDMDataset
from neural_pos.datasets.siamese_cc_ds import SiameseCCDataset
from neural_pos.datasets.time_triplet_cc_ds import TimeTripletCCDataset


Datasets = {
    "FingerprintingDataset": FingerprintingDataset,
    "SiameseCCDataset": SiameseCCDataset,
    "TimeTripletCCDataset": TimeTripletCCDataset,
    "GDMDataset": GDMDataset,
}