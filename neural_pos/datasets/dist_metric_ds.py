import torch
#import h5pickle as h5py
import h5py
from neural_pos.datasets.augmentations import Augmentations
import numpy as np

from neural_pos.datasets.smoothing import Smoothing
from neural_pos.datasets.utils import calculate_norm_factors, fft_shift_vectorized, tdoa_corrections

FP_DATASET_NAME = "FP" 
CSI_STRUCT_KEY = "CIR"
TIME_KEY = "timestamp"
DIST_METRIC_DS_NAME = "distances"
REFERENCE_STRUCT_KEY = "reference"
VALIDITY_MASK_KEY = "valid_mask"
CC_DIST_DATASET_NAME = "CC_DIST"
TDOA_STRUCT_KEY = "toa"


class DistMetricDataset(torch.utils.data.Dataset):
    # Dataset used to create and evaluate distance-metrics
    
    def __init__(self, file_path: str, reduction_factor = 1, smoothing_type = None, smoothing_params= {}, norm = None, norm_range="self", align_tdoa = False, tdoa_correction = None, sampling_frequency = 8.138020833333334e-09
):

        self.attrs = {}
        with h5py.File(file_path) as f:

            self.ds = f[FP_DATASET_NAME][:]
            self.csi = self.ds[CSI_STRUCT_KEY][:]
            self.timestamps = self.ds[TIME_KEY][:]

            if VALIDITY_MASK_KEY in self.ds.dtype.names:
                self.valid_mask = self.ds[VALIDITY_MASK_KEY][:]
            else:
                self.valid_mask = None

            if TDOA_STRUCT_KEY in self.ds.dtype.names:
                self.tdoas = self.ds[TDOA_STRUCT_KEY][:]
            else:
                self.tdoas = None

            # First smoohting
            if smoothing_type is not None:
                self.csi, self.valid_mask = Smoothing[smoothing_type](self.csi, timestamps = self.timestamps, validity_mask=self.valid_mask, **smoothing_params)

            if norm is not None:
                norm_factors = calculate_norm_factors(self.csi, norm, norm_range)
            
                # Normalisiere in-place
                self.csi /= norm_factors

            if tdoa_correction is not None:
                if self.tdoas is None:
                    print("[WARN] tdoa correction is set, but the dataset has no tdoa values")
                else:
                    print("Applying TDOA corrections")
                    self.tdoas = tdoa_corrections(self.tdoas, self.timestamps, **tdoa_correction)

            if align_tdoa:
                if self.tdoas is None:
                    print("[WARN] tdoa alignment is set, but the dataset has no tdoa values")
                else:
                    print("Shifting CIRS")
                    self.csi = fft_shift_vectorized(self.csi, self.tdoas / sampling_frequency)

            self.csi = self.csi[::reduction_factor]
            if self.valid_mask is not None:
                self.valid_mask = self.valid_mask[::reduction_factor]

            self.timestamps = self.timestamps[::reduction_factor]
            self.ds = self.ds[::reduction_factor]

            if self.tdoas is not None:
                self.tdoas = self.tdoas[::reduction_factor]

            if REFERENCE_STRUCT_KEY in self.ds.dtype.names:
                self.ref = self.ds[REFERENCE_STRUCT_KEY][:]
            else:
                self.ref = None
            
            self.attrs = {key: value for key, value in f[FP_DATASET_NAME].attrs.items()}
 

    def write(self, file_path, distances, gen_params):
        with h5py.File(file_path, "w") as f:
            f.create_dataset(FP_DATASET_NAME, data=self.ds)
            f.create_dataset(CC_DIST_DATASET_NAME, data=distances)
            f[CC_DIST_DATASET_NAME].attrs["gen_params"] = gen_params
            for key, val in self.attrs.items():
                f[FP_DATASET_NAME].attrs[key] = val