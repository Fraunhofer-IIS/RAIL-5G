import torch
#import h5pickle as h5py
import h5py
from neural_pos.datasets.augmentations import Augmentations
import numpy as np

from neural_pos.datasets.smoothing import Smoothing
from neural_pos.datasets.utils import calculate_norm_factors, fft_shift_vectorized, tdoa_corrections

FP_DATASET_NAME = "FP"
CSI_STRUCT_KEY = "CIR"
REFERENCE_STRUCT_KEY = "reference"
VALIDITY_MASK_KEY = "valid_mask"
TIME_KEY = "timestamp"
INTERPOLATION_DIFF_KEY = "interpolation_diff"
TOA_STRUCT_KEY = "toa"

ATTRS_KEY_CIR_RES = "cir_res"
ATTRS_KEY_RU_IDS = "ru_ids"

class FingerprintingDataset(torch.utils.data.Dataset):
    
    def __init__(self, file_path: str, augmentations = None, dtype="float32", smoothing_type = None, smoothing_params = {}, norm = None, norm_range = "self", align_tdoa = False, tdoa_correction = None, filter_rus = None, return_invalid_reference = True, reduction=1):

        with h5py.File(file_path) as f:

            # csi and reference are both required for a fingerprinting dataset
            ds = f[FP_DATASET_NAME][::reduction]
            self.csi = ds[CSI_STRUCT_KEY][:]
            self.reference = ds[REFERENCE_STRUCT_KEY][:]
            self.timestamps = ds[TIME_KEY][:]
            self.timestamps = self.timestamps - self.timestamps.min() # Make them relative to the first sample, so it will fit in float32

            self.cir_res = f[FP_DATASET_NAME].attrs[ATTRS_KEY_CIR_RES]
            self.ru_ids: list = f[FP_DATASET_NAME].attrs[ATTRS_KEY_RU_IDS]

            if TOA_STRUCT_KEY in ds.dtype.names:
                self.toa = ds[TOA_STRUCT_KEY][:]
            else:
                self.toa = None

            # validity_mask is optional
            if VALIDITY_MASK_KEY in ds.dtype.names:
                self.valid_mask = None # ds[VALIDITY_MASK_KEY][:] (We use nan_to_num to set masked values to 0)
            else:
                self.valid_mask = None

            if INTERPOLATION_DIFF_KEY in ds.dtype.names:
                self.ref_valid = np.sum(ds[INTERPOLATION_DIFF_KEY], axis=1) < 1 # Reference is only valid if we interpolate less than 1 second
            else:
                self.ref_valid = np.ones_like(self.timestamps, dtype=np.bool)
            
            if filter_rus is not None:
                ru_indices = sorted([ self.ru_ids.index(ru_id) for ru_id in filter_rus])
                self.ru_ids = self.ru_ids[ru_indices]
                self.csi = self.csi[:, :, ru_indices, :]
                if self.toa is not None:
                    self.toa = self.toa[:, ru_indices]
                if self.valid_mask is not None:
                    self.valid_mask = self.valid_mask[:, ru_indices]

        if smoothing_type is not None:
            self.csi, self.valid_mask = Smoothing[smoothing_type](self.csi, timestamps = self.timestamps, validity_mask=self.valid_mask, **smoothing_params)

        if norm is not None:
            norm_factors = calculate_norm_factors(self.csi, norm, norm_range)
            
            # Normalisiere in-place
            self.csi /= norm_factors

        if tdoa_correction is not None:
            if self.toa is None:
                print("[WARN] tdoa correction is set, but the dataset has no tdoa values")
            else:
                print("Applying TDOA corrections")
                self.toa = tdoa_corrections(self.toa, self.timestamps, **tdoa_correction)

        if align_tdoa:
            if self.toa is None:
                print("[WARN] tdoa alignment is set, but the dataset has no tdoa values")
            else:
                print("Aĺigning CIRs to TDOA")
                self.csi = fft_shift_vectorized(self.csi, self.toa / self.cir_res)

        self.csi = np.nan_to_num(self.csi)

        self.augmentations = []

        if augmentations is not None:
            for augmentation in augmentations:
                self.augmentations.append(Augmentations[augmentation["augmentation_name"]](**augmentation["augmentation_params"]))
        
        self.dtype = np.dtype(dtype)
        self.return_invalid_reference = return_invalid_reference

    def get_data_shape(self):
        return self[0]["inputs"].shape

    def __len__(self):
        if self.return_invalid_reference:
            return len(self.csi)
        else:
            return self.return_invalid_reference.sum()
    
    def __getitem__(self, index):
        if not self.return_invalid_reference:
            index = np.where(self.ref_valid)[0][index]

        ret = {
            "inputs": self.csi[index].astype(self.dtype), 
            "targets": self.reference[index].astype(self.dtype),
            "timestamps": self.timestamps[index].astype(self.dtype)
        }

        if self.valid_mask is not None:
            ret["valid_mask"] = self.valid_mask[index]

        for augmentation in self.augmentations:
            ret = augmentation(ret)

        return ret
