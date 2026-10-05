import torch
#import h5pickle as h5py
import h5py
import numpy as np
import random

CC_DIST_DS_NAME = "CC_DIST"

class SiameseCCDataset:

    def __init__(self, file_path, base_dataset_type, base_dataset_params: dict = {}):
        """
        Args:
            file_path: Path to the HDF5 file containing distance matrix
            base_dataset: Instance of FingerprintingDataset to use as base
            seed: Random seed for reproducibility
        """
        from . import Datasets
        self.base_dataset = Datasets[base_dataset_type](file_path, **base_dataset_params)
        
        with h5py.File(file_path) as f:
            self.distances = f[CC_DIST_DS_NAME][:]
        
    def get_distance(self, index, second_index):
        """Access condensed distance matrix."""
        # Handle diagonal (distance to itself)
        if index == second_index:
            return np.zeros(1, dtype=self.base_dataset.dtype)[0]
        
        # Ensure index < second_index (condensed format stores upper triangle)
        if index > second_index:
            index, second_index = second_index, index
        
        # Convert to condensed index
        n = len(self.base_dataset.csi)  # Total number of samples
        condensed_index = n * index - (index * (index + 1)) // 2 + second_index - index - 1
        
        return self.distances[condensed_index].astype(self.base_dataset.dtype)

    def get_data_shape(self):
        return self.base_dataset[0]["inputs"].shape

    def __len__(self):
        return len(self.base_dataset.csi)
    
    def __getitem__(self, index):
        second_index = np.random.randint(0, len(self.base_dataset) - 1)
        
        ret = {
            "inputs": self.base_dataset.csi[index].astype(self.base_dataset.dtype), 
            "inputs2": self.base_dataset.csi[second_index].astype(self.base_dataset.dtype), 
            "targets": self.get_distance(index, second_index)
        }

        if self.base_dataset.valid_mask is not None:
            ret["valid_mask"] = self.base_dataset.valid_mask[index].astype(self.base_dataset.dtype)
            ret["valid_mask2"] = self.base_dataset.valid_mask[second_index].astype(self.base_dataset.dtype)

        for augmentation in self.base_dataset.augmentations:
            ret = augmentation(ret)

        return ret
