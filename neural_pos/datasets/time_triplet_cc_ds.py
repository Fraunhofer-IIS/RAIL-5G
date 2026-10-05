import numpy as np

class TimeTripletCCDataset:

    def __init__(self, time_range, recording_frequency, base_dataset_type, base_dataset_params: dict = {}):
        """
        Args:
            base_dataset: Instance of FingerprintingDataset to use as base
            time_range: Time range for positive sample selection
            recording_frequency: Recording frequency for calculating index offsets
        """
        from . import Datasets

        self.base_dataset = Datasets[base_dataset_type](**base_dataset_params)
        self.time_range = time_range
        self.recording_frequency = recording_frequency
        
        # Precompute maximum index offset for efficient positive selection
        self.max_index_offset = int(np.ceil(time_range * recording_frequency))

    def __len__(self):
        return len(self.base_dataset)
    
    def _get_positive_index(self, anchor_idx):
        """
        Get a positive sample index within the time range of the anchor.
        Uses precomputed index offset for efficient search, then vectorized
        operations to calculate actual time differences.
        """
        # Calculate index range to search (based on max possible offset)
        min_idx = max(0, anchor_idx - self.max_index_offset)
        max_idx = min(len(self.base_dataset.csi) - 1, anchor_idx + self.max_index_offset)
        
        # Get candidate indices (excluding the anchor itself)
        candidate_indices = np.arange(min_idx, max_idx + 1)
        candidate_indices = candidate_indices[candidate_indices != anchor_idx]
        
        if len(candidate_indices) == 0:
            # Fallback: return the anchor itself if no other candidates
            return anchor_idx
        
        # Vectorized calculation of actual time differences
        anchor_time = self.base_dataset.timestamps[anchor_idx]
        candidate_times = self.base_dataset.timestamps[candidate_indices]
        time_diffs = np.abs(candidate_times - anchor_time)
        
        # Filter candidates within actual time range
        valid_mask = time_diffs <= self.time_range
        valid_candidates = candidate_indices[valid_mask]
        
        if len(valid_candidates) == 0:
            # Fallback: return closest candidate if none within time range
            closest_idx = candidate_indices[np.argmin(time_diffs)]
            return closest_idx
        
        # Randomly select one valid positive candidate
        positive_idx = np.random.choice(valid_candidates)
        return positive_idx
    
    def _get_negative_index(self, anchor_idx):
        """
        Get a random negative sample from the entire dataset.
        """
        negative_idx = np.random.randint(0, len(self.base_dataset.csi))
        return negative_idx
    
    def __getitem__(self, index):
        # Anchor (the query sample)
        anchor = self.base_dataset.csi[index].astype(self.base_dataset.dtype)
        
        # Positive: temporally close sample
        positive_idx = self._get_positive_index(index)
        positive = self.base_dataset.csi[positive_idx].astype(self.base_dataset.dtype)
        
        # Negative: random sample
        negative_idx = self._get_negative_index(index)
        negative = self.base_dataset.csi[negative_idx].astype(self.base_dataset.dtype)
        
        ret = {
            "inputs": anchor,
            "inputs_positive": positive,
            "inputs_negative": negative
        }

        # Include validity masks if available
        if self.base_dataset.valid_mask is not None:
            ret["valid_mask"] = self.base_dataset.valid_mask[index].astype(self.base_dataset.dtype)
            ret["valid_mask_positive"] = self.base_dataset.valid_mask[positive_idx].astype(self.base_dataset.dtype)
            ret["valid_mask_negative"] = self.base_dataset.valid_mask[negative_idx].astype(self.base_dataset.dtype)

        # Apply augmentations
        for augmentation in self.base_dataset.augmentations:
            ret = augmentation(ret)

        return ret
    
    def get_data_shape(self):
        return self.base_dataset[0]["inputs"].shape