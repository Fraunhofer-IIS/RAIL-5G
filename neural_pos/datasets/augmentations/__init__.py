from neural_pos.datasets.augmentations.cir_augmentations import IQToAbsAugmentation, TimeToFrequencyAugmentation
from neural_pos.datasets.augmentations.random_mask_fixed_k import RandomMaskFixedK

Augmentations = {
    "IQToAbs": IQToAbsAugmentation,
    "RandomMaskFixedK": RandomMaskFixedK,
    "TimeToFrequency": TimeToFrequencyAugmentation
}