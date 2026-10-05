import re

import numpy as np

class RandomMaskFixedK:

    def __init__(self, k: bool = False, mask_name = None):
        self.k = k
        self.mask_name = mask_name

    def __call__(self, item: dict) -> dict:
        for key in list(item.keys()):
            # Simple string check - faster than regex
            if not key.startswith('inputs'):
                continue
            
            suffix = key[6:]  # Remove 'inputs' prefix
            cir = item[key]
            valid_mask_key = f"valid_mask{suffix}"
            valid_mask = item.get(valid_mask_key, np.ones(cir.shape[1]))
            
            num_valid = valid_mask.sum()
            
            if self.k < num_valid:
                valid_indices = np.where(valid_mask)[0]
                set_invalid = np.random.choice(valid_indices, int(num_valid - self.k))
                new_mask = valid_mask.copy()
                new_mask[set_invalid] = False
                
                if self.mask_name is not None:
                    output_mask_key = f"{self.mask_name}{suffix}"
                else:
                    output_mask_key = valid_mask_key
                
                item[output_mask_key] = new_mask
        
        return item