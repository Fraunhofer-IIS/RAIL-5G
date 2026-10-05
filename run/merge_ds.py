#!/usr/bin/env python3
"""
Merge datasets

Usage:
  python merge_ds.py --base-folder /path/to/data \
      --ds-indices 1,2,3 --output-file /path/to/out.h5 \

"""
import argparse
import json
import os
import numpy as np
import h5py

def find_first_dataset(obj):
    if isinstance(obj, h5py.Dataset):
        return obj
    if isinstance(obj, h5py.Group):
        for key in obj:
            ds = find_first_dataset(obj[key])
            if ds is not None:
                return ds
    return None

def main():
    parser = argparse.ArgumentParser(description="Merge datasets.")
    parser.add_argument("--base-folder", required=True, help="Path containing takeXX.h5")
    parser.add_argument("--ds-indices", required=True, help="Comma-separated list of take indices, e.g., 1,2,3")
    parser.add_argument("--output-file", required=True, help="Path to write FP (HDF5)")
    parser.add_argument("--optimized", default=False, action="store_true", help="Will reduce file size, but also produce slightly different results than reported in the paper.")

    args = parser.parse_args()

    ds_indices = [int(x) for x in args.ds_indices.split(",") if x.strip()]
    CIR_all = []
    mask_all = []
    timestamp_all = []
    reference_all = []
    toa_all = []
    interp_diff_all = []
    info = None

    for idx in ds_indices:
        processed_path = os.path.join(args.base_folder, f"take{idx}.h5")
        if not os.path.exists(processed_path):
            raise FileNotFoundError(f"Processed file not found: {processed_path}")

        with h5py.File(processed_path, "r") as f:
            dset = find_first_dataset(f)
            if dset is None:
                raise RuntimeError(f"No dataset found in {processed_path}")

            data = dset[ ... ]
            CIR_i = data["CIR"]                
            mask_i = data["antenna_mask"]      
            timestamp_i = data["timestamp"]    
            reference_i = data["reference"]    
            tdoa_i = data["toa"] 
            interp_diff_i = data["interpolation_diff"] 
            info = f.attrs["description"]

        L, R, A, F = CIR_i.shape

        CIR_all.append(CIR_i)
        mask_all.append(mask_i)
        timestamp_all.append(timestamp_i)
        reference_all.append(reference_i)
        toa_all.append(tdoa_i)
        interp_diff_all.append(interp_diff_i)

    # Concatenate across datasets 
    CIR_concat = np.concatenate(CIR_all, axis=0)           
    mask_concat = np.concatenate(mask_all, axis=0)         
    timestamp_concat = np.concatenate(timestamp_all, axis=0)  
    reference_concat = np.concatenate(reference_all, axis=0)  
    toa_concat = np.concatenate(toa_all, axis=0)  
    interp_diff_concat = np.concatenate(interp_diff_all, axis=0)    

    toa_concat[~mask_concat] = np.nan

    # Determine final dimensions
    N_total, R, A, F = CIR_concat.shape

    if args.optimized:
        dtype_out = np.dtype([
            ("CIR", np.float32, (R, A, F)),
            ("valid_mask", np.bool_, (A,)),
            ("timestamp", np.float64),
            ("reference", np.float32, (reference_concat.shape[1],)),
            ("toa", np.float32, (A,)),
            ("interpolation_diff", np.float32, (interp_diff_concat.shape[1],))
        ])
    else:
        dtype_out = np.dtype([
            ("CIR", np.float64, (R, A, F)),
            ("valid_mask", np.bool_, (A,)),
            ("timestamp", np.float64),
            ("reference", np.float64, (reference_concat.shape[1],)),
            ("toa", np.float64, (A,)),
            ("interpolation_diff", np.float64, (interp_diff_concat.shape[1],))
        ])

    recs = np.empty(N_total, dtype=dtype_out)
    recs["CIR"] = CIR_concat
    recs["valid_mask"] = mask_concat
    recs["timestamp"] = timestamp_concat
    recs["reference"] = reference_concat
    recs["toa"] = toa_concat
    recs["interpolation_diff"] = interp_diff_concat

    with h5py.File(args.output_file, "w") as f_out:
        f_out.create_dataset("FP", data=recs)
        if info is not None:
            info = json.loads(info)
            if "additional_info" in info.keys():
                info = info["additional_info"]
                if "cir_res" in info.keys():
                    f_out["FP"].attrs["cir_res"] = info["cir_res"]
                else:
                    print("WARN: no cir_res")
                if "ru_ids" in info.keys():
                    f_out["FP"].attrs["ru_ids"] = info["ru_ids"]
                else:
                    print("WARN: no ru_ids")
            else:
                print("WARN: No additional infos")

    print(f"Wrote {args.output_file} with {N_total} records (dataset 'FP').")

if __name__ == "__main__":
    main()