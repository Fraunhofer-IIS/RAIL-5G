import numpy as np
from scipy.fft import fft, ifft, fftfreq
from scipy.ndimage import convolve1d, median_filter


def fft_shift_vectorized(input_signal, delay_samples):
    """
    Apply FFT shift theorem to align CIR signals.
    
    Parameters:
    -----------
    input_signal : np.ndarray
        Shape (N, 2, A, T) where:
        - N: number of samples
        - 2: real/imaginary components
        - A: number of antennas
        - T: delay taps (time samples)

    delay_samples : np.ndarray
        Shape (N, A) - delay for each sample and antenna
    
    Returns:
    --------
    np.ndarray
        Aligned signal in shape (N, 2, A, T)
    """
    N, _, A, T = input_signal.shape
    
    # Convert from (N, 2, A, T) to complex (N, A, T)
    complex_signal = input_signal[:, 0, :, :] + 1j * input_signal[:, 1, :, :]
    
    # Apply FFT along time axis (last dimension)
    signal_fft = fft(complex_signal, axis=-1)  # Shape: (N, A, T)
    
    # Get frequency bins for the FFT
    freqs = fftfreq(T)  # Shape: (T,)
    
    # Reshape for broadcasting: delays (N, A, 1), freqs (1, 1, T)
    delays_reshaped = delay_samples[:, :, np.newaxis]  # (N, A, 1)
    freqs_reshaped = freqs[np.newaxis, np.newaxis, :]  # (1, 1, T)
    
    # Compute phase shift: exp(-1j * 2π * f * delay)
    phase_shift = np.exp(-1j * 2 * np.pi * freqs_reshaped * delays_reshaped)
    
    # Apply phase shift
    shifted_fft = signal_fft * phase_shift  # Broadcasting: (N, A, T)
    
    # Inverse FFT
    shifted_signal = ifft(shifted_fft, axis=-1)
    
    # Convert back to (N, 2, A, T) format
    result = np.stack([shifted_signal.real, shifted_signal.imag], axis=1)
    
    return result


def interpolate_nan(array, axis=0):
    """
    Linearly interpolate NaN values in a numpy array along a specified axis.
    
    Parameters:
    -----------
    array : np.ndarray
        Input array with NaN values
    axis : int, optional
        Axis along which to interpolate (default: 0)
        
    Returns:
    --------
    np.ndarray
        Array with NaN values interpolated
    """
    array = np.array(array, dtype=float)
    
    # Move the axis to interpolate to the first position
    array = np.moveaxis(array, axis, 0)
    
    # Get the shape and create output array
    result = array.copy()
    
    # Iterate over all other dimensions
    for idx in np.ndindex(array.shape[1:]):
        # Extract 1D slice along the interpolation axis
        slice_data = array[(slice(None),) + idx]
        
        # Find valid (non-NaN) indices
        valid_mask = ~np.isnan(slice_data)
        
        if valid_mask.any():
            # Get indices
            indices = np.arange(len(slice_data))
            valid_indices = indices[valid_mask]
            valid_values = slice_data[valid_mask]
            
            # Interpolate (extrapolation will use nearest boundary value)
            interpolated = np.interp(indices, valid_indices, valid_values)
            
            # Update result
            result[(slice(None),) + idx] = interpolated
    
    # Move axis back to original position
    result = np.moveaxis(result, 0, axis)
    
    return result


def tdoa_corrections(tdoa, timestamps, kernel_size, max_time_diff=None, filter_type="mean"):
    if filter_type not in ("mean", "median"):
        raise ValueError(f"filter_type must be 'mean' or 'median', got '{filter_type}'")

    def apply_filter(data):
        if filter_type == "mean":
            return convolve1d(data, np.ones(kernel_size) / kernel_size, axis=0)
        else:
            return median_filter(data, size=(kernel_size, 1))

    if max_time_diff is None:
        tdoa = interpolate_nan(tdoa, axis=0)
        tdoa = apply_filter(tdoa)
        return tdoa

    diffs = np.diff(timestamps)
    split_indices = np.where(diffs > max_time_diff)[0] + 1
    segments = np.split(np.arange(len(tdoa)), split_indices)

    print(f"TDOA Correction: Processing {len(segments)} different subsets")

    result = tdoa.copy()
    for seg in segments:
        if len(seg) == 0:
            continue
        chunk = tdoa[seg]
        chunk = interpolate_nan(chunk, axis=0)
        chunk = apply_filter(chunk)
        result[seg] = chunk

    return result


def calculate_norm_factors(cirs, norm, norm_range):
    cir_abs = np.nan_to_num(np.linalg.norm(cirs, axis=1, keepdims=True))
    assert norm in ["unit", "max"], f"Invalid norm {norm}!"
    assert norm_range in ["self", "set", "antenna", "full"], f"Invalid norm_range {norm_range}!"

    if norm == "unit":
        if norm_range == "self":
            axis = 3
        elif norm_range == "set":
            axis = (2,3)
        elif norm_range == "full":
            raise NotImplementedError("Unit norm over the whole dataset does not make sense!")
        elif norm_range == "antenna":
            raise NotImplementedError("Unit norm over a antenna does not make sense!")
        
        norm_factors = np.linalg.norm(
            cir_abs, 
            axis=axis, 
            keepdims=True
        )
    elif norm == "max":
        if norm_range == "self":
            axis = 3
        elif norm_range == "set":
            axis = (2,3)
        elif norm_range == "antenna":
            axis = (0,1,3)
        elif norm_range == "full":
            axis = (0,1,2,3)
        
        norm_factors = np.max(cir_abs, axis=axis, keepdims=True)
        
    return norm_factors