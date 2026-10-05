import numpy as np
from scipy.ndimage import convolve1d

def kernel_moving_average(values, kernel_size_mag, kernel_size_phase=None, 
               mode="mean", validity_mask=None, min_valid_samples=1,
               timestamps=None, max_time_diff=1, **kwargs):
    if kwargs:
        print("Warn: Unknown smoothing params", kwargs.keys(), "for kernel_moving_average")

    values = np.nan_to_num(values)
    L, C, A, T = values.shape

    if kernel_size_phase is None:
        kernel_size_phase = kernel_size_mag

    # Handle validity mask
    if validity_mask is None:
        validity_mask = np.ones((L, A, T), dtype=bool)
    else:
        if validity_mask.shape == (L, A):
            validity_mask = np.broadcast_to(
                validity_mask[:, :, np.newaxis], (L, A, T)
            ).copy()

    # Split by time gaps if requested
    if timestamps is not None and max_time_diff is not None:
        time_diffs = np.diff(timestamps)
        split_indices = np.where(time_diffs > max_time_diff)[0] + 1
        
        if len(split_indices) > 0:
            print(f"Splitting dataset into {len(split_indices) + 1} subsets")
            segments = np.split(np.arange(L), split_indices)
            smoothed = np.empty_like(values)
            output_validity = np.empty((L, A, T), dtype=bool)
            
            for seg in segments:
                s_vals, s_val = kernel_moving_average(
                    values[seg], kernel_size_mag, kernel_size_phase,
                    mode=mode, validity_mask=validity_mask[seg],
                    min_valid_samples=min_valid_samples
                )
                smoothed[seg] = s_vals
                output_validity[seg] = s_val[:, :, np.newaxis] if s_val.ndim == 2 else s_val
            
            output_validity = np.all(output_validity, axis=-1)
            return smoothed, output_validity

    # --- Original smoothing logic below ---
    if C == 2:
        complex_values = values[:, 0, :, :] + 1j * values[:, 1, :, :]
        magnitude = np.abs(complex_values)
        phase = np.angle(complex_values)

        smoothed_mag, validity_mag = _smooth_with_mask(
            magnitude, kernel_size_mag, mode, validity_mask, min_valid_samples
        )
        smoothed_phase, validity_phase = _smooth_phase_with_mask(
            phase, kernel_size_phase, mode, validity_mask, min_valid_samples
        )

        output_validity = validity_mag & validity_phase
        smoothed_complex = smoothed_mag * np.exp(1j * smoothed_phase)
        smoothed = np.stack([smoothed_complex.real, smoothed_complex.imag], axis=1)
    else:
        magnitude = values[:, 0, :, :]
        smoothed_mag, output_validity = _smooth_with_mask(
            magnitude, kernel_size_mag, mode, validity_mask, min_valid_samples
        )
        smoothed = smoothed_mag[:, np.newaxis, :, :]

    output_validity = np.all(output_validity, axis=-1)
    return smoothed, output_validity


def _create_kernel(kernel_size, mode):
    """Create smoothing kernel."""
    if mode == "mean":
        kernel = np.ones(kernel_size) / kernel_size
    elif mode == "exponential":
        alpha = 2.0 / (kernel_size + 1)
        indices = np.arange(kernel_size)
        kernel = alpha * (1 - alpha) ** indices
        kernel = kernel[::-1]
        kernel /= kernel.sum()
    else:
        raise ValueError(f"mode must be 'mean' or 'exponential', got '{mode}'")
    return kernel


def _smooth_with_mask(data, kernel_size, mode, validity_mask, min_valid_samples):
    """
    Smooth magnitude data with validity tracking.
    
    Parameters
    ----------
    data : np.ndarray
        Shape (L, A, T)
    
    Returns
    -------
    smoothed : np.ndarray
        Shape (L, A, T)
    output_validity : np.ndarray
        Shape (L, A, T), bool
    """
    kernel = _create_kernel(kernel_size, mode)
    
    # Mask invalid data to zero
    masked_data = data * validity_mask
    
    # Convolve data and mask
    smoothed_numerator = convolve1d(masked_data, kernel, axis=0, mode='constant')
    mask_weights = convolve1d(validity_mask.astype(float), kernel, axis=0, mode='constant')
    
    # Count valid samples in each window
    valid_counts = convolve1d(
        validity_mask.astype(float), 
        np.ones(kernel_size), 
        axis=0, 
        mode='constant'
    )
    
    # Normalize
    smoothed = smoothed_numerator / mask_weights
    
    # Output is valid if at least min_valid_samples in the window
    output_validity = valid_counts >= min_valid_samples
    
    return smoothed, output_validity


def _smooth_phase_with_mask(phase, kernel_size, mode, validity_mask, min_valid_samples):
    """
    Smooth phase data with proper wrapping handling.
    
    Phase wrapping is handled by converting to unit complex numbers,
    averaging them, then extracting the angle.
    """
    kernel = _create_kernel(kernel_size, mode)
    
    # Convert phase to complex unit vectors
    phase_complex = np.exp(1j * phase)
    
    # Apply validity mask
    masked_real = phase_complex.real * validity_mask
    masked_imag = phase_complex.imag * validity_mask
    
    # Convolve real and imaginary parts separately
    smoothed_real = convolve1d(masked_real, kernel, axis=0, mode='constant')
    smoothed_imag = convolve1d(masked_imag, kernel, axis=0, mode='constant')
    
    # Get normalization weights
    mask_weights = convolve1d(validity_mask.astype(float), kernel, axis=0, mode='constant')
    
    # Normalize
    smoothed_real /= mask_weights
    smoothed_imag /= mask_weights
    
    # Convert back to phase
    smoothed_phase = np.angle(smoothed_real + 1j * smoothed_imag)
    
    # Count valid samples
    valid_counts = convolve1d(
        validity_mask.astype(float),
        np.ones(kernel_size),
        axis=0,
        mode='constant'
    )
    output_validity = valid_counts >= min_valid_samples
    
    return smoothed_phase, output_validity