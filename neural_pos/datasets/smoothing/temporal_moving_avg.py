from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

import numpy as np
from tqdm import tqdm


def temporal_moving_average(values, timestamps, window_seconds = 0.1, 
                           window_seconds_phase=0.03,
                           mode="normal", alpha=None, alpha_phase=None,
                           validity_mask=None, n_workers=None, use_processes=False,
                           sampling_frequency=100.0, drift_margin=1.15):
    """
    Compute moving average based on timestamp windows (centered).
    Smooths magnitude and phase separately, then converts back to complex.
    
    Parameters:
    -----------
    values : np.ndarray
        Input array of shape (L, 2, A, T) where dim 1 is [real, imag].
    timestamps : np.ndarray
        Timestamps array of shape (L,) in seconds (must be ordered).
    window_seconds : float
        Window size in seconds for magnitude smoothing.
    window_seconds_phase : float, optional
        Window size in seconds for phase smoothing. If None, uses window_seconds.
    mode : str, default="normal"
        "normal" - Simple moving average
        "exponential" - Exponential moving average
    alpha : float, optional
        Decay parameter for exponential mode (magnitude).
    alpha_phase : float, optional
        Decay parameter for exponential mode (phase). If None, uses alpha.
    validity_mask : np.ndarray, optional
        Boolean mask of shape (L, A) indicating valid samples.
    n_workers : int, optional
        Number of parallel workers. If None, uses CPU count.
    use_processes : bool, default=False
        If True, uses ProcessPoolExecutor instead of ThreadPoolExecutor.
    sampling_frequency : float, default=100.0
        Expected sampling frequency in Hz.
    drift_margin : float, default=1.15
        Margin factor to account for clock drift (e.g., 1.15 = 15% margin).
        
    Returns:
    --------
    result : np.ndarray
        Smoothed values in complex form (L, 2, A, T) where dim 1 is [real, imag].
    updated_mask : np.ndarray or None
        Updated validity mask of shape (L, A).
    """
    if mode == "mean":
        mode = "normal"

    values = np.asarray(values)
    timestamps = np.asarray(timestamps)
    
    if values.ndim != 4:
        raise ValueError(f"Values must be 4D (L, C, A, T), got shape {values.shape}")
    
    L, C, A, T = values.shape
    
    if C != 2:
        raise ValueError(f"Second dimension must be 2 (real, imag), got {C}")
    
    if timestamps.shape[0] != L:
        raise ValueError(
            f"Timestamps length ({timestamps.shape[0]}) must match "
            f"first dimension of values ({L})"
        )
    
    # Set phase window if not provided
    if window_seconds_phase is None:
        window_seconds_phase = window_seconds
    
    # Set phase alpha if not provided
    if alpha_phase is None:
        alpha_phase = alpha
    
    # Handle validity mask
    if validity_mask is not None:
        validity_mask = np.asarray(validity_mask, dtype=bool)
        if validity_mask.shape != (L, A):
            raise ValueError(
                f"Validity mask shape {validity_mask.shape} must be ({L}, {A})"
            )
        use_mask = True
    else:
        validity_mask = np.ones((L, A), dtype=bool)
        use_mask = False
    
    # Convert to complex, then to magnitude and phase
    complex_values = values[:, 0, :, :] + 1j * values[:, 1, :, :]  # Shape: (L, A, T)
    magnitude = np.abs(complex_values)  # Shape: (L, A, T)
    phase = np.angle(complex_values)  # Shape: (L, A, T)
    
    # Pre-compute sin and cos for phase smoothing (vectorized)
    sin_phase = np.sin(phase)  # Shape: (L, A, T)
    cos_phase = np.cos(phase)  # Shape: (L, A, T)
    
    # Prepare output arrays
    smoothed_magnitude = np.zeros_like(magnitude, dtype=np.float64)
    smoothed_phase = np.zeros_like(phase, dtype=np.float64)
    updated_mask = np.zeros((L, A), dtype=bool)
    
    # Set alpha values for exponential mode
    if mode == "exponential":
        if alpha is None:
            alpha = 2.0 / (window_seconds + 1.0)
        if alpha_phase is None:
            alpha_phase = 2.0 / (window_seconds_phase + 1.0)
    
    # Calculate window sizes in samples (with margin for clock drift)
    window_samples_mag = int(np.ceil(window_seconds * sampling_frequency * drift_margin))
    window_samples_phase = int(np.ceil(window_seconds_phase * sampling_frequency * drift_margin))
    half_window_mag = window_samples_mag // 2
    half_window_phase = window_samples_phase // 2
    
    # Set number of workers
    if n_workers is None:
        n_workers = multiprocessing.cpu_count()
    
    # Pre-compute exponential weights if needed (for centered window)
    exp_weights_mag = None
    exp_weights_phase = None
    if mode == "exponential":
        # Generate symmetric weights for centered window
        offsets_mag = np.arange(-half_window_mag, half_window_mag + 1)
        distances_mag = np.abs(offsets_mag) / sampling_frequency
        exp_weights_mag = np.exp(-alpha * distances_mag)
        exp_weights_mag /= np.sum(exp_weights_mag)
        
        offsets_phase = np.arange(-half_window_phase, half_window_phase + 1)
        distances_phase = np.abs(offsets_phase) / sampling_frequency
        exp_weights_phase = np.exp(-alpha_phase * distances_phase)
        exp_weights_phase /= np.sum(exp_weights_phase)
    
    # Worker function to process a single time point
    def process_time_point(l):
        # Calculate window bounds using sample indices
        start_mag = max(0, l - half_window_mag)
        end_mag = min(L, l + half_window_mag + 1)
        
        start_phase = max(0, l - half_window_phase)
        end_phase = min(L, l + half_window_phase + 1)
        
        local_smoothed_mag = np.zeros((A, T), dtype=np.float64)
        local_smoothed_phase = np.zeros((A, T), dtype=np.float64)
        local_mask = np.zeros(A, dtype=bool)
        
        # For each antenna/path dimension
        for a in range(A):
            # Check if current point is valid
            if not validity_mask[l, a]:
                local_smoothed_mag[a, :] = np.nan
                local_smoothed_phase[a, :] = np.nan
                local_mask[a] = False
                continue
            
            # === MAGNITUDE SMOOTHING ===
            valid_in_window_mag = validity_mask[start_mag:end_mag, a]
            
            if np.any(valid_in_window_mag):
                # Get valid magnitude values in window
                window_mag_values = magnitude[start_mag:end_mag, a, :]
                window_mag_values = window_mag_values[valid_in_window_mag, :]
                
                if mode == "normal":
                    local_smoothed_mag[a, :] = np.mean(window_mag_values, axis=0)
                    
                elif mode == "exponential":
                    # Get corresponding weights for valid samples
                    offset_start = start_mag - l + half_window_mag
                    offset_end = end_mag - l + half_window_mag
                    weights = exp_weights_mag[offset_start:offset_end][valid_in_window_mag]
                    weights = weights / np.sum(weights)  # Renormalize
                    weights_reshaped = weights.reshape(-1, 1)
                    local_smoothed_mag[a, :] = np.sum(
                        window_mag_values * weights_reshaped, axis=0
                    )
            else:
                local_smoothed_mag[a, :] = magnitude[l, a, :]
            
            # === PHASE SMOOTHING (circular mean) ===
            valid_in_window_phase = validity_mask[start_phase:end_phase, a]
            
            if np.any(valid_in_window_phase):
                # Use pre-computed sin/cos values
                window_sin = sin_phase[start_phase:end_phase, a, :]
                window_cos = cos_phase[start_phase:end_phase, a, :]
                window_sin = window_sin[valid_in_window_phase, :]
                window_cos = window_cos[valid_in_window_phase, :]
                
                if mode == "normal":
                    sin_mean = np.mean(window_sin, axis=0)
                    cos_mean = np.mean(window_cos, axis=0)
                    local_smoothed_phase[a, :] = np.arctan2(sin_mean, cos_mean)
                    
                elif mode == "exponential":
                    # Get corresponding weights for valid samples
                    offset_start = start_phase - l + half_window_phase
                    offset_end = end_phase - l + half_window_phase
                    weights = exp_weights_phase[offset_start:offset_end][valid_in_window_phase]
                    weights = weights / np.sum(weights)  # Renormalize
                    weights_reshaped = weights.reshape(-1, 1)
                    
                    sin_mean = np.sum(window_sin * weights_reshaped, axis=0)
                    cos_mean = np.sum(window_cos * weights_reshaped, axis=0)
                    local_smoothed_phase[a, :] = np.arctan2(sin_mean, cos_mean)
            else:
                local_smoothed_phase[a, :] = phase[l, a, :]
            
            local_mask[a] = True
        
        return l, local_smoothed_mag, local_smoothed_phase, local_mask
    
    # Execute in parallel
    if use_processes:
        # Use multiprocessing.Pool for CPU-bound work
        with multiprocessing.Pool(processes=n_workers) as pool:
            results = list(tqdm(
                pool.imap(process_time_point, range(L)),
                total=L,
                desc="Smoothing CIRs"
            ))
    else:
        # Use ThreadPoolExecutor (default)
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(process_time_point, l): l for l in range(L)}
            results = []
            
            for future in tqdm(as_completed(futures), total=L, desc="Smoothing CIRs"):
                results.append(future.result())
            
            # Sort results by index
            results.sort(key=lambda x: x[0])
    
    # Collect results
    for l, local_smoothed_mag, local_smoothed_phase, local_mask in results:
        smoothed_magnitude[l, :, :] = local_smoothed_mag
        smoothed_phase[l, :, :] = local_smoothed_phase
        updated_mask[l, :] = local_mask
    
    # Convert back to complex (real, imag)
    complex_smoothed = smoothed_magnitude * np.exp(1j * smoothed_phase)
    
    result = np.zeros((L, 2, A, T), dtype=np.float64)
    result[:, 0, :, :] = np.real(complex_smoothed)
    result[:, 1, :, :] = np.imag(complex_smoothed)
    
    return result, (updated_mask if use_mask else None)