from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations

import numpy as np
from tqdm import tqdm


def tdoa_least_squares_exhaustive(receiver_positions, relative_toas,
                                  speed_of_signal=3e8,
                                  fixed_dim=2, fixed_value=1.5,
                                  inlier_threshold=2.0):
    """
    TDOA estimator with exhaustive search over all antenna combinations.
    
    Deterministic - tests all C(N,3) combinations sequentially.
    
    Parameters:
    -----------
    receiver_positions : ndarray, shape (N, D)
        Antenna positions in meters
    relative_toas : ndarray, shape (N,)
        Time of arrival relative to earliest (seconds)
    speed_of_signal : float
        Signal speed (default: 3e8 m/s for RF)
    fixed_dim : int
        Dimension to fix (2 = z-axis for height)
    fixed_value : float
        Value for fixed dimension (height in meters)
    inlier_threshold : float
        NLOS detection threshold in meters (default: 2.0m)
        
    Returns:
    --------
    position : ndarray, shape (D,)
        Estimated source position
    inliers : ndarray, shape (N,)
        Boolean mask of LOS antennas
    """
    
    receiver_positions = np.array(receiver_positions)
    relative_toas = np.array(relative_toas, dtype=float)
    
    valid_mask = ~np.isnan(relative_toas)
    n_valid = np.sum(valid_mask)
    N, D = receiver_positions.shape
    
    if n_valid < 3:
        raise ValueError(f"Need at least 3 valid measurements, got {n_valid}")
    
    valid_receivers = receiver_positions[valid_mask]
    valid_toas = relative_toas[valid_mask]
    range_differences = valid_toas * speed_of_signal
    
    # Generate all combinations of 3 antennas (including reference)
    all_combinations = [combo for combo in combinations(range(n_valid), 3) if 0 in combo]
    
    if not all_combinations:
        # Fallback if reference not in valid set
        all_combinations = list(combinations(range(n_valid), 3))
    
    # Evaluate all combinations
    best_n_inliers = 0
    best_position = None
    best_inliers = None
    best_residual = np.inf
    
    for combo in all_combinations:
        try:
            # Fit model on this subset
            sample_pos = _fit_model(
                valid_receivers[list(combo)],
                range_differences[list(combo)],
                fixed_dim, fixed_value
            )
            
            # Compute residuals for all measurements
            distances = np.linalg.norm(valid_receivers - sample_pos, axis=1)
            predicted_rd = distances - distances[0]
            residuals = np.abs(range_differences - predicted_rd)
            
            # Find inliers
            inliers_local = residuals < inlier_threshold
            n_inliers = np.sum(inliers_local)
            
            # Mean residual of inliers (for tie-breaking)
            inlier_residuals = residuals[inliers_local]
            mean_residual = np.mean(inlier_residuals) if len(inlier_residuals) > 0 else np.inf
            
            # Update best (most inliers, then lowest residual)
            if (n_inliers > best_n_inliers or 
                (n_inliers == best_n_inliers and mean_residual < best_residual)):
                best_n_inliers = n_inliers
                best_position = sample_pos
                best_inliers = inliers_local
                best_residual = mean_residual
                
        except (ValueError, np.linalg.LinAlgError):
            continue
    
    if best_position is None:
        raise ValueError("All combinations failed to converge")
    
    # Refine using all inliers
    if best_n_inliers >= 3:
        final_position = _fit_model(
            valid_receivers[best_inliers],
            range_differences[best_inliers],
            fixed_dim, fixed_value
        )
    else:
        final_position = best_position
    
    # Map inliers back to full array
    full_inliers = np.zeros(N, dtype=bool)
    full_inliers[valid_mask] = best_inliers
    
    return final_position, full_inliers


def _fit_model(receivers, range_differences, fixed_dim, fixed_value,
               max_iterations=20, tolerance=1e-6):
    """
    Fit TDOA model using Gauss-Newton.
    """
    n_receivers = len(receivers)
    D = receivers.shape[1]
    
    # Initial guess
    x_full = np.mean(receivers, axis=0)
    x_full[fixed_dim] = fixed_value
    
    # Free dimensions (x, y) when height is fixed
    free_dims = np.ones(D, dtype=bool)
    free_dims[fixed_dim] = False
    x_free = x_full[free_dims]
    
    r0 = receivers[0]  # Reference receiver
    
    # Gauss-Newton iteration
    for iteration in range(max_iterations):
        # Update full position
        x_full[free_dims] = x_free
        x_full[fixed_dim] = fixed_value
        
        # Compute distances
        distances = np.linalg.norm(receivers - x_full, axis=1)
        
        # Predicted range differences
        predicted_rd = distances - distances[0]
        
        # Residuals (skip reference receiver)
        residuals = range_differences[1:] - predicted_rd[1:]
        
        # Check convergence
        if np.linalg.norm(residuals) < tolerance:
            break
        
        # Build Jacobian (only free dimensions)
        J = np.zeros((n_receivers - 1, 2))  # 2D when height fixed
        
        for i in range(1, n_receivers):
            dir_i = (x_full - receivers[i]) / (distances[i] + 1e-10)
            dir_0 = (x_full - r0) / (distances[0] + 1e-10)
            jacobian_full = dir_i - dir_0
            J[i-1] = jacobian_full[free_dims]
        
        # Least squares update
        JTJ = J.T @ J + np.eye(2) * 1e-8
        delta = np.linalg.solve(JTJ, J.T @ residuals)
        x_free = x_free + delta
    
    # Return full position
    x_full[free_dims] = x_free
    x_full[fixed_dim] = fixed_value
    
    return x_full


def _process_single_timestep(args):
    """Process single timestep - must be top-level for pickling."""
    t, receiver_positions, relative_toas, speed_of_signal, fixed_dim, fixed_value, inlier_threshold = args
    
    try:
        pos, inliers = tdoa_least_squares_exhaustive(
            receiver_positions,
            relative_toas,
            speed_of_signal=speed_of_signal,
            fixed_dim=fixed_dim,
            fixed_value=fixed_value,
            inlier_threshold=inlier_threshold
        )
        return t, pos, inliers, True
    except ValueError:
        return t, None, None, False


def tdoa_batch(receiver_positions, relative_toas_batch,
               speed_of_signal=3e8,
               fixed_dim=2, fixed_value=1.5,
               inlier_threshold=2.0,
               max_workers=4):
    """
    Process multiple timesteps in parallel.
    """
    relative_toas_batch = np.array(relative_toas_batch)
    T = relative_toas_batch.shape[0]
    N, D = receiver_positions.shape
    
    positions = np.full((T, D), np.nan)
    inliers_batch = np.zeros((T, N), dtype=bool)
    success = np.zeros(T, dtype=bool)
    
    # Prepare arguments for each timestep
    args_list = [
        (t, receiver_positions, relative_toas_batch[t], 
         speed_of_signal, fixed_dim, fixed_value, inlier_threshold)
        for t in range(T)
    ]
    
    # Parallelize over timesteps
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_process_single_timestep, args): args[0] 
                   for args in args_list}
        
        total_completed = 0
        successfully_completed = 0
        
        with tqdm(total=T, desc="Processing") as pbar:
            for future in as_completed(futures):
                t, pos, inliers, succ = future.result()
                total_completed += 1
                
                if succ:
                    positions[t] = pos
                    inliers_batch[t] = inliers
                    success[t] = True
                    successfully_completed += 1
                
                success_rate = (successfully_completed / total_completed) * 100
                pbar.set_description(f"Processing (Success: {success_rate:.1f}%)")
                pbar.update(1)
    
    return positions, inliers_batch, success