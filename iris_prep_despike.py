import numpy as np
from scipy.ndimage import convolve
import time


def iris_prep_despike(data, sigmas=4.5, n_iter=10, kernel=None, min_std=1.0, 
                     silent=False, return_goodmap=False, mode='bright'):
    """
    Generalized data despiking tool. Removes spikes/outliers from data arrays of 1 to 4 dimensions.
    
    Parameters
    ----------
    data : np.ndarray
        Input data array with 1-4 dimensions
    sigmas : float, optional
        Threshold for designating a bad pixel, as multiple of neighborhood std dev. Default: 4.5
    n_iter : int, optional
        Maximum number of iterations for identifying bad pixels. Default: 10
    kernel : np.ndarray, optional
        Convolution kernel for calculating neighborhood statistics. Default kernels are 
        automatically created based on data dimensions.
    min_std : float, optional
        Minimum value for local standard deviation to prevent excessive detection in flat regions. 
        Default: 1.0
    silent : bool, optional
        If True, suppress verbose output. Default: False
    return_goodmap : bool, optional
        If True, return tuple of (result, goodmap). Default: False
    mode : str, optional
        Detection mode: 'bright', 'dark', or 'both'. Default: 'bright'
    
    Returns
    -------
    result : np.ndarray
        Processed data with spikes removed
    goodmap : np.ndarray (optional)
        Map of good pixels (1.0 for good, 0.0 for bad) if return_goodmap=True
    """
    
    if not silent:
        print(f"{time.ctime()} IRIS_PREP_DESPIKE started on array of {data.size} elements.")
        print("Step (1): Iteratively identifying bad pixels.")
    
    # Make a copy to avoid modifying input
    data = data.copy().astype(np.float64)
    
    # Deal with NaN and Inf
    bad_mask = ~np.isfinite(data)
    if np.any(bad_mask):
        bad_values = data[bad_mask].copy()
        data[bad_mask] = 1e-6  # Small number << 1 DN
    
    # Get data dimensions
    ndim = data.ndim
    
    # Create default kernels if not provided
    if kernel is None:
        if ndim == 1:
            kernel = np.ones(11)
        elif ndim == 2:
            kernel = np.ones((9, 9))
        elif ndim == 3:
            kernel = np.ones((5, 5, 5))
        elif ndim == 4:
            kernel = np.ones((3, 3, 3, 3))
        else:
            raise ValueError(f"Data dimensionality {ndim} is too great. Cannot construct kernel.")
    
    t_begin = time.time()
    
    # Initialize good pixel map
    goodmap = np.ones_like(data, dtype=np.float64)
    
    # Iteratively identify bad pixels
    for i in range(1, n_iter + 1):
        # Calculate neighborhood mean using only good pixels
        numerator = convolve(goodmap * data, kernel, mode='nearest')
        denominator = convolve(goodmap, kernel, mode='nearest')
        # Avoid division by zero
        denominator = np.where(denominator > 0, denominator, 1.0)
        neighborhood_mean = numerator / denominator
        
        # Calculate deviation based on mode
        if mode == 'bright':
            deviation = data - neighborhood_mean
        elif mode == 'dark':
            deviation = neighborhood_mean - data
        elif mode == 'both':
            deviation = np.abs(data - neighborhood_mean)
        else:
            raise ValueError(f"Undefined mode: {mode}")
        
        # Calculate neighborhood standard deviation
        numerator = convolve(goodmap * deviation**2, kernel, mode='nearest')
        denominator = convolve(goodmap, kernel, mode='nearest')
        denominator = np.where(denominator > 0, denominator, 1.0)
        neighborhood_std = np.sqrt(numerator / denominator)
        neighborhood_std = np.maximum(neighborhood_std, min_std)
        
        # Find bad pixels
        bad_pixels = deviation > (sigmas * neighborhood_std)
        
        # Check for newly bad pixels
        newly_bad = bad_pixels & (goodmap > 0)
        n_newly_bad = np.sum(newly_bad)
        
        if n_newly_bad == 0:
            break
        
        if not silent:
            print(f"Iteration {i:4d} found {np.sum(bad_pixels):12d} bad pixels, "
                  f"{n_newly_bad:12d} of them new.")
        
        # Update goodmap
        goodmap[bad_pixels] = 0.0
    
    if not silent:
        print("Step (2): Replacing bad pixels")
    
    # Construct weighted kernel for replacement (exp(-r)/(1+r^ndim))
    nk2 = 5  # Size of very-near-local smoothing kernel
    middle = (nk2 - 1) // 2
    
    if ndim == 1:
        k2 = np.zeros(nk2)
        for i in range(nk2):
            x = i - middle
            k2[i] = np.exp(-abs(x))
    elif ndim == 2:
        k2 = np.zeros((nk2, nk2))
        for i in range(nk2):
            for j in range(nk2):
                x, y = i - middle, j - middle
                r = np.sqrt(x**2 + y**2)
                if r == 0:
                    k2[i, j] = 1.0
                else:
                    k2[i, j] = np.exp(-r) / (1 + r**ndim)
    elif ndim == 3:
        k2 = np.zeros((nk2, nk2, nk2))
        for i in range(nk2):
            for j in range(nk2):
                for k in range(nk2):
                    x, y, z = i - middle, j - middle, k - middle
                    r = np.sqrt(x**2 + y**2 + z**2)
                    if r == 0:
                        k2[i, j, k] = 1.0
                    else:
                        k2[i, j, k] = np.exp(-r) / (1 + r**ndim)
    elif ndim == 4:
        k2 = np.zeros((nk2, nk2, nk2, nk2))
        for i in range(nk2):
            for j in range(nk2):
                for k in range(nk2):
                    for m in range(nk2):
                        x, y, z, t = i - middle, j - middle, k - middle, m - middle
                        r = np.sqrt(x**2 + y**2 + z**2 + t**2)
                        if r == 0:
                            k2[i, j, k, m] = 1.0
                        else:
                            k2[i, j, k, m] = np.exp(-r) / (1 + r**ndim)
    
    # Calculate replacement values
    numerator = convolve(goodmap * data, k2, mode='nearest')
    denominator = convolve(goodmap, k2, mode='nearest')
    denominator = np.where(denominator > 0, denominator, 1.0)
    replacement_values = numerator / denominator
    
    # Replace bad pixels
    result = data.copy()
    bad_pixel_mask = goodmap == 0
    result[bad_pixel_mask] = replacement_values[bad_pixel_mask]
    
    # Restore original NaN/Inf values
    if np.any(bad_mask):
        result[bad_mask] = bad_values
        data[bad_mask] = bad_values  # Also restore in original for consistency
    
    t_end = time.time()
    if not silent:
        print(f"{time.ctime()} IRIS_PREP_DESPIKE finished, {t_end - t_begin:.2f} sec elapsed.")
    
    if return_goodmap:
        return result, goodmap
    else:
        return result


# Example usage and test
if __name__ == "__main__":
    # Create test data with some spikes
    np.random.seed(42)
    
    # 2D example (like an image)
    test_data = np.random.randn(100, 100) * 10 + 100
    
    # Add some artificial spikes
    spike_positions = np.random.choice(10000, 50, replace=False)
    test_data.flat[spike_positions] = 1000  # Bright spikes
    
    # Add some dark spikes
    dark_spike_positions = np.random.choice(10000, 30, replace=False)
    test_data.flat[dark_spike_positions] = -500  # Dark spikes
    
    # Add some NaN values
    nan_positions = np.random.choice(10000, 10, replace=False)
    test_data.flat[nan_positions] = np.nan
    
    print("\n=== Testing 2D despike ===")
    print(f"Original data: min={np.nanmin(test_data):.2f}, max={np.nanmax(test_data):.2f}, "
          f"mean={np.nanmean(test_data):.2f}, std={np.nanstd(test_data):.2f}")
    
    # Despike for bright spikes
    despiked_bright = iris_prep_despike(test_data, mode='bright', silent=False)
    print(f"After bright despike: min={np.nanmin(despiked_bright):.2f}, "
          f"max={np.nanmax(despiked_bright):.2f}, mean={np.nanmean(despiked_bright):.2f}, "
          f"std={np.nanstd(despiked_bright):.2f}")
    
    # Despike for both bright and dark spikes
    despiked_both, goodmap = iris_prep_despike(test_data, mode='both', 
                                               return_goodmap=True, silent=True)
    n_bad = np.sum(goodmap == 0)
    print(f"After both despike: min={np.nanmin(despiked_both):.2f}, "
          f"max={np.nanmax(despiked_both):.2f}, mean={np.nanmean(despiked_both):.2f}, "
          f"std={np.nanstd(despiked_both):.2f}")
    print(f"Total bad pixels identified: {n_bad}")
    
    # Test with 1D data
    print("\n=== Testing 1D despike ===")
    test_1d = np.random.randn(1000) * 5 + 50
    test_1d[100] = 500  # Add spike
    test_1d[500] = -300  # Add dark spike
    
    despiked_1d = iris_prep_despike(test_1d, mode='both', silent=True)
    print(f"1D test successful: shape={despiked_1d.shape}")