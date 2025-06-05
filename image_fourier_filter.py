
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, ifft
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import time
from functools import partial
import gc

def create_filter_response(nt, dt, target_period_sec=300.0,
    filter_width_sec=60.0, filter_type='notch', smooth=True):
    """
    Create the filter response for 5-minute filtering
    
    Parameters:
    -----------
    nt : int
        Number of time points
    dt : float
        Time step in seconds
    target_period_sec : float
        Target period in seconds, default is 5 minutes
    filter_width_sec : float
        Filter width in seconds
    filter_type : str
        'notch' for notch filter, 'highpass' for high-pass, 'lowpass' for low-pass
    smooth : bool
        Use smooth Gaussian filter to reduce ringing
    
    Returns:
    --------
    filter_response : array
        Filter frequency response
    frequencies : array
        Frequency array in mHz
    """
    
    # Create frequency array
    frequencies = fftfreq(nt, dt)  # Hz
    freq_mHz = frequencies * 1000  # Convert to mHz
    
    # 5-minute period = 300 seconds = 1/300 Hz ≈ 3.33 mHz
    target_freq_mHz = 1000.0 / target_period_sec
    filter_width_mHz = 1000.0 * filter_width_sec/target_period_sec**2
    # filter_width_mHz = 1000.0 / filter_width_sec
    
    if smooth:
        # Smooth Gaussian filter
        filter_response = np.ones(nt, dtype=complex)
        if filter_type == 'notch':
            # Gaussian notch around 5-minute frequency
            gaussian_notch = np.exp(-0.5 * ((np.abs(freq_mHz) - target_freq_mHz) / (filter_width_mHz/4))**2)
            # filter_response *= (1 - gaussian_notch)
            filter_response *= gaussian_notch
        elif filter_type == 'highpass':
            # Smooth high-pass using error function
            from scipy.special import erf
            cutoff_freq = target_freq_mHz - filter_width_mHz/2
            filter_response = 0.5 * (1 + erf((np.abs(freq_mHz) - cutoff_freq) / (filter_width_mHz/4)))
        elif filter_type == 'lowpass':
            # Smooth low-pass using error function  
            from scipy.special import erf
            cutoff_freq = target_freq_mHz + filter_width_mHz/2
            filter_response = 0.5 * (1 - erf((np.abs(freq_mHz) - cutoff_freq) / (filter_width_mHz/4)))
    else:
        # Sharp filter
        filter_response = np.ones(nt, dtype=complex)
        
        if filter_type == 'notch':
            # Find indices near 5-minute frequency
            freq_mask = np.abs(np.abs(freq_mHz) - target_freq_mHz) <= filter_width_mHz/2
            filter_response[~freq_mask] = 0.0
        elif filter_type == 'highpass':
            high_freq_cutoff = target_freq_mHz - filter_width_mHz/2
            filter_response[np.abs(freq_mHz) <= high_freq_cutoff] = 0.0
        elif filter_type == 'lowpass':
            low_freq_cutoff = target_freq_mHz + filter_width_mHz/2
            filter_response[np.abs(freq_mHz) >= low_freq_cutoff] = 0.0
    
    return filter_response, freq_mHz

def filter_pixel_chunk(args):
    """
    Filter a chunk of pixels - designed for multiprocessing
    
    Parameters:
    -----------
    args : tuple
        (pixel_data_chunk, filter_response, chunk_indices)
        
    Returns:
    --------
    tuple : (filtered_chunk, chunk_indices)
    """
    pixel_data_chunk, filter_response, chunk_indices = args
    
    ny_chunk, nx_chunk, nt = pixel_data_chunk.shape
    filtered_chunk = np.zeros_like(pixel_data_chunk)
    
    for j in range(ny_chunk):
        for i in range(nx_chunk):
            pixel_timeseries = pixel_data_chunk[j, i, :]
            
            # Skip if all NaN
            if np.all(np.isnan(pixel_timeseries)):
                filtered_chunk[j, i, :] = pixel_timeseries
                continue
                
            # Check for minimum valid points
            valid_mask = ~np.isnan(pixel_timeseries)
            if np.sum(valid_mask) < 10:  # Need minimum points
                filtered_chunk[j, i, :] = pixel_timeseries
                continue
            
            # Handle NaN values by interpolation or zero-padding
            if np.any(~valid_mask):
                # Simple approach: replace NaN with mean
                pixel_clean = pixel_timeseries.copy()
                pixel_clean[~valid_mask] = np.nanmean(pixel_timeseries)
            else:
                pixel_clean = pixel_timeseries
            
            # Apply filter in frequency domain
            pixel_fft = fft(pixel_clean)
            filtered_fft = pixel_fft * filter_response
            filtered_pixel = np.real(ifft(filtered_fft))
            
            # Restore NaN values if they existed
            if np.any(~valid_mask):
                filtered_pixel[~valid_mask] = np.nan
                
            filtered_chunk[j, i, :] = filtered_pixel
    
    return filtered_chunk, chunk_indices

def filter_solar_timeseries_parallel(data_cube, cadence_seconds=12, target_period_sec=300.0,
                                   filter_width_sec=60.0, filter_type='notch',
                                   smooth=True, n_workers=None, chunk_size=64):
    """
    Filter a 3D solar data cube (ny, nx, nt) to remove 5-minute oscillations using parallel processing
    
    Parameters:
    -----------
    data_cube : array (ny, nx, nt)
        3D data cube with time as last dimension
    cadence_seconds : float
        Time cadence in seconds
    target_period_sec : float
        Target period in seconds, default is 300 seconds
    filter_width_sec : float
        Filter width in seconds, default is 60 seconds
    filter_type : str
        Type of filter ('notch', 'highpass', 'lowpass')
    smooth : bool
        Use smooth filter to reduce ringing
    n_workers : int or None
        Number of worker processes (None = auto-detect)
    chunk_size : int
        Size of spatial chunks for parallel processing
        
    Returns:
    --------
    filtered_cube : array
        Filtered data cube
    """
    
    ny, nx, nt = data_cube.shape
    
    if n_workers is None:
        n_workers = min(mp.cpu_count(), 8)  # Cap at 8 to avoid memory issues
    
    print(f"Filtering {ny}x{nx} pixels with {nt} time steps using {n_workers} workers...")
    print(f"Using chunk size: {chunk_size}x{chunk_size}")
    
    # Create filter response once
    filter_response, frequencies = create_filter_response(
        nt, cadence_seconds, target_period_sec, filter_width_sec, filter_type, smooth
    )
    
    # Create chunks for parallel processing
    chunks = []
    chunk_indices = []
    
    for j_start in range(0, ny, chunk_size):
        for i_start in range(0, nx, chunk_size):
            j_end = min(j_start + chunk_size, ny)
            i_end = min(i_start + chunk_size, nx)
            
            chunk_data = data_cube[j_start:j_end, i_start:i_end, :]
            chunk_idx = (j_start, j_end, i_start, i_end)
            
            chunks.append((chunk_data, filter_response, chunk_idx))
            chunk_indices.append(chunk_idx)
    
    print(f"Created {len(chunks)} chunks for processing")
    
    # Initialize output array
    filtered_cube = np.zeros_like(data_cube)
    
    # Process chunks in parallel
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Submit all jobs
        future_to_chunk = {
            executor.submit(filter_pixel_chunk, chunk): i 
            for i, chunk in enumerate(chunks)
        }
        
        # Collect results as they complete
        completed = 0
        for future in as_completed(future_to_chunk):
            try:
                filtered_chunk, (j_start, j_end, i_start, i_end) = future.result()
                
                # Place result in output array
                filtered_cube[j_start:j_end, i_start:i_end, :] = filtered_chunk
                
                completed += 1
                if completed % max(1, len(chunks)//10) == 0:
                    elapsed = time.time() - start_time
                    progress = completed / len(chunks)
                    eta = elapsed / progress - elapsed if progress > 0 else 0
                    print(f"Progress: {completed}/{len(chunks)} chunks ({progress*100:.1f}%) "
                          f"- Elapsed: {elapsed:.1f}s, ETA: {eta:.1f}s")
                    
            except Exception as e:
                chunk_idx = future_to_chunk[future]
                print(f"Error processing chunk {chunk_idx}: {e}")
                # Continue with other chunks
    
    total_time = time.time() - start_time
    print(f"Filtering completed in {total_time:.2f} seconds")
    
    return filtered_cube



