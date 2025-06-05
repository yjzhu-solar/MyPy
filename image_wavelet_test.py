import warnings
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2
from copy import deepcopy
import multiprocessing as mp
from astropy.visualization import AsinhStretch, ImageNormalize


class WPSImage(object):
    """Fixed and improved WPSImage class for wavelet power spectrum analysis"""

    def __init__(self, scales, image_seq, time, mother="MORLET", pad=1, lag1=0.0, siglvl=0.95):
        # Input validation
        self.scales = np.asarray(scales, dtype=float)
        self.time = np.asarray(time, dtype=float)
        self.image_seq = np.asarray(image_seq, dtype=float)
        
        # Validate dimensions
        if len(self.image_seq.shape) != 3:
            raise ValueError("image_seq must be 3D array (ny, nx, nt)")
        
        self.ny, self.nx, self.n_times = self.image_seq.shape
        
        if len(self.time) != self.n_times:
            raise ValueError(f"Time array length ({len(self.time)}) must match image sequence time dimension ({self.n_times})")
        
        # Calculate time step with improved robustness
        if len(self.time) > 1:
            time_diffs = np.diff(self.time)
            self.dt = np.median(time_diffs)  # More robust than taking first difference
            
            # Warn if time sampling is irregular
            if np.std(time_diffs) / self.dt > 0.01:  # 1% tolerance
                warnings.warn("Time sampling appears irregular. Consider interpolating to uniform grid.")
        else:
            raise ValueError("Need at least 2 time points")
        
        self.mother = mother.upper()
        self.pad = int(pad)
        self.lag1 = float(lag1)
        self.siglvl = float(siglvl)
        self.n_scales = len(self.scales)

        # Pre-calculate significance levels
        self._calculate_significance_levels()

    def _calculate_significance_levels(self):
        """Pre-calculate significance levels to avoid repeated computation"""
        try:
            self.signif, _ = wave_signif(1.0, self.dt, self.scales, 0, self.lag1, 
                                       self.siglvl, -1, self.mother)
            self.ws_sigmap = np.outer(self.signif, np.ones(self.n_times))
            
            self.global_signif, _ = wave_signif(1.0, self.dt, self.scales, 1, self.lag1, 
                                              self.siglvl, self.n_times - self.scales, self.mother)
            # Fixed division - ensure no division by zero
            self.global_signif_unbias = np.divide(self.global_signif, self.scales, 
                                                 out=np.zeros_like(self.global_signif), 
                                                 where=self.scales!=0)
        except Exception as e:
            warnings.warn(f"Could not calculate significance levels: {e}")
            self.signif = np.ones_like(self.scales)
            self.ws_sigmap = np.ones((self.n_scales, self.n_times))
            self.global_signif = np.ones_like(self.scales)
            self.global_signif_unbias = np.ones_like(self.scales)
    
    def _first_attempt(self):
        """Initialize with first pixel to get COI and period - FIXED VERSION"""
        try:
            # Use the actual signal to get correct dimensions
            test_signal = self.image_seq[0, 0, :]
            
            # Get the wavelet transform to determine actual output dimensions
            cwtmatr, period, coi = wavelet(test_signal, self.dt, self.scales, self.pad, self.mother)
            
            # Store the period and actual signal length
            self.period = period
            self.coi = coi
            
            # Calculate COI mask with correct dimensions
            # Use the actual length of the output (which may be padded)
            actual_length = len(coi)
            
            # Create COI mask based on actual output dimensions
            period_mesh = np.tile(self.period[:, np.newaxis], (1, actual_length))
            coi_mesh = np.tile(coi[np.newaxis, :], (len(self.period), 1))
            self.coi_mask = period_mesh > coi_mesh
            
            # Store the actual length for later use
            self.actual_length = actual_length
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize wavelet analysis: {e}")
    
    def _iterate_column(self, index_column):
        """Process a single column with improved error handling"""
        if index_column >= self.nx:
            raise IndexError(f"Column index {index_column} out of range [0, {self.nx})")
        
        results = []
        
        for ii in range(self.ny):
            try:
                signal = self.image_seq[ii, index_column, :]
                
                # Check for valid signal
                if np.all(np.isnan(signal)) or np.all(signal == 0):
                    # Handle degenerate case
                    global_ws_unbias = np.full(self.n_scales, np.nan)
                    global_ws_unbias_coi = np.full(self.n_scales, np.nan)
                    global_ws_unbias_coi_sig = np.full(self.n_scales, np.nan)
                else:
                    global_ws_unbias, global_ws_unbias_coi, global_ws_unbias_coi_sig = self._wavelet(
                        self.dt, signal, self.scales, self.mother, self.pad, 
                        self.ws_sigmap, self.n_times, coi_mask=self.coi_mask
                    )
                
                results.append((global_ws_unbias, global_ws_unbias_coi, global_ws_unbias_coi_sig))
                
            except Exception as e:
                warnings.warn(f"Error processing pixel ({ii}, {index_column}): {e}")
                # Fill with NaN for failed pixels
                nan_result = np.full(self.n_scales, np.nan)
                results.append((nan_result, nan_result, nan_result))
        
        # Stack results
        if results:
            global_ws_unbias_column = np.vstack([r[0] for r in results])
            global_ws_unbias_coi_column = np.vstack([r[1] for r in results])
            global_ws_unbias_coi_sig_column = np.vstack([r[2] for r in results])
        else:
            # Fallback for empty results
            nan_array = np.full((self.ny, self.n_scales), np.nan)
            global_ws_unbias_column = nan_array
            global_ws_unbias_coi_column = nan_array
            global_ws_unbias_coi_sig_column = nan_array

        return global_ws_unbias_column, global_ws_unbias_coi_column, global_ws_unbias_coi_sig_column
        
    def _wps_image(self, ncpu="max"):
        """Main processing function with improved parallelization"""
        
        # Determine number of CPUs
        if str(ncpu).lower() == "max" or ncpu is None:
            ncpu = mp.cpu_count()
        else:
            ncpu = max(1, int(ncpu))  # Ensure at least 1 CPU

        print(f"Processing with {ncpu} CPU(s)...")
        
        # Initialize
        self._first_attempt()
        
        if ncpu == 1:
            # Serial processing
            all_results = []
            for ii in range(self.nx):
                try:
                    result = self._iterate_column(ii)
                    all_results.append(result)
                    if (ii + 1) % 10 == 0:  # Progress indicator
                        print(f"Processed {ii + 1}/{self.nx} columns")
                except Exception as e:
                    warnings.warn(f"Failed to process column {ii}: {e}")
                    # Add NaN result for failed column
                    nan_array = np.full((self.ny, self.n_scales), np.nan)
                    all_results.append((nan_array, nan_array, nan_array))
            
        else:
            # Parallel processing with error handling
            try:
                with mp.Pool(ncpu) as pool:
                    all_results = pool.map(self._iterate_column, range(self.nx))
            except Exception as e:
                warnings.warn(f"Parallel processing failed: {e}. Falling back to serial processing.")
                return self._wps_image(ncpu=1)
        
        # Stack results into final arrays
        try:
            self.global_ws_unbias = np.stack([r[0] for r in all_results], axis=1)
            self.global_ws_unbias_coi = np.stack([r[1] for r in all_results], axis=1)  
            self.global_ws_unbias_coi_sig = np.stack([r[2] for r in all_results], axis=1)
            
            print(f"Completed processing. Output shape: {self.global_ws_unbias.shape}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to stack results: {e}")

    def plot_single_ws(self, x, y, index=10, logy=True, global_signif=True, 
                      cmap='viridis', figsize=(12, 8)):
        """Improved plotting with better error handling and visualization"""
        
        # Validate inputs
        if not (0 <= x < self.nx and 0 <= y < self.ny):
            raise ValueError(f"Coordinates ({x}, {y}) out of bounds. Image shape: ({self.ny}, {self.nx})")
        
        if not (0 <= index < self.n_times):
            raise ValueError(f"Time index {index} out of bounds. Time series length: {self.n_times}")
        
        signal = self.image_seq[y, x, :]
        
        # Check for valid signal
        if np.all(np.isnan(signal)):
            raise ValueError(f"Signal at ({x}, {y}) contains only NaN values")
        
        try:
            result = self._wavelet(
                self.dt, signal, self.scales, self.mother, self.pad, 
                self.ws_sigmap, self.n_times, coi_mask=self.coi_mask, return_all=True
            )
            
            (global_ws_unbias, global_ws_unbias_coi, global_ws_unbias_coi_sig, 
             power_unbias, power_unbias_coi, power_unbias_coi_sig, ws_sig_level, coi) = result
             
        except Exception as e:
            raise RuntimeError(f"Wavelet analysis failed for pixel ({x}, {y}): {e}")
        
        # Create improved figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize, layout="constrained")

        # Wavelet power spectrum
        try:
            # Use log scale for better visualization
            power_plot = np.log10(np.maximum(power_unbias, np.finfo(float).eps))
            
            # Ensure time array matches the power array dimensions
            time_for_plot = self.time
            if power_plot.shape[1] != len(self.time):
                # If there's padding, truncate or extend time array as needed
                if power_plot.shape[1] < len(self.time):
                    time_for_plot = self.time[:power_plot.shape[1]]
                else:
                    # If padded, extend time array
                    extra_points = power_plot.shape[1] - len(self.time)
                    dt = self.time[1] - self.time[0] if len(self.time) > 1 else 1.0
                    extra_time = self.time[-1] + dt * np.arange(1, extra_points + 1)
                    time_for_plot = np.concatenate([self.time, extra_time])
            
            im1 = ax1.pcolormesh(time_for_plot, self.period / 60., power_plot, cmap=cmap, shading='auto')
            ax1.set_title(f'Wavelet Power Spectrum (x={x}, y={y})')
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Period (min)')
            
            if logy:
                ax1.set_yscale("log")
            
            # Add significance contour
            sig_levels = np.where(ws_sig_level >= 1, 1, 0)
            ax1.contour(time_for_plot, self.period/60., sig_levels, levels=[0.5], 
                       colors=['red'], linewidths=2, alpha=0.8)
            
            # Add colorbar
            plt.colorbar(im1, ax=ax1, label='Log10(Power)')
            
        except Exception as e:
            ax1.text(0.5, 0.5, f'Plot error: {str(e)[:50]}...', 
                    transform=ax1.transAxes, ha='center', va='center')

        # Global wavelet spectrum
        try:
            ax2.loglog(global_ws_unbias_coi, self.period/60., 'b-', linewidth=2, 
                      label='Global WS (COI masked)')
            
            if global_signif and hasattr(self, 'global_signif_unbias'):
                ax2.loglog(self.global_signif_unbias, self.period/60., 'r--', 
                          linewidth=2, label='95% Significance')
            
            ax2.set_title('Global Wavelet Spectrum')
            ax2.set_xlabel('Power')
            ax2.set_ylabel('Period (min)')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            ax2.sharey(ax1)
            
        except Exception as e:
            ax2.text(0.5, 0.5, f'Plot error: {str(e)[:50]}...', 
                    transform=ax2.transAxes, ha='center', va='center')

        # Image at selected time
        try:
            img_data = self.image_seq[:, :, index]
            
            # Handle potential NaN or infinite values
            if np.all(np.isnan(img_data)):
                img_data = np.zeros_like(img_data)
            else:
                # Robust normalization
                vmin, vmax = np.nanpercentile(img_data, [1, 99])
                img_data = np.clip(img_data, vmin, vmax)
            
            im3 = ax3.imshow(img_data, origin="lower", 
                           norm=ImageNormalize(stretch=AsinhStretch(0.2)),
                           cmap='inferno')
            ax3.scatter(x, y, color="white", s=50, marker="x", linewidths=3)
            ax3.set_title(f'Image at t={self.time[index]:.2f}s (index={index})')
            ax3.set_xlabel('X pixel')
            ax3.set_ylabel('Y pixel')
            plt.colorbar(im3, ax=ax3)
            
        except Exception as e:
            ax3.text(0.5, 0.5, f'Plot error: {str(e)[:50]}...', 
                    transform=ax3.transAxes, ha='center', va='center')

        # Coefficient of variation map
        try:
            mean_img = np.nanmean(self.image_seq, axis=2)
            std_img = np.nanstd(self.image_seq, axis=2)
            
            # Avoid division by zero
            cv_img = np.divide(std_img, mean_img, out=np.zeros_like(std_img), 
                              where=mean_img!=0)
            
            im4 = ax4.imshow(cv_img, origin="lower", cmap='cividis')
            ax4.scatter(x, y, color="white", s=50, marker="x", linewidths=3)
            ax4.set_title('Coefficient of Variation')
            ax4.set_xlabel('X pixel')
            ax4.set_ylabel('Y pixel')
            plt.colorbar(im4, ax=ax4)
            
        except Exception as e:
            ax4.text(0.5, 0.5, f'Plot error: {str(e)[:50]}...', 
                    transform=ax4.transAxes, ha='center', va='center')

        # Add cone of influence to wavelet spectrum
        try:
            ax1_ylim = ax1.get_ylim()
            coi_minutes = coi / 60.
            
            # Ensure COI length matches time array for plot
            if len(coi_minutes) != len(time_for_plot):
                # Truncate or pad COI to match time array
                if len(coi_minutes) > len(time_for_plot):
                    coi_minutes = coi_minutes[:len(time_for_plot)]
                else:
                    # Extend COI with last value
                    extra_points = len(time_for_plot) - len(coi_minutes)
                    coi_minutes = np.concatenate([coi_minutes, np.full(extra_points, coi_minutes[-1])])
            
            # Ensure COI doesn't exceed plot bounds
            coi_clipped = np.clip(coi_minutes, ax1_ylim[0], ax1_ylim[1])
            
            ax1.fill_between(time_for_plot, 
                           np.full_like(time_for_plot, ax1_ylim[1]), 
                           coi_clipped, 
                           facecolor="none", edgecolor="#00000040", 
                           hatch='///', alpha=0.5, label='COI')
            ax1.plot(time_for_plot, coi_clipped, 'k-', linewidth=1, alpha=0.8)
            ax1.set_ylim(ax1_ylim)
            
        except Exception as e:
            warnings.warn(f"Could not plot COI: {e}")

        plt.tight_layout()
        return fig

    @staticmethod
    def _wavelet(dt, signal, scale, mother, pad, ws_sigmap, n_times, coi_mask=None, return_all=False):
        """Improved wavelet analysis with better error handling and consistent dimensions"""
        
        # Input validation
        signal = np.asarray(signal, dtype=float)
        if len(signal) == 0:
            raise ValueError("Signal cannot be empty")
        
        # Handle NaN values
        nan_mask = np.isnan(signal)
        if np.all(nan_mask):
            raise ValueError("Signal contains only NaN values")
        
        if np.any(nan_mask):
            warnings.warn("Signal contains NaN values. Consider interpolation or gap filling.")
        
        # Robust normalization
        signal_clean = signal.copy()
        
        # Calculate statistics ignoring NaN
        variance = np.nanvar(signal_clean)
        mean = np.nanmean(signal_clean)
        
        if variance == 0:
            warnings.warn("Signal has zero variance. Results may be unreliable.")
            variance = 1.0  # Avoid division by zero
        
        # Normalize signal
        signal_clean = (signal_clean - mean) / np.sqrt(variance)
        
        # Replace NaN with zero (will be masked later)
        signal_clean[nan_mask] = 0
        
        try:
            cwtmatr, period, coi = wavelet(signal_clean, dt, scale, pad, mother)
            power = np.abs(cwtmatr) ** 2
            
        except Exception as e:
            raise RuntimeError(f"Wavelet transform failed: {e}")

        if coi_mask is None:
            # Initialize COI mask - this is the FIXED part
            try:
                # Use actual dimensions from the wavelet output
                actual_time_length = power.shape[1]
                
                period_mesh = np.tile(period[:, np.newaxis], (1, actual_time_length))
                coi_mesh = np.tile(coi[np.newaxis, :], (len(period), 1))
                coi_mask = period_mesh > coi_mesh
                
                return period, coi, coi_mask
                
            except Exception as e:
                raise RuntimeError(f"COI mask calculation failed: {e}")
        
        else:
            try:
                # Ensure ws_sigmap dimensions match power dimensions
                if ws_sigmap.shape[1] != power.shape[1]:
                    # Adjust ws_sigmap to match power dimensions
                    if ws_sigmap.shape[1] > power.shape[1]:
                        # Truncate ws_sigmap
                        ws_sigmap_adjusted = ws_sigmap[:, :power.shape[1]]
                    else:
                        # Extend ws_sigmap by repeating last column
                        extra_cols = power.shape[1] - ws_sigmap.shape[1]
                        last_col = ws_sigmap[:, -1:] if ws_sigmap.shape[1] > 0 else np.ones((ws_sigmap.shape[0], 1))
                        ws_sigmap_adjusted = np.concatenate([ws_sigmap, np.tile(last_col, (1, extra_cols))], axis=1)
                else:
                    ws_sigmap_adjusted = ws_sigmap
                
                # Ensure coi_mask dimensions match power dimensions
                if coi_mask.shape[1] != power.shape[1]:
                    # Adjust coi_mask to match power dimensions
                    if coi_mask.shape[1] > power.shape[1]:
                        # Truncate coi_mask
                        coi_mask_adjusted = coi_mask[:, :power.shape[1]]
                    else:
                        # Extend coi_mask by repeating last column
                        extra_cols = power.shape[1] - coi_mask.shape[1]
                        last_col = coi_mask[:, -1:] if coi_mask.shape[1] > 0 else np.ones((coi_mask.shape[0], 1), dtype=bool)
                        coi_mask_adjusted = np.concatenate([coi_mask, np.tile(last_col, (1, extra_cols))], axis=1)
                else:
                    coi_mask_adjusted = coi_mask
                
                # Significance testing
                ws_sig_level = np.divide(power, ws_sigmap_adjusted, 
                                       out=np.full_like(power, np.inf), 
                                       where=ws_sigmap_adjusted!=0)
                ws_sig_mask = ws_sig_level < 1

                # Bias correction
                power_unbias = np.divide(power, scale[:, np.newaxis], 
                                       out=np.zeros_like(power), 
                                       where=scale[:, np.newaxis]!=0)
                
                # Apply masks
                power_unbias_coi = power_unbias.copy()
                power_unbias_coi[coi_mask_adjusted] = np.nan
                
                power_unbias_coi_sig = power_unbias_coi.copy()
                power_unbias_coi_sig[ws_sig_mask] = np.nan

                # Global statistics
                global_ws_unbias = np.nanmean(power_unbias, axis=1)
                global_ws_unbias_coi = np.nanmean(power_unbias_coi, axis=1)
                global_ws_unbias_coi_sig = np.nanmean(power_unbias_coi_sig, axis=1)
                
                if return_all:
                    return (global_ws_unbias, global_ws_unbias_coi, global_ws_unbias_coi_sig, 
                           power_unbias, power_unbias_coi, power_unbias_coi_sig, ws_sig_level, coi)
                else:
                    return global_ws_unbias, global_ws_unbias_coi, global_ws_unbias_coi_sig
                    
            except Exception as e:
                raise RuntimeError(f"Wavelet analysis computation failed: {e}")
        

def calculate_wps(dt, signal, scale, mother="MORLET", pad=1, lag1=0., siglvl=0.95):
    """Improved WPS calculation with better error handling"""
    
    # Input validation
    signal = np.asarray(signal, dtype=float)
    scale = np.asarray(scale, dtype=float)
    
    if len(signal) == 0:
        raise ValueError("Signal cannot be empty")
    
    if len(scale) == 0:
        raise ValueError("Scale array cannot be empty")
    
    if np.any(scale <= 0):
        raise ValueError("All scales must be positive")
    
    n = len(signal)
    
    # Handle NaN values
    nan_mask = np.isnan(signal)
    if np.all(nan_mask):
        raise ValueError("Signal contains only NaN values")
    
    # Robust normalization
    variance = np.nanvar(signal)
    mean = np.nanmean(signal)
    
    if variance == 0:
        warnings.warn("Signal has zero variance")
        variance = 1.0
    
    signal_norm = (signal - mean) / np.sqrt(variance)
    signal_norm[nan_mask] = 0  # Replace NaN with zero

    try:
        cwtmatr, period, coi = wavelet(signal_norm, dt, scale, pad, mother)
        power = np.abs(cwtmatr) ** 2
        
    except Exception as e:
        raise RuntimeError(f"Wavelet transform failed: {e}")

    try:
        # Significance testing
        signif, _ = wave_signif(1.0, dt, scale, 0, lag1, siglvl, -1, mother)
        sig95 = np.outer(signif, np.ones(n))
        sig95 = np.divide(power, sig95, out=np.full_like(power, np.inf), where=sig95!=0)

        # Global wavelet spectrum
        global_ws = np.nanmean(power, axis=1)
        
        # Degrees of freedom correction
        dof = np.maximum(n - scale, 1)  # Ensure positive DOF
        global_signif, _ = wave_signif(1.0, dt, scale, 1, lag1, siglvl, dof, mother)

        # Bias correction
        power_unbias = np.divide(power, scale[:, np.newaxis], 
                               out=np.zeros_like(power), where=scale[:, np.newaxis]!=0)
        global_ws_unbias = np.divide(global_ws, scale, 
                                   out=np.zeros_like(global_ws), where=scale!=0)
        global_signif_unbias = np.divide(global_signif, scale, 
                                       out=np.zeros_like(global_signif), where=scale!=0)

        return power_unbias, period, coi, global_ws_unbias, global_signif_unbias, sig95
        
    except Exception as e:
        raise RuntimeError(f"Significance calculation failed: {e}")


def find_max_power(x):
    """Improved parabolic interpolation for peak finding"""
    
    x = np.asarray(x)
    
    if len(x) < 3:
        return np.nanargmax(x)
    
    # Find the maximum and check bounds
    x_argmax = np.nanargmax(x)
    
    if x_argmax == 0 or x_argmax == len(x) - 1:
        return x_argmax  # Peak at boundary, no interpolation possible
    
    # Extract neighborhood
    x_select = x[x_argmax-1:x_argmax+2]
    
    # Check for valid parabolic fit
    if np.any(np.isnan(x_select)) or np.any(np.isinf(x_select)):
        return x_argmax
    
    denominator = 2 * (x_select[0] - 2 * x_select[1] + x_select[2])
    
    if abs(denominator) < np.finfo(float).eps:
        return x_argmax  # Nearly flat, no interpolation
    
    x_argmax_para = -(x_select[2] - x_select[0]) / denominator
    
    # Validate interpolation result
    if abs(x_argmax_para) > 1:
        return x_argmax  # Interpolation failed, return discrete maximum
    
    return x_argmax + x_argmax_para


def wave_bases(mother, k, scale, param):
    """Improved wave_bases function with better error handling"""
    
    mother = mother.upper()
    k = np.asarray(k, dtype=complex)  # Ensure complex for proper arithmetic
    n = len(k)
    
    if n == 0:
        raise ValueError("Frequency array k cannot be empty")
    
    if scale <= 0:
        raise ValueError("Scale must be positive")
    
    # Define Heaviside step function with improved numerical stability
    def ksign(x):
        return (x > 0).astype(float)
    
    # Initialize outputs
    daughter = np.zeros_like(k, dtype=complex)
    
    try:
        if mother == 'MORLET':
            if param == -1: 
                param = 6.
            k0 = param
            
            # Improved numerical stability
            expnt = -(scale * k - k0) ** 2 / 2. * ksign(k.real)
            expnt = np.clip(expnt, -700, 700)  # Prevent overflow
            
            norm = np.sqrt(scale * k[1]) * (np.pi ** (-0.25)) * np.sqrt(n)
            daughter = norm * np.exp(expnt) * ksign(k.real)
            
            fourier_factor = (4. * np.pi) / (k0 + np.sqrt(2. + k0 ** 2))
            coi = fourier_factor / np.sqrt(2)
            dofmin = 2.
            
        elif mother == 'PAUL':
            if param == -1: 
                param = 4.
            m = param
            
            expnt = -(scale * k) * ksign(k.real)
            expnt = np.clip(expnt, -700, 700)  # Prevent overflow
            
            norm = (np.sqrt(scale * k[1]) * (2. ** m / np.sqrt(m * np.prod(np.arange(2, 2 * m)))) 
                   * np.sqrt(n))
            daughter = norm * ((scale * k) ** m) * np.exp(expnt) * ksign(k.real)
            
            fourier_factor = 4 * np.pi / (2. * m + 1.)
            coi = fourier_factor * np.sqrt(2)
            dofmin = 2.
            
        elif mother == 'DOG':
            if param == -1: 
                param = 2.
            m = param
            
            expnt = -(scale * k) ** 2 / 2.0
            expnt = np.clip(expnt, -700, 700)  # Prevent overflow
            
            from scipy.special import gamma
            norm = np.sqrt(scale * k[1] / gamma(m + 0.5)) * np.sqrt(n)
            daughter = -norm * (1j ** m) * ((scale * k) ** m) * np.exp(expnt)
            
            fourier_factor = 2. * np.pi * np.sqrt(2. / (2. * m + 1.))
            coi = fourier_factor / np.sqrt(2)
            dofmin = 1.
            
        else:
            raise ValueError("Mother must be one of MORLET, PAUL, DOG")
            
    except Exception as e:
        raise RuntimeError(f"Wavelet basis calculation failed for {mother}: {e}")

    return daughter, fourier_factor, coi, dofmin


def wavelet(Y, dt, scale, pad=0., mother="MORLET", param=-1):
    """Improved wavelet function with better error handling and validation"""
    
    # Input validation
    Y = np.asarray(Y, dtype=float)
    scale = np.asarray(scale, dtype=float)
    
    if len(Y) == 0:
        raise ValueError("Input signal Y cannot be empty")
    
    if len(scale) == 0:
        raise ValueError("Scale array cannot be empty")
    
    if np.any(scale <= 0):
        raise ValueError("All scales must be positive")
    
    if dt <= 0:
        raise ValueError("Time step dt must be positive")
    
    n1 = len(Y)
    mother = mother.upper()
    J1 = len(scale) - 1
    
    # Construct time series to analyze, pad if necessary
    x = Y - np.nanmean(Y)  # Handle NaN values
    
    if pad == 1:
        base2 = int(np.fix(np.log(n1) / np.log(2) + 0.4999))
        pad_length = int(2 ** (base2 + 1) - n1)
        if pad_length > 0:
            x = np.concatenate((x, np.zeros(pad_length)))
    
    n = len(x)
    
    # Construct wavenumber array
    k = np.arange(1, int(n/2) + 1, dtype=float)
    k = k * (2. * np.pi) / (n * dt)
    k = np.concatenate(([0.], k, -k[-2::-1]))
    
    # Compute FFT
    try:
        f = np.fft.fft(x)
    except Exception as e:
        raise RuntimeError(f"FFT computation failed: {e}")
    
    # Initialize output arrays
    period = scale.copy()
    wave = np.zeros((int(J1) + 1, n), dtype=complex)
    
    # Loop through scales
    for a1 in range(int(J1) + 1):
        try:
            daughter, fourier_factor, coi, dofmin = wave_bases(mother, k, scale[a1], param)
            wave[a1, :] = np.fft.ifft(f * daughter)
        except Exception as e:
            warnings.warn(f"Wavelet computation failed for scale {scale[a1]}: {e}")
            wave[a1, :] = np.nan
    
    # Calculate period and COI
    period = fourier_factor * scale
    
    # Improved COI calculation
    try:
        coi_base = coi * dt
        coi_ramp = np.arange(1., (n1 + 1.) / 2.)
        coi_ramp = np.minimum(coi_ramp, np.flipud(coi_ramp))
        coi = coi_base * np.concatenate(([1.E-5], coi_ramp, [1.E-5]))
        coi = coi[:n1]  # Ensure correct length
    except Exception as e:
        warnings.warn(f"COI calculation failed: {e}")
        coi = np.full(n1, coi * dt)
    
    # Remove padding
    wave = wave[:, :n1]
    
    return wave, period, coi


def wave_signif(Y, dt, scale1, sigtest=-1, lag1=-1, siglvl=-1, dof=-1, mother=-1, param=-1):
    """Improved significance testing with better error handling"""
    
    # Input validation
    scale1 = np.asarray(scale1, dtype=float)
    
    if len(scale1) == 0:
        raise ValueError("Scale array cannot be empty")
    
    if np.any(scale1 <= 0):
        raise ValueError("All scales must be positive")
    
    try:
        n1 = len(Y) if hasattr(Y, '__len__') else 1
    except:
        n1 = 1
        
    J1 = len(scale1) - 1
    scale = scale1.copy()
    s0 = np.min(scale)
    
    if len(scale) > 1:
        dj = np.log(scale[1] / scale[0]) / np.log(2.)
    else:
        dj = 0.25  # Default value

    # Set defaults
    if n1 == 1:
        variance = float(Y)
    else:
        Y_array = np.asarray(Y, dtype=float)
        variance = np.nanvar(Y_array)
        if variance == 0:
            warnings.warn("Signal has zero variance")
            variance = 1.0

    if sigtest == -1: sigtest = 0
    if lag1 == -1: lag1 = 0.0
    if siglvl == -1: siglvl = 0.95
    if mother == -1: mother = 'MORLET'

    mother = mother.upper()

    # Get wavelet parameters with error handling
    try:
        if mother == 'MORLET':
            if param == -1: param = 6.
            k0 = param
            fourier_factor = (4. * np.pi) / (k0 + np.sqrt(2. + k0 ** 2))
            empir = [2., -1, -1, -1]
            if k0 == 6: 
                empir[1:4] = [0.776, 2.32, 0.60]
        elif mother == 'PAUL':
            if param == -1: param = 4.
            m = param
            fourier_factor = 4. * np.pi / (2. * m + 1.)
            empir = [2., -1, -1, -1]
            if m == 4: 
                empir[1:4] = [1.132, 1.17, 1.5]
        elif mother == 'DOG':
            if param == -1: param = 2.
            m = param
            fourier_factor = 2. * np.pi * np.sqrt(2. / (2. * m + 1.))
            empir = [1., -1, -1, -1]
            if m == 2: 
                empir[1:4] = [3.541, 1.43, 1.4]
            if m == 6: 
                empir[1:4] = [1.966, 1.37, 0.97]
        else:
            raise ValueError("Mother must be one of MORLET, PAUL, DOG")
    except Exception as e:
        raise RuntimeError(f"Parameter calculation failed: {e}")

    period = scale * fourier_factor
    dofmin = empir[0]
    Cdelta = empir[1]
    gamma_fac = empir[2]
    dj0 = empir[3]

    # Calculate theoretical spectrum
    freq = dt / period
    fft_theor = (1. - lag1 ** 2) / (1. - 2. * lag1 * np.cos(freq * 2. * np.pi) + lag1 ** 2)
    fft_theor = variance * fft_theor
    signif = fft_theor.copy()

    # Handle DOF
    if dof == -1:
        dof = dofmin

    try:
        # Significance testing
        if sigtest == 0:  # No smoothing
            dof = dofmin
            chisquare = chi2.ppf(siglvl, dof) / dof
            signif = fft_theor * chisquare
            
        elif sigtest == 1:  # Time-averaged
            if not hasattr(dof, '__len__'):
                dof = np.full(J1 + 1, dof)
            
            dof = np.maximum(dof, 1.)  # Ensure positive DOF
            dof = dofmin * np.sqrt(1. + (dof * dt / gamma_fac / scale) ** 2)
            dof = np.maximum(dof, dofmin)
            
            for a1 in range(J1 + 1):
                chisquare = chi2.ppf(siglvl, dof[a1]) / dof[a1]
                signif[a1] = fft_theor[a1] * chisquare
                
        elif sigtest == 2:  # Scale-averaged
            if len(dof) != 2:
                raise ValueError("DOF must be [S1, S2] for scale-averaged test")
            if Cdelta == -1:
                raise ValueError(f'Cdelta not defined for {mother} with param = {param}')
                
            s1, s2 = dof[0], dof[1]
            avg = (scale >= s1) & (scale <= s2)
            navg = np.sum(avg)
            
            if navg == 0:
                raise ValueError(f'No valid scales between {s1} and {s2}')
                
            Savg = 1. / np.sum(1. / scale[avg])
            Smid = np.exp((np.log(s1) + np.log(s2)) / 2.)
            dof = (dofmin * navg * Savg / Smid) * np.sqrt(1. + (navg * dj / dj0) ** 2)
            fft_theor = Savg * np.sum(fft_theor[avg] / scale[avg])
            chisquare = chi2.ppf(siglvl, dof) / dof
            signif = (dj * dt / Cdelta / Savg) * fft_theor * chisquare
            
        else:
            raise ValueError('sigtest must be 0, 1, or 2')
            
    except Exception as e:
        raise RuntimeError(f"Significance calculation failed: {e}")

    return signif, fft_theor