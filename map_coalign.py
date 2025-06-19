import sunpy
import sunpy.map
from sunpy.map import Map, MapSequence
from sunpy.util.exceptions import SunpyUserWarning
import numpy as np
import warnings
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, Button, CheckButtons
import matplotlib.patches as patches
import matplotlib.animation
from skimage.feature import match_template
from scipy.ndimage import shift
import astropy.units as u
from astropy.visualization import (ImageNormalize, AsinhStretch)
from astropy.io.fits import CompImageHDU
from tqdm import tqdm
from glob import glob
from copy import deepcopy
import os
import h5py
import argparse

class MapSequenceCoalign(MapSequence):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nt = len(self)

    def __setitem__(self, key, value):
        if isinstance(value, sunpy.map.GenericMap):
            self.maps[key] = value

    def plot(self, axes=None, resample=None, annotate=True,
             interval=200, plot_function=None, no_wcs=False, **kwargs):
        if no_wcs:
            if not axes:
                fig, axes = plt.subplots(layout='constrained')

            if not plot_function:
                def plot_function(fig, ax, smap):
                    return []
            removes = []

            # Normal plot
            def annotate_frame(i):
                axes.set_title(f"{self[i].name}")
                axes.set_xlabel('X [pixel]')
                axes.set_ylabel('Y [pixel]')

            if resample:
                if self.all_maps_same_shape():
                    resample = u.Quantity(self.maps[0].dimensions) * np.array(resample)
                    ani_data = [amap.resample(resample) for amap in self.maps]
                else:
                    raise ValueError('Maps in mapsequence do not all have the same shape.')
            else:
                ani_data = self.maps

            if 'norm' in kwargs:
                norm = kwargs['norm']
                kwargs.pop('norm',None)
            else:
                try:
                    norm = ani_data[0].plot_settings['norm']
                except:
                    norm = None

            if 'cmap' in kwargs:
                cmap = kwargs['cmap']
                kwargs.pop('cmap',None)
            else:
                try:
                    cmap = ani_data[0].plot_settings['cmap']
                except:
                    cmap = None

            im = axes.imshow(ani_data[0].data, origin='lower', norm=norm, cmap=cmap, **kwargs)

            def updatefig(i, im, annotate, ani_data, removes, update_norm=False):
                while removes:
                    removes.pop(0).remove()

                im.set_array(ani_data[i].data)
                im.set_cmap(kwargs.get('cmap', ani_data[i].plot_settings['cmap']))

                if update_norm:
                    norm = deepcopy(kwargs.get('norm', ani_data[i].plot_settings['norm']))
                # The following explicit call is for bugged versions of Astropy's
                # ImageNormalize
                    norm.autoscale_None(ani_data[i].data)
                    im.set_norm(norm)

                if annotate:
                    annotate_frame(i)
                removes += list(plot_function(fig, axes, ani_data[i]))

            ani = matplotlib.animation.FuncAnimation(fig, updatefig,
                                                    frames=list(range(0, len(ani_data))),
                                                    fargs=[im, annotate, ani_data, removes],
                                                    interval=interval,
                                                    blit=False)

            return ani
        else:
            super().plot(axes, resample, annotate,
             interval, plot_function, **kwargs)

    def coalign(self, reference_index=0, bottom_left=None, top_right=None, check_header=True,
                nframes=10,iter=3):

        if check_header:
            self._check_header()

        self.nx = self[0].data.shape[1]
        self.ny = self[0].data.shape[0]

        if bottom_left is None or top_right is None:
            self.bottom_left = [self.nx//4, self.ny//4]
            self.top_right = [3*self.nx//4, 3*self.ny//4]

            self._get_common_map_extent()
        else:
            self.bottom_left = bottom_left
            self.top_right = top_right

        self.reference_index = reference_index
        self.xshifts_pixel = np.zeros((self.nt,iter))
        self.yshifts_pixel = np.zeros((self.nt,iter))
        self.nframes = nframes
        self.iter = iter
        self.n_segment = np.ceil(self.nt/nframes).astype(int)

        for ii in range(iter):
            print(f'--------Starting iteration {ii+1}/{iter}--------')
            self._calculate_shifts(ii)
            self._apply_shifts(ii)
            print(f'--------Iteration {ii+1}/{iter} finished--------')

    def submap(self, *args, **kwargs) -> 'MapSequenceCoalign':
        new_map_seq = []
        for ii in range(self.nt):
            new_map_seq.append(self[ii].submap(*args, **kwargs))
        return MapSequenceCoalign(new_map_seq)
    
    def save(self, filepath, filetype='auto', **kwargs):
        filedir = os.path.dirname(filepath)
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        with h5py.File(os.path.join(filedir,"coalign_info.h5"), 'w') as hf:
            hf.create_dataset('xshifts_pixel', data=self.xshifts_pixel)
            hf.create_dataset('yshifts_pixel', data=self.yshifts_pixel)
            hf.create_dataset('bottom_left', data=self.bottom_left)
            hf.create_dataset('top_right', data=self.top_right)
            hf.create_dataset('reference_index', data=self.reference_index)
            hf.create_dataset('nframes', data=self.nframes)
            hf.create_dataset('iter', data=self.iter)
            hf.create_dataset('n_segment', data=self.n_segment)

        super().save(filepath, filetype, **kwargs)

    def _calculate_shifts(self,iter_index):
        for ii_seg in tqdm(range(self.n_segment)):
            start_index_ = ii_seg*self.nframes
            if ii_seg == self.n_segment-1:
                end_index_ = (ii_seg+1)*self.nframes if self.nt%self.nframes == 0 else self.nt - 1
            else:
                end_index_ = (ii_seg+1)*self.nframes + 1

            template_ = self[start_index_].data[self.bottom_left[1]:self.top_right[1],
                                    self.bottom_left[0]:self.top_right[0]]

            for jj in range(start_index_, end_index_):
                if jj == start_index_:
                    yshift_keep, xshift_keep = _calculate_shift(self[jj].data, template_)
                else:
                    yshift, xshift = _calculate_shift(self[jj].data, template_)
                    self.yshifts_pixel[jj,iter_index] = yshift.value + self.yshifts_pixel[start_index_,iter_index] - yshift_keep.value
                    self.xshifts_pixel[jj,iter_index] = xshift.value + self.xshifts_pixel[start_index_,iter_index] - xshift_keep.value

        self.yshifts_pixel[:,iter_index] = self.yshifts_pixel[:,iter_index] - self.yshifts_pixel[self.reference_index,iter_index]
        self.xshifts_pixel[:,iter_index] = self.xshifts_pixel[:,iter_index] - self.xshifts_pixel[self.reference_index,iter_index]

    def _apply_shifts(self,iter_index):
        for ii in range(self.nt):
            self[ii] = sunpy.map.Map(shift(self[ii].data, (-self.yshifts_pixel[ii,iter_index],-self.xshifts_pixel[ii,iter_index])),
                                     self[ii].meta)
    
        
    def _check_header(self):
        
        if self.all_maps_same_shape() is False:
            raise ValueError("All maps in the sequence must have the same shape")
        
        if 'cdelt1' in self.all_meta()[0].keys():
            cdelt1_all = [meta['cdelt1'] for meta in self.all_meta()]
            cdelt2_all = [meta['cdelt2'] for meta in self.all_meta()]

            if np.allclose(cdelt1_all, cdelt1_all[0]) and np.allclose(cdelt2_all, cdelt2_all[0]):
                pass 
            else:
                raise ValueError("All maps in the sequence must have the same CDELT1 values")
        else:
            warnings.warn("CDELT1 not found in header. Assuming equal pixel scales")

        if 'cdelt2' in self.all_meta()[0].keys():
            cdelt2_all = [meta['cdelt2'] for meta in self.all_meta()]

            if np.allclose(cdelt2_all, cdelt2_all[0]):
                pass 
            else:
                raise ValueError("All maps in the sequence must have the same CDELT2 value")
        else:
            warnings.warn("CDELT2 not found in header. Assuming equal pixel scales")


        rot_flag = False
        if 'crota' in self.all_meta()[0].keys():
            rot_flag = True
            crota_all = [meta['crota'] for meta in self.all_meta()]

            if np.allclose(crota_all, crota_all[0], atol=1e-2):
                pass 
            else:
                raise ValueError("All maps in the sequence must have the same CROTA value")
        elif 'crota2' in self.all_meta()[0].keys():
            rot_flag = True
            crota2_all = [meta['crota2'] for meta in self.all_meta()]

            if np.allclose(crota2_all, crota2_all[0]):
                pass 
            else:
                raise ValueError("All maps in the sequence must have the same CROTA2 value")
        elif 'pc1_1' in self.all_meta()[0].keys():
            rot_flag = True
            pc1_1_all = [meta['pc1_1'] for meta in self.all_meta()]
            pc1_2_all = [meta['pc1_2'] for meta in self.all_meta()]
            pc2_1_all = [meta['pc2_1'] for meta in self.all_meta()]
            pc2_2_all = [meta['pc2_2'] for meta in self.all_meta()]

            if np.allclose(pc1_1_all, pc1_1_all[0]) and np.allclose(pc1_2_all, pc1_2_all[0]) and np.allclose(pc2_1_all, pc2_1_all[0]) and np.allclose(pc2_2_all, pc2_2_all[0]):
                pass
            else:
                raise ValueError("All maps in the sequence must have the same PCi_j values")

        if rot_flag is False:
            warnings.warn("No rotation information found in header. Assuming no rotation")

    def _get_common_map_extent(self) -> None:

        fig = plt.figure(figsize=(7,5))

        self.fig = fig

        ax = fig.add_axes([0.1,0.1,0.6,0.9])

        self.ax = ax

        ax.imshow(self[0].data, origin='lower', cmap='gray',norm=ImageNormalize(stretch=AsinhStretch()))

        ax_bottom_left_x = fig.add_axes([0.75,0.7,0.1,0.08])
        ax_bottom_left_y = fig.add_axes([0.87,0.7,0.1,0.08])
        ax_top_right_x = fig.add_axes([0.75,0.5,0.1,0.08])
        ax_top_right_y = fig.add_axes([0.87,0.5,0.1,0.08])

        ax_bottom_left_x.text(1.25,1.5,"Bottom Left", fontsize=12, ha='center',
                va='center', transform=ax_bottom_left_x.transAxes)
        
        ax_top_right_x.text(1.25,1.5,"Top Right", fontsize=12, ha='center',
                va='center', transform=ax_top_right_x.transAxes)

        self.textbox_bottom_left_x = TextBox(ax_bottom_left_x,None,textalignment='center',
                                        initial=str(self.bottom_left[0]))
        self.textbox_bottom_left_y = TextBox(ax_bottom_left_y,None,textalignment='center',
                                        initial=str(self.bottom_left[1]))
        self.textbox_top_right_x = TextBox(ax_top_right_x,None,textalignment='center',
                                        initial=self.top_right[0])
        self.textbox_top_right_y = TextBox(ax_top_right_y,None,textalignment='center',
                                        initial=self.top_right[1])
        
        self.rectangle = patches.Rectangle((self.bottom_left[0],self.bottom_left[1]),
                                             self.top_right[0]-self.bottom_left[0],
                                             self.top_right[1]-self.bottom_left[1],
                                             edgecolor='red', facecolor='none') 
        ax.add_patch(self.rectangle)

        self.textbox_bottom_left_x.on_submit(self._update_bottom_left_x)
        self.textbox_bottom_left_y.on_submit(self._update_bottom_left_y)
        self.textbox_top_right_x.on_submit(self._update_top_right_x)
        self.textbox_top_right_y.on_submit(self._update_top_right_y)

        ax_close_button = fig.add_axes([0.87,0.3,0.1,0.08])
        close_button = Button(ax_close_button, 'Close')
        close_button.on_clicked(self._close_window)

        ax_select_button = fig.add_axes([0.75,0.3,0.1,0.08])
        self.selecting_mode = False
        self.select_button = CheckButtons(ax_select_button, ['Select'],
                                     frame_props={'sizes':[50]})

        self.select_rectangle = patches.Rectangle((0,0),1,1,edgecolor='blue', facecolor='none',
                                                  ls='--')
        ax.add_patch(self.select_rectangle)
        self.select_rectangle.set_visible(False)

        self.is_selecting = False
        self.fig.canvas.mpl_connect('button_press_event', self._on_press)
        self.fig.canvas.mpl_connect('button_release_event', self._on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self._on_select)
        

        plt.show()
        if plt.get_backend() in ['inline','nbAgg','ipympl']:
            plt.ion()
            print("did it?")
        return None
    
    def _update_bottom_left_x(self, expression):
        self.bottom_left[0] = int(expression)
        self._update_common_extent()
    
    def _update_bottom_left_y(self, expression):
        self.bottom_left[1] = int(expression)
        self._update_common_extent()
    
    def _update_top_right_x(self, expression):
        self.top_right[0] = int(expression)
        self._update_common_extent()

    def _update_top_right_y(self, expression):
        self.top_right[1] = int(expression)
        self._update_common_extent()

    def _update_text_from_rectangle(self):
        self.textbox_bottom_left_x.set_val(str(self.bottom_left[0]))
        self.textbox_bottom_left_y.set_val(str(self.bottom_left[1]))
        self.textbox_top_right_x.set_val(str(self.top_right[0]))
        self.textbox_top_right_y.set_val(str(self.top_right[1]))
    
    def _update_common_extent(self):
        self.rectangle.set_bounds(self.bottom_left[0],self.bottom_left[1],
                                  self.top_right[0]-self.bottom_left[0],
                                  self.top_right[1]-self.bottom_left[1])
        plt.draw()

    def _start_select(self, event):
        self.fig.canvas.mpl_disconnect('motion_notify_event',self._on_select)

    def _on_press(self, event):
        if self.select_button.get_status()[0] and event.inaxes == self.ax:
            self.bottom_left_selecting = [event.xdata, event.ydata]
            self.is_selecting = True
        else:
            pass

    def _on_select(self, event):
        if self.is_selecting and self.select_button.get_status()[0] and event.inaxes == self.ax:
            self.top_right_selecting = [event.xdata, event.ydata]
            self._draw_select_rectangle()
            self.fig.canvas.draw_idle()
            
    def _draw_select_rectangle(self):
        self.select_rectangle.set_bounds(self.bottom_left_selecting[0],self.bottom_left_selecting[1],
                                            self.top_right_selecting[0]-self.bottom_left_selecting[0],
                                            self.top_right_selecting[1]-self.bottom_left_selecting[1])  
        self.select_rectangle.set_visible(True)

    def _on_release(self, event):
        if self.is_selecting and self.select_button.get_status()[0] and event.inaxes == self.ax:
            self.is_selecting = False
            self.bottom_left = [int(np.min([self.bottom_left_selecting[0],self.top_right_selecting[0]])), 
                                int(np.min([self.bottom_left_selecting[1],self.top_right_selecting[1]]))]
            self.top_right = [int(np.max([self.bottom_left_selecting[0],self.top_right_selecting[0]])),
                              int(np.max([self.bottom_left_selecting[1],self.top_right_selecting[1]]))]
            self.select_rectangle.set_visible(False)
            self._update_text_from_rectangle()
            self._update_common_extent()


    def _close_window(self, event):
        print('The selected region is: ', 'Bottom Left', self.bottom_left, 'Top Right', self.top_right)
        if plt.get_backend() in ['inline','nbAgg','ipympl']:
            plt.ioff()
        plt.close()



# Functions copied from sunpy.image.coalignment version 0.5.1


def _calculate_shift(this_layer, template):
    """
    Calculates the pixel shift required to put the template in the "best"
    position on a layer.

    Parameters
    ----------
    this_layer : `numpy.ndarray`
        A numpy array of size ``(ny, nx)``, where the first two dimensions are
        spatial dimensions.
    template : `numpy.ndarray`
        A numpy array of size ``(N, M)`` where ``N < ny`` and ``M < nx``.

    Returns
    -------
    `tuple`
        Pixel shifts ``(yshift, xshift)`` relative to the offset of the template
        to the input array.
    """
    # Warn user if any NANs, Infs, etc are present in the layer or the template
    _check_for_nonfinite_entries(this_layer, template)
    # Calculate the correlation array matching the template to this layer
    corr = match_template(this_layer, template)
    # Calculate the y and x shifts in pixels
    return _find_best_match_location(corr)


def _find_best_match_location(corr):
    """
    Calculate an estimate of the location of the peak of the correlation result
    in image pixels.

    Parameters
    ----------
    corr : `numpy.ndarray`
        A 2D correlation array.

    Returns
    -------
    `~astropy.units.Quantity`
        The shift amounts ``(y, x)`` in image pixels. Subpixel values are
        possible.
    """
    # Get the index of the maximum in the correlation function
    ij = np.unravel_index(np.argmax(corr), corr.shape)
    cor_max_x, cor_max_y = ij[::-1]

    # Get the correlation function around the maximum
    array_maximum = corr[
        np.max([0, cor_max_y - 1]) : np.min([cor_max_y + 2, corr.shape[0] - 1]),
        np.max([0, cor_max_x - 1]) : np.min([cor_max_x + 2, corr.shape[1] - 1]),
    ]
    y_shift_maximum, x_shift_maximum = _get_correlation_shifts(array_maximum)

    # Get shift relative to correlation array
    y_shift_correlation_array = y_shift_maximum + cor_max_y * u.pix
    x_shift_correlation_array = x_shift_maximum + cor_max_x * u.pix

    return y_shift_correlation_array, x_shift_correlation_array


def _get_correlation_shifts(array):
    """
    Estimate the location of the maximum of a fit to the input array. The
    estimation in the "x" and "y" directions are done separately. The location
    estimates can be used to implement subpixel shifts between two different
    images.

    Parameters
    ----------
    array : `numpy.ndarray`
        An array with at least one dimension that has three elements. The
        input array is at most a 3x3 array of correlation values calculated
        by matching a template to an image.

    Returns
    -------
    `~astropy.units.Quantity`
        The ``(y, x)`` location of the peak of a parabolic fit, in image pixels.
    """
    # Check input shape
    ny = array.shape[0]
    nx = array.shape[1]
    if nx > 3 or ny > 3:
        msg = "Input array dimension should not be greater than 3 in any dimension."
        raise ValueError(msg)

    # Find where the maximum of the input array is
    ij = np.unravel_index(np.argmax(array), array.shape)
    x_max_location, y_max_location = ij[::-1]

    # Estimate the location of the parabolic peak if there is enough data.
    # Otherwise, just return the location of the maximum in a particular
    # direction.
    y_location = _parabolic_turning_point(array[:, x_max_location]) if ny == 3 else 1.0 * y_max_location

    x_location = _parabolic_turning_point(array[y_max_location, :]) if nx == 3 else 1.0 * x_max_location

    return y_location * u.pix, x_location * u.pix


def _parabolic_turning_point(y):
    """
    Find the location of the turning point for a parabola ``y(x) = ax^2 + bx +
    c``, given input values ``y(-1), y(0), y(1)``. The maximum is located at
    ``x0 = -b / 2a``. Assumes that the input array represents an equally spaced
    sampling at the locations ``y(-1), y(0) and y(1)``.

    Parameters
    ----------
    y : `numpy.ndarray`
        A one dimensional numpy array of shape "3" with entries that sample the
        parabola at "-1", "0", and "1".

    Returns
    -------
    `float`
        A float, the location of the parabola maximum.
    """
    numerator = -0.5 * y.dot([-1, 0, 1])
    denominator = y.dot([1, -2, 1])
    return numerator / denominator

def _check_for_nonfinite_entries(layer_image, template_image):
    """
    Issue a warning if there is any nonfinite entry in the layer or template
    images.

    Parameters
    ----------
    layer_image : `numpy.ndarray`
        A two-dimensional `numpy.ndarray`.
    template_image : `numpy.ndarray`
        A two-dimensional `numpy.ndarray`.
    """
    if not np.all(np.isfinite(layer_image)):
        warnings.warn(
            "The layer image has nonfinite entries. "
            "This could cause errors when calculating shift between two "
            "images. Please make sure there are no infinity or "
            "Not a Number values. For instance, replacing them with a "
            "local mean.",
            SunpyUserWarning,
            stacklevel=3,
        )

    if not np.all(np.isfinite(template_image)):
        warnings.warn(
            "The template image has nonfinite entries. "
            "This could cause errors when calculating shift between two "
            "images. Please make sure there are no infinity or "
            "Not a Number values. For instance, replacing them with a "
            "local mean.",
            SunpyUserWarning,
            stacklevel=3,
        )



if __name__ == "__main__":
    '''
    Example usage:
    python map_coalign.py /path/to/map_sequence/*.fits /path/to/output_dir -ref 0 -i 3 -nh 
    '''
    parser = argparse.ArgumentParser(description='Coalign a sequence of maps using a cross-correlation method')
    parser.add_argument('filename', type=str, help='input map sequence filename')
    parser.add_argument('output_dir', type=str, help='output directory')
    parser.add_argument('-ref','--reference_index', type=int, default=0, help='Index of the reference map')
    parser.add_argument('-bl','--bottom_left', type=int, nargs=2, default=None, help='Bottom left corner of the region to align')
    parser.add_argument('-tr','--top_right', type=int, nargs=2, default=None, help='Top right corner of the region to align')
    parser.add_argument('-n','--nframes', type=int, default=10, help='Number of frames in each segment')
    parser.add_argument('-i','--iter', type=int, default=3, help='Number of iterations')
    parser.add_argument('-nh','--no_header_check', action='store_true', help='Do not check the header of the maps')
    parser.add_argument('-np','--no_preview', action='store_true', help='Do not show the video preview of the coaligned map sequence in the selected region')
    args = parser.parse_args()

    map_files = sorted(glob(args.filename))
    ms = MapSequenceCoalign(sunpy.map.Map(map_files))
    ms.coalign(reference_index=args.reference_index, bottom_left=args.bottom_left, top_right=args.top_right,
                check_header=not args.no_header_check, nframes=args.nframes, iter=args.iter)
    
    if not args.no_preview:
        anim = ms.plot(no_wcs=True)
        plt.show()

    do_save = input('Do you want to save the coaligned map sequence? (y/n): ')
    if do_save == 'y':
        do_compress = input('Do you want to save the coaligned map in compressed FITS? (y/n): ')
        if do_compress == 'y':
            ms.save(os.path.join(args.output_dir,"map_seq_coalign_{index:03}.fits"), overwrite=True, hdu_type=CompImageHDU)
        else:
            ms.save(os.path.join(args.output_dir,"map_seq_coalign_{index:03}.fits"), overwrite=True)
    else:
        pass




    # test_maps = sorted(glob('/home/yjzhu/Solar/EIS_DKIST_SolO/src/EUI/HRI/euv174/20221024/*.fits'))[10:100]

    # ms = MapSequenceCoalign(sunpy.map.Map(test_maps))

    # ms.coalign(iter=3,check_header=True)
    # ms_crop = ms.submap([500,600]*u.pix, top_right=[670,760]*u.pix)
    # ms.save("/home/yjzhu/Downloads/eui_coalign_test/eui_map_seq_coalign_{index:03}.fits", overwrite=True)
    # anim = ms_crop.plot(no_wcs=True)
    # plt.show()

    # print(_calculate_shift(ms[0].data, ms[0].data[100:200,100:200]))

    # fig, axes = plt.subplots(1,2,figsize=(10,5))
    # for ii in range(3):
    #     axes[0].plot(ms.xshifts_pixel[:,ii],label=f'Iteration {ii}')
    #     axes[1].plot(ms.yshifts_pixel[:,ii],label=f'Iteration {ii}')

    #     axes[0].legend()

    # plt.show()





