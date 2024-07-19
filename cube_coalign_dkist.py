import sunpy
import sunpy.map
from sunpy.map import Map, MapSequence
from sunpy.util.exceptions import SunpyUserWarning
import numpy as np
import warnings
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, Button, CheckButtons
import matplotlib.patches as patches
import matplotlib.animation as animation
from skimage.feature import match_template
from scipy.ndimage import shift
import astropy.units as u
from astropy.visualization import (ImageNormalize, AsinhStretch)
from astropy.time import Time
from datetime import datetime
import dkist
from ndcube import NDCube
from dask.diagnostics import ProgressBar

from glob import glob
from copy import deepcopy
import os
import h5py
import argparse
from IPython.display import HTML, display


class DatasetCoalign(dkist.Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        

    @classmethod
    def from_dkist_ds(cls, parent):
        return cls(parent.data, parent.wcs, parent.uncertainty,
                    parent.mask, parent.meta, parent.unit)

    def flicker(self, index=0, fps = 5):
        date_avg = Time(self.headers['DATE-AVG'])

        if isinstance(index, int):
            pass
        elif isinstance(index, str):
            time_select = Time(index)
            index = np.argmin(np.abs(date_avg - time_select))
        elif isinstance(index, Time):
            index = np.argmin(np.abs(date_avg - index))
        elif isinstance(index, datetime):
            index = np.argmin(np.abs(date_avg - Time(index)))

        if index < 0 or index >= self.dimensions[0].value - 1:
            raise ValueError('Index out of bounds')
        
        data_1 = self[index].data.compute()
        data_2 = self[index+1].data.compute()

        norm_1 = ImageNormalize(vmin=np.nanpercentile(data_1, 0.1), 
                                vmax=np.nanpercentile(data_1, 99.9),
                                stretch=AsinhStretch())
        norm_2 = ImageNormalize(vmin=np.nanpercentile(data_2, 0.1), 
                                vmax=np.nanpercentile(data_2, 99.9),
                                stretch=AsinhStretch())
        

        fig, ax = plt.subplots(figsize=(10,10),layout='constrained')

        im = ax.imshow(data_1, norm=norm_1, cmap='gray', origin='lower')

        plt.axis('off')

        def init_plot():
            return im,

        def update_plot(ii):
            if ii == 0:
                im.set_array(data_1)
                im.set_norm(norm_1)
            else:
                im.set_array(data_2)
                im.set_norm(norm_2)

            return im,


        anim = animation.FuncAnimation(fig, update_plot, interval=1000/fps, blit=True,frames=2,
                            repeat=True, init_func=init_plot)
        
        if matplotlib.get_backend().lower() == 'module://matplotlib_inline.backend_inline':
            anim_html = HTML(anim.to_jshtml())
            
            fig.clf()
            plt.close()
            display(anim_html)
        else:
            plt.show()

    def self_coalign_single(self, single_threaded=True):
        self.data_rechunk = self.data[:,100:-100,100:-100].rechunk((2, self.data.chunksize[1], self.data.chunksize[2]))
        if single_threaded:
            with ProgressBar():
                self.shifts = self.data_rechunk.map_overlap(_self_calculate_shift, 
                                                    depth=(1,0,0), # We only want to cross-correlate in the time dimension
                                                    boundary='reflect', 
                                                    chunks=(3,1), # To match depth = 1, so the overlap chunksize is chuncksize + 2*depth
                                                    drop_axis=(1,2), # Drop the spatial dimensions
                                                    new_axis=1, # Add a new axis to store the y- and x-shifts
                                                    dtype='float64').compute(scheduler='single-threaded')
        else:
            self.shifts = self.data_rechunk.map_overlap(_self_calculate_shift, 
                                                depth=(1,0,0), # We only want to cross-correlate in the time dimension
                                                boundary='reflect', 
                                                chunks=(3,1), # To match depth = 1, so the overlap chunksize is chuncksize + 2*depth
                                                drop_axis=(1,2), # Drop the spatial dimensions
                                                new_axis=1, # Add a new axis to store the y- and x-shifts
                                                dtype='float64').compute()


def _self_calculate_shift(x):
    template = x[0,974:2922,974:2922]
    this_layer = x[1,:,:]
    yshift_pixel, xshift_pixel = _calculate_shift(this_layer, template)

    #We need to deceive the map_overlap function to let it return the x and y shifts 
    #at the correct position in the array

    return np.array([[0,yshift_pixel.value,0], [0,xshift_pixel.value,0]]).T



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
        


    





