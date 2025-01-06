import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import AutoMinorLocator
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.offsetbox import (AnchoredOffsetbox, AuxTransformBox,
                                  TextArea, VPacker)
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from matplotlib.patches import Rectangle
import numpy as np

import astropy.units as u
from astropy.wcs.utils import proj_plane_pixel_scales

def plot_colorbar(im, ax, bbox_to_anchor=(1.02, 0., 0.1, 1),fontsize=10,
                  orientation="vertical",
                  title=None,scilimits=(-4,4),**kwargs):
    # clb_ax = inset_axes(ax,width=width,height=height,loc=loc,
    #             bbox_to_anchor=bbox_to_anchor,
    #              bbox_transform=ax.transAxes,
    #              borderpad=0)

    clb_ax = ax.inset_axes(bbox_to_anchor,transform=ax.transAxes)
    
    clb = plt.colorbar(im,pad = 0.05,orientation=orientation,ax=ax,cax=clb_ax,**kwargs)
    
    clb_ax.tick_params(labelsize=fontsize)
    
    if orientation == "vertical":
        clb_ax.ticklabel_format(axis="y", style="sci", scilimits=scilimits)
        clb_ax.yaxis.get_offset_text().set_fontsize(fontsize)
        clb_ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    elif orientation == "horizontal":
        clb_ax.ticklabel_format(axis="x", style="sci", scilimits=scilimits)
        clb_ax.xaxis.get_offset_text().set_fontsize(fontsize)
        clb_ax.xaxis.set_minor_locator(AutoMinorLocator(5))

    if title is not None:
        clb.set_label(title,fontsize=fontsize)

    return clb, clb_ax

# a more advanced version of astropy.visualization.wcsaxes.add_scalebar()
# which allows for more customization including bbox_props
# also allows for the scalebar accept a length-like quantity
# and automatically converts it to degrees using the dsun attribute
# of the WCS using a Cartesian approximation

def wcs_scalebar(ax,
    length,
    label=None,
    corner="bottom right",
    frame=False,
    borderpad=0.4,
    pad=0.5,
    bbox_props=None,
    dsun=None,
    correct_rectangle_pixels=True,
    **kwargs,
    ):
    """Add a scale bar.

    Parameters
    ----------
    ax : :class:`~astropy.visualization.wcsaxes.WCSAxes`
        WCSAxes instance in which the scale bar is displayed. The WCS must be
        celestial.
    length : float or :class:`~astropy.units.Quantity`
        The length of the scalebar in degrees or an angular quantity
    label : str, optional
        Label to place below the scale bar
    corner : str, optional
        Where to place the scale bar. Acceptable values are:, ``'left'``,
        ``'right'``, ``'top'``, ``'bottom'``, ``'top left'``, ``'top right'``,
        ``'bottom left'`` and ``'bottom right'`` (default)
    frame : bool, optional
        Whether to display a frame behind the scale bar (default is ``False``)
    borderpad : float, optional
        Border padding, in fraction of the font size. Default is 0.4.
    pad : float, optional
        Padding around the scale bar, in fraction of the font size. Default is 0.5.
    bbox_props : dict, optional
        A dictionary of properties to be passed to the :class:`~matplotlib.patches.FancyBboxPatch`
        that is used to draw the scale bar. Default is ``None``.
    dsun : :class:`~astropy.units.Quantity`, optional
        The distance to the Sun. Only used when the length is a length-like quantity. 
        If not provided, dsun is get from the WCSAxes.wcs.wcs.aux.dsun_obs attribute.
        Default is ``None``.
    correct_rectangle_pixels : bool, optional
        Whether to correct for rectangular pixels (cdelt1 != cdelt2).
        Assume the aspect ratio of the pixels is correct, i.e.,
        aspect = cdelt2 / cdelt1.
        Default is ``True``.
    kwargs
        Additional arguments are passed to
        :class:`mpl_toolkits.axes_grid1.anchored_artists.AnchoredSizeBar`.

    Notes
    -----
    This function may be inaccurate when:

    - The pixel scales at the reference pixel are different from the pixel scales
      within the image extent (e.g., when the reference pixel is well outside of
      the image extent and the projection is non-linear)
    - The pixel scales in the two directions are very different from each other
      (e.g., rectangular pixels)

    """

    CORNERS = {
    "top right": 1,
    "top left": 2,
    "bottom left": 3,
    "bottom right": 4,
    "right": 5,
    "left": 6,
    "bottom": 8,
    "top": 9,
    }

    if dsun is None:
        dsun = ax.wcs.wcs.aux.dsun_obs * u.m
    
    if isinstance(length, u.Quantity):
        if length.unit.physical_type == "angle":
            length = length.to(u.degree).value
        elif length.unit.physical_type == "length":
            length = (length/dsun).decompose()*u.rad
            length = length.to_value(u.degree)  
        else:
            raise ValueError("Length must be an angular quantity or a length quantity ()")

    if ax.wcs.is_celestial:
        pix_scale = proj_plane_pixel_scales(ax.wcs)
        sx = pix_scale[0]
        sy = pix_scale[1]
        if correct_rectangle_pixels:
            degrees_per_pixel = sx
        else:
            degrees_per_pixel = np.sqrt(sx * sy)
    else:
        raise ValueError("Cannot show scalebar when WCS is not celestial")

    length = length / degrees_per_pixel

    corner = CORNERS[corner]

    scalebar = AnchoredSizeBarFancybox(
        ax.transData,
        length,
        label,
        corner,
        pad=pad,
        borderpad=borderpad,
        sep=5,
        frameon=frame,
        bbox_props=bbox_props,
        **kwargs,
    )

    ax.add_artist(scalebar)

    return scalebar

class AnchoredOffsetFancybox(AnchoredOffsetbox):

    def __init__(self, loc, *,
                 pad=0.4, borderpad=0.5,
                 child=None, prop=None, frameon=True,
                 bbox_to_anchor=None,
                 bbox_transform=None,
                 bbox_props=None,
                 **kwargs):
        
        super().__init__(loc, pad=pad, borderpad=borderpad, child=child,
                         prop=prop, frameon=frameon, bbox_to_anchor=bbox_to_anchor,
                         bbox_transform=bbox_transform, **kwargs)

        if bbox_props is not None:
            default_bbox_props = dict(xy=(0.0, 0.0), width=1., height=1.,
            facecolor='w', edgecolor='k',
            mutation_scale=self.prop.get_size_in_points(),
            snap=True,
            visible=frameon,
            boxstyle="square,pad=0",)

            default_bbox_props.update(bbox_props)
            self.patch = FancyBboxPatch(**default_bbox_props)

class AnchoredSizeBarFancybox(AnchoredOffsetFancybox):
    def __init__(self, transform, size, label, loc,
                 pad=0.1, borderpad=0.1, sep=2,
                 frameon=True, size_vertical=0, color='black',
                 label_top=False, fontproperties=None, fill_bar=None,
                 bbox_props=None, **kwargs):

        if fill_bar is None:
            fill_bar = size_vertical > 0

        self.size_bar = AuxTransformBox(transform)
        self.size_bar.add_artist(Rectangle((0, 0), size, size_vertical,
                                           fill=fill_bar, facecolor=color,
                                           edgecolor=color))

        if fontproperties is None and 'prop' in kwargs:
            fontproperties = kwargs.pop('prop')

        if fontproperties is None:
            textprops = {'color': color}
        else:
            textprops = {'color': color, 'fontproperties': fontproperties}

        self.txt_label = TextArea(label, textprops=textprops)

        if label_top:
            _box_children = [self.txt_label, self.size_bar]
        else:
            _box_children = [self.size_bar, self.txt_label]

        self._box = VPacker(children=_box_children,
                            align="center",
                            pad=0, sep=sep)

        super().__init__(loc, pad=pad, borderpad=borderpad, child=self._box,
                         prop=fontproperties, frameon=frameon, bbox_props=bbox_props,
                         **kwargs)
        

# if __name__ == '__main__':
#     fig, ax = plt.subplots()

#     bar = AnchoredSizeBarFancybox(ax.transData, 0.1, '10 arcsec', 'lower right',
#                                     pad=0.1, borderpad=0.1, sep=2,
#                                     frameon=True, size_vertical=0.01, color='black',
#                                     label_top=False, fontproperties=None, fill_bar=None,
#                                     bbox_props=dict(facecolor='w', edgecolor='blue', boxstyle="round"))
    
#     ax.add_artist(bar)
#     plt.show()