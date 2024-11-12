import numpy as np
import matplotlib 
import matplotlib.pyplot as plt
from matplotlib.widgets import (TextBox, Button, 
                                CheckButtons, RangeSlider,
                                Slider, LassoSelector)
from matplotlib.backend_bases import NavigationToolbar2
import matplotlib.lines as mlines
from matplotlib.transforms import Bbox
import sunpy
import sunpy.map
from sunpy.map import GenericMap, MapSequence
from map_coalign import MapSequenceCoalign
import warnings
from astropy.time import Time
from astropy.visualization import (ImageNormalize, AsinhStretch,
                                    LinearStretch, PercentileInterval,
                                    ZScaleInterval)
import astropy.units as u
from astropy.io.misc.hdf5 import write_table_hdf5
from skimage import draw, measure
import skimage.measure.profile
from ndcube import NDCube
from ndcube.extra_coords import (TimeTableCoordinate,
                                 QuantityTableCoordinate)
from PyQt5.QtWidgets import QFileDialog
import h5py
import os
import cv2
from watroo import wow
import multiprocessing


class SlitPick:
    """
    A class to make interactive slit picking and fitting on spacetime/stacked plots. 

    Accept Sunpy Map, MapSequence, or NDArray as input.  
    """

    def __init__(self, image_seq):
        if isinstance(image_seq, GenericMap):
            self.image_seq = MapSequenceCoalign(image_seq)
            self.image_type = 'SunpyMap'
            self.ny, self.nx = self.image_seq.maps[0].data.shape
            self.nt = len(self.image_seq.maps)
            self.dates = Time([map.date for map in self.image_seq.maps])
        elif isinstance(image_seq, MapSequence):
            self.image_seq = MapSequenceCoalign(image_seq.maps)
            self.image_type = 'SunpyMap'
            self.ny, self.nx = self.image_seq.maps[0].data.shape
            self.nt = len(self.image_seq.maps)
            self.dates = Time([map.date for map in self.image_seq.maps])
        elif isinstance(image_seq, np.ndarray):
            self.image_seq = image_seq
            self.image_type = 'NDArray'
            self.ny, self.nx, self.nt = self.image_seq.shape

    def __call__(self, bottom_left=None, top_right=None, wcs_index=0, 
                 wcs_shift=None, norm=None, line_width=5, img_wow=False,
                 init_gui=True):

        if init_gui:
            matplotlib.use('Qt5Agg')

        self.bottom_left = bottom_left
        self.top_right = top_right
        self.wcs_index = wcs_index
        self.frame_index = wcs_index
        self.wcs_shift = wcs_shift
        self.plot_asinha = 0.5
        if norm is None:
            self.norm = ImageNormalize(stretch=AsinhStretch(0.5))
        else:
            self.norm = norm
        self.in_selection = False
        self.successful = False
        self.in_moving = False
        self.in_fitting = False
        self.fit_poly_order = 2
        self.select_x = []
        self.select_y = []
        self.line_width = line_width
        self.bg_remove_on = False
        self.img_wow = img_wow

        if self.image_type == 'SunpyMap':
            if bottom_left is not None and top_right is not None:
                self.image_seq_prep = self.image_seq.submap(bottom_left, top_right=top_right)
            else:
                self.image_seq_prep  = self.image_seq

            if wcs_shift is not None:
                self.map_wcs = self.image_seq_prep[wcs_index].shift_reference_coord(*wcs_shift).wcs
            else:
                self.map_wcs = self.image_seq_prep[wcs_index].wcs
            
            self.projection = self.map_wcs

            if img_wow:
                for ii, map in enumerate(self.image_seq_prep):
                    self.image_seq_prep[ii] = sunpy.map.Map(wow(map.data)[0], map.meta)

        elif self.image_type == 'NDArray':
            if bottom_left is not None and top_right is not None:
                self.image_seq_prep = self.image_seq[bottom_left[1]:top_right[1]+1, bottom_left[0]:top_right[0]+1]
            else:
                self.image_seq_prep = self.image_seq

            if wcs_shift is not None:
                warnings.warn('wcs_shift is not supported for NDArray input')
            
            self.projection = None

            if img_wow:
                for ii in range(self.nt):
                    self.image_seq_prep[:,:,ii] = wow(self.image_seq_prep[:,:,ii])[0]

        if init_gui:
            self._init_gui()

    
    def _init_gui(self):

        NavigationToolbar2.home = self._new_home

        self.select_ax1_collection = []
        self.select_ax2_collection = []

        self.fig = plt.figure(figsize=(8,6))
        self.fig.canvas.manager.set_window_title('Interactive Spacetime Plot Maker')


        self.ax1 = self.fig.add_axes([0.09, 0.55, 0.3, 0.4], projection=self.projection)
        self.ax2 = self.fig.add_axes([0.48, 0.55, 0.3, 0.4], projection=self.projection)
        self.ax3 = self.fig.add_axes([0.09, 0.08, 0.66, 0.36], projection=None)

        self.ax_text_all = self.fig.add_axes([0.795,0,0.2,1])
        self.ax_text_all.axis('off')
        

        self.ax2.sharex(self.ax1)
        self.ax2.sharey(self.ax1)

        if self.image_type == 'SunpyMap':
            self.ax1.imshow(self.image_seq_prep[self.frame_index].data, cmap='magma', norm=self.norm,
                            origin='lower')
            self.ax1.set_xlabel('Solar-X [arcsec]')
            self.ax1.set_ylabel('Solar-Y [arcsec]')
            self.ax2.set_xlabel('Solar-X [arcsec]')
            self.ax2.set_ylabel(' ') 
        elif self.image_type == 'NDArray':
            self.ax1.imshow(self.image_seq_prep[:,:,self.frame_index], cmap='magma', norm=self.norm,
                            origin='lower')
            self.ax1.set_xlabel('Pixel-X')
            self.ax1.set_ylabel('Pixel-Y')
            self.ax2.set_xlabel('Pixel-X')
            self.ax1.set_aspect('equal')
            self.ax2.set_aspect('equal')

        self.simple_std = self._get_simple_std(every_nth=1)
        self.ax2.imshow(self.simple_std, cmap='magma', origin='lower',
                        norm = ImageNormalize(vmin=np.nanpercentile(self.simple_std, 2),
                                              vmax=np.nanpercentile(self.simple_std, 98),
                                              stretch=AsinhStretch(0.5),))


        self.ax1.set_title('Image')
        self.ax2.set_title(r'$\sigma/\mu$')

        self.ax1_axis = self.ax1.axis()
        self.ax2_axis = self.ax2.axis()

        self.ax_text_frame_index = self.fig.add_axes([0.795, 0.9, 0.2, 0.04])
        self.ax_text_frame_index.set_title('Frame Index', fontsize=10)

        self.ax_text_time = self.fig.add_axes([0.795, 0.81, 0.2, 0.04])
        self.ax_text_time.set_title('Time', fontsize=10)

        self.ax_text_lw = self.fig.add_axes([0.795, 0.72, 0.2, 0.04])
        self.ax_text_lw.set_title('Line Width', fontsize=10)

        self.ax_start_button = self.fig.add_axes([0.795, 0.61, 0.095, 0.05])
        self.ax_spline_button = self.fig.add_axes([0.90, 0.61, 0.095, 0.05])
        self.ax_end_button = self.fig.add_axes([0.795, 0.545, 0.095, 0.05])
        self.ax_clean_button = self.fig.add_axes([0.90, 0.545, 0.095, 0.05])

        self.ax_text_all.text(0.5, 0.67, 'Slit Pick', ha='center', va='bottom', fontsize=10)

        self.ax_asinha = self.fig.add_axes([0.795, 0.47, 0.2, 0.03])
        self.ax_asinha.set_title(r'Asinh $a$', fontsize=10, pad=0)

        self.ax_vmin_vmax = self.fig.add_axes([0.795, 0.39, 0.2, 0.03])
        self.ax_vmin_vmax.set_title('Vmin/Vmax', fontsize=10, pad=0)

        self.ax_bg_remove_checkbutton = self.fig.add_axes([0.795, 0.31, 0.1, 0.04])
        self.ax_bg_remove_checkbutton.axis('off')

        self.ax_text_all.text(0.5, 0.275, 'Spacetime Fitting', ha='center', va='bottom', fontsize=10)

        self.ax_text_ploy_order = self.fig.add_axes([0.86, 0.22, 0.03, 0.04])
        self.ax_reloc_checkbutton = self.fig.add_axes([0.90, 0.22, 0.095, 0.04])
        self.ax_reloc_checkbutton.axis('off')
        
        self.ax_st_start_button = self.fig.add_axes([0.795, 0.155, 0.095, 0.05])
        self.ax_st_end_button = self.fig.add_axes([0.90, 0.155, 0.095, 0.05])
        self.ax_st_delete_button = self.fig.add_axes([0.795, 0.09, 0.095, 0.05])
        self.ax_st_clean_button = self.fig.add_axes([0.90, 0.09, 0.095, 0.05])
        self.ax_st_save_button = self.fig.add_axes([0.795, 0.025, 0.095, 0.05])
        self.ax_close_button = self.fig.add_axes([0.90, 0.025, 0.095, 0.05])

        self.text_box_frame_index = TextBox(self.ax_text_frame_index, None, initial=str(self.frame_index),
                                            textalignment='center')

        if self.image_type == 'SunpyMap':
            self.text_box_time = TextBox(self.ax_text_time, None, initial=str(self.image_seq_prep[self.frame_index].date.iso[:-4]),
                                         textalignment='center')
        elif self.image_type == 'NDArray':
            self.text_box_time = TextBox(self.ax_text_time, None, initial=str(self.frame_index),
                                         textalignment='center')
            
        self.text_box_lw = TextBox(self.ax_text_lw, None, initial='5', textalignment='center')
            

        self.text_box_frame_index.on_submit(lambda x: self._update_time_index('frame_index'))
        self.text_box_time.on_submit(lambda x: self._update_time_index('time'))
        self.text_box_lw.on_submit(lambda x: self._update_line_width())

        self.button_start = Button(self.ax_start_button, 'Start')
        self.button_spline = Button(self.ax_spline_button, 'Spline')
        self.button_end = Button(self.ax_end_button, 'End')
        self.button_clean = Button(self.ax_clean_button, 'Clean')
    

        self.button_start.on_clicked(self._start_selection)
        self.button_end.on_clicked(self._make_slit)
        self.button_clean.on_clicked(self._clean_points)
        

        self.plot_vmin, self.plot_vmax = self.ax1.get_images()[0].get_clim()
        self.slider_asinha = Slider(self.ax_asinha, None, 0, 1, valinit=self.plot_asinha,
                                    valstep=np.linspace(0.05,1,20))
        self.slider_asinha.valtext.set_position((0.5,-0.1))
        self.slider_asinha.valtext.set_horizontalalignment('center')
        self.slider_asinha.valtext.set_verticalalignment('top')

        self.slider_vmin_vmax = RangeSlider(self.ax_vmin_vmax, None, 0, self.plot_vmax*2,
                                             valinit=[self.plot_vmin, self.plot_vmax])
        self.slider_vmin_vmax.valtext.set_position((0.5,-0.1))
        self.slider_vmin_vmax.valtext.set_horizontalalignment('center')
        self.slider_vmin_vmax.valtext.set_verticalalignment('top')

        self.slider_asinha.on_changed(self._update_asinha)
        self.slider_vmin_vmax.on_changed(self._update_vmin_vmax)

        self.checkbutton_bg_remove = CheckButtons(self.ax_bg_remove_checkbutton, ['BG Remove'], [False],
                                              frame_props=dict(sizes=(50,)))
        self.checkbutton_bg_remove.on_clicked(self._switch_bg_remove)


        self.text_box_ploy_order = TextBox(self.ax_text_ploy_order, 'Order', initial=str(self.fit_poly_order), 
                                           textalignment='center', label_pad = 0.4)
        self.checkbutton_reloc = CheckButtons(self.ax_reloc_checkbutton, ['Relocate'], [False],
                                              frame_props=dict(sizes=(50,)))

        self.button_st_start = Button(self.ax_st_start_button, 'Start')
        self.button_st_end = Button(self.ax_st_end_button, 'End')
        self.button_st_delete = Button(self.ax_st_delete_button, 'Delete')
        self.button_st_clean = Button(self.ax_st_clean_button, 'Clean')
        self.button_st_save = Button(self.ax_st_save_button, 'Save')
        self.button_close = Button(self.ax_close_button, 'Close')


        self.text_box_ploy_order.on_submit(lambda x: self._update_fit_order())
        self.button_st_start.on_clicked(self._start_st_fitting)
        self.button_st_end.on_clicked(self._end_st_fitting)
        self.button_st_delete.on_clicked(self._delete_st_fit)
        self.button_st_clean.on_clicked(self._clean_st_fit)
        self.button_st_save.on_clicked(self._save_all)


        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
        self.fig.canvas.mpl_connect('motion_notify_event', self._on_move)
        self.fig.canvas.mpl_connect('button_release_event', self._on_release)
        self.fig.canvas.mpl_connect('pick_event', self._pick_artist)

        self.button_close.on_clicked(lambda x: plt.close())
        
        plt.show()

    def _new_home(self):
        self.ax1.axis(self.ax1_axis)
        self.ax2.axis(self.ax2_axis)

        if self.successful:
            self.ax3.axis(self.ax3_axis)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()


    def _get_simple_std(self, every_nth=10):
        if self.image_type == 'SunpyMap':
            data_array = self.image_seq_prep[::every_nth].as_array()
        elif self.image_type == 'NDArray':
            data_array = self.image_seq_prep[:,:,:]
        return np.nanstd(data_array, axis=2)/np.nanmean(data_array, axis=2)
    
    def _update_time_index(self,which):
        if self.image_type == 'SunpyMap':
            if which == 'frame_index':
                if int(self.text_box_frame_index.text) > 0 and int(self.text_box_frame_index.text) < self.nt:
                    self.frame_index = int(self.text_box_frame_index.text)
                    self.text_box_time.set_val(self.image_seq_prep[self.frame_index].date.iso[:-4])
                else:
                    warnings.warn('Frame index out of range!')
            elif which == 'time':
                self.frame_index = np.argmin(np.abs(self.dates - Time(self.text_box_time.text)))
                self.text_box_frame_index.set_val(str(self.frame_index))
            self.ax1.get_images()[0].set_data(self.image_seq_prep[self.frame_index].data)
        elif self.image_type == 'NDArray':
            if which == 'frame_index':
                if int(self.text_box_frame_index.text) > 0 and int(self.text_box_frame_index.text) < self.nt:
                    self.frame_index = int(self.text_box_frame_index.text)
                    self.text_box_time.set_val(str(self.frame_index))
                else:
                    warnings.warn('Frame index out of range!')
            elif which == 'time':
                self.frame_index = int(self.text_box_time.text)
                self.text_box_frame_index.set_val(str(self.frame_index))
            self.ax1.get_images()[0].set_data(self.image_seq_prep[:,:,self.frame_index])

        if self.successful:
            try:
                self.ax3_timeline.set_xdata([self.frame_index, self.frame_index])
            except:
                pass
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def _update_line_width(self,):
        self.line_width = int(self.text_box_lw.text)

    def _update_asinha(self,val):
        self.plot_asinha = val
        self._update_norm()

    def _update_vmin_vmax(self,val):
        self.plot_vmin, self.plot_vmax = val
        self._update_norm()

    def _update_norm(self,):
        self.norm = ImageNormalize(vmin=self.plot_vmin, vmax=self.plot_vmax,stretch=AsinhStretch(self.plot_asinha))
        self.ax1.get_images()[0].set_norm(self.norm)

        if self.successful and not self.bg_remove_on:
            self.ax3.get_images()[0].set_norm(self.norm)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()


    def _start_selection(self,event):
        self.in_selection = True
        self.successful = False

    def _on_click(self,event):
        if self.in_selection:
            if event.inaxes in (self.ax1, self.ax2) and event.button == 1:
                self._add_points(event)
        elif self.in_fitting and event.button == 1 and event.inaxes == self.ax3:
            self._get_st_curve(event)

    def _on_move(self,event):
        if event.button == 1 and self.in_fitting and event.inaxes == self.ax3:
            self._get_st_curve(event)
        if self.in_moving and event.button == 2 and event.inaxes in (self.ax1, self.ax2):
            self._drag_points(event)
            

    def _on_release(self,event):
        if event.button == 2 and self.in_selection and self.in_moving:
            self._stop_drag_points(event) 

    def _pick_artist(self,event):
        if self.in_selection:
            if event.mouseevent.inaxes in (self.ax1, self.ax2) and event.mouseevent.button == 3 \
                and isinstance(event.artist, mlines.Line2D):
                self._delete_points(event)
            if event.mouseevent.inaxes in (self.ax1, self.ax2) and event.mouseevent.button == 2 \
                and isinstance(event.artist, mlines.Line2D):
                self._pick_points(event)

                self.in_moving = True
                
    def _add_points(self,event):
        self.select_x.append(event.xdata)
        self.select_y.append(event.ydata)
        cross_marker_ax1 = mlines.Line2D([event.xdata], [event.ydata], marker='x', color='white',
                                          markersize=6,linewidth=2, picker=True, pickradius=3)
        self.select_ax1_collection.append(self.ax1.add_line(cross_marker_ax1))                
        cross_marker_ax2 = mlines.Line2D([event.xdata], [event.ydata], marker='x', color='white',
                                          markersize=6,linewidth=2, picker=True, pickradius=3)
        self.select_ax2_collection.append(self.ax2.add_line(cross_marker_ax2))
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def _delete_points(self,event):
        if event.mouseevent.inaxes == self.ax1:
            picked_point_index = self.select_ax1_collection.index(event.artist)
        elif event.mouseevent.inaxes == self.ax2:
            picked_point_index = self.select_ax2_collection.index(event.artist)

        self.select_x.pop(picked_point_index)
        self.select_y.pop(picked_point_index)
        self.select_ax1_collection[picked_point_index].remove()
        self.select_ax2_collection[picked_point_index].remove()
        self.select_ax1_collection.pop(picked_point_index)
        self.select_ax2_collection.pop(picked_point_index)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def _pick_points(self,event):
        if event.mouseevent.inaxes == self.ax1:
            self._point_to_drag_index = self.select_ax1_collection.index(event.artist)
        elif event.mouseevent.inaxes == self.ax2:
            self._point_to_drag_index = self.select_ax2_collection.index(event.artist)

        self._points_to_drag = [self.select_ax1_collection[self._point_to_drag_index],
                                    self.select_ax2_collection[self._point_to_drag_index]]
        
        for point in self._points_to_drag:
            point.set_color('#81C7D4')
            point.set_alpha(0.8)
            
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def _drag_points(self,event):
        self._points_to_drag[0].set_xdata([event.xdata])
        self._points_to_drag[0].set_ydata([event.ydata])
        self._points_to_drag[1].set_xdata([event.xdata])
        self._points_to_drag[1].set_ydata([event.ydata])

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def _stop_drag_points(self,event):
        
        self.select_x[self._point_to_drag_index] = event.xdata
        self.select_y[self._point_to_drag_index] = event.ydata

        for point in self._points_to_drag:
            point.set_color('white')
            point.set_alpha(1)
        self.in_moving = False

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()


    def _make_slit(self,event):
        if self.select_x and self.select_y:
            self._clean_previous_slit()
            self._generate_slit_data()
            self._plot_slit_position()
            self._plot_slit_intensity()
            self.successful = True
        else:
            warnings.warn('Please select points first!')


    def _generate_slit_data(self):
        self.in_selection = False
        
        for ii in range(len(self.select_x)-1):
            pixels_idy_, pixels_idx_ = measure.profile._line_profile_coordinates((self.select_y[ii], self.select_x[ii]),
                                     (self.select_y[ii+1], self.select_x[ii+1]), linewidth=self.line_width)
            if ii == 0:
                self.pixels_idy, self.pixels_idx = pixels_idy_, pixels_idx_
            else:
                self.pixels_idy = np.vstack((self.pixels_idy,pixels_idy_[1:]))
                self.pixels_idx = np.vstack((self.pixels_idx,pixels_idx_[1:]))

        self.pixels_idy_center = np.nanmean(self.pixels_idy,axis=1)
        self.pixels_idx_center = np.nanmean(self.pixels_idx,axis=1)

        if self.image_type == 'SunpyMap':
            self.world_coord_center = self.map_wcs.pixel_to_world(self.pixels_idx_center,self.pixels_idy_center)
            self.world_coord_all = self.map_wcs.pixel_to_world(self.pixels_idx,self.pixels_idy) 

            world_coord_center_distance = []

            for ii, pixels_center_ in enumerate(self.world_coord_center):
                if ii == 0:
                    world_coord_center_distance.append(0*u.arcsec)
                else:
                    world_coord_center_distance.append(self.world_coord_center[ii].separation(self.world_coord_center[ii-1]).to(u.arcsec) + \
                                                    world_coord_center_distance[ii-1])
            self.world_coord_center_distance = u.Quantity(world_coord_center_distance).to_value(u.rad)*self.image_seq_prep[self.wcs_index].dsun
            self.world_coord_center_distance_interp = np.linspace(self.world_coord_center_distance[0],self.world_coord_center_distance[-1],
                                                                  len(self.world_coord_center_distance))
        
        elif self.image_type == 'NDArray':
            self.world_coord_center = None
            self.world_coord_all = None
            self.world_coord_center_distance = None

        self.pixel_distance = np.cumsum(np.sqrt(np.diff(self.pixels_idx_center)**2 + np.diff(self.pixels_idy_center)**2))
        self.pixel_distance = np.insert(self.pixel_distance,0,0)
        self.pixel_distance_interp = np.linspace(self.pixel_distance[0],self.pixel_distance[-1],len(self.pixel_distance))

        intensity = []
        for tt in range(self.nt):
            for ii in range(len(self.select_x)-1):
                if self.image_type == 'SunpyMap':
                    line = measure.profile_line(self.image_seq_prep[tt].data, (self.select_y[ii], self.select_x[ii]),
                                                (self.select_y[ii+1], self.select_x[ii+1]), linewidth=self.line_width,
                                                reduce_func=np.nanmean)
                elif self.image_type == 'NDArray':
                    line = skimage.measure.profile_line(self.image_seq_prep[:,:,tt], (self.select_y[ii], self.select_x[ii]),
                                                (self.select_y[ii+1], self.select_x[ii+1]), linewidth=self.line_width,
                                                reduce_func=np.nanmean)
                if ii == 0:
                    intensity_ = line
                else:
                    intensity_ = np.concatenate((intensity_,line[1:]))

            if self.image_type == 'SunpyMap':
                intensity_interp = np.interp(self.world_coord_center_distance_interp,self.world_coord_center_distance,intensity_)
            elif self.image_type == 'NDArray':
                intensity_interp = np.interp(self.pixel_distance_interp,self.pixel_distance,intensity_)
        
            intensity.append(intensity_interp)

        self.slit_intensity = u.Quantity(intensity).T

        if self.image_type == 'SunpyMap':
            self.spacetime_wcs = (TimeTableCoordinate(Time([map_.date for map_ in self.image_seq_prep]),
                                            physical_types="time",names="time") & 
                                QuantityTableCoordinate(self.world_coord_center_distance_interp.to(u.Mm),
                                                physical_types="length",names="distance")).wcs
            self.slit_cube = NDCube(self.slit_intensity,self.spacetime_wcs)

    def _plot_slit_position(self):
        boundary_x = np.concatenate((self.pixels_idx[:,0],self.pixels_idx[-1,1:],
                                     self.pixels_idx[-1::-1,-1],self.pixels_idx[0,-1::-1]))
        boundary_y = np.concatenate((self.pixels_idy[:,0],self.pixels_idy[-1,1:],
                                        self.pixels_idy[-1::-1,-1],self.pixels_idy[0,-1::-1]))
        
        self.slit_boundary_collection = []
        boundary_x_line2d_ax1 = mlines.Line2D(boundary_x, boundary_y, color='#58B2DC', lw=1, alpha=0.8)
        boundary_x_line2d_ax2 = mlines.Line2D(boundary_x, boundary_y, color='#58B2DC', lw=1, alpha=0.8)
        self.slit_boundary_collection.append(self.ax1.add_line(boundary_x_line2d_ax1))
        self.slit_boundary_collection.append(self.ax2.add_line(boundary_x_line2d_ax2))

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def _plot_slit_intensity(self):
        if self.image_type == 'SunpyMap':
            self.ax3.remove()
            self.ax3 = self.fig.add_axes([0.12, 0.08, 0.65, 0.36], projection=self.slit_cube.wcs)
            self.slit_cube.plot(axes=self.ax3, aspect='auto', cmap='magma', norm=self.norm)
        elif self.image_type == 'NDArray':
            self.ax3.imshow(self.slit_intensity, aspect='auto', cmap='magma', norm=self.norm, origin='lower')

        self.ax3.get_images()[0].format_cursor_data = lambda e: ""

        if self.bg_remove_on:
            self.bg_remove_on = False
            self.checkbutton_bg_remove.set_active(0)

        self.ax3_axis = self.ax3.axis()

        self.ax3_timeline = mlines.Line2D([self.frame_index, self.frame_index], [0, self.slit_intensity.shape[0]],
                                           color='white', linewidth=1, alpha=0.5, zorder = 2, ls = ':')
        self.ax3.add_line(self.ax3_timeline)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def _switch_bg_remove(self,label):
        if label == 'BG Remove' and self.successful:
            self.bg_remove_on = not self.bg_remove_on
            if self.bg_remove_on:
                self.slit_intensity_bg_removed = self.slit_intensity - cv2.GaussianBlur(self.slit_intensity,(1,29),0,10)
                
                self.ax3.get_images()[0].set_data(self.slit_intensity_bg_removed)
                self.ax3.get_images()[0].set_norm(ImageNormalize(interval=ZScaleInterval(),
                                                                 stretch=AsinhStretch(0.5)))
            else:
                self.ax3.get_images()[0].set_data(self.slit_intensity)
                self.ax3.get_images()[0].set_norm(self.norm)

            self.ax3.get_images()[0].format_cursor_data = lambda e: ""

            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()

    def _clean_points(self,event):
        self.select_x = []
        self.select_y = []
        for collection in self.select_ax1_collection:
            collection.remove()
        for collection in self.select_ax2_collection:
            collection.remove()
        self.select_ax1_collection = []
        self.select_ax2_collection = []
        
        try:
            self.ax3_timeline.remove()
            self.ax3_timeline = None
        except:
            pass

        if not self.in_selection:
            self._clean_previous_slit()
        else:
            self.in_selection = False

        self.successful = False

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def _clean_previous_slit(self):
        if self.successful:
            self.pixels_idy = None
            self.pixels_idx = None
            self.pixels_idy_center = None
            self.pixels_idx_center = None
            self.world_coord_center = None
            self.world_coord_all = None
            self.world_coord_center_distance = None
            self.pixel_distance = None
            self.slit_intensity = None
        try:
            for collection in self.slit_boundary_collection:
                collection.remove()
        except:
            pass
        try:
            self.ax3.get_images()[0].remove()
        except:
            pass

    def _start_st_fitting(self,event):
        if self.successful:
            self.in_selection = False
            self.in_fitting = True
            try:
                self.fit_params
                self.fit_xdata
                self.fit_curves
                self.fit_curves_collection
                if self.image_type == 'SunpyMap':
                    self.fit_params_world
                    self.fit_xdata_world
                    self.fit_curves_world
            except AttributeError:
                self.fit_params = []
                self.fit_xdata = []
                self.fit_curves = []
                self.fit_curves_collection = [] 
                if self.image_type == 'SunpyMap':
                    self.fit_params_world = []
                    self.fit_xdata_world = []
                    self.fit_curves_world = []

            try:
                self.latest_st_line.set_data([],[])
            except:
                self.latest_st_line = mlines.Line2D([], [], color='white', linewidth=1, alpha=1, zorder = 3)
                self.ax3.add_line(self.latest_st_line)

            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()

        else:
            warnings.warn('Please make a slit first!')

    def _get_st_curve(self,event):
        self.latest_st_line.set_xdata(np.append(self.latest_st_line.get_xdata(),event.xdata))
        self.latest_st_line.set_ydata(np.append(self.latest_st_line.get_ydata(),event.ydata))

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def _end_st_fitting(self,event):
        if self.in_fitting:
            self.in_fitting = False
            self._fit_spacetime()
            self._plot_st_fit()
        else:
            warnings.warn('Please start fitting first!')

    def _update_fit_order(self):
        self.fit_poly_order = int(self.text_box_ploy_order.text)

    def _fit_spacetime(self):
        xdata, ydata = self.latest_st_line.get_data()
        if self.checkbutton_reloc.get_status()[0]:
            xdata = np.round(xdata).astype(int)
            ydata = np.round(ydata).astype(int)
            ydata_new = np.zeros_like(ydata,dtype=np.float64)
            for ii in range(len(xdata)):
                window_half_size = 1
                window_max_arg = np.nanargmax(self.slit_intensity[ydata[ii] - window_half_size:ydata[ii] + window_half_size + 1,
                                            xdata[ii]]) + ydata[ii] - window_half_size
                try:                
                    max_quadratic_param = np.polyfit(np.arange(window_max_arg - window_half_size, window_max_arg + window_half_size + 1),
                        self.slit_intensity[np.arange(window_max_arg - window_half_size, window_max_arg + window_half_size + 1,dtype=int),
                                                    xdata[ii]],2)
                    ydata_new[ii] = -max_quadratic_param[1]/(2*max_quadratic_param[0])
                except:
                    ydata_new[ii] = window_max_arg
            fit_weights = None
            ydata = ydata_new
            xdata = xdata.astype(np.float64)
        else:
            fit_weights = None
        
        fit_param = np.polyfit(xdata,ydata,self.fit_poly_order,w=fit_weights)
        self.fit_params.append(fit_param)
        self.fit_xdata.append(xdata)
        fit_curve = np.polyval(fit_param,xdata)
        self.fit_curves.append(fit_curve)

        if self.image_type == 'SunpyMap':
            xdata_world, ydata_world = self.slit_cube.wcs.pixel_to_world(xdata,ydata)
            fit_param_world = np.polyfit((xdata_world - xdata_world[0]).to_value(u.s),
                                         ydata_world.to_value(u.km),self.fit_poly_order,w=fit_weights)
            print(f"Fit parameters, polynomial coefficients in decending orders: {fit_param_world}")
            self.fit_params_world.append(fit_param_world)
            fit_curve_world = np.polyval(fit_param_world,(xdata_world - xdata_world[0]).to_value(u.s))
            self.fit_curves_world.append(fit_curve_world)
            self.fit_xdata_world.append((xdata_world - xdata_world[0]).to_value(u.s))




    def _plot_st_fit(self):
        fit_line_2d = mlines.Line2D(self.fit_xdata[-1], self.fit_curves[-1], 
                            color='#81C7D4', linewidth=1, alpha=1, zorder = 3)
        self.fit_curves_collection.append(self.ax3.add_line(fit_line_2d))

        self.latest_st_line.set_xdata([])
        self.latest_st_line.set_ydata([])

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        

    def _delete_st_fit(self,event):
        if self.successful:
            if self.in_fitting:
                self.latest_st_line.set_xdata([])
                self.latest_st_line.set_ydata([])

                self.fig.canvas.draw_idle()
                self.fig.canvas.flush_events()
            else:
                try:
                    self.latest_st_line.set_xdata([])
                    self.latest_st_line.set_ydata([])
                    self.fit_curves_collection[-1].remove()
                    self.fit_curves_collection.pop()
                    self.fit_params.pop()
                    self.fit_curves.pop()
                    self.fit_xdata.pop()
                    self.fit_params_world.pop()
                    self.fit_curves_world.pop()
                    self.fit_xdata_world.pop()
                except:
                    pass

                self.fig.canvas.draw_idle()
                self.fig.canvas.flush_events()
        else:
            warnings.warn('Please make a slit first!')

    def _clean_st_fit(self,event):
        if self.successful:
            self.latest_st_line.set_xdata([])
            self.latest_st_line.set_ydata([])
            try:
                for collection in self.fit_curves_collection:
                    collection.remove()
            except:
                pass
            self.fit_params = []
            self.fit_curves = []
            self.fit_curves_collection = []
            self.fit_params_world = []
            self.fit_curves_world = []

            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
        else:
            warnings.warn('Please make a slit first!')

    def _save_all(self,event):
        self.save_dir = str(QFileDialog.getExistingDirectory(None, "Select Directory",
                         '/home/yjzhu/Solar/EIS_DKIST_SolO/sav/dynamic_fibrils/'))
        if self.successful:
            with h5py.File(os.path.join(self.save_dir,'slit_info.h5'), 'w') as hf:
                if self.bottom_left is not None:
                    hf.create_dataset('bottom_left', data=self.bottom_left.value)
                if self.top_right is not None:
                    hf.create_dataset('top_right', data=self.top_right.value)
                hf.create_dataset('wcs_index', data=self.wcs_index)
                if self.wcs_shift is not None:
                    hf.create_dataset('wcs_shift', data=self.wcs_shift.to_value(u.arcsec))
                hf.create_dataset('line_width', data=self.line_width)

                hf.create_dataset('select_x', data=self.select_x)
                hf.create_dataset('select_y', data=self.select_y)

                hf.create_dataset('pixels_idy', data=self.pixels_idy)
                hf.create_dataset('pixels_idx', data=self.pixels_idx)
                hf.create_dataset('pixels_idy_center', data=self.pixels_idy_center)
                hf.create_dataset('pixels_idx_center', data=self.pixels_idx_center)
                hf.create_dataset('pixel_distance', data=self.pixel_distance)
                hf.create_dataset('pixel_distance_interp', data=self.pixel_distance_interp)

                if self.image_type == 'SunpyMap':
                    hf.create_dataset('world_coord_center_distance', data=self.world_coord_center_distance.to_value(u.km))
                    hf.create_dataset('world_coord_center_distance_interp', data=self.world_coord_center_distance_interp.to_value(u.km))
                    hf.create_dataset('time', data=Time([map_.date for map_ in self.image_seq_prep]).mjd)

                hf.create_dataset('slit_intensity', data=self.slit_intensity)

            # if self.image_type == 'SunpyMap':
            #     write_table_hdf5(self.world_coord_center.to_table(), os.path.join(self.save_dir, 'slit_info.h5'),
            #                      'world_coord_center', append=True)
            #     write_table_hdf5(self.world_coord_all.to_table(), os.path.join(self.save_dir, 'slit_info.h5'),
            #                         'world_coord_all', append=True)

            with h5py.File(os.path.join(self.save_dir, 'spacetime_fit.h5'), 'w') as hf:
                hf.create_dataset('fit_params', data=np.asarray(self.fit_params))

                for ii, array in enumerate(self.fit_xdata):
                    hf.create_dataset(f'fit_xdata_{ii}', data=array)
                for ii, array in enumerate(self.fit_curves):
                    hf.create_dataset(f'fit_curves_{ii}', data=array)

                if self.image_type == 'SunpyMap':
                    hf.create_dataset('fit_params_world', data=np.asarray(self.fit_params_world))
                    
                    for ii, array in enumerate(self.fit_xdata_world):
                        hf.create_dataset(f'fit_xdata_world_{ii}', data=array)
                    for ii, array in enumerate(self.fit_curves_world):
                        hf.create_dataset(f'fit_curves_world_{ii}', data=array)
                
                hf.create_dataset('fit_number', data=len(self.fit_xdata))


            bbox_to_save = Bbox([[0,0],[0.79,0.99]])
            bbox_to_save = bbox_to_save.transformed(self.fig.transFigure).transformed(self.fig.dpi_scale_trans.inverted())
            self.fig.savefig(os.path.join(self.save_dir, 'slit_plot.png'), dpi=300,
                             bbox_inches=bbox_to_save)
            print(f'Data saved successfully in {self.save_dir}')

    def generate_all_slit_preview(self, x_num=9, y_num=9, angle_num=4, length=15,
                                  line_width=5, ncpu=None, save_path=None):
        
        self.simple_std = self._get_simple_std(every_nth=1)
        
        if self.image_type == 'SunpyMap':
            data_shape = self.image_seq_prep[0].data.shape
        elif self.image_type == 'NDArray':
            data_shape = self.image_seq_prep.shape

        xcen_array = np.linspace(0,data_shape[1],x_num+2)[0:-1]
        ycen_array = np.linspace(0,data_shape[0],y_num+2)[0:-1]

        args_array = []

        for xcen in xcen_array:
            for ycen in ycen_array:
                args_array.append((xcen, ycen, angle_num, length, line_width, save_path))
        
        # # test one 
        # self._generate_single_slit_work(*args_array[36])

        if ncpu is None:
            ncpu = os.cpu_count()


        with multiprocessing.Pool(ncpu) as pool:
            pool.starmap(self._generate_single_slit_work, args_array)

        # with ProcessPoolExecutor(max_workers=ncpu) as executor:
        #     executor.map(self._generate_single_slit_work, args_array)

    def _generate_single_slit_work(self, xcen, ycen, angle_num, length, line_width,
                                   save_path):
        for angle in np.linspace(0, np.pi, angle_num+1)[:-1]:
            x_select = np.array([xcen - length/2*np.sin(angle), xcen + length/2*np.sin(angle)])
            y_select = np.array([ycen - length/2*np.cos(angle), ycen + length/2*np.cos(angle)])

            pixels_idy, pixels_idx = measure.profile._line_profile_coordinates((y_select[0], x_select[0]),
                                        (y_select[1], x_select[1]), linewidth=line_width)
            
            pixels_idy_center = np.nanmean(pixels_idy,axis=1)
            pixels_idx_center = np.nanmean(pixels_idx,axis=1)

            if self.image_type == 'SunpyMap':
                world_coord_center = self.map_wcs.pixel_to_world(pixels_idx_center,pixels_idy_center)
                world_coord_all = self.map_wcs.pixel_to_world(pixels_idx,pixels_idy) 

                world_coord_center_distance = []

                for ii, pixels_center_ in enumerate(world_coord_center):
                    if ii == 0:
                        world_coord_center_distance.append(0*u.arcsec)
                    else:
                        world_coord_center_distance.append(world_coord_center[ii].separation(world_coord_center[ii-1]).to(u.arcsec) + \
                                                        world_coord_center_distance[ii-1])
                world_coord_center_distance = u.Quantity(world_coord_center_distance).to_value(u.rad)*self.image_seq_prep[self.wcs_index].dsun
                world_coord_center_distance_interp = np.linspace(world_coord_center_distance[0],world_coord_center_distance[-1],
                                                                    len(world_coord_center_distance))
                
            elif self.image_type == 'NDArray':
                world_coord_center = None
                world_coord_all = None
                world_coord_center_distance = None
            
            pixel_distance = np.cumsum(np.sqrt(np.diff(pixels_idx_center)**2 + np.diff(pixels_idy_center)**2))
            pixel_distance = np.insert(pixel_distance,0,0)
            pixel_distance_interp = np.linspace(pixel_distance[0],pixel_distance[-1],len(pixel_distance))

            intensity = []
            for tt in range(self.nt):
                if self.image_type == 'SunpyMap':
                    line = measure.profile_line(self.image_seq_prep[tt].data, (y_select[0], x_select[0]),
                                                (y_select[1], x_select[1]), linewidth=line_width,
                                                reduce_func=np.nanmean)
                elif self.image_type == 'NDArray':
                    line = skimage.measure.profile_line(self.image_seq_prep[:,:,tt], (y_select[0], x_select[0]),
                                                (y_select[1], x_select[1]), linewidth=line_width,
                                                reduce_func=np.nanmean)
                    
                intensity_ = line
            
                if self.image_type == 'SunpyMap':
                    intensity_interp = np.interp(world_coord_center_distance_interp,world_coord_center_distance,intensity_)
                elif self.image_type == 'NDArray':
                    intensity_interp = np.interp(pixel_distance_interp,pixel_distance,intensity_)


                intensity.append(intensity_interp)
            
            slit_intensity = u.Quantity(intensity).T
            slit_intensity = slit_intensity - cv2.GaussianBlur(slit_intensity,(1,15),0,5)


            if self.image_type == 'SunpyMap':
                spacetime_wcs = (TimeTableCoordinate(Time([map_.date for map_ in self.image_seq_prep]),
                                                physical_types="time",names="time") & 
                                QuantityTableCoordinate(world_coord_center_distance_interp.to(u.Mm),
                                                physical_types="length",names="distance")).wcs
                slit_cube = NDCube(slit_intensity,spacetime_wcs)

                fig = plt.figure(figsize=(7,6), layout='constrained')
                gs = fig.add_gridspec(2,2)

                ax1 = fig.add_subplot(gs[0,0], projection=self.map_wcs)
                ax2 = fig.add_subplot(gs[0,1], projection=self.map_wcs)
                ax3 = fig.add_subplot(gs[1,:], projection=slit_cube.wcs)
                
                ax1.imshow(self.image_seq_prep[self.wcs_index].data, cmap='magma',
                           norm=self.norm, origin='lower')
                
                ax2.imshow(self.simple_std, cmap='magma', origin='lower',
                           norm=ImageNormalize(vmin=np.nanpercentile(self.simple_std,1),
                                               vmax=np.nanpercentile(self.simple_std,99),
                                               stretch=AsinhStretch(0.5)))
                
                boundary_x = np.concatenate((pixels_idx[:,0],pixels_idx[-1,1:],
                                            pixels_idx[-1::-1,-1],pixels_idx[0,-1::-1]))
                
                boundary_y = np.concatenate((pixels_idy[:,0],pixels_idy[-1,1:],
                                            pixels_idy[-1::-1,-1],pixels_idy[0,-1::-1]))
                
                boundary_x_line2d_ax1 = mlines.Line2D(boundary_x, boundary_y, color='#58B2DC', lw=1, alpha=0.8)
                boundary_x_line2d_ax2 = mlines.Line2D(boundary_x, boundary_y, color='#58B2DC', lw=1, alpha=0.8)

                ax1.add_line(boundary_x_line2d_ax1)
                ax2.add_line(boundary_x_line2d_ax2)

                ax3.imshow(slit_intensity, aspect='auto', cmap='magma', norm=ImageNormalize(interval=ZScaleInterval(),
                                                                                           stretch=AsinhStretch(0.5)),
                           origin='lower')
                
                fig.savefig(os.path.join(save_path,f'slit_{int(xcen)}_{int(ycen)}_{int(angle*180/np.pi)}.png'), dpi=300)
                plt.close(fig)

            if self.image_type == 'NDArray':
                fig = plt.figure(figsize=(7,6), layout='constrained')
                gs = fig.add_gridspec(2,2)

                ax1 = fig.add_subplot(gs[0,0])
                ax2 = fig.add_subplot(gs[0,1])
                ax3 = fig.add_subplot(gs[1,:])
                
                ax1.imshow(self.image_seq_prep[:,:,self.wcs_index], cmap='magma',
                           norm=self.norm, origin='lower')
                
                ax2.imshow(self.simple_std, cmap='magma', origin='lower',
                           norm=ImageNormalize(vmin=np.nanpercentile(self.simple_std,1),
                                               vmax=np.nanpercentile(self.simple_std,99),
                                               stretch=AsinhStretch(0.5)))
                
                boundary_x = np.concatenate((pixels_idx[:,0],pixels_idx[-1,1:],
                                            pixels_idx[-1::-1,-1],pixels_idx[0,-1::-1]))
                
                boundary_y = np.concatenate((pixels_idy[:,0],pixels_idy[-1,1:],
                                            pixels_idy[-1::-1,-1],pixels_idy[0,-1::-1]))
                
                boundary_x_line2d_ax1 = mlines.Line2D(boundary_x, boundary_y, color='#58B2DC', lw=1, alpha=0.8)
                boundary_x_line2d_ax2 = mlines.Line2D(boundary_x, boundary_y, color='#58B2DC', lw=1, alpha=0.8)

                ax1.add_line(boundary_x_line2d_ax1)
                ax2.add_line(boundary_x_line2d_ax2)

                ax3.imshow(slit_intensity, aspect='auto', cmap='magma', norm=ImageNormalize(interval=ZScaleInterval(),
                                                                                           stretch=AsinhStretch(0.5)),
                           origin='lower')
                
                fig.savefig(os.path.join(save_path,f'slit_{int(xcen)}_{int(ycen)}_{int(angle*180/np.pi)}.png'), dpi=300)
                plt.close(fig)
  

            

if  __name__ == "__main__":
    from glob import glob

    # eui_files = sorted(glob("/home/yjzhu/Solar/EIS_DKIST_SolO/src/EUI/HRI/euv174/20221024/coalign_step/*.fits"))
    # eui_files = sorted(glob("/home/yjzhu/Solar/EIS_DKIST_SolO/src/EUI/HRI/euv174/20221020/coalign_step_boxcar/*.fits"))
    # eui_files = sorted(glob("/home/yjzhu/Solar/EIS_DKIST_SolO/src/EUI/HRI/euv174/20221026/coalign_step_boxcar/*.fits"))
    # eui_files = sorted(glob("/home/yjzhu/Downloads/JSOC_20240919_003607/*.fits"))


    # eui_map_seq_coalign = MapSequenceCoalign(sunpy.map.Map(eui_files[:])) 

    # eui_map_seq_coalign_unsharp = []


    # for map in eui_map_seq_coalign.maps:
    #     eui_map_seq_coalign_unsharp.append(sunpy.map.Map(unsharp_mask(map.data, radius=10, amount=1),
    #                                                      map.meta))
    
    # eui_map_seq_coalign_unsharp = MapSequenceCoalign(sunpy.map.Map(eui_map_seq_coalign_unsharp))

    # eui_map_seq_coalign = np.ones((50,50,30))

    # for ii in range(30):
    #     eui_map_seq_coalign[ii:ii+2,ii:ii+2,ii] = np.ones((2,2))*10

    dkist_cube = np.ones((200,200,210))
    dkist_files = sorted(glob("/home/yjzhu/Solar/EIS_DKIST_SolO/sav/DKIST_of/BJOLO/33_npy/*.npy"))

    for ii in range(210):
        # dkist_cube[:,:,ii] = np.load(dkist_files[ii])[300+32:500+32,200+32:400+32]
        dkist_cube[:,:,ii] = np.load(dkist_files[ii])[300+32:500+32,400+32:600+32]

    #slit_pick = SlitPick(eui_map_seq_coalign)
    slit_pick = SlitPick(dkist_cube)

    # slit_pick(wcs_index=0, img_wow=True) #1024 east 1
    # slit_pick(bottom_left=[500,600]*u.pix, top_right=[670,760]*u.pix,wcs_index=0, img_wow=False) #1024 east 1
    # slit_pick(bottom_left=[500,600]*u.pix, top_right=[670,760]*u.pix,wcs_index=181, img_wow=False, init_gui=False) #1024 east 1 all test
    # slit_pick.generate_all_slit_preview(x_num=9, y_num=9, angle_num=4, length=25, line_width=5, save_path='/home/yjzhu/Solar/EIS_DKIST_SolO/sav/dynamic_fibrils/east_1_generate_all_test/')
    
    slit_pick(wcs_index=0, img_wow=False, init_gui=False) #1024 east 1

    slit_pick.generate_all_slit_preview(x_num=2, y_num=2, angle_num=4, length=50, line_width=5, save_path='/home/yjzhu/Downloads/', ncpu=1)
    # slit_pick(bottom_left=[1600,300]*u.pix, top_right=[2048,700]*u.pix,wcs_index=0,) #1026 west
    # slit_pick(bottom_left=[850,800]*u.pix, top_right=[1050,1000]*u.pix,wcs_index=0)
    # slit_pick(bottom_left=[700,550]*u.pix, top_right=[900,750]*u.pix,wcs_index=0)
    # slit_pick(bottom_left=[300,750]*u.pix, top_right=[500,950]*u.pix,wcs_index=0) # 1020 east
    # slit_pick(bottom_left=[300,750]*u.pix, top_right=[500,950]*u.pix,wcs_index=0) # 1020 west
    # slit_pick(bottom_left=[250,750]*u.pix, top_right=[420,920]*u.pix,wcs_index=0)
        
    # files = sorted(glob('/home/yjzhu/Solar/EIS_DKIST_SolO/sav/DKIST_of/BJOLO/33_npy/*.npy'))
    # files = sorted(glob('/home/yjzhu/Solar/EIS_DKIST_SolO/src/IRIS/20221024/1904/sji_1400_wow_to_vbi/*vbi*.npy'))
    # files = sorted(glob('/home/yjzhu/Solar/EIS_DKIST_SolO/src/AIA/20221024/171/ARcutout_lvl15/to_vbi/*vbi*.npy'))
    # data_cube = np.zeros((120,120,len(files)))
    # data_cube = np.zeros((200,200,len(files)))

    # for ii in range(len(files)):
        # print(np.load(files[ii]).shape)
        # data_cube[:,:,ii] = (np.load(files[ii]))[380:500,400:520]
        # data_cube[:,:,ii] = (np.load(files[ii]))[300+32:500+32,200+32:400+32]

    # data_cube = data_cube/np.nanmean(data_cube, axis=-1)[:,:,np.newaxis]
    # slit_pick = SlitPick(data_cube)
    # slit_pick()



        
                


        


