import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, Button, CheckButtons
import sunpy
import sunpy.map
from sunpy.map import GenericMap, MapSequence
from map_coalign import MapSequenceCoalign
import warnings




class SlitPick:
    def __init__(self, image_seq):
        if isinstance(image_seq, GenericMap):
            self.image_seq = MapSequenceCoalign(image_seq)
            self.image_type = 'SunpyMap'
            self.ny, self.nx = self.image_seq.maps[0].data.shape
            self.nt = len(self.image_seq.maps)
        elif isinstance(image_seq, MapSequence):
            self.image_seq = MapSequenceCoalign(image_seq.maps)
            self.image_type = 'SunpyMap'
            self.ny, self.nx = self.image_seq.maps[0].data.shape
            self.nt = len(self.image_seq.maps)
        elif isinstance(image_seq, np.ndarray):
            self.image_seq = image_seq
            self.image_type = 'NDArray'
            self.ny, self.nx, self.nt = self.image_seq.shape

        
    def __call__(self, bottom_left=None, top_right=None, wcs_index=0, wcs_shift=None, norm=None):

        self.bottom_left = bottom_left
        self.top_right = top_right
        self.wcs_index = wcs_index
        self.wcs_shift = wcs_shift
        self.norm = norm

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

        elif self.image_type == 'NDArray':
            if bottom_left is not None and top_right is not None:
                self.image_seq_prep = self.image_seq[bottom_left[1]:top_right[1]+1, bottom_left[0]:top_right[0]+1]
            else:
                self.image_seq_prep = self.image_seq

            if wcs_shift is not None:
                warnings.warn('wcs_shift is not supported for NDArray input')
            
            self.projection = None

        self._init_gui()

    
    def _init_gui(self):

        self.fig = plt.figure(figsize=(8,6),layout='constrained')
        self.gs = self.fig.add_gridspec(2, 3)
        self.ax1 = self.fig.add_subplot(self.gs[0,:1], projection=self.projection)
        self.ax2 = self.fig.add_subplot(self.gs[0,1:2], projection=self.projection)
        self.ax3 = self.fig.add_subplot(self.gs[1,:2], projection=None)

        self.ax2.sharex(self.ax1)
        self.ax2.sharey(self.ax1)

        if self.image_type == 'SunpyMap':
            self.ax1.imshow(self.image_seq_prep[self.wcs_index].data, cmap='magma', norm=self.norm)
            self.ax1.set_xlabel('Solar-X [arcsec]')
            self.ax1.set_ylabel('Solar-Y [arcsec]')
            self.ax2.set_xlabel('Solar-X [arcsec]')
            self.ax2.set_ylabel(' ') 
        elif self.image_type == 'NDArray':
            self.ax1.imshow(self.image_seq_prep[self.wcs_index], cmap='magma', norm=self.norm)
            self.ax1.set_xlabel('Pixel-X')
            self.ax1.set_ylabel('Pixel-Y')
            self.ax2.set_xlabel('Pixel-X')

        self.ax2.imshow(self._get_simple_std(), cmap='magma')

        gs_panel_sub1 = self.gs[0,2:].subgridspec(5,1)
        self.ax_text_frame_index = self.fig.add_subplot(gs_panel_sub1[1,:])
        self.ax_text_frame_index.set_title('Frame Index')

        self.ax_text_time = self.fig.add_subplot(gs_panel_sub1[3,:])
        self.ax_text_time.set_title('Time')

        gs_panel_sub2 = self.gs[1,2:].subgridspec(5,2)
        self.ax_start_button = self.fig.add_subplot(gs_panel_sub2[0,0])

        self.ax_end_button = self.fig.add_subplot(gs_panel_sub2[0,1])

        self.ax_clean_button = self.fig.add_subplot(gs_panel_sub2[1,0])

        self.ax_close_button = self.fig.add_subplot(gs_panel_sub2[1,1])

        self.text_box_frame_index = TextBox(self.ax_text_frame_index, None, initial=str(self.wcs_index),
                                            textalignment='center')

        if self.image_type == 'SunpyMap':
            self.text_box_time = TextBox(self.ax_text_time, None, initial=str(self.image_seq_prep[self.wcs_index].date.iso[:-4]),
                                         textalignment='center')
        elif self.image_type == 'NDArray':
            self.text_box_time = TextBox(self.ax_text_time, 'Time', initial=str(self.wcs_index),
                                         textalignment='center')

        self.button_start = Button(self.ax_start_button, 'Start')
        self.button_end = Button(self.ax_end_button, 'End')
        self.button_clean = Button(self.ax_clean_button, 'Clean')
        self.button_close = Button(self.ax_close_button, 'Close')



        plt.show()

    def _get_simple_std(self, every_nth=10):
        if self.image_type == 'SunpyMap':
            data_array = self.image_seq_prep[::every_nth].as_array()
        elif self.image_type == 'NDArray':
            data_array = self.image_seq_prep[::every_nth]
        return np.nanstd(data_array, axis=2)/np.nanmean(data_array, axis=2)
    

if  __name__ == "__main__":
    from glob import glob
    import astropy.units as u
    eui_files = sorted(glob("/home/yjzhu/Solar/EIS_DKIST_SolO/src/EUI/HRI/euv174/20221024/coalign_step_boxcar/*.fits"))
    eui_map_seq_coalign = MapSequenceCoalign(sunpy.map.Map(eui_files[:])) 

    slit_pick = SlitPick(eui_map_seq_coalign)
    slit_pick(bottom_left=[500,600]*u.pix, top_right=[670,760]*u.pix, wcs_index=0)
        



        
                


        


