import sunpy 
import sunpy.map
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from astropy.visualization import (ImageNormalize, AsinhStretch,
                                   ) 
from IPython.display import HTML, display

# from mpl_animators import ArrayAnimatorWCS


class SunBlinker():
    def __init__(self, map1, map2, reproject=False, fps=5, figsize=(5,5),
                 norm1=None, norm2=None) -> None:
        self.map1 = map1
        if reproject:
            self.map2 = map2.reproject_to(map1.wcs)
        else:
            self.map2 = map2
        self.fps = fps
        self.figsize = figsize

        self._init_plot()

        if norm1 is None:
            self.norm1 = self.map1.plot_settings['norm']
        else:
            self.norm1 = norm1

        if norm2 is None:
            self.norm2 = self.map2.plot_settings['norm']
        else:
            self.norm2 = norm2

        self.anim = FuncAnimation(self.fig, self._update_plot, interval=1000/self.fps, blit=True,frames=2,
                                  repeat=True)
        
        self.anim_html = HTML(self.anim.to_jshtml())

        self.fig.clf()
        plt.close()

        display(self.anim_html)


    
    def _init_plot(self):
        self.fig = plt.figure(figsize=self.figsize,constrained_layout=True)
        self.ax = self.fig.add_subplot(111, projection=self.map1)
        self.im = self.map1.plot(axes=self.ax)
        self.ax.set_title(None)

    def _update_plot(self,i):
        # self.ax.clear()   
        if i == 0:
            self.im.set_array(self.map1.data)
            self.im.set_norm(self.norm1)
            self.im.set_cmap(self.map1.plot_settings['cmap'])
        else:
            self.im.set_array(self.map2.data)
            self.im.set_norm(self.norm2)
            self.im.set_cmap(self.map2.plot_settings['cmap'])
        
        return [self.im]









