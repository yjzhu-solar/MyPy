import sunpy 
import sunpy.map
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from astropy.visualization import (ImageNormalize, AsinhStretch,
                                   ) 
from IPython.display import HTML, display

# from mpl_animators import ArrayAnimatorWCS


#: Backends that cannot open a window at all, so there is nothing to show.
_NON_GUI_BACKENDS = frozenset({'agg', 'cairo', 'pdf', 'pgf', 'ps', 'svg', 'template'})

#: Notebook backends, which get the HTML animation rather than a window.
_NOTEBOOK_BACKENDS = ('inline', 'nbagg', 'ipympl', 'widget')


def _backend_can_show():
    """
    Whether the active matplotlib backend can open a figure window.

    Testing for one hard-coded backend name ('qtagg') missed every other GUI
    backend -- macosx, tkagg, gtk4agg, and even the 'QtAgg' spelling, since the
    comparison was case-sensitive -- and silently closed the figure instead of
    showing it.

    Returns
    -------
    bool
    """
    backend = matplotlib.get_backend().lower()
    if backend in _NON_GUI_BACKENDS:
        return False
    return not any(nb in backend for nb in _NOTEBOOK_BACKENDS)


class SunBlinker():
    def __init__(self, map1, map2, reproject=False, fps=5, figsize=(5,5),
                 norm1=None, norm2=None, save_fname=None) -> None:
        self.map1 = map1
        if reproject:
            self.map2 = map2.reproject_to(map1.wcs)
        else:
            self.map2 = map2
        self.fps = fps
        self.figsize = figsize

        if norm1 is None:
            self.norm1 = self.map1.plot_settings['norm']
        else:
            self.norm1 = norm1

        if norm2 is None:
            self.norm2 = self.map2.plot_settings['norm']
        else:
            self.norm2 = norm2

        self._init_plot()

        self.anim = FuncAnimation(self.fig, self._update_plot, interval=1000/self.fps, blit=True,frames=2,
                                  repeat=True)
        
        self.anim_html = HTML(self.anim.to_jshtml())

        if save_fname is not None:
            self.anim.save(save_fname, writer='imagemagick', fps=self.fps)

        if _backend_can_show():
            plt.show()
        else:
            self.fig.clf()
            plt.close()

            display(self.anim_html)


    
    def _init_plot(self):
        self.fig = plt.figure(figsize=self.figsize)
        self.ax = self.fig.add_subplot(111, projection=self.map1)
        if 'aspect' in self.map1.plot_settings.keys():
            self.ax.set_aspect(self.map1.plot_settings['aspect'])
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
    

class ImageBlinker():
    def __init__(self, image1, image2, fps=5, figsize=(5,5),
                 norm1=None, norm2=None, aspect=1,
                 save_fname=None, cmap1=None,
                 cmap2=None,**kwargs) -> None:
        self.image1 = image1
        self.image2 = image2
        self.cmap1 = cmap1
        self.cmap2 = cmap2
        self.fps = fps
        self.figsize = figsize
        self.aspect = aspect    
        self.kwargs = kwargs
        
        if norm1 is None:
            self.norm1 = ImageNormalize(vmin=np.nanpercentile(image1, 0.1), 
                                        vmax=np.nanpercentile(image1, 99.9),
                                        stretch=AsinhStretch())
        else:
            self.norm1 = norm1

        if norm2 is None:
            self.norm2 = ImageNormalize(vmin=np.nanpercentile(image2, 0.1), 
                                        vmax=np.nanpercentile(image2, 99.9),
                                        stretch=AsinhStretch())
        else:
            self.norm2 = norm2
        
        self._init_plot()

        self.anim = FuncAnimation(self.fig, self._update_plot, interval=1000/self.fps, blit=True,frames=2,
                                  repeat=True)
        
        self.anim_html = HTML(self.anim.to_jshtml())

        if save_fname is not None:
            self.anim.save(save_fname, writer='imagemagick', fps=self.fps)

        if _backend_can_show():
            plt.show()
        else:
            self.fig.clf()
            plt.close()

            display(self.anim_html)


    def _init_plot(self):
        self.fig = plt.figure(figsize=self.figsize)
        self.ax = self.fig.add_subplot(111)
        self.im = self.ax.imshow(self.image1, norm=self.norm1, origin='lower',
                                 aspect=self.aspect, cmap=self.cmap1, **self.kwargs)

    def _update_plot(self,i):
        # self.ax.clear()   
        if i == 0:
            self.im.set_array(self.image1)
            self.im.set_norm(self.norm1)
            if self.cmap1 is not None:
                self.im.set_cmap(self.cmap1)
        else:
            self.im.set_array(self.image2)
            self.im.set_norm(self.norm2)
            if self.cmap2 is not None:
                self.im.set_cmap(self.cmap2)
        
        return [self.im]









