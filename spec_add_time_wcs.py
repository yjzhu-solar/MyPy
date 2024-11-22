import sunpy
import sunpy.map
from astropy.wcs import WCS
from astropy.time import Time
import astropy.units as u
from sunpy.time import parse_time
from ndcube import NDCube
import numpy as np
from ndcube.extra_coords import TimeTableCoordinate


def add_temporal_axis(map, date_beg=None, date_end=None,
                      scan_direction='west_east'):
    map_meta = map.meta.copy()
    if date_beg is None:
        try:
            date_beg = map_meta['date_beg']
        except:
            raise ValueError("date_beg is not provided in the meta or as an argument.")
    date_beg = parse_time(date_beg)

    if date_end is None:
        try:
            date_end = map_meta['date_end']
        except:
            raise ValueError("date_end is not provided in the meta or as an argument.")
    date_end = parse_time(date_end)
    
    # rotation_matrix_0 = map.rotation_matrix.copy()

    # for key in ['CROTA1', 'CROTA2', 'PC1_1', 'PC1_2', 'PC2_1', 'PC2_2',
    #             'CD1_1', 'CD1_2', 'CD2_1', 'CD2_2']:
    #     map_meta.pop(key, None)
    
    # map_meta['DATEREF'] = date_beg.isot
    # map_meta['DATE_BEG'] = date_beg.isot
    # map_meta['DATE_END'] = date_end.isot

    # map_meta['NAXIS'] = 3
    # map_meta['NAXIS3'] = 1
    # map_meta['CTYPE3'] = 'TIME'
    # map_meta['CNAME3'] = 'Time (Degenerate Dimension)'
    # map_meta['CUNIT3'] = 's'

    # map_meta['CRPIX3'] = 1
    # map_meta['CDELT3'] = 1
    # map_meta['PC3_3'] = 1
    # map_meta['CRVAL3'] = (date_end - date_beg).to_value(u.s)

    # if scan_direction == 'west_east':
    #     map_meta['PC3_1'] = -(date_end - date_beg).to_value(u.s)/map_meta['NAXIS1']
    # elif scan_direction == 'east_west':
    #     map_meta['PC3_1'] = (date_end - date_beg).to_value(u.s)/map_meta['NAXIS1']
    
    # map_meta['COMMENT'] = "The third axis is a degenerate dimension representing time added manually."

    # map_meta['PC1_1'] = rotation_matrix_0[0, 0]
    # map_meta['PC1_2'] = rotation_matrix_0[0, 1]
    # map_meta['PC2_1'] = rotation_matrix_0[1, 0]
    # map_meta['PC2_2'] = rotation_matrix_0[1, 1]

    # new_map = sunpy.map.Map(np.array([map.data]), map_meta)
    # new_map = NDCube(np.array([map.data]), wcs=WCS(map_meta),
    #                 )

    new_map = NDCube(map.data, wcs=WCS(map.meta),
                     meta=map.meta)

    if scan_direction == 'west_east':
        timestamps = date_end - np.linspace(0, map_meta['NAXIS1']-1, map_meta['NAXIS1'])*u.s*(date_end - date_beg).to_value(u.s)/map_meta['NAXIS1']
    elif scan_direction == 'east_west':
        timestamps = date_beg + np.linspace(0, map_meta['NAXIS1']-1, map_meta['NAXIS1'])*u.s*(date_end - date_beg).to_value(u.s)/map_meta['NAXIS1']
    timestamps = TimeTableCoordinate(timestamps, names="Time", reference_time=date_beg)
    new_map.extra_coords.add('time', (1,), timestamps)
    new_map = NDCube(map.data, wcs=new_map.combined_wcs,
                     meta=new_map.meta)

    return new_map




if __name__ == "__main__":
    import eispac
    import matplotlib.pyplot as plt
    from collections import defaultdict
    # map = sunpy.map.Map("~/Solar/EIS_DKIST_SolO/src/EIS/DHB_007_v2/20221025T0023/sunpymaps/eis_195_intmap_shift.fits")
    eis_fitres = eispac.read_fit("/home/yjzhu/Solar/EIS_DKIST_SolO/src/EIS/DHB_007_v2/20221025T0023/eis_20221025_002341.fe_12_195_119.1c-0.fit.h5")
    map = eis_fitres.get_map()
    new_map = add_temporal_axis(map)


    time_ticks = Time('2022-10-25T00:30') + np.linspace(0,40,3)*u.min
    time_ticks_from_beg = time_ticks - Time(new_map.meta['date_beg'])
    print(time_ticks_from_beg.to_value(u.s))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection=new_map.wcs)
    new_map.plot(axes=ax, aspect=1/4)

    ax.coords[2].set_ticks(time_ticks_from_beg.to(u.s))
    


    fig.canvas.draw()

    # ax.coords[2].ticklabels.text['t'] = ['1','2','3']
    # ax.coords[2].ticklabels.text['b'] = ['1','2','3']
    ax.coords[2].set_ticklabel(text=['1','2','3'])
    # ax.coords[2].ticklabels.text = 
    # print(ax.coords[2].ticklabels.text)
    # ax.coords[2].ticklabels.clear()
    # # ax.coords[2]._update_ticks()
    # ax.coords[2].ticklabels.add('t', 1330*u.s, 0, '1', 0.5, [0.5,0.5])
    fig.canvas.draw()
    print(ax.coords[2].ticklabels.text)
    plt.show()

    