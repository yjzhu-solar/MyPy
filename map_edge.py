# get map edge SkyCoord
from itertools import chain
import sunpy.map
from astropy import units as u

def get_map_edge_coords(map, step=1):
    map_edges = sunpy.map.map_edges(map)

    x_pix = []
    y_pix = []

    if map_edges[1].shape[0] % step != 0:
        iter_1 = chain(range(0, map_edges[1].shape[0], step), [map_edges[1].shape[0]-1])
    else:
        iter_1 = range(0, map_edges[1].shape[0], step)
    for ii in iter_1:
        x_pix.append(map_edges[1][ii,0].value)
        y_pix.append(map_edges[1][ii,1].value)

    if map_edges[3].shape[0] % step != 0:
        iter_3 = chain(range(0, map_edges[3].shape[0], step), [map_edges[3].shape[0]-1])
    else:
        iter_3 = range(0, map_edges[3].shape[0], step)

    for ii in iter_3:
        x_pix.append(map_edges[3][ii,0].value)
        y_pix.append(map_edges[3][ii,1].value)

    if map_edges[0].shape[0] % step != 0:
        iter_0 = chain(range(map_edges[0].shape[0]-1, -1, -step), [0])
    else:
        iter_0 = range(map_edges[0].shape[0]-1, -1, -step)

    for ii in iter_0:
        x_pix.append(map_edges[0][ii,0].value)
        y_pix.append(map_edges[0][ii,1].value)

    iter_2 = chain(range(map_edges[2].shape[0]-1, -1, -step), [0])

    for ii in iter_2:
        x_pix.append(map_edges[2][ii,0].value)
        y_pix.append(map_edges[2][ii,1].value)
    
    return map.pixel_to_world(x_pix*u.pix,y_pix*u.pix)