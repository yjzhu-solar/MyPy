import numpy as np
import sunpy
import sunpy.map
from sunpy.coordinates import Helioprojective, propagate_with_solar_surface
import astropy
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.time import Time
import astropy.units as u
from astropy.wcs import WCS
from scipy.interpolate import LinearNDInterpolator
from copy import deepcopy
import warnings 
from shapely import minimum_rotated_rectangle, MultiPoint


def iris_spec_xymesh_from_header(win_header, aux_header, aux_data):
    deltay = win_header["CDELT2"]
    crpixy = win_header["CRPIX2"]
    ny = win_header["NAXIS2"]
    nx = win_header["NAXIS3"]
    pc_ij = np.zeros((nx,2,2))
    pc_ij[:,0,0] = aux_data[:,aux_header["PC3_3IX"]]
    pc_ij[:,0,1] = aux_data[:,aux_header["PC3_2IX"]]
    pc_ij[:,1,0] = aux_data[:,aux_header["PC2_3IX"]]
    pc_ij[:,1,1] = aux_data[:,aux_header["PC2_2IX"]]
    xcen = aux_data[:,aux_header["XCENIX"]]
    ycen = aux_data[:,aux_header["YCENIX"]]

    xmesh = np.zeros((ny, nx))
    ymesh = (np.tile(np.arange(ny), (nx,1)).T - (crpixy - 1))*deltay

    xmesh_rot = np.zeros((ny, nx))
    ymesh_rot = np.zeros((ny, nx))

    for ii in range(nx):
        xmesh_rot[:,ii] = pc_ij[ii,0,0]*xmesh[:,ii] + pc_ij[ii,0,1]*ymesh[:,ii]
        ymesh_rot[:,ii] = pc_ij[ii,1,0]*xmesh[:,ii] + pc_ij[ii,1,1]*ymesh[:,ii]

    return xmesh_rot + xcen[np.newaxis,:], ymesh_rot + ycen[np.newaxis,:]


def iris_spec_map_merge_header(filename, win_ext=1):

    with fits.open(filename) as hdul:
        win_header = hdul[win_ext].header.copy()
        prim_header = hdul[0].header.copy()

        header_merge = prim_header.copy()
        header_merge.update(win_header)
    
    header_merge["DATE-OBS"] = header_merge["DATE_OBS"]
    header_merge["DATE-END"] = header_merge["DATE_END"]
    return header_merge


def iris_spec_map_interp_from_header(filename,data,mask=None,win_ext=1,aux_ext=-2,
                                    synchronize="mid",sdo_rsun=True,xbin=1,ybin=1,
                                    tr_mode="on",scan_start="west",rotate=True):
    data = deepcopy(data)	

    with fits.open(filename) as hdul:
        win_header = hdul[win_ext].header.copy()
        aux_data = hdul[aux_ext].data.copy()
        aux_header = hdul[aux_ext].header.copy()
        prim_header = hdul[0].header.copy()

        detector_type = prim_header[f"TDET{win_ext:1d}"][:3]

        xmesh, ymesh = iris_spec_xymesh_from_header(win_header, aux_header, aux_data)


        nx = win_header["NAXIS3"]
        ny = win_header["NAXIS2"]

        if xbin > 1:
            xmesh = np.nanmean(xmesh.reshape(xmesh.shape[0], -1, xbin), axis=2)
            ymesh = np.nanmean(ymesh.reshape(ymesh.shape[0], -1, xbin), axis=2)
            nx = nx//xbin
        if ybin > 1:
            xmesh = np.nanmean(xmesh.reshape(-1, ybin, xmesh.shape[1]), axis=1)
            ymesh = np.nanmean(ymesh.reshape(-1, ybin, ymesh.shape[1]), axis=1)
            ny = ny//ybin

        if (ny, nx) != data.shape[:2]:
            raise ValueError("Data shape does not match the shape of the mesh")

        deltax = win_header["CDELT3"]
        if deltax < 0:
            deltax = -deltax
            xmesh = np.flip(xmesh, axis=1)
            ymesh = np.flip(ymesh, axis=1)
            warnings.warn("Negative CDELT3 found (raster from west to east). Changing sign to positive. "
                    "Because we assume `iris_auto_fit` has flipped the data, we will not flip it back.",
                    UserWarning,stacklevel=2)
        deltay = win_header["CDELT2"]

        if xbin > 1:
            deltax *= xbin
        if ybin > 1:
            deltay *= ybin

        if detector_type == "FUV":
            exposure_time = aux_data[:,aux_header["EXPTIMEF"]]*u.s
        elif detector_type == "NUV":
            exposure_time = aux_data[:,aux_header["EXPTIMEN"]]*u.s
        else:
            raise ValueError("Detector type not recognized")
        
        date_obs_start = Time(prim_header["DATE_OBS"])
        date_obs_end = Time(prim_header["DATE_END"])
        date_average = date_obs_start + (date_obs_end - date_obs_start)/2
        date_obs_each_exposure = date_obs_start + aux_data[:,aux_header["TIME"]]*u.s + exposure_time/2

        if scan_start == "west":
            date_obs_each_exposure = np.flip(date_obs_each_exposure)


        if sdo_rsun:
            rsun = 696000000.0*u.m
        else:
            rsun = None

        if synchronize in ["mid", "start", "end"]: 

            if synchronize == "mid":
                synchronize_date = date_average
            elif synchronize == "start":
                synchronize_date = date_obs_start
            elif synchronize == "end":
                synchronize_date = date_obs_end
            else:
                raise ValueError("Synchronize keyword not recognized")
            
            unify_helioprojective_frame = Helioprojective(observer='earth', obstime=synchronize_date,rsun=rsun)


            for ii in range(nx):
                if tr_mode == "on":
                    helioprojective_frame_ii = Helioprojective(observer='earth', obstime=date_obs_each_exposure[0],rsun=rsun)
                else:
                    helioprojective_frame_ii = Helioprojective(observer='earth', obstime=date_obs_each_exposure[ii],rsun=rsun)
                coords_ii = SkyCoord(xmesh[:,ii]*u.arcsec, ymesh[:,ii]*u.arcsec, frame=helioprojective_frame_ii)
                with propagate_with_solar_surface(rotation_model='rigid'):
                    coords_ii_unified = coords_ii.transform_to(unify_helioprojective_frame)
                xmesh[:,ii] = coords_ii_unified.Tx.to_value(u.arcsec)
                ymesh[:,ii] = coords_ii_unified.Ty.to_value(u.arcsec)
            
        if synchronize in ["mid", "start", "end"]:
            wcs_time = synchronize_date
        else:
            wcs_time = date_obs_start

        if rotate:
            x_interp = np.linspace(xmesh.min(), xmesh.max(), np.ceil(xmesh.ptp()/deltax).astype(int))
            y_interp = np.linspace(ymesh.min(), ymesh.max(), np.ceil(ymesh.ptp()/deltay).astype(int))

            xi_interp = np.moveaxis(np.array(np.meshgrid(x_interp, y_interp)), 0, -1)

            wcs = xy_to_wcs(x_interp, y_interp, wcs_time, detector_type, rsun=rsun)
        else:
            mesh_envolope = minimum_rotated_rectangle(MultiPoint([(xmesh[0,0], ymesh[0,0]), (xmesh[0,-1], ymesh[0,-1]),
                                                                  (xmesh[-1,-1], ymesh[-1,-1]), (xmesh[-1,0], ymesh[-1,0])])).normalize()
            
            
            wcs, xi_interp = envolope_to_wcs(mesh_envolope, deltax, deltay, wcs_time, detector_type, rsun=rsun)
            

        points_flatten = (xmesh.flatten(), ymesh.flatten())

        if mask is not None:
            if mask.shape == data.shape:
                data[mask] = np.nan
            elif len(data.shape) == 3 and mask.shape == data.shape[:2]:
                data[mask] = np.nan
        
        if len(data.shape) == 2:
            data_interp_linear_func = LinearNDInterpolator(points_flatten, data.flatten())
        elif len(data.shape) == 3:
            data_interp_linear_func = LinearNDInterpolator(points_flatten, data.reshape(-1, data.shape[2]))

        data_interp_linear = data_interp_linear_func(xi_interp)

        # fig, ax = plt.subplots(layout="constrained")
        # ax.scatter(xmesh, ymesh, s=1, color="k")
        # ax.scatter(xi_interp[:,:,0], xi_interp[:,:,1], s=1, color="r")
        # plt.show()
        
        if len(data.shape) == 2:
            return sunpy.map.Map(data_interp_linear, wcs)
        elif len(data.shape) == 3:
            return data_interp_linear, wcs


def xy_to_wcs(x,y,date_obs,detector,rsun=None):
    nx = len(x)
    ny = len(y)
    wcs_header = sunpy.map.make_fitswcs_header(
        (ny, nx),
        coordinate=SkyCoord(x[nx//2], y[ny//2], unit=u.arcsec,
                            frame="helioprojective", obstime=date_obs,
                            rsun=rsun),
        reference_pixel=[nx//2, ny//2]*u.pix,
        scale=[np.abs(x[-1] - x[0])/(nx - 1), np.abs(y[-1] - y[0])/(ny - 1)] * u.arcsec/u.pix,
        telescope="IRIS",
        instrument="SPEC",
        detector=detector,
        ) 
    return wcs_header          

def envolope_to_wcs(mesh_envolope, deltax, deltay, date_obs, detector, rsun=None):
    crval1 = mesh_envolope.centroid.x
    crval2 = mesh_envolope.centroid.y

    fovx = np.sqrt((mesh_envolope.exterior.xy[0][-2] - mesh_envolope.exterior.xy[0][0])**2 + \
                   (mesh_envolope.exterior.xy[1][-2] - mesh_envolope.exterior.xy[1][0])**2)

    fovy = np.sqrt((mesh_envolope.exterior.xy[0][1] - mesh_envolope.exterior.xy[0][0])**2 + \
                   (mesh_envolope.exterior.xy[1][1] - mesh_envolope.exterior.xy[1][0])**2)

    nx = np.ceil(fovx/deltax).astype(int)
    ny = np.ceil(fovy/deltay).astype(int)

    crota = np.arctan2(mesh_envolope.exterior.xy[1][-2] - mesh_envolope.exterior.xy[1][0],
                        mesh_envolope.exterior.xy[0][-2] - mesh_envolope.exterior.xy[0][0])
    crota = np.rad2deg(crota)

    wcs_header = sunpy.map.make_fitswcs_header(
        (ny, nx),
        coordinate=SkyCoord(crval1, crval2, unit=u.arcsec,
                            frame="helioprojective",
                            obstime=date_obs, rsun=rsun),
        reference_pixel=[nx//2, ny//2]*u.pix,
        scale=[fovx/nx, fovy/ny] * u.arcsec/u.pix,
        rotation_angle=crota*u.deg,
        telescope="IRIS",
        instrument="SPEC",
        detector=detector,
    )

    # generate the meshgrid for the wcs

    y_mesh, x_mesh = np.indices((ny, nx))

    wcs_obj = WCS(wcs_header) 
    all_coords = wcs_obj.pixel_to_world(x_mesh, y_mesh)
    x_interp = all_coords.Tx.to_value(u.arcsec)
    y_interp = all_coords.Ty.to_value(u.arcsec)
    xi_interp = np.moveaxis(np.array([x_interp, y_interp]), 0, -1)

    return wcs_header, xi_interp
                                                  

if __name__ == "__main__":
    from scipy.io import readsav
    from sun_blinker import SunBlinker
    from sjireader import read_iris_sji
    from astropy.visualization import ImageNormalize, AsinhStretch
    import matplotlib.pyplot as plt

    filename = "/home/yjzhu/Solar/EIS_DKIST_SolO/src/IRIS/20221024/2322/iris_l2_20221024_232249_3600609177_raster_t000_r00000.fits"
    SiIV_1393_fitres_file = readsav("/home/yjzhu/Solar/EIS_DKIST_SolO/src/IRIS/20221024/2322/fit_res/SiIV_1393_raster0.sav",verbose=True)


    # map = iris_spec_map_interp_from_header(filename, np.zeros((548,320)), win_ext=1, aux_ext=-2)
    SiIV_1393_int_map = iris_spec_map_interp_from_header("/home/yjzhu/Solar/EIS_DKIST_SolO/src/IRIS/20221024/2322/iris_l2_20221024_232249_3600609177_raster_t000_r00000.fits",
                    win_ext=3,data=SiIV_1393_fitres_file["int"].copy(), tr_mode="off", rotate=False)
    
    # iris_1400_sji_2322_map = read_iris_sji("/home/yjzhu/Solar/EIS_DKIST_SolO/src/IRIS/20221024/2322/iris_l2_20221024_232249_3600609177_SJI_1400_t000.fits",
    #                                     index=SiIV_1393_int_map.date,sdo_rsun=True)
    
    # SiIV_1393_int_map.plot_settings["norm"] = ImageNormalize(vmin=0,vmax=1e4,stretch=AsinhStretch(0.1))
    # iris_1400_sji_2322_map.plot_settings["norm"] = ImageNormalize(vmin=10,vmax=200,stretch=AsinhStretch(0.05))
    
    # with propagate_with_solar_surface():
    #     SunBlinker(SiIV_1393_int_map, iris_1400_sji_2322_map, reproject=True, fps=0.5)
    


#     map.plot()

    plt.show()
    
    
        

