import sunpy 
import sunpy.map
import sunpy.data.sample
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage

'''

This is an example of how to rotate a sunpy image array, without modifying the rotation matrix 
of the WCS, which is equivalent to modifying the rotation matrix but not touching the image array. 
I did not use the sunpy.map.Map.rotate() method because, first, it modifies the WCS rotation matrix,
and second, it is not implemented for data when CDELT1 != CDELT2.


'''

# Load in a sample AIA image
aia_map = sunpy.map.Map(sunpy.data.sample.AIA_171_IMAGE)

# Resample to create a map, of which CDELT1 != CDELT2
aia_map_resampled = aia_map.resample([1024, 2048]*u.pix)

rotation_angle = 45*u.deg

aia_map_falsely_rotated = aia_map_resampled.rotate(rotation_angle)

# We need to first calculate the rotation matrix which takes into account the difference in pixel scales
# along x and y axes

rotation_matrix = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle)*aia_map_resampled.scale[1]/aia_map_resampled.scale[0]],
                            [np.sin(rotation_angle)*aia_map_resampled.scale[0]/aia_map_resampled.scale[1], np.cos(rotation_angle)]])



# Here we assume we rotate around the CRPIX, so the actual mapping of new (x_prime, y_prime) to (x, y) is
# (x_prime, y_prime)^T = rotation_matrix @ (x - (CRPIX1-1), y - (CRPIX2-1)) + (CRPIX1-1, CRPIX2-1)
# Note that the CRPIX is 1-based, so we need to subtract 1 from it to make it 0-based

# However, we need to map the new (x_prime, y_prime) to the old (x, y) in order to use scipy.ndimage.map_coordinates
# So we need to calculate the inverse of the above equation, which is
# (x, y)^T = rotation_matrix_inv @ (x_prime - (CRPIX1-1), y_prime - (CRPIX2-1)) + (CRPIX1-1, CRPIX2-1)

rotation_matrix_inv = np.linalg.inv(rotation_matrix)

xy_grid_prime = np.flip(np.indices(aia_map_resampled.data.shape),axis=0)

rot_center = np.array([aia_map_resampled.reference_pixel.x.to_value(u.pix), aia_map_resampled.reference_pixel.y.to_value(u.pix)])

xy_grid_0 = np.einsum('ij,jkl->ikl', rotation_matrix_inv, xy_grid_prime) + (rot_center - np.dot(rotation_matrix_inv, rot_center))[:,np.newaxis,np.newaxis]

yx_grid_0 = np.flip(xy_grid_0,axis=0)

aia_resample_rotated = ndimage.map_coordinates(aia_map_resampled.data, yx_grid_0, order=1)

plt.imshow(aia_resample_rotated, origin='lower',aspect=1/2, norm=aia_map.plot_settings['norm'])
plt.show()

# Since in scipy.ndimage.map_coordinates, 
