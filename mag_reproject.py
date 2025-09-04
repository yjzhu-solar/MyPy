import numpy as np
from sunpy.coordinates.spice import get_rotation_matrix
import spiceypy as spice
import os

# SPICE Frame names and NAIF ID Codes
# ------------------------------------------------------------------------------

#    The following names and NAIF ID codes are defined in this kernel file:

#        Name                   ID
#        ---------------------  --------
#        EARTH_NORTH_POLE       399901


#    The following generic frames are defined in this kernel file:

#       SPICE Frame Name            Long-name
#       -------------------------   --------------------------------------------

#    SOLO mission specific generic frames:

#       SOLO_SUN_RTN                Sun Solar Orbiter Radial-Tangential-Normal
#       SOLO_SOLAR_MHP              S/C-centred mirror helioprojective
#       SOLO_IAU_SUN_2009           Sun Body-Fixed based on IAU 2009 report
#       SOLO_IAU_SUN_2003           Sun Body-Fixed based on IAU 2003 report
#       SOLO_GAE                    Geocentric Aries Ecliptic at J2000 (GAE)
#       SOLO_GSE                    Geocentric Solar Ecliptic at J2000 (GSE)
#       SOLO_HEE                    Heliocentric Earth Ecliptic at J2000 (HEE)
#       SOLO_HOR                    Heliocentric orbital reference frame (HOR)
#       SOLO_VSO                    Venus-centric Solar Orbital (VSO)

#    Heliospheric Coordinate Frames developed for the NASA STEREO mission:

#       SOLO_ECLIPDATE              Mean Ecliptic of Date Frame
#       SOLO_HCI                    Heliocentric Inertial Frame
#       SOLO_HEE_NASA               Heliocentric Earth Ecliptic Frame
#       SOLO_HEEQ                   Heliocentric Earth Equatorial Frame
#       SOLO_GEORTN                 Geocentric Radial Tangential Normal Frame

#    Heliocentric Generic Frames(*):

#       SUN_ARIES_ECL               Heliocentric Aries Ecliptic   (HAE)
#       SUN_EARTH_CEQU              Heliocentric Earth Equatorial (HEEQ)
#       SUN_EARTH_ECL               Heliocentric Earth Ecliptic   (HEE)
#       SUN_INERTIAL                Heliocentric Inertial         (HCI)

#    Geocentric Generic Frames:

#       EARTH_SUN_ECL   (*)         Geocentric Solar Ecliptic     (GSE)
#       EARTH_MECL_MEQX (*)         Earth Mean Ecliptic and Equinox of date
#                                   frame (Auxiliary frame for EARTH_SUN_ECL)
#       EARTH_MECL_MEQX_J2000       Earth Mean Ecliptic and Equinox at J2000
#                                   frame (Auxiliary frame for SOLO_GSE and
#                                   SOLO_HEE)


#    (*) These frames are commonly used by other missions for data analysis
#        and scientific research. In the future NAIF may include them
#        in their official generic frames kernel for the Sun and Earth systems.
#        When this happens the frames will be removed from this kernel.


#    These frames have the following centers, frame class and NAIF
#    IDs:

#       SPICE Frame Name              Center      Class    NAIF ID
#       --------------------------    ----------  -------  ----------
#       SOLO_SUN_RTN                  SOLO        DYNAMIC     -144991
#       SOLO_SOLAR_MHP                SOLO        DYNAMIC     -144992
#       SOLO_IAU_SUN_2009             SUN         FIXED       -144993
#       SOLO_IAU_SUN_2003             SUN         FIXED       -144994

#       SOLO_GAE                      EARTH       DYNAMIC     -144995
#       SOLO_GSE                      EARTH       DYNAMIC     -144996
#       SOLO_HEE                      SUN         DYNAMIC     -144997
#       SOLO_HOR                      SUN         DYNAMIC     -144985
#       SOLO_VSO                      VENUS       DYNAMIC     -144999
#       SOLO_GSM                      EARTH       DYNAMIC     -144962

#       SOLO_ECLIPDATE                EARTH       PARAM       -144980
#       SOLO_HCI                      SUN         DYNAMIC     -144981
#       SOLO_HEE_NASA                 SUN         DYNAMIC     -144982
#       SOLO_HEEQ                     SUN         DYNAMIC     -144983
#       SOLO_GEORTN                   SUN         DYNAMIC     -144984

#       SUN_ARIES_ECL                 SUN         DYNAMIC  1000010000
#       SUN_EARTH_CEQU                SUN         DYNAMIC  1000010001
#       SUN_EARTH_ECL                 SUN         DYNAMIC  1000010002
#       SUN_INERTIAL                  SUN         FIXED    1000010004

#       EARTH_SUN_ECL                 EARTH       DYNAMIC   300399005
#       EARTH_MECL_MEQX               EARTH       PARAM     300399000
#       EARTH_MECL_MEQX_J2000         EARTH       DYNAMIC     -144998


#    These frames have the following common names and other designators
#    in literature:

#       SPICE Frame Name            Common names and other designators
#       --------------------        --------------------------------------
#       SUN_ARIES_ECL               HAE, Solar Ecliptic (SE)
#       SUN_EARTH_CEQU              HEEQ, Stonyhurst Heliographic
#       SUN_INERTIAL                HCI, Heliographic Inertial (HGI)
#       EARTH_SUN_ECL               GSE of Date, Hapgood
#       SUN_EARTH_ECL               HEE of Date
#       EARTH_MECL_MEQX             Mean Ecliptic of Date (ECLIPDATE)


#    The keywords implementing these frame definitions are located in the
#    section "Generic Dynamic Frames" and "Generic Inertial Frames".


#   A self-defined Earth-centered mirror Helioprojective Cartesian (HPC) frame:
#       SPICE Frame Name            Common names and other designators
#       --------------------        --------------------------------------
#       
# ------------------------------------------------------------------------------

spice_kernel_path = "../spice_kernel/"
spice_kernels = [
    "naif0012.tls",           # leapseconds
    "de442s.bsp",             # planetary ephemeris
    "pck00010.tpc",           # PCK for IAU_SUN orientation
    "solo_ANC_soc-sci-fk_V08.tf",  # heliospheric frames
    "solo_ANC_soc-orbit-stp_20200210-20301120_377_V1_00465_V01.bsp", # Solar Orbiter ephemeris
    "earth_hpc.tf"    # Earth-centered Helioprojective Cartesian
]

# load kernels
for kernel in spice_kernels:
    spice.furnsh(os.path.join(spice_kernel_path, kernel))

def hgs_local_to_heeq_cart(Bw, Bn, Br, lon_deg, lat_deg):
    """
    Convert local HGS components (west, north, radial) at (lon, lat)
    to HEEQ Cartesian (X, Y, Z). Angles: degrees; west-positive lon.
    """
    lam = np.deg2rad(lon_deg)
    B   = np.deg2rad(lat_deg)
    sl, cl = np.sin(lam), np.cos(lam)
    sB, cB = np.sin(B),  np.cos(B)

    BX = -Bw*sl - Bn*sB*cl + Br*cB*cl
    BY =  Bw*cl - Bn*sB*sl + Br*cB*sl
    BZ =  Bn*cB + Br*sB
    return BX, BY, BZ

if __name__ == "__main__":
    Bw, Bn, Br = 0, 0, 1
    lon, lat = 45, 45
    BX, BY, BZ = hgs_local_to_heeq_cart(Bw, Bn, Br, lon, lat)
    print("BX, BY, BZ (HEEQ) = {}".format((BX, BY, BZ)))

    obstime = "2022-10-24T19:15:00"

    rotation_matrix = get_rotation_matrix("SUN_EARTH_CEQU", "EARTH_SOLAR_MHP", obstime)
    B_hpc = rotation_matrix @ np.array([BX, BY, BZ])
    print("B_hpc = {}".format(B_hpc))