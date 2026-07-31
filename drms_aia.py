# directly download level 1.5 AIA data from JSOC


import os 
from pathlib import Path
import drms
from sunpy.time import parse_time 
from astropy.time import Time
import astropy.units as u
from datetime import datetime, timedelta

def time_to_input_string(
        start_time: str | Time | datetime,
        end_time: str | Time | datetime | None = None,
        duration: u.Quantity | timedelta | None = None,
        cadence: u.Quantity | timedelta | None = None
    ):
    """
    Convert time inputs to a string format suitable for JSOC queries.
    """

    if duration is not None and end_time is not None:
        raise ValueError("Specify either 'end_time' or 'duration', not both.")

    if duration is None and end_time is None and cadence is not None:
        raise ValueError("If 'cadence' is specified, either 'end_time' or 'duration' must also be provided.")

    if not isinstance(start_time, Time):
        start_time = parse_time(start_time)

    if duration is not None:
        if isinstance(duration, timedelta):
            duration = duration.total_seconds() * u.s

    if end_time is not None:
        if not isinstance(end_time, Time):
            end_time = parse_time(end_time)
        duration = end_time - start_time

    if cadence is not None:
        if isinstance(cadence, timedelta):
            cadence = cadence.total_seconds() * u.s
    

    jsoc_time_str = start_time.strftime('%Y-%m-%dT%H:%M:%S')

    if duration is not None:
        if duration > 1*u.day:
            jsoc_time_str += f'/{duration.to_value(u.day):.2f}d'
        elif duration > 1*u.hour:
            jsoc_time_str += f'/{duration.to_value(u.hour):.2f}h'
        elif duration > 1*u.minute:
            jsoc_time_str += f'/{duration.to_value(u.minute):.2f}m'
        else:
            jsoc_time_str += f'/{duration.to_value(u.s):.2f}s'

    if cadence is not None:
        if cadence > 1*u.day:
            jsoc_time_str += f'@{cadence.to_value(u.day):.2f}d'
        elif cadence > 1*u.hour:
            jsoc_time_str += f'@{cadence.to_value(u.hour):.2f}h'
        elif cadence > 1*u.minute:
            jsoc_time_str += f'@{cadence.to_value(u.minute):.2f}m'
        else:
            jsoc_time_str += f'@{cadence.to_value(u.s):.2f}s'

    return jsoc_time_str

def cutout_coords_dict(bottom_left: u.Quantity,
                  top_right: u.Quantity):
    """
    convert cutout coordinates to a dictionary format suitable for JSOC queries.
    """

    if not isinstance(bottom_left, u.Quantity) or not isinstance(top_right, u.Quantity):
        raise ValueError("Both 'bottom_left' and 'top_right' must be astropy Quantity objects.")

    center_x = (bottom_left[0] + top_right[0]) / 2
    center_y = (bottom_left[1] + top_right[1]) / 2
    width = top_right[0] - bottom_left[0]
    height = top_right[1] - bottom_left[1]

    center_x_pixel = center_x.to_value(u.arcsec)/0.6 # AIA level 1.5 data has a pixel scale of 0.6 arcsec/pixel
    center_y_pixel = center_y.to_value(u.arcsec)/0.6
    width_pixel = width.to_value(u.arcsec)/0.6
    height_pixel = height.to_value(u.arcsec)/0.6

    return {
        "xc": center_x_pixel,
        "yc": center_y_pixel,
        "wide": width_pixel,
        "high": height_pixel
    }

def parse_wavelength(wavelength: int | list[int] | str):
    """
    Convert wavelength input to a string format suitable for JSOC queries.
    """

    euv_wavelengths = [94, 131, 171, 193, 211, 304, 335]
    uv_wavelengths = [1600, 1700]

    if isinstance(wavelength, str):
        if wavelength.lower() == "euv":
            data_product_str = "aia.lev1_euv_12s"
            wavelength_str = "[" + ",".join(str(w) for w in euv_wavelengths) + "]"
        elif wavelength.lower() == "uv":
            data_product_str = "aia.lev1_uv_24s"
            wavelength_str = "[" + ",".join(str(w) for w in uv_wavelengths) + "]"
        elif wavelength.lower() == "all":
            data_product_str = ["aia.lev1_euv_12s", "aia.lev1_uv_24s"]
            wavelength_str = [
                "[" + ",".join(str(w) for w in euv_wavelengths) + "]",
                "[" + ",".join(str(w) for w in uv_wavelengths) + "]"
            ]
        else:
            raise ValueError(f"Invalid wavelength string: {wavelength}. Must be 'euv', 'uv', or 'all'.")
    elif isinstance(wavelength, int):
        if wavelength in euv_wavelengths:
            data_product_str = "aia.lev1_euv_12s"
            wavelength_str = f"[{wavelength:d}]"
        elif wavelength in uv_wavelengths:
            data_product_str = "aia.lev1_uv_24s"
            wavelength_str = f"[{wavelength:d}]"
        else:
            raise ValueError(f"Invalid wavelength: {wavelength}. Must be one of {euv_wavelengths + uv_wavelengths}.")
    else:
        if all(w in euv_wavelengths for w in wavelength):
            data_product_str = "aia.lev1_euv_12s"
            wavelength_str = "[" + ",".join(str(w) for w in wavelength) + "]"
        elif all(w in uv_wavelengths for w in wavelength):
            data_product_str = "aia.lev1_uv_24s"
            wavelength_str = "[" + ",".join(str(w) for w in wavelength) + "]"
        else:
            if any(w in euv_wavelengths for w in wavelength) and any(w in uv_wavelengths for w in wavelength):
                data_product_str = ["aia.lev1_euv_12s", "aia.lev1_uv_24s"]
                wavelength_str = [
                    "[" + ",".join(str(w) for w in wavelength if w in euv_wavelengths) + "]",
                    "[" + ",".join(str(w) for w in wavelength if w in uv_wavelengths) + "]"
                ]
            else:
                raise ValueError(f"Invalid wavelength list: {wavelength}. Must be a list of valid AIA wavelengths {euv_wavelengths + uv_wavelengths}.")

    return data_product_str, wavelength_str

def construct_query_string(
        start_time: str | Time | datetime,
        wavelength: int | list[int] | str,
        end_time: str | Time | datetime | None = None,
        duration: u.Quantity | timedelta | None = None,
        cadence: u.Quantity | timedelta | None = None,
    ):

    """
    Construct a JSOC query string for downloading AIA data.
    """

    time_str = time_to_input_string(start_time, end_time, duration, cadence)

    data_product_str, wavelength_str = parse_wavelength(wavelength)

    if isinstance(data_product_str, list):
        query_str = []
        for dp, wl in zip(data_product_str, wavelength_str):
            query_str.append(f"{dp}[{time_str}]{wl}")
    else:
        query_str = f"{data_product_str}[{time_str}]{wavelength_str}"

    return query_str

def construct_option_dict(
        cutout_bottom_left: u.Quantity | None = None,
        cutout_top_right: u.Quantity | None = None,
        mpo: bool = True
    ) -> dict:
    if cutout_bottom_left is None and cutout_top_right is None and mpo is False:
        return {None: None}
    if cutout_bottom_left is not None and cutout_top_right is not None:
        cutout_dict = cutout_coords_dict(cutout_bottom_left, cutout_top_right)
        if mpo is True:
            return {**cutout_dict, "mpt": "aia.master_pointing3h"}
        else:
            return cutout_dict

def query_and_download_aia_lev15(
        email: str,
        start_time: str | Time | datetime,
        wavelength: int | list[int] | str,
        end_time: str | Time | datetime | None = None,
        duration: u.Quantity | timedelta | None = None,
        cadence: u.Quantity | timedelta | None = None,
        cutout_bottom_left: u.Quantity | None = None,
        cutout_top_right: u.Quantity | None = None,
        mpo: bool = True,
        download_dir: str | Path = ".",
        method: str = "url",
        protocol: str = "fits"
    ):
    """
    Query and download SDO/AIA Level 1.5 data from JSOC.

    This helper constructs the JSOC record query from time and wavelength
    inputs, optionally applies a rectangular cutout in arcsec coordinates,
    submits one or more export requests via DRMS, waits for completion, and
    downloads all files into ``download_dir``.

    Parameters
    ----------
    email : str
        Email address required by JSOC export requests.
    start_time : str or astropy.time.Time or datetime.datetime
        Start time of the query.
    wavelength : int, list[int], or str
        AIA wavelength selector.
        Supported integer channels are: 94, 131, 171, 193, 211, 304, 335,
        1600, and 1700.
        Supported string groups are: ``"euv"``, ``"uv"``, and ``"all"``.
    end_time : str or astropy.time.Time or datetime.datetime, optional
        End time of the query. Mutually exclusive with ``duration``.
    duration : astropy.units.Quantity or datetime.timedelta, optional
        Time span from ``start_time``. Mutually exclusive with ``end_time``.
    cadence : astropy.units.Quantity or datetime.timedelta, optional
        Sampling cadence for the query (for example, ``1*u.minute``).
        Requires either ``end_time`` or ``duration``.
    cutout_bottom_left : astropy.units.Quantity, optional
        Bottom-left cutout coordinate ``[x, y]`` in arcsec.
    cutout_top_right : astropy.units.Quantity, optional
        Top-right cutout coordinate ``[x, y]`` in arcsec.
    mpo : bool, default=True
        If True, include master pointing correction
        (``aia.master_pointing3h``) in the processing options.
        By default, ``aia_prep`` in SolarSoft updates the pointing information
        using the latest JSOC MPO data series ``aia.master_pointing3h``. 
        If you want to use the pointing information of the original level 1 header, set ``mpo=False``.
    download_dir : str or pathlib.Path, default="."
        Output directory for downloaded FITS files.
    method : str, default="url"
        DRMS export method passed to ``drms.Client.export``.
        Supported values are ``"url"``, ``"url-tar"``, ``"ftp"``,
        ``"ftp-tar"``, ``"url_direct"``, and ``"url_quick"``. 
    protocol : str, default="fits"
        Export protocol passed to ``drms.Client.export``.
        Supported values are ``"as-is"``, ``"FITS"``, ``"JPEG"``, ``"MPEG"``, and ``"MP4"``.

    Examples
    --------
    Download one EUV channel for 30 minutes at 1-minute cadence:

    >>> import astropy.units as u
    >>> query_and_download_aia_lev15(
    ...     email="you@example.com",
    ...     start_time="2024-01-01T00:00:00",
    ...     duration=30*u.minute,
    ...     cadence=1*u.minute,
    ...     wavelength=171,
    ...     download_dir="~/Downloads/aia_171"
    ... )

    Download mixed EUV+UV channels with a spatial cutout:

    >>> import astropy.units as u
    >>> query_and_download_aia_lev15(
    ...     email="you@example.com",
    ...     start_time="2024-01-01T00:00:00",
    ...     end_time="2024-01-01T00:20:00",
    ...     wavelength=[171, 1600],
    ...     cutout_bottom_left=u.Quantity([-500, -500], u.arcsec),
    ...     cutout_top_right=u.Quantity([500, 500], u.arcsec),
    ...     download_dir="~/Downloads/aia_cutout"
    ... )

    Download all standard AIA channels without cutout:

    >>> query_and_download_aia_lev15(
    ...     email="you@example.com",
    ...     start_time="2024-01-01T00:00:00",
    ...     duration=1*u.hour,
    ...     wavelength="all",
    ...     download_dir="~/Downloads/aia_all_channels"
    ... )
    """

    query_string = construct_query_string(
        start_time=start_time,
        wavelength=wavelength,
        end_time=end_time,
        duration=duration,
        cadence=cadence
    )

    option_dict = construct_option_dict(
        cutout_bottom_left=cutout_bottom_left,
        cutout_top_right=cutout_top_right,
        mpo=mpo
    )

    if isinstance(download_dir, str):
        download_dir = Path(download_dir).expanduser()
    if not download_dir.exists():
        download_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(query_string, list):
        for qs in query_string:
            print(f"Querying JSOC with: {qs}")
            client = drms.Client()
            result = client.export(
                qs,
                method=method,
                email=email,
                protocol=protocol,
                process={"aia_scale_aialev1": option_dict},
            )

            result.wait()
            result.download(download_dir)
    else:
        print(f"Querying JSOC with: {query_string}")
        client = drms.Client()
        result = client.export(
            query_string,
            method=method,
            email=email,
            protocol=protocol,
            process={"aia_scale_aialev1": option_dict},
        )

        result.wait()
        result.download(download_dir)

if __name__ == "__main__":
    # # Example usage
    # start_time = "2024-01-01T00:00:00"
    # end_time = "2024-01-01T01:00:00"
    # wavelength = [171, 1600]
    # # wavelength = "all"
    # cutout_bottom_left = u.Quantity([-500, -500], u.arcsec)
    # cutout_top_right = u.Quantity([500, 500], u.arcsec)

    # print((parse_time(start_time)-parse_time(end_time)).to_value(u.minute))

    # query_string = construct_query_string(
    #     start_time=start_time,
    #     end_time=end_time,
    #     wavelength=wavelength,
    # )

    # print("JSOC Query String:", query_string)

    query_and_download_aia_lev15(
        email="yingjie.zhu@pmodwrc.ch",
        start_time="2024-01-01T00:00:00",
        cutout_bottom_left=u.Quantity([-500, -500], u.arcsec),
        cutout_top_right=u.Quantity([500, 500], u.arcsec),
        wavelength=[171, 1600],
        download_dir="~/Downloads/",
    )




    