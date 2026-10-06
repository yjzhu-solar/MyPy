#!/usr/bin/env python3
"""
Download Solar Orbiter/EUI FITS files from the ROB data archive
(https://data.observatory.be/science/solo/eui/) for a given product and
time range.

The archive is a plain Apache directory listing organised as
<level>/YYYY/MM/DD/, so this script fetches the index page of every day in
the requested range, parses the file names, and downloads the matching ones.

Dependencies
------------
pip install requests

Example
-------
python eui_downloader.py hrieuv174 2023-04-10T03:30 2023-04-10T03:35 \
    --output ./eui_data

Product names match either exactly or with an "-image" suffix, so
"hrieuv174" -> "hrieuv174-image" and "fsi174" -> "fsi174-image".
Pass the full name (e.g. "fsi174-image-short") for the other products.
Use --dry-run to only list the matching files.
"""

from __future__ import annotations

import argparse
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import requests


BASE_URL = "https://data.observatory.be/science/solo/eui/"

FILE_RE = re.compile(
    r'href="(?P<name>solo_(?P<level>L\d)_eui-(?P<product>[a-z0-9-]+?)_'
    r'(?P<time>\d{8}T\d{6})(?P<ms>\d{3})?_V(?P<version>\d+)\.fits)"'
)


def _parse_time(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(value.replace("Z", ""))


def _product_matches(product: str, wanted: str) -> bool:
    return product == wanted or product == f"{wanted}-image"


def _day_url(day: datetime, level: str, base_url: str) -> str:
    return f"{base_url}{level}/{day:%Y/%m/%d}/"


def list_eui_files(
    product: str,
    start: str | datetime,
    end: str | datetime,
    *,
    level: str = "L2",
    base_url: str = BASE_URL,
    session: requests.Session | None = None,
    timeout: float = 60.0,
) -> list[dict]:
    """Return the matching files (newest version only) sorted by time.

    Each entry is a dict with keys name, url, product, time, version.
    """
    start, end = _parse_time(start), _parse_time(end)
    if end < start:
        raise ValueError(f"end ({end}) is before start ({start})")

    session = session or requests.Session()
    files: dict[tuple[str, datetime], dict] = {}

    day = start.replace(hour=0, minute=0, second=0, microsecond=0)
    while day <= end:
        url = _day_url(day, level, base_url)
        response = session.get(url, timeout=timeout)
        if response.status_code == 404:
            print(f"No data directory for {day:%Y-%m-%d}: {url}")
            day += timedelta(days=1)
            continue
        response.raise_for_status()

        for match in FILE_RE.finditer(response.text):
            if not _product_matches(match["product"], product):
                continue
            obs_time = datetime.strptime(match["time"], "%Y%m%dT%H%M%S")
            if match["ms"]:
                obs_time += timedelta(milliseconds=int(match["ms"]))
            if not start <= obs_time <= end:
                continue

            entry = {
                "name": match["name"],
                "url": url + match["name"],
                "product": match["product"],
                "time": obs_time,
                "version": int(match["version"]),
            }
            # keep only the newest version of each observation
            key = (entry["product"], obs_time)
            if key not in files or files[key]["version"] < entry["version"]:
                files[key] = entry

        day += timedelta(days=1)

    return sorted(files.values(), key=lambda f: f["time"])


def _download_one(
    session: requests.Session,
    url: str,
    path: Path,
    *,
    overwrite: bool,
    retries: int,
    timeout: float,
) -> str:
    if path.exists() and not overwrite:
        return "skipped"

    tmp_path = path.with_name(path.name + ".part")
    for attempt in range(1, retries + 1):
        try:
            with session.get(url, stream=True, timeout=timeout) as response:
                response.raise_for_status()
                with open(tmp_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
            tmp_path.replace(path)
            return "downloaded"
        except requests.RequestException:
            tmp_path.unlink(missing_ok=True)
            if attempt == retries:
                raise
            time.sleep(2 * attempt)
    return "failed"


def download_eui(
    product: str,
    start: str | datetime,
    end: str | datetime,
    output_dir: str | Path = ".",
    *,
    level: str = "L2",
    workers: int = 4,
    overwrite: bool = False,
    dry_run: bool = False,
    retries: int = 3,
    timeout: float = 60.0,
    base_url: str = BASE_URL,
) -> list[Path]:
    """Download EUI files of `product` between `start` and `end`.

    Returns the local paths of the matching files (also in dry-run mode,
    where nothing is written).
    """
    session = requests.Session()
    session.headers.update({"User-Agent": "eui-downloader-python/0.1"})

    files = list_eui_files(
        product, start, end,
        level=level, base_url=base_url, session=session, timeout=timeout,
    )
    output_dir = Path(output_dir).expanduser()
    paths = [output_dir / f["name"] for f in files]
    print(f"Found {len(files)} {level} '{product}' files "
          f"between {_parse_time(start)} and {_parse_time(end)}.")

    if dry_run or not files:
        for f in files:
            print(f["url"])
        return paths

    output_dir.mkdir(parents=True, exist_ok=True)
    n_done = 0
    failed = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _download_one, session, f["url"], path,
                overwrite=overwrite, retries=retries, timeout=timeout,
            ): f["name"]
            for f, path in zip(files, paths)
        }
        for future in as_completed(futures):
            name = futures[future]
            n_done += 1
            try:
                status = future.result()
            except Exception as err:
                failed.append(name)
                status = f"FAILED ({err})"
            print(f"[{n_done}/{len(files)}] {status}: {name}")

    if failed:
        print(f"{len(failed)} file(s) failed to download:")
        for name in failed:
            print(f"  {name}")
    return paths


def main():
    parser = argparse.ArgumentParser(
        description="Download Solar Orbiter/EUI data from data.observatory.be"
    )
    parser.add_argument("product",
                        help='EUI product, e.g. "hrieuv174", "hrilya1216", '
                             '"fsi174", "fsi304-image-short"')
    parser.add_argument("start", help="start time (ISO), e.g. 2023-04-10T03:30")
    parser.add_argument("end", help="end time (ISO), e.g. 2023-04-10T04:00")
    parser.add_argument("-o", "--output", default=".",
                        help="output directory (default: current directory)")
    parser.add_argument("--level", default="L2",
                        help="data level directory, e.g. L1, L2 (default: L2)")
    parser.add_argument("-w", "--workers", type=int, default=4,
                        help="number of parallel downloads (default: 4)")
    parser.add_argument("--overwrite", action="store_true",
                        help="re-download files that already exist")
    parser.add_argument("--dry-run", action="store_true",
                        help="only list the matching files")
    args = parser.parse_args()

    download_eui(
        args.product, args.start, args.end, args.output,
        level=args.level, workers=args.workers,
        overwrite=args.overwrite, dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
