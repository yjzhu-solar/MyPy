#!/usr/bin/env python3
"""
Download PUNCH files from UMBRA:
https://umbra.nascom.nasa.gov/punch/<level>/<product>/<YYYY>/<MM>/<DD>/

Examples:
    python punch_downloader.py \
        --start "2025-09-22T00:00:00" \
        --end   "2025-09-22T00:30:00" \
        --level 3 --product CIM --format fits --out ./data

    python punch_downloader.py \
        -s 2025-09-22T00:00:00 -e 2025-09-22T00:30:00 \
        -L 3 -P CIM -F jp2 -o ./jp2
"""

import argparse
import concurrent.futures as cf
import datetime as dt
import os
import re
import sys
from pathlib import Path
from typing import Iterable, List, Tuple

import requests

BASE = "https://umbra.nascom.nasa.gov/punch"

# Example filename:
# PUNCH_L3_CIM_20250922000029_v0g.fits
FNAME_RE = re.compile(
    r"^PUNCH_L(?P<level>\d+)_(?P<product>[A-Z0-9]+)_(?P<ts>\d{14})_v[^.]+\.(?P<ext>fits|jp2)$"
)

def parse_iso8601(ts: str) -> dt.datetime:
    """Parse an ISO8601 timestamp without timezone; assume UTC."""
    try:
        # Python 3.11+ handles 'Z' too, but we assume naive -> UTC
        return dt.datetime.fromisoformat(ts)
    except ValueError:
        raise SystemExit(f"Invalid time '{ts}'. Use e.g. 2025-09-22T00:00:00")

def ymd_from_dt(t: dt.datetime) -> Tuple[int, int, int]:
    return t.year, t.month, t.day

def build_day_url(level: int, product: str, day: dt.date) -> str:
    return f"{BASE}/{level}/{product}/{day:%Y}/{day:%m}/{day:%d}/"

def fetch_listing(url: str, session: requests.Session) -> List[str]:
    """Fetch directory listing and return href names (filenames)."""
    headers = {"User-Agent": "punch-downloader/1.0 (+https://example)"}
    r = session.get(url, headers=headers, timeout=30)
    if r.status_code == 404:
        # Directory may be missing if no data that day
        return []
    r.raise_for_status()
    # Extract href="...". Works with typical Apache-style listings.
    hrefs = re.findall(r'href="([^"]+)"', r.text, flags=re.IGNORECASE)
    # Keep only plausible data files
    files = [h for h in hrefs if FNAME_RE.match(os.path.basename(h))]
    return files

def parse_filename(fname: str):
    """Return dict with level, product, ts (datetime), ext if pattern matches, else None."""
    m = FNAME_RE.match(fname)
    if not m:
        return None
    d = m.groupdict()
    ts = dt.datetime.strptime(d["ts"], "%Y%m%d%H%M%S")
    d["ts_dt"] = ts
    return d

def filter_files(
    files: Iterable[str],
    start: dt.datetime,
    end: dt.datetime,
    level: int,
    product: str,
    ext: str,
) -> List[str]:
    out = []
    for f in files:
        meta = parse_filename(os.path.basename(f))
        if not meta:
            continue
        if int(meta["level"]) != int(level):
            continue
        if meta["product"].upper() != product.upper():
            continue
        if meta["ext"].lower() != ext.lower():
            continue
        if start <= meta["ts_dt"] <= end:
            out.append(f)
    return sorted(out)

def ensure_outdir(path: str) -> Path:
    p = Path(path).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p

def download_one(url: str, outdir: Path, session: requests.Session, retries: int = 3) -> Tuple[str, bool, str]:
    """Download a single file with basic retries. Returns (filename, success, message)."""
    fname = os.path.basename(url)
    dest = outdir / fname
    if dest.exists():
        return (fname, True, "already exists")
    headers = {"User-Agent": "punch-downloader/1.0"}
    for attempt in range(1, retries + 1):
        try:
            with session.get(url, headers=headers, timeout=120, stream=True) as r:
                if r.status_code == 404:
                    return (fname, False, "404 not found")
                r.raise_for_status()
                tmp = dest.with_suffix(dest.suffix + ".part")
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 256):
                        if chunk:
                            f.write(chunk)
                tmp.replace(dest)
                return (fname, True, "downloaded")
        except Exception as e:
            last_err = str(e)
    return (fname, False, last_err)

def daterange_days(start: dt.datetime, end: dt.datetime) -> List[dt.date]:
    """Inclusive list of dates spanning [start, end]."""
    days = []
    d0 = start.date()
    d1 = end.date()
    n = (d1 - d0).days
    for i in range(n + 1):
        days.append(d0 + dt.timedelta(days=i))
    return days

def download_punch(
    start_time: str,
    end_time: str,
    level: int = 3,
    product: str = "CIM",
    file_format: str = "fits",
    outdir: str = "./downloads",
    max_workers: int = 4,
) -> None:
    """
    Download PUNCH files between start_time and end_time (inclusive).
    Times should be ISO 8601 (e.g., 2025-09-22T00:00:00).
    """
    start = parse_iso8601(start_time)
    end = parse_iso8601(end_time)
    if end < start:
        raise SystemExit("end_time must be >= start_time")
    ext = file_format.lower()
    if ext not in ("fits", "jp2"):
        raise SystemExit("file_format must be 'fits' or 'jp2'")
    product = product.upper()

    out_path = ensure_outdir(outdir)

    session = requests.Session()

    # Gather candidate files by crawling each day folder in the range
    all_matches: List[Tuple[str, str]] = []  # (url, fname)
    for day in daterange_days(start, end):
        day_url = build_day_url(level, product, day)
        try:
            files = fetch_listing(day_url, session)
        except Exception as e:
            print(f"[WARN] Failed to list {day_url}: {e}", file=sys.stderr)
            continue
        matches = filter_files(files, start, end, level, product, ext)
        for m in matches:
            # m might be a relative link; ensure absolute URL
            url = m if m.startswith("http") else day_url + m
            all_matches.append((url, os.path.basename(m)))

    if not all_matches:
        print("No files matched your query.")
        return

    # Download in parallel
    print(f"Found {len(all_matches)} file(s). Starting downloads to: {out_path}")
    successes, failures = 0, 0
    with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(download_one, url, out_path, session) for url, _ in all_matches]
        for fut in cf.as_completed(futures):
            fname, ok, msg = fut.result()
            status = "OK" if ok else "FAIL"
            print(f"[{status}] {fname} - {msg}")
            successes += int(ok)
            failures += int(not ok)

    print(f"\nDone. Success: {successes}, Failed: {failures}. Files saved in: {out_path}")

def main():
    ap = argparse.ArgumentParser(description="Download PUNCH files from UMBRA by time window.")
    ap.add_argument("-s", "--start", dest="start", required=True, help="ISO start, e.g., 2025-09-22T00:00:00")
    ap.add_argument("-e", "--end", dest="end", required=True, help="ISO end, e.g., 2025-09-22T00:30:00")
    ap.add_argument("-L", "--level", dest="level", type=int, default=3, help="Data level (e.g., 3)")
    ap.add_argument("-P", "--product", dest="product", default="CIM", help="Data product (e.g., CIM)")
    ap.add_argument("-F", "--format", dest="format", choices=["fits", "jp2"], default="fits", help="File format")
    ap.add_argument("-o", "--out", dest="outdir", default="./downloads", help="Output directory")
    ap.add_argument("-w", "--workers", dest="workers", type=int, default=4, help="Parallel downloads")
    args = ap.parse_args()

    download_punch(
        start_time=args.start,
        end_time=args.end,
        level=args.level,
        product=args.product,
        file_format=args.format,
        outdir=args.outdir,
        max_workers=args.workers,
    )

if __name__ == "__main__":
    main()
