#!/usr/bin/env python3
"""
Small helper for querying the Stockholm Solar Telescope (SST) archive.

The SST archive currently exposes a web interface rather than a documented
public JSON API, so this script queries the same HTTP endpoints as the browser
and parses the returned HTML.

Dependencies
------------
pip install requests beautifulsoup4 lxml pandas

Example
-------
python sst_archive.py \
    --start-date 2000-01-01 \
    --end-date 2026-08-27 \
    --spectral-line 4846 \
    --spectral-line 6563 \
    --output sst_observations.csv

Use --details to also visit every observation-detail page and extract
high-level metadata and FITS-header metadata.
"""

from __future__ import annotations

import argparse
import re
import time
from typing import Any
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup


BASE_URL = "https://dubshen.astro.su.se/sst_archive/"


class SSTArchive:
    def __init__(
        self,
        base_url: str = BASE_URL,
        request_delay: float = 0.5,
        timeout: float = 60.0,
    ):
        self.base_url = base_url
        self.request_delay = request_delay
        self.timeout = timeout

        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "sst-archive-python-client/0.1"}
        )

    def _get(self, url: str, **kwargs) -> requests.Response:
        response = self.session.get(url, timeout=self.timeout, **kwargs)
        response.raise_for_status()
        return response

    def get_search_page(
        self,
        *,
        start_date: str,
        end_date: str,
        instrument: str = "all",
        spectral_lines: list[str] | None = None,
        polarimetry: str = "any",
        advanced_query: str = "",
        page: int = 1,
    ) -> BeautifulSoup:
        params: dict[str, Any] = {
            "start_date": start_date,
            "end_date": end_date,
            "instrument": instrument,
            "polarimetry": polarimetry,
            "advanced_query": advanced_query,
            "page": page,
        }

        if spectral_lines:
            params["spectral_lines"] = spectral_lines

        response = self._get(
            urljoin(self.base_url, "search"),
            params=params,
        )

        return BeautifulSoup(response.text, "lxml")

    @staticmethod
    def _get_number_of_results(soup: BeautifulSoup) -> int:
        text = soup.get_text(" ", strip=True)

        # The wording has changed over time (for example, "Showing results
        # 1 to 25 of 100" and "Showing 1-25 of 100 results").
        m = re.search(
            r"Showing(?:\s+results)?\s+\d+\s*(?:to|[-–])\s*\d+"
            r"\s+of\s+([\d,]+)",
            text,
            flags=re.IGNORECASE,
        )

        if m:
            return int(m.group(1).replace(",", ""))

        return 0

    def _get_next_page_url(self, soup: BeautifulSoup) -> str | None:
        """Return the archive-provided next-page URL, if present."""
        # Prefer semantic markup, then handle the labels used by common
        # Bootstrap pagination templates.
        link = soup.find("a", rel=lambda value: value and "next" in value)
        if link is None:
            for candidate in soup.find_all("a", href=True):
                label = " ".join(
                    filter(
                        None,
                        (
                            candidate.get_text(" ", strip=True),
                            candidate.get("aria-label", ""),
                            candidate.get("title", ""),
                        ),
                    )
                ).strip()
                if re.search(r"\bnext\b", label, flags=re.IGNORECASE):
                    link = candidate
                    break

        # Some pagination widgets use only chevrons and expose neither rel nor
        # an accessible label. In that case, use the link immediately after
        # the active page item.
        if link is None:
            active = soup.select_one(
                ".pagination .active, nav[aria-label*='aginat'] .active"
            )
            if active is not None:
                next_item = active.find_next_sibling()
                if next_item is not None:
                    link = next_item.find("a", href=True)

        if link is None or not link.get("href"):
            return None

        # A disabled "Next" control is sometimes still rendered as an anchor.
        classes = set(link.get("class", []))
        parent_classes = set(link.parent.get("class", [])) if link.parent else set()
        if "disabled" in classes or "disabled" in parent_classes:
            return None

        # Resolve query-only links ("?page=2") against /search, not the
        # archive directory.
        return urljoin(urljoin(self.base_url, "search"), link["href"])

    def parse_search_page(self, soup: BeautifulSoup) -> list[dict[str, Any]]:
        observations: list[dict[str, Any]] = []

        # Locate a table containing links to observation-detail pages.
        result_table = None
        for table in soup.find_all("table"):
            if table.find("a", href=re.compile(r"/observations/\d+")):
                result_table = table
                break

        if result_table is None:
            return observations

        for row in result_table.find_all("tr"):
            cells = row.find_all("td")
            if not cells:
                continue

            obs_link = row.find("a", href=re.compile(r"/observations/\d+"))
            if obs_link is None:
                continue

            href = obs_link["href"]
            obs_url = urljoin(self.base_url, href)

            match = re.search(r"/observations/(\d+)", href)
            obs_id = int(match.group(1)) if match else None

            cell_text = [
                cell.get_text(" ", strip=True)
                for cell in cells
            ]

            observations.append(
                {
                    "observation_id": obs_id,
                    "observation_url": obs_url,
                    "search_columns": cell_text,
                }
            )

        return observations

    def search(
        self,
        *,
        start_date: str,
        end_date: str,
        instrument: str = "all",
        spectral_lines: list[str] | None = None,
        polarimetry: str = "any",
        advanced_query: str = "",
    ) -> pd.DataFrame:
        first_page = self.get_search_page(
            start_date=start_date,
            end_date=end_date,
            instrument=instrument,
            spectral_lines=spectral_lines,
            polarimetry=polarimetry,
            advanced_query=advanced_query,
            page=1,
        )

        observations = self.parse_search_page(first_page)
        n_results = self._get_number_of_results(first_page)
        next_url = self._get_next_page_url(first_page)
        visited_urls: set[str] = set()
        page = 1

        # Follow the URLs emitted by the archive itself. This remains correct if
        # its page size or pagination parameter/path changes.
        while next_url and next_url not in visited_urls:
            visited_urls.add(next_url)
            page += 1
            total = f" (of {n_results} results)" if n_results else ""
            print(f"Reading search page {page}{total}")

            response = self._get(next_url)
            soup = BeautifulSoup(response.text, "lxml")
            observations.extend(self.parse_search_page(soup))
            next_url = self._get_next_page_url(soup)
            time.sleep(self.request_delay)

        df = pd.DataFrame(observations)

        if len(df):
            df = df.drop_duplicates(
                subset=["observation_id"],
                keep="first",
            ).reset_index(drop=True)

        return df

    def get_observation_soup(self, observation: int | str) -> BeautifulSoup:
        if isinstance(observation, int) or str(observation).isdigit():
            url = urljoin(
                self.base_url,
                f"observations/{observation}",
            )
        else:
            url = str(observation)

        response = self._get(url)
        return BeautifulSoup(response.text, "lxml")

    def get_observation_details(
        self,
        observation: int | str,
    ) -> dict[str, str]:
        """
        Extract high-level key/value metadata from an observation page.

        The parser is deliberately generic: any two-column metadata table
        outside the FITS-header tables is included.
        """
        soup = self.get_observation_soup(observation)

        details: dict[str, str] = {}

        for table in soup.find_all("table"):
            rows = table.find_all("tr")
            if not rows:
                continue

            first_cells = rows[0].find_all(["th", "td"])
            first_text = [
                c.get_text(" ", strip=True)
                for c in first_cells
            ]

            # Skip FITS keyword/value tables here.
            if len(first_text) >= 2 and first_text[:2] == ["Keyword", "Value"]:
                continue

            for row in rows:
                cells = row.find_all(["th", "td"])
                if len(cells) != 2:
                    continue

                key = cells[0].get_text(" ", strip=True).rstrip(":")
                value = cells[1].get_text(" ", strip=True)

                if key and value and key not in details:
                    details[key] = value

        return details

    def get_fits_metadata(
        self,
        observation: int | str,
    ) -> list[dict[str, Any]]:
        """
        Extract FITS-header metadata exposed on an observation-detail page.

        Returns
        -------
        list of dict
            Each item contains:
                filename : str | None
                metadata : dict[str, str]
        """
        soup = self.get_observation_soup(observation)
        records: list[dict[str, Any]] = []

        for table in soup.find_all("table"):
            rows = table.find_all("tr")
            if not rows:
                continue

            first_cells = rows[0].find_all(["th", "td"])
            header = [
                cell.get_text(" ", strip=True)
                for cell in first_cells
            ]

            if len(header) < 2 or header[:2] != ["Keyword", "Value"]:
                continue

            metadata: dict[str, str] = {}

            for row in rows[1:]:
                cells = row.find_all(["th", "td"])
                if len(cells) < 2:
                    continue

                key = cells[0].get_text(" ", strip=True)
                value = cells[1].get_text(" ", strip=True)

                if key:
                    metadata[key] = value

            filename = None

            # The archive commonly labels a FITS metadata section with a
            # nearby button such as "All FITS Metadata (filename.fits)".
            previous_button = table.find_previous("button")
            if previous_button:
                button_text = previous_button.get_text(" ", strip=True)

                match = re.search(
                    r"All FITS Metadata\s*\((.*?)\)",
                    button_text,
                    flags=re.IGNORECASE,
                )

                if match:
                    filename = match.group(1)

            records.append(
                {
                    "filename": filename,
                    "metadata": metadata,
                }
            )

        return records

    def build_metadata_catalog(
        self,
        observations: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Visit each observation page and create one row per FITS metadata block.
        """
        rows: list[dict[str, Any]] = []
        status_width = 0
        failures = 0

        for position, (_, obs) in enumerate(observations.iterrows(), start=1):
            obs_id = obs.get("observation_id")
            obs_url = obs.get("observation_url")

            status = (
                f"Fetching observations: {position}/{len(observations)} "
                f"(observation {obs_id})"
            )
            status_width = max(status_width, len(status))
            print(f"\r{status:<{status_width}}", end="", flush=True)

            try:
                details = self.get_observation_details(obs_url)
                fits_records = self.get_fits_metadata(obs_url)

            except requests.RequestException as exc:
                print(f"\r{status} failed: {exc}")
                failures += 1
                continue

            # If no FITS-header table exists, still keep one high-level row.
            if not fits_records:
                rows.append(
                    {
                        "observation_id": obs_id,
                        "observation_url": obs_url,
                        **{f"detail_{k}": v for k, v in details.items()},
                    }
                )
            else:
                for fits_record in fits_records:
                    metadata = fits_record["metadata"]

                    row = {
                        "observation_id": obs_id,
                        "observation_url": obs_url,
                        "filename": fits_record["filename"],
                        **{f"detail_{k}": v for k, v in details.items()},
                        **metadata,
                    }

                    rows.append(row)

            time.sleep(self.request_delay)

        if len(observations):
            final_status = (
                f"Fetched {len(observations) - failures}/{len(observations)} "
                "observations."
            )
            print(f"\r{final_status:<{status_width}}")

        return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Query the SST archive web interface."
    )

    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)

    parser.add_argument(
        "--instrument",
        default="all",
        help="Archive instrument value; default: all",
    )

    parser.add_argument(
        "--spectral-line",
        action="append",
        dest="spectral_lines",
        help=(
            "Spectral-line query value. May be supplied multiple times, "
            "e.g. --spectral-line 4846 --spectral-line 6563"
        ),
    )

    parser.add_argument(
        "--polarimetry",
        default="any",
        help="Archive polarimetry value; default: any",
    )

    parser.add_argument(
        "--advanced-query",
        default="",
    )

    parser.add_argument(
        "--details",
        action="store_true",
        help=(
            "Visit every observation page and extract detailed/FITS metadata."
        ),
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Delay between archive requests in seconds; default: 0.5",
    )

    parser.add_argument(
        "--output",
        default="sst_observations.csv",
        help="Output .csv or .xlsx filename",
    )

    return parser.parse_args()


def save_dataframe(df: pd.DataFrame, filename: str) -> None:
    lower = filename.lower()

    if lower.endswith(".xlsx"):
        df.to_excel(filename, index=False)
    else:
        df.to_csv(filename, index=False)


def main() -> None:
    args = parse_args()

    archive = SSTArchive(request_delay=args.delay)

    observations = archive.search(
        start_date=args.start_date,
        end_date=args.end_date,
        instrument=args.instrument,
        spectral_lines=args.spectral_lines,
        polarimetry=args.polarimetry,
        advanced_query=args.advanced_query,
    )

    print(f"Found {len(observations)} unique observations.")

    if args.details:
        result = archive.build_metadata_catalog(observations)
    else:
        result = observations

    save_dataframe(result, args.output)
    print(f"Saved {len(result)} rows to {args.output}")


if __name__ == "__main__":
    main()
