#!/usr/bin/env python3
"""
Rewrite .ipynb hyperlinks in converted HTML files for GitHub Pages hosting.

Finds all <a href="..."> links pointing to local .ipynb files and rewrites
them to point to the corresponding .html files on your GitHub Pages site.

Examples:
    href="other_notebook.ipynb"           -> href="https://user.github.io/repo/other_notebook.html"
    href="subdir/analysis.ipynb"          -> href="https://user.github.io/repo/subdir/analysis.html"
    href="notebook.ipynb#section"         -> href="https://user.github.io/repo/notebook.html#section"
    href="https://example.com/foo.ipynb"  -> (unchanged, external URL)

Usage:
    # Single file
    python fix_links.py page_url file.html

    # All HTML files in a directory (recursive)
    python fix_links.py page_url output_dir/

    # Dry run — show what would change without modifying files
    python fix_links.py page_url output_dir/ --dry-run

    # With a subdirectory prefix (if HTMLs live under a subpath on the site)
    python fix_links.py page_url output_dir/ --prefix notes/2024

    page_url is the GitHub Pages base URL, e.g.:
        https://username.github.io/repo-name
"""

import argparse
import re
import sys
from pathlib import Path
from urllib.parse import urlparse


def is_local_ipynb_link(href: str) -> bool:
    """Return True if href is a relative link to a .ipynb file."""
    # Skip anchors-only, empty, javascript:, mailto:, data:
    if not href or href.startswith(("#", "javascript:", "mailto:", "data:")):
        return False
    # Skip absolute URLs (http://, https://, //, ftp://)
    parsed = urlparse(href)
    if parsed.scheme or href.startswith("//"):
        return False
    # Check that the path part (before any #fragment) ends with .ipynb
    path_part = parsed.path
    return path_part.lower().endswith(".ipynb")


def rewrite_href(href: str, base_url: str, prefix: str = "") -> str:
    """
    Rewrite a relative .ipynb href to an absolute .html URL on GitHub Pages.

    href    : e.g. "subdir/notebook.ipynb#Section-1"
    base_url: e.g. "https://user.github.io/repo"
    prefix  : e.g. "notes/2024"  (optional subdirectory on the site)

    Returns : e.g. "https://user.github.io/repo/notes/2024/notebook.html#Section-1"
    """
    parsed = urlparse(href)
    path = parsed.path.rsplit("/", 1)[-1]      # "notebook.ipynb"
    fragment = parsed.fragment                  # "Section-1" or ""

    # .ipynb -> .html
    path = re.sub(r'\.ipynb$', '.html', path, flags=re.IGNORECASE)

    # Build the full URL
    base = base_url.rstrip("/")
    if prefix:
        base = f"{base}/{prefix.strip('/')}"
    url = f"{base}/{path}"

    if fragment:
        url = f"{url}#{fragment}"

    return url


# Regex to match href="..." or href='...' inside <a> tags
# Captures the quote char and the href value
HREF_PATTERN = re.compile(
    r'''(<a\b[^>]*?\bhref\s*=\s*)(["'])(.*?)\2''',
    re.IGNORECASE | re.DOTALL,
)


def fix_links_in_html(html: str, base_url: str, prefix: str = "") -> tuple[str, int]:
    """
    Rewrite all local .ipynb links in the HTML string.
    Returns (new_html, count_of_replacements).
    """
    count = 0

    def _replacer(m):
        nonlocal count
        before = m.group(1)   # '<a ... href='
        quote = m.group(2)    # '"' or "'"
        href = m.group(3)     # the URL

        if is_local_ipynb_link(href):
            new_href = rewrite_href(href, base_url, prefix)
            count += 1
            tag = f'{before}{quote}{new_href}{quote}'
            # Add target="_blank" if not already present in the <a> tag
            if 'target=' not in before.lower():
                tag += ' target="_blank" rel="noopener noreferrer"'
            return tag
        return m.group(0)

    new_html = HREF_PATTERN.sub(_replacer, html)
    return new_html, count


def process_file(filepath: Path, base_url: str, prefix: str = "",
                 dry_run: bool = False) -> int:
    """Process a single HTML file. Returns number of links rewritten."""
    html = filepath.read_text(encoding="utf-8")
    new_html, count = fix_links_in_html(html, base_url, prefix)

    if count == 0:
        return 0

    if dry_run:
        print(f"  [dry-run] {filepath}: {count} link(s) would be rewritten")
        # Show the specific rewrites
        for m in HREF_PATTERN.finditer(html):
            href = m.group(3)
            if is_local_ipynb_link(href):
                new = rewrite_href(href, base_url, prefix)
                print(f"            {href}  ->  {new}")
    else:
        filepath.write_text(new_html, encoding="utf-8")
        print(f"  {filepath}: {count} link(s) rewritten")

    return count


def main():
    parser = argparse.ArgumentParser(
        description="Rewrite .ipynb links in HTML files for GitHub Pages.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python fix_links.py https://user.github.io/repo output/\n"
            "  python fix_links.py https://user.github.io/repo file.html --dry-run\n"
            "  python fix_links.py https://user.github.io/repo output/ --prefix notes/2024\n"
        ),
    )
    parser.add_argument("page_url",
                        help="GitHub Pages base URL, e.g. https://user.github.io/repo")
    parser.add_argument("target",
                        help="HTML file or directory to process (recursive)")
    parser.add_argument("--prefix", default="",
                        help="Subdirectory prefix on the site (e.g. 'notes/2024')")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would change without modifying files")
    args = parser.parse_args()

    target = Path(args.target)
    if not target.exists():
        print(f"Error: {target} not found.", file=sys.stderr)
        sys.exit(1)

    # Collect HTML files
    if target.is_file():
        files = [target]
    else:
        files = sorted(target.rglob("*.html"))

    if not files:
        print(f"No HTML files found in {target}.")
        sys.exit(0)

    total = 0
    for f in files:
        total += process_file(f, args.page_url, args.prefix, args.dry_run)

    action = "would be rewritten" if args.dry_run else "rewritten"
    print(f"\n{total} link(s) {action} across {len(files)} file(s).")


if __name__ == "__main__":
    main()
