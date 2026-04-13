#!/usr/bin/env python3
"""
Convert a Jupyter Notebook (.ipynb) to HTML with:
  - A left sidebar navigation for markdown headings
  - Markdown cell images (attachments & <img src=...>) embedded as base64
  - Optional slides export

Usage:
    python nb2html.py notebook.ipynb output_dir/           # -> output_dir/notebook.html
    python nb2html.py notebook.ipynb output_dir/ --slides   # -> notebook.slides.html
    python nb2html.py notebook.ipynb output_dir/ --no-sidebar  # plain, no sidebar

Requirements:
    pip install nbconvert nbformat
"""

import argparse
import base64
import os
import re
import sys
from pathlib import Path

import nbformat
import nbconvert
from nbconvert import HTMLExporter, SlidesExporter


# ---------------------------------------------------------------------------
# 1.  Embed images from markdown cells (attachments + <img src=...>)
#     Adapted from @imcomking — https://github.com/jupyter/nbconvert/issues/699
# ---------------------------------------------------------------------------

def collect_embedded_images(notebook, nb_dir: str = "."):
    """
    Scan markdown cells for:
      (a) cell attachments  — attachment:filename.png
      (b) <img src="...">   — local file paths
    Return a list of [original_src, data_uri] pairs.
    """
    images = []
    for cell in notebook["cells"]:
        # --- (a) cell attachments ---
        if "attachments" in cell:
            for filename, attachment in cell["attachments"].items():
                for mime, b64 in attachment.items():
                    images.append(
                        [f"attachment:{filename}", f"data:{mime};base64,{b64}"]
                    )

        # --- (b) <img src="local_path"> in markdown source ---
        if "img src=" in cell["source"]:
            for line in cell["source"].split("\n"):
                if "img src=" not in line:
                    continue
                # Extract the path between the quotes
                match = re.search(r'img src="([^"]+)"', line)
                if not match:
                    continue
                img_path = match.group(1)
                # Skip if already a data URI or remote URL
                if img_path.startswith(("data:", "http://", "https://")):
                    continue
                # Resolve relative to the notebook directory
                abs_path = os.path.join(nb_dir, img_path)
                if not os.path.isfile(abs_path):
                    print(f"  Warning: image not found, skipping: {abs_path}",
                          file=sys.stderr)
                    continue
                # Guess MIME from extension
                ext = os.path.splitext(img_path)[1].lower()
                mime_map = {
                    ".png": "image/png", ".jpg": "image/jpeg",
                    ".jpeg": "image/jpeg", ".gif": "image/gif",
                    ".svg": "image/svg+xml", ".webp": "image/webp",
                }
                mime = mime_map.get(ext, "image/png")
                with open(abs_path, "rb") as f:
                    encoded = base64.b64encode(f.read()).decode("utf-8")
                images.append([img_path, f"data:{mime};base64,{encoded}"])

    return images


def replace_image_srcs(body: str, images: list) -> str:
    """Replace each original src with its base64 data URI in the HTML body."""
    for src, data_uri in images:
        body = body.replace(f'src="{src}"', f'src="{data_uri}"', 1)
    return body


# ---------------------------------------------------------------------------
# 2.  Parse headings from the notebook's markdown cells
# ---------------------------------------------------------------------------

def extract_headings(nb):
    """Return a list of (level, text, anchor_id) for every markdown heading."""
    headings = []
    slug_counts: dict[str, int] = {}

    for cell in nb.cells:
        if cell.cell_type != "markdown":
            continue
        for line in cell.source.splitlines():
            m = re.match(r'^(#{1,6})\s+(.+)', line)
            if m:
                level = len(m.group(1))
                raw_text = m.group(2).strip()

                # --- Build the anchor the same way nbconvert does ---
                slug = raw_text
                # Strip markdown inline markup:
                #   backticks (`code`), bold (**b** / __b__), italic (*i* / _i_)
                slug = re.sub(r'`([^`]*)`', r'\1', slug)   # `code` -> code
                slug = re.sub(r'\*\*([^*]*)\*\*', r'\1', slug)  # **bold**
                slug = re.sub(r'__([^_]*)__', r'\1', slug)      # __bold__
                slug = re.sub(r'\*([^*]*)\*', r'\1', slug)      # *italic*
                slug = re.sub(r'_([^_]*)_', r'\1', slug)        # _italic_
                # Replace spaces with hyphens (nbconvert preserves most
                # other characters like = ( ) $ etc.)
                base_slug = re.sub(r'\s+', '-', slug)
                # nbconvert HTML-encodes certain chars in the id attribute
                base_slug = base_slug.replace('&', '&amp;')
                base_slug = base_slug.replace('"', '&quot;')
                base_slug = base_slug.replace('<', '&lt;')
                base_slug = base_slug.replace('>', '&gt;')

                # Display text: also strip markup for the sidebar label
                display = raw_text
                display = re.sub(r'`([^`]*)`', r'\1', display)
                display = re.sub(r'\*\*([^*]*)\*\*', r'\1', display)
                display = re.sub(r'__([^_]*)__', r'\1', display)
                display = re.sub(r'\*([^*]*)\*', r'\1', display)
                display = re.sub(r'_([^_]*)_', r'\1', display)

                # Handle duplicate headings (nbconvert appends a counter)
                if base_slug in slug_counts:
                    slug_counts[base_slug] += 1
                    anchor = f"{base_slug}-{slug_counts[base_slug]}"
                else:
                    slug_counts[base_slug] = 0
                    anchor = base_slug
                headings.append((level, display, anchor))
    return headings


# ---------------------------------------------------------------------------
# 3.  Build the sidebar HTML
# ---------------------------------------------------------------------------

def build_sidebar_html(headings):
    """Return an HTML string for the sidebar navigation."""
    if not headings:
        return '<nav id="nb-sidebar"><p style="color:#999;">No headings found</p></nav>'

    items = []
    for level, text, anchor in headings:
        indent = (level - 1) * 16
        items.append(
            f'<a class="nav-link nav-h{level}" '
            f'style="padding-left:{12 + indent}px" '
            f'href="#{anchor}">{text}</a>'
        )
    links = "\n".join(items)
    return f"""\
<nav id="nb-sidebar">
  <div class="sidebar-title">Contents</div>
  <div class="nav-links">
{links}
  </div>
</nav>"""


# ---------------------------------------------------------------------------
# 4.  CSS & JS for the sidebar
# ---------------------------------------------------------------------------

SIDEBAR_CSS = r"""
<style>
/* ---------- sidebar ---------- */
#nb-sidebar {
  position: fixed;
  top: 0; left: 0;
  width: 280px;
  height: 100vh;
  overflow-y: auto;
  background: #f7f7f8;
  border-right: 1px solid #ddd;
  padding: 12px 0;
  box-sizing: border-box;
  z-index: 1000;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
  font-size: 13px;
  transition: transform 0.25s ease;
}
#nb-sidebar .sidebar-title {
  font-weight: 700;
  font-size: 15px;
  padding: 8px 16px 12px;
  color: #333;
  border-bottom: 1px solid #e0e0e0;
  margin-bottom: 6px;
}
#nb-sidebar .nav-links {
  display: flex;
  flex-direction: column;
}
#nb-sidebar a.nav-link {
  text-decoration: none;
  color: #444;
  padding: 5px 12px;
  line-height: 1.45;
  border-left: 3px solid transparent;
  transition: background 0.15s, border-color 0.15s;
}
#nb-sidebar a.nav-link:hover {
  background: #eaeaec;
  color: #000;
}
#nb-sidebar a.nav-link.active {
  border-left-color: #4a90d9;
  color: #1a56a0;
  background: #e8f0fb;
  font-weight: 600;
}
/* Heading-level styling */
#nb-sidebar a.nav-h1 { font-weight: 700; font-size: 14px; }
#nb-sidebar a.nav-h2 { font-weight: 600; }
#nb-sidebar a.nav-h3 { font-size: 12.5px; }
#nb-sidebar a.nav-h4,
#nb-sidebar a.nav-h5,
#nb-sidebar a.nav-h6 { font-size: 12px; color: #666; }

/* ---------- push body content right ---------- */
#notebook-container, .jp-Notebook, body > div.container {
  margin-left: 300px !important;
  max-width: calc(100% - 320px) !important;
}
body {
  margin-left: 290px;
}

/* ---------- toggle button (for narrow screens) ---------- */
#sidebar-toggle {
  display: none;
  position: fixed;
  top: 8px; left: 8px;
  z-index: 1100;
  background: #4a90d9;
  color: #fff;
  border: none;
  border-radius: 4px;
  padding: 6px 10px;
  cursor: pointer;
  font-size: 16px;
}
@media (max-width: 860px) {
  #nb-sidebar { transform: translateX(-100%); }
  #nb-sidebar.open { transform: translateX(0); }
  #sidebar-toggle { display: block; }
  #notebook-container, .jp-Notebook, body > div.container {
    margin-left: 0 !important;
    max-width: 100% !important;
  }
  body { margin-left: 0; }
}
</style>
"""

SIDEBAR_JS = r"""
<script>
document.addEventListener("DOMContentLoaded", function () {
  var toggle = document.getElementById("sidebar-toggle");
  var sidebar = document.getElementById("nb-sidebar");
  if (toggle && sidebar) {
    toggle.addEventListener("click", function () {
      sidebar.classList.toggle("open");
    });
  }

  var links = document.querySelectorAll("#nb-sidebar a.nav-link");
  if (!links.length) return;

  var targets = [];
  links.forEach(function (a) {
    var id = a.getAttribute("href").replace("#", "");
    var el = document.getElementById(id);
    if (el) targets.push({id: id, el: el, link: a});
  });

  function setActive() {
    var current = null;
    for (var i = 0; i < targets.length; i++) {
      if (targets[i].el.getBoundingClientRect().top <= 120) {
        current = targets[i];
      }
    }
    links.forEach(function (a) { a.classList.remove("active"); });
    if (current) current.link.classList.add("active");
  }
  window.addEventListener("scroll", setActive, {passive: true});
  setActive();
});
</script>
"""


# ---------------------------------------------------------------------------
# 5.  Main conversion pipeline
# ---------------------------------------------------------------------------

def convert(filename: str, output_dir: str, slides: bool = False,
            sidebar: bool = True):
    infile = Path(filename)
    if not infile.exists():
        print(f"Error: {infile} not found.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    nb_dir = str(infile.parent)  # for resolving relative image paths

    # Read notebook
    notebook = nbformat.read(str(infile), as_version=4)

    # Choose exporter
    if slides:
        outname = os.path.join(
            output_dir,
            os.path.basename(filename).replace(".ipynb", ".slides.html"),
        )
        print(f"Converting to slides: {outname}")
        exporter = SlidesExporter()
        sidebar = False  # sidebar doesn't make sense for reveal.js slides
    else:
        outname = os.path.join(
            output_dir,
            os.path.basename(filename).replace(".ipynb", ".html"),
        )
        print(f"Converting to HTML: {outname}")
        exporter = HTMLExporter()
        exporter.template_name = "classic"

    body, _resources = exporter.from_notebook_node(notebook)

    # --- Embed images from markdown cells ---
    images = collect_embedded_images(notebook, nb_dir=nb_dir)
    if images:
        body = replace_image_srcs(body, images)
        print(f"  Embedded {len(images)} image(s) as base64.")

    # --- Inject sidebar ---
    if sidebar:
        headings = extract_headings(notebook)
        sidebar_block = (
            SIDEBAR_CSS
            + '<button id="sidebar-toggle">&#9776;</button>\n'
            + build_sidebar_html(headings)
            + SIDEBAR_JS
        )
        if "</body>" in body:
            body = body.replace("</body>", sidebar_block + "\n</body>")
        else:
            body += sidebar_block
        print(f"  {len(headings)} heading(s) added to sidebar.")

    Path(outname).write_text(body, encoding="utf-8")
    print(f"Done: {outname}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Convert Jupyter Notebook to HTML with sidebar navigation "
            "and embedded images."
        )
    )
    parser.add_argument("filename", help="Input .ipynb file")
    parser.add_argument("output_dir", help="Output directory")
    parser.add_argument("--slides", action="store_true",
                        help="Export as reveal.js slides (disables sidebar)")
    parser.add_argument("--no-sidebar", action="store_true",
                        help="Disable the sidebar navigation")
    args = parser.parse_args()
    convert(args.filename, args.output_dir,
            slides=args.slides, sidebar=not args.no_sidebar)


if __name__ == "__main__":
    main()
