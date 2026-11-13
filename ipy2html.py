#!/usr/bin/env python3
"""
Convert a Jupyter Notebook (.ipynb) to HTML with:
  - A left sidebar navigation for markdown headings
  - Images embedded as base64 (cell attachments and local image files)
  - A "copy" button on every code cell (and markdown code block)
  - Optional slides export

Usage:
    python ipy2html.py notebook.ipynb output_dir/           # -> output_dir/notebook.html
    python ipy2html.py notebook.ipynb output_dir/ --slides   # -> notebook.slides.html
    python ipy2html.py notebook.ipynb output_dir/ --no-sidebar  # plain, no sidebar
    python ipy2html.py notebook.ipynb output_dir/ --no-copy-button  # no copy buttons

Requirements:
    pip install nbconvert nbformat beautifulsoup4
"""

import argparse
import base64
import html
import mimetypes
import re
import sys
from pathlib import Path
from urllib.parse import unquote

import nbformat
from bs4 import BeautifulSoup
from nbconvert import HTMLExporter, SlidesExporter


# ---------------------------------------------------------------------------
# 1.  Embed images (cell attachments and local image files)
# ---------------------------------------------------------------------------

# <img ... src="attachment:NAME" ...> written as raw HTML in a markdown cell
_HTML_ATTACHMENT = re.compile(
    r"""(<img\b[^>]*?\bsrc\s*=\s*)(["'])attachment:(.+?)\2""", re.IGNORECASE
)


def inline_html_attachments(notebook) -> int:
    """
    Replace <img src="attachment:name"> in markdown cells with a data URI.

    nbconvert resolves the markdown form ![alt](attachment:name) by itself
    but leaves raw <img> tags alone.  This is done on the notebook, before
    conversion, because an attachment name is only unique within its cell
    (Jupyter calls every pasted image "image.png").

    Modifies ``notebook`` in place; returns the number of images embedded.
    """
    count = 0
    for cell in notebook.cells:
        attachments = cell.get("attachments")
        if cell.cell_type != "markdown" or not attachments:
            continue

        def to_data_uri(match, attachments=attachments):
            nonlocal count
            bundle = attachments.get(match.group(3))
            if not bundle:
                return match.group(0)
            mime, b64 = next(iter(bundle.items()))
            count += 1
            quote = match.group(2)
            return f"{match.group(1)}{quote}data:{mime};base64,{b64}{quote}"

        cell.source = _HTML_ATTACHMENT.sub(to_data_uri, cell.source)
    return count


def embed_local_images(soup, nb_dir: Path) -> int:
    """
    Embed every <img> that points to a local file as a base64 data URI.

    Works on the rendered page, so it covers every way of writing an image:
    ![alt](path), <img> with any attribute order or quoting, percent-encoded
    names, HTML outputs.  Paths are relative to the notebook's directory.
    Remote and already embedded images are left alone.

    Modifies ``soup`` in place; returns the number of images embedded.
    """
    count = 0
    for img in soup.find_all("img"):
        src = img.get("src", "").strip()
        if not src or src.startswith(("data:", "http://", "https://", "//")):
            continue
        # "my%20fig.png" names the file "my fig.png"
        path = next((p for p in (nb_dir / src, nb_dir / unquote(src))
                     if p.is_file()), None)
        if path is None:
            print(f"  Warning: image not found, left as a link: {src}",
                  file=sys.stderr)
            continue
        # Browsers sniff the real type of a raster image, so PNG is a safe
        # label for an extension that mimetypes does not know.
        mime = mimetypes.guess_type(path.name)[0] or "image/png"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        img["src"] = f"data:{mime};base64,{encoded}"
        count += 1
    return count


# ---------------------------------------------------------------------------
# 2.  Collect the headings of the markdown cells
# ---------------------------------------------------------------------------

def extract_headings(soup):
    """
    Return a list of (level, text, anchor_id) for every markdown-cell heading.

    The headings are read from the rendered page instead of being re-parsed
    from the markdown source, so each anchor is exactly the id nbconvert
    gave the heading.  nbconvert gives identical headings the same id; the
    repeats are renamed here (modifying ``soup``) so that every sidebar link
    has a target of its own.
    """
    # nbconvert marks the headings it renders from markdown with an anchor
    # link; headings inside a cell's output are not part of the outline.
    found = []
    for tag in soup.find_all(re.compile(r"^h[1-6]$")):
        link = tag.find("a", class_="anchor-link")
        if (link is not None and tag.get("id")
                and tag.find_parent(class_="jp-OutputArea") is None):
            found.append((tag, link))

    headings = []
    taken = {tag["id"] for tag, _ in found}
    seen = set()
    for tag, link in found:
        anchor = tag["id"]
        if anchor in seen:
            n = 1
            while f"{anchor}-{n}" in taken:
                n += 1
            anchor = f"{anchor}-{n}"
            taken.add(anchor)
            tag["id"] = anchor
            link["href"] = f"#{anchor}"
        seen.add(anchor)
        # the anchor link ("¶") is the last thing inside the heading
        text = tag.get_text().removesuffix(link.get_text()).strip()
        headings.append((int(tag.name[1]), text or anchor, anchor))
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
            f'href="#{html.escape(anchor)}">{html.escape(text)}</a>'
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

/* ---------- toggle: a "Contents" bar, on narrow screens only ---------- */
#sidebar-toggle {
  display: none;
  position: fixed;
  top: 0; left: 0;
  z-index: 1100;
  width: 100%; height: 40px;
  background: #4a90d9;
  color: #fff;
  border: none;
  padding: 0 14px;
  text-align: left;
  cursor: pointer;
  font-size: 16px;
}
#sidebar-toggle::after { content: " Contents"; font-size: 14px; }
@media screen and (max-width: 860px) {
  /* The toggle is a bar across the top and the page starts below it, so
     the bar never sits on top of the notebook. */
  #sidebar-toggle { display: block; }
  #nb-sidebar {
    top: 40px;
    height: calc(100vh - 40px);
    transform: translateX(-100%);
  }
  #nb-sidebar.open { transform: translateX(0); }
  #notebook-container, .jp-Notebook, body > div.container {
    margin-left: 0 !important;
    max-width: 100% !important;
  }
  body { margin-left: 0; padding-top: 50px !important; }
  /* keep a heading clear of the bar when jumping to it */
  h1, h2, h3, h4, h5, h6 { scroll-margin-top: 48px; }
}

/* ---------- printing: the notebook only ---------- */
@media print {
  #nb-sidebar, #sidebar-toggle { display: none; }
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

  // A clicked entry stays highlighted while its heading is at the top, even
  // when the next heading is close enough below it to count as current.
  var clicked = null;
  targets.forEach(function (t) {
    t.link.addEventListener("click", function () {
      clicked = t;
      sidebar.classList.remove("open");  // narrow screens: back to the page
      setTimeout(setActive, 0);  // the jump may not scroll the page at all
    });
  });

  function setActive() {
    var current = null;
    for (var i = 0; i < targets.length; i++) {
      if (targets[i].el.getBoundingClientRect().top <= 120) {
        current = targets[i];
      }
    }
    if (clicked) {
      var top = clicked.el.getBoundingClientRect().top;
      if (top >= -1 && top <= 120) current = clicked;
      else clicked = null;  // scrolled away from it
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
# 5.  CSS & JS for the "copy code" buttons
# ---------------------------------------------------------------------------

COPY_BUTTON_CSS = r"""
<style>
/* ---------- copy-code button ---------- */
.nb-copy-host { position: relative; }
.nb-copy-btn {
  position: absolute;
  top: 3px; right: 3px;
  top: min(3px, calc(50% - 11px));  /* centre on blocks shorter than the button */
  display: flex;
  align-items: center;
  justify-content: center;
  width: 22px; height: 22px;
  margin: 0; padding: 0;
  box-sizing: border-box;
  color: var(--jp-ui-font-color2, #666);
  background: var(--jp-layout-color1, #fff);
  border: 1px solid var(--jp-border-color1, #bdbdbd);
  border-radius: 4px;
  cursor: pointer;
  opacity: 0;
  transition: opacity 0.15s, color 0.15s, border-color 0.15s;
}
/* Hidden until the code is hovered, the button has keyboard focus, or it is
   showing the result of a copy.  (:focus-visible is a separate rule so that
   a browser without it does not discard the hover rule as well.) */
.nb-copy-host:hover > .nb-copy-btn,
.nb-copy-btn[data-state="done"],
.nb-copy-btn[data-state="fail"] { opacity: 1; }
.nb-copy-btn:focus-visible { opacity: 1; }
.nb-copy-btn:hover {
  color: var(--jp-ui-font-color0, #000);
  border-color: var(--jp-ui-font-color2, #666);
}
.nb-copy-btn[data-state="done"] { color: #2ea043; border-color: #2ea043; }
.nb-copy-btn[data-state="fail"] { color: #cf222e; border-color: #cf222e; }
/* Touch screens cannot hover: keep the button visible. */
@media (hover: none) { .nb-copy-btn { opacity: 1; } }
@media print { .nb-copy-btn { display: none; } }
</style>
"""

COPY_BUTTON_JS = r"""
<script>
(function () {
  // Code-cell inputs (lab/reveal and classic templates) and code blocks in
  // markdown cells.  Cell outputs are deliberately left alone, and so are
  // Mermaid sources, which nbconvert replaces with the rendered diagram.
  var SELECTOR = [
    ".jp-CodeCell .jp-InputArea-editor pre",
    ".code_cell .input_area pre",
    ".jp-RenderedMarkdown pre:not(.mermaid)",
    ".text_cell_render pre:not(.mermaid)"
  ].join(",");

  function icon(shapes) {
    return '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" ' +
           'stroke="currentColor" stroke-width="2" stroke-linecap="round" ' +
           'stroke-linejoin="round" aria-hidden="true">' + shapes + '</svg>';
  }
  var STATES = {
    copy: {label: "Copy code", icon: icon(
      '<rect x="9" y="9" width="12" height="12" rx="2"/>' +
      '<path d="M5 15H4a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1h10a1 1 0 0 1 1 1v1"/>')},
    done: {label: "Copied!", icon: icon('<path d="M5 13l4 4L19 7"/>')},
    fail: {label: "Copy failed", icon: icon('<path d="M6 6l12 12M18 6L6 18"/>')}
  };

  function setState(btn, state) {
    btn.innerHTML = STATES[state].icon;
    btn.title = STATES[state].label;
    btn.setAttribute("aria-label", STATES[state].label);
    btn.setAttribute("data-state", state);
  }

  // Fallback for pages where the async Clipboard API is unavailable (it
  // needs a secure context: https, localhost or, in most browsers, file://).
  function legacyCopy(text) {
    var active = document.activeElement;
    var ta = document.createElement("textarea");
    ta.value = text;
    ta.setAttribute("readonly", "");
    ta.style.cssText = "position:fixed;top:0;left:0;opacity:0;";
    document.body.appendChild(ta);
    ta.focus({preventScroll: true});
    ta.setSelectionRange(0, text.length);
    var ok = false;
    try { ok = document.execCommand("copy"); } catch (e) { ok = false; }
    document.body.removeChild(ta);
    if (active && active.focus) active.focus({preventScroll: true});
    return ok ? Promise.resolve() : Promise.reject(new Error("copy failed"));
  }

  function copyText(text) {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      return navigator.clipboard.writeText(text).catch(function () {
        return legacyCopy(text);
      });
    }
    return legacyCopy(text);
  }

  function addButton(pre) {
    // Pygments ends every block with one newline that is not in the source.
    var text = pre.textContent.replace(/\n$/, "");
    if (!text.trim()) return;  // empty cell: nothing to copy

    // Highlighted code sits in <div class="highlight"><pre>: anchor the
    // button to that div.  A bare <pre> (un-highlighted markdown block) is
    // its own anchor, so no wrapper element has to be inserted.
    var parent = pre.parentNode;
    var host = parent.classList.contains("highlight") ? parent : pre;
    host.classList.add("nb-copy-host");

    var btn = document.createElement("button");
    btn.type = "button";
    btn.className = "nb-copy-btn";
    setState(btn, "copy");

    var timer = null;
    btn.addEventListener("click", function () {
      function show(state) {
        setState(btn, state);
        clearTimeout(timer);
        timer = setTimeout(function () { setState(btn, "copy"); }, 1500);
      }
      copyText(text).then(
        function () { show("done"); },
        function () { show("fail"); }
      );
    });
    host.appendChild(btn);
  }

  document.addEventListener("DOMContentLoaded", function () {
    document.querySelectorAll(SELECTOR).forEach(addButton);
  });
})();
</script>
"""


# ---------------------------------------------------------------------------
# 6.  CSS for the page itself (always added)
# ---------------------------------------------------------------------------

PAGE_CSS = r"""
<style>
/* nbconvert's lab template cuts long code lines off at the edge of the cell:
   let them scroll sideways instead. */
.jp-InputArea-editor pre { overflow-x: auto; }
</style>
"""


# ---------------------------------------------------------------------------
# 7.  Main conversion pipeline
# ---------------------------------------------------------------------------

def convert(filename: str, output_dir: str, slides: bool = False,
            sidebar: bool = True, copy_button: bool = True):
    infile = Path(filename)
    if not infile.is_file():
        raise FileNotFoundError(f"{infile} not found.")

    # Named after the stem, whatever the extension, so that the output can
    # never land on top of the notebook itself.
    outfile = Path(output_dir) / (
        infile.stem + (".slides.html" if slides else ".html"))
    outfile.parent.mkdir(parents=True, exist_ok=True)

    # Read notebook
    notebook = nbformat.read(str(infile), as_version=4)
    embedded = inline_html_attachments(notebook)

    # Choose exporter
    if slides:
        print(f"Converting to slides: {outfile}")
        exporter = SlidesExporter()
        sidebar = False  # sidebar doesn't make sense for reveal.js slides
    else:
        print(f"Converting to HTML: {outfile}")
        # The template has to be chosen here: nbconvert ignores a
        # template_name that is assigned after construction.
        exporter = HTMLExporter(template_name="lab")

    # What nbconvert's own from_filename() passes along; "name" becomes the
    # page <title>, which is otherwise "Notebook" for every file.
    resources = {"metadata": {"name": infile.stem, "path": str(infile.parent)}}
    body, _resources = exporter.from_notebook_node(notebook, resources=resources)

    # --- Work on the rendered page: local images, then headings ---
    soup = BeautifulSoup(body, "html.parser")
    embedded += embed_local_images(soup, infile.parent)
    if embedded:
        print(f"  Embedded {embedded} image(s) as base64.")
    headings = extract_headings(soup) if sidebar else []
    body = str(soup)

    # --- Collect the blocks to inject: page CSS, copy buttons, sidebar ---
    injected = PAGE_CSS
    if copy_button:
        injected += COPY_BUTTON_CSS + COPY_BUTTON_JS
    if sidebar:
        injected += (
            SIDEBAR_CSS
            + '<button id="sidebar-toggle">&#9776;</button>\n'
            + build_sidebar_html(headings)
            + SIDEBAR_JS
        )
        print(f"  {len(headings)} heading(s) added to sidebar.")

    # --- Inject them just before the closing </body> (the last one) ---
    head, closing_tag, tail = body.rpartition("</body>")
    if closing_tag:
        body = head + injected + "\n" + closing_tag + tail
    else:
        body += injected

    outfile.write_text(body, encoding="utf-8")
    print(f"Done: {outfile}")


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
    parser.add_argument("--no-copy-button", action="store_true",
                        help="Disable the copy button on code cells")
    args = parser.parse_args()
    try:
        convert(args.filename, args.output_dir,
                slides=args.slides, sidebar=not args.no_sidebar,
                copy_button=not args.no_copy_button)
    except FileNotFoundError as err:
        print(f"Error: {err}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
