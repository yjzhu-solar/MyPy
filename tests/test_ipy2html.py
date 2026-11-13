"""
Tests for ipy2html.py. Run with:  python -m pytest tests/test_ipy2html.py
"""
import base64
import os
import struct
import sys
import zlib

import nbformat
import pytest
from bs4 import BeautifulSoup
from nbformat.v4 import (new_code_cell, new_markdown_cell, new_notebook,
                         new_output)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ipy2html


def _png(rgb, size=8):
    """A valid single-colour PNG."""
    def chunk(tag, data):
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))
    raw = b"".join(b"\x00" + bytes(rgb) * size for _ in range(size))
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


RED, GREEN, BLUE = _png((220, 60, 60)), _png((60, 180, 60)), _png((60, 60, 220))


def _uri(png):
    return "data:image/png;base64," + base64.b64encode(png).decode()


def _attachment(png):
    return {"image.png": {"image/png": base64.b64encode(png).decode()}}


def _write_notebook(path, cells):
    nb = new_notebook(cells=cells, metadata={"language_info": {"name": "python"}})
    nbformat.write(nb, str(path))
    return path


def _soup(path):
    return BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")


def _sidebar_links(soup):
    return soup.select("#nb-sidebar a.nav-link")


# Headings that the old regex-based slug builder got wrong, plus the
# different ways an image can be written in a markdown cell.
CELLS = [
    new_markdown_cell("# Title\n\nIntro."),
    new_markdown_cell("## Using slit_interactive_plus module"),
    new_markdown_cell("## Fitting the H$\\alpha$ line"),
    new_markdown_cell("## Fe XII 195.12 Å"),
    new_markdown_cell("## Setup"),
    new_code_cell("x = 1 < 2", execution_count=1),
    new_markdown_cell("## Setup"),
    new_markdown_cell("## A & B < C"),
    new_markdown_cell("## Link to [SunPy](https://sunpy.org)"),
    new_markdown_cell("## Trailing hashes ##"),
    new_markdown_cell("```bash\n# not a heading\npip install sunpy\n```"),
    new_markdown_cell("Setext heading\n--------------\n\ntext"),
    new_code_cell("HTML('<h3>In an output</h3>')", execution_count=2, outputs=[
        new_output("execute_result", execution_count=2,
                   data={"text/html": "<h3>In an output</h3>", "text/plain": "<HTML>"})]),
    new_markdown_cell(
        "### Images\n\n"
        "![md-syntax](fig.png)\n\n"
        '<img src="fig.png" alt="html">\n\n'
        '<img alt="attr-first" src="fig.png" width="8">\n\n'
        "<img src='fig.png' alt='single-quote'>\n\n"
        "![percent-encoded](my%20fig.png)\n\n"
        "![remote](https://example.org/remote.png)\n"),
    # the same attachment name in two cells, with different contents
    new_markdown_cell("![md-attachment](attachment:image.png)", attachments=_attachment(GREEN)),
    new_markdown_cell('<img src="attachment:image.png" alt="html-attachment">',
                      attachments=_attachment(BLUE)),
]

HEADINGS = [
    (1, "Title"),
    (2, "Using slit_interactive_plus module"),
    (2, "Fitting the H$\\alpha$ line"),
    (2, "Fe XII 195.12 Å"),
    (2, "Setup"),
    (2, "Setup"),
    (2, "A & B < C"),
    (2, "Link to SunPy"),
    (2, "Trailing hashes"),
    (2, "Setext heading"),
    (3, "Images"),
]


@pytest.fixture(scope="module")
def page(tmp_path_factory):
    """The CELLS notebook converted once, with the output in another folder."""
    src = tmp_path_factory.mktemp("notebook")
    (src / "fig.png").write_bytes(RED)
    (src / "my fig.png").write_bytes(RED)
    notebook = _write_notebook(src / "demo.ipynb", CELLS)
    out = tmp_path_factory.mktemp("html")
    ipy2html.convert(str(notebook), str(out))
    return out / "demo.html"


# ---------------------------------------------------------------- sidebar

def test_every_sidebar_link_has_exactly_one_target(page):
    soup = _soup(page)
    links = _sidebar_links(soup)
    assert links
    for link in links:
        assert link["href"].startswith("#")
        targets = soup.find_all(id=link["href"][1:])
        assert len(targets) == 1, f"{link['href']} -> {len(targets)} elements"


def test_sidebar_lists_the_markdown_headings(page):
    links = _sidebar_links(_soup(page))
    listed = [(int(a["class"][1][-1]), a.get_text()) for a in links]
    assert listed == HEADINGS


def test_sidebar_links_point_at_their_own_heading(page):
    soup = _soup(page)
    for link in _sidebar_links(soup):
        heading = soup.find(id=link["href"][1:])
        assert heading.get_text().replace("¶", "").strip() == link.get_text()


def test_repeated_headings_get_separate_targets(page):
    soup = _soup(page)
    first, second = [a["href"] for a in _sidebar_links(soup) if a.get_text() == "Setup"]
    assert first != second
    # the heading's own permalink follows the renamed id
    heading = soup.find(id=second[1:])
    assert heading.find("a", class_="anchor-link")["href"] == second


def test_sidebar_labels_are_escaped(page):
    html = page.read_text(encoding="utf-8")
    sidebar = html[html.index('<nav id="nb-sidebar">'):]
    assert ">A &amp; B &lt; C</a>" in sidebar


def test_no_sidebar_option(tmp_path):
    notebook = _write_notebook(tmp_path / "plain.ipynb", [new_markdown_cell("# Title")])
    ipy2html.convert(str(notebook), str(tmp_path / "out"), sidebar=False)
    assert _soup(tmp_path / "out" / "plain.html").find(id="nb-sidebar") is None


# ----------------------------------------------------------------- images

def test_local_images_are_embedded_however_they_are_written(page):
    srcs = {img["alt"]: img["src"] for img in _soup(page).find_all("img")}
    for alt in ("md-syntax", "html", "attr-first", "single-quote", "percent-encoded"):
        assert srcs[alt] == _uri(RED), alt
    assert srcs["remote"] == "https://example.org/remote.png"


def test_attachments_come_from_their_own_cell(page):
    srcs = {img["alt"]: img["src"] for img in _soup(page).find_all("img")}
    assert srcs["md-attachment"] == _uri(GREEN)
    assert srcs["html-attachment"] == _uri(BLUE)


def test_missing_image_is_reported_and_left_alone(tmp_path, capsys):
    notebook = _write_notebook(tmp_path / "broken.ipynb",
                               [new_markdown_cell("![gone](nowhere.png)")])
    ipy2html.convert(str(notebook), str(tmp_path / "out"))
    assert "nowhere.png" in capsys.readouterr().err
    assert _soup(tmp_path / "out" / "broken.html").find("img")["src"] == "nowhere.png"


# ------------------------------------------------------------ output file

def test_title_is_the_notebook_name(page):
    assert _soup(page).title.string == "demo"


def test_output_never_replaces_the_notebook(tmp_path):
    # an extension other than lower-case ".ipynb", converted into its own folder
    notebook = _write_notebook(tmp_path / "Demo.IPYNB", [new_markdown_cell("# Title")])
    before = notebook.read_bytes()
    ipy2html.convert(str(notebook), str(tmp_path))
    assert notebook.read_bytes() == before
    assert (tmp_path / "Demo.html").read_text(encoding="utf-8").lstrip().startswith("<!DOCTYPE html>")


def test_slides_export(tmp_path):
    notebook = _write_notebook(tmp_path / "deck.ipynb",
                               [new_markdown_cell("# Title"), new_code_cell("x = 1")])
    ipy2html.convert(str(notebook), str(tmp_path / "out"), slides=True)
    html = (tmp_path / "out" / "deck.slides.html").read_text(encoding="utf-8")
    assert 'id="nb-sidebar"' not in html     # no sidebar on slides
    assert "nb-copy-btn" in html


def test_missing_notebook_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ipy2html.convert(str(tmp_path / "nope.ipynb"), str(tmp_path))


def test_cli_reports_a_missing_notebook(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["ipy2html.py", str(tmp_path / "nope.ipynb"), str(tmp_path)])
    with pytest.raises(SystemExit) as exit_info:
        ipy2html.main()
    assert exit_info.value.code == 1
    assert "not found" in capsys.readouterr().err


# ------------------------------------------------------------ copy button

def test_copy_button_is_added_by_default(page):
    html = page.read_text(encoding="utf-8")
    assert "nb-copy-btn" in html
    # everything injected sits inside <body>
    assert html.rindex("nb-copy-btn") < html.rindex("</body>")


def test_copy_button_can_be_switched_off(tmp_path):
    notebook = _write_notebook(tmp_path / "plain.ipynb", [new_code_cell("x = 1")])
    ipy2html.convert(str(notebook), str(tmp_path / "out"), copy_button=False)
    assert "nb-copy-btn" not in (tmp_path / "out" / "plain.html").read_text(encoding="utf-8")
