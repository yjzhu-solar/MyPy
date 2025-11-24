#!/usr/bin/env python3
"""
Minimal IDL Notebook (.idlnb) -> standalone HTML converter.

Usage:
    python idlnb_to_html.py input.idlnb [output.html]

Notes:
- Handles markdown and code cells.
- Renders text outputs and embedded PNG images produced by the IDL notebook renderer.
- Produces a single self-contained HTML file with inline images.
"""
import sys, json, re, html
from pathlib import Path
from datetime import datetime

def md_minimal_to_html(text: str) -> str:
    lines = text.splitlines()
    html_lines = []
    in_code = False
    code_buf = []
    code_lang = ""
    for ln in lines:
        if ln.strip().startswith("```"):
            if not in_code:
                in_code = True
                code_lang = ln.strip().lstrip("`").strip() or "idl"
                code_buf = []
            else:
                code_html = html.escape("\n".join(code_buf))
                html_lines.append(f'<pre class="code"><code class="language-{code_lang}">{code_html}</code></pre>')
                in_code = False
                code_buf = []
            continue
        if in_code:
            code_buf.append(ln); continue
        m = re.match(r"^(#{1,6})\s+(.*)$", ln)
        if m:
            level = len(m.group(1))
            html_lines.append(f"<h{level}>{html.escape(m.group(2))}</h{level}>"); continue
        if re.match(r"^---+$", ln.strip()):
            html_lines.append("<hr/>"); continue
        if ln.strip() == "":
            html_lines.append("")
        else:
            esc = html.escape(ln)
            esc = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", esc)
            esc = re.sub(r"\*(.+?)\*", r"<em>\1</em>", esc)
            esc = re.sub(r"`([^`]+)`", r"<code>\1</code>", esc)
            html_lines.append(f"<p>{esc}</p>")
    if in_code and code_buf:
        code_html = html.escape("\n".join(code_buf))
        html_lines.append(f'<pre class="code"><code class="language-{code_lang}">{code_html}</code></pre>')
    return "\n".join(html_lines)

def convert_idlnb_to_html(nb: dict, title: str) -> str:
    css = """
    <style>
      body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; line-height: 1.5; max-width: 900px; margin: 2rem auto; padding: 0 1rem; }
      h1,h2,h3 { margin-top: 1.6rem; }
      .cell { border: 1px solid #eee; border-radius: 10px; padding: 0.8rem 1rem; margin: 1rem 0; box-shadow: 0 1px 4px rgba(0,0,0,0.04); }
      .cell .prompt { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; font-size: 12px; color: #666; }
      pre.code { background: #f7f7f9; padding: 0.75rem; border-radius: 8px; overflow-x: auto; }
      pre.output { background: #111; color: #ddd; padding: 0.75rem; border-radius: 8px; overflow-x: auto; }
      .output-block { margin-top: 0.5rem; }
      img.output { max-width: 100%; height: auto; display: block; margin: 0.5rem 0; border-radius: 6px; }
      .footer { color: #888; font-size: 12px; margin-top: 2rem; text-align: center; }
      .md { margin: 0.2rem 0; }
      .badge { display:inline-block; font-size: 11px; padding: 2px 6px; background:#eef; border-radius: 6px; color:#335; margin-left:8px; }
    </style>
    """
    head = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{html.escape(title)}</title>
{css}
</head>
<body>
<h1>{html.escape(title)} <span class="badge">exported {datetime.utcnow().isoformat(timespec='seconds')}Z</span></h1>
"""
    body_parts = []
    for idx, cell in enumerate(nb.get("cells", []), 1):
        ctype = cell.get("type", "code")
        content = cell.get("content")
        if isinstance(content, list):
            content_text = "\n".join(str(x) for x in content)
        else:
            content_text = str(content) if content is not None else ""
        block = [f'<div class="cell" id="cell-{idx}">']
        if ctype == "markdown":
            block.append(f'<div class="md">{md_minimal_to_html(content_text)}</div>')
        else:
            block.append(f'<div class="prompt">In [{idx}]:</div>')
            block.append(f'<pre class="code"><code class="language-idl">{html.escape(content_text)}</code></pre>')
        # outputs
        for out in cell.get("outputs", []):
            for item in out.get("items", []):
                mime = item.get("mime")
                if mime == "text/plain":
                    lines = item.get("content", [])
                    if isinstance(lines, list):
                        text = "\n".join(str(x) for x in lines)
                    else:
                        text = str(lines)
                    block.append(f'<div class="output-block"><pre class="output">{html.escape(text)}</pre></div>')
                elif mime == "idl/notebook-renderer":
                    payload = item.get("content")
                    try:
                        payload_json = json.loads(payload) if isinstance(payload, str) else payload
                        if payload_json and payload_json.get("type") == "idlnotebookimage_png":
                            img_item = payload_json.get("item", {})
                            data_b64 = img_item.get("data", "")
                            block.append(f'<div class="output-block"><img class="output" src="data:image/png;base64,{data_b64}"/></div>')
                    except Exception as e:
                        block.append(f'<div class="output-block"><pre class="output">{html.escape(str(e))}</pre></div>')
                else:
                    content_val = item.get("content")
                    text = content_val if isinstance(content_val, str) else repr(content_val)
                    block.append(f'<div class="output-block"><pre class="output">{html.escape(text[:5000])}</pre></div>')
        block.append("</div>")
        body_parts.append("\n".join(block))
    foot = f"""<div class="footer">IDL notebook version {html.escape(str(nb.get('version','?')))} • Converted by idlnb_to_html.py</div>
</body>
</html>"""
    return head + "\n".join(body_parts) + foot

def main(argv):
    if len(argv) < 2 or argv[1].startswith("-h"):
        print(__doc__); return 0
    in_path = Path(argv[1])
    out_path = Path(argv[2]) if len(argv) >= 3 else in_path.with_suffix(".html")
    nb = json.loads(in_path.read_text(encoding="utf-8"))
    title = out_path.stem.replace("_"," ").title()
    html_str = convert_idlnb_to_html(nb, title=title)
    out_path.write_text(html_str, encoding="utf-8")
    print("Wrote", out_path)
    return 0

if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
