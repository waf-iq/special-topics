"""Render a Markdown report to a styled HTML file for headless-Chrome PDF printing.

Usage: python scripts/md_to_pdf.py reports/D4/D4_report.md
Writes <stem>.html next to the source. A headless Chrome/Edge call then prints it
to <stem>.pdf (see the Makefile target / the command in the D4 conversion).

Relative image paths in the Markdown (e.g. ../D1/foo.png) are preserved, so they
resolve when Chrome loads the local HTML file from the same directory.
"""
import sys
from pathlib import Path

import markdown

CSS = """
@page { size: A4; margin: 18mm 16mm; }
* { box-sizing: border-box; }
body {
  font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
  font-size: 10.5pt; line-height: 1.5; color: #1a1a1a;
  max-width: 100%; margin: 0;
}
h1 { font-size: 20pt; color: #0b3d66; border-bottom: 2px solid #0b3d66;
     padding-bottom: 6px; margin-top: 0; }
h2 { font-size: 14pt; color: #0b3d66; margin-top: 22px;
     border-bottom: 1px solid #ccc; padding-bottom: 3px; }
h3 { font-size: 11.5pt; color: #14507f; margin-top: 16px; }
h2, h3, h4 { page-break-after: avoid; }
p { margin: 6px 0; }
a { color: #14507f; text-decoration: none; }
code { background: #f3f3f3; padding: 1px 4px; border-radius: 3px;
       font-family: "Consolas", monospace; font-size: 9pt; }
pre { background: #f6f8fa; border: 1px solid #e1e4e8; border-radius: 5px;
      padding: 10px; overflow-x: auto; font-size: 9pt; page-break-inside: avoid; }
pre code { background: none; padding: 0; }
table { border-collapse: collapse; width: 100%; margin: 10px 0;
        font-size: 9pt; page-break-inside: avoid; }
th, td { border: 1px solid #cfd8e3; padding: 4px 7px; text-align: left;
         vertical-align: top; }
th { background: #eef3f8; color: #0b3d66; }
tr:nth-child(even) td { background: #fafbfd; }
img { max-width: 92%; height: auto; display: block; margin: 12px auto;
      page-break-inside: avoid; border: 1px solid #e1e4e8; }
hr { border: none; border-top: 1px solid #d0d0d0; margin: 18px 0; }
blockquote { border-left: 3px solid #cfd8e3; margin: 8px 0; padding: 2px 12px;
             color: #444; background: #fafbfd; }
strong { color: #111; }
"""

HTML = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>{title}</title><style>{css}</style></head>
<body>{body}</body></html>"""


def main() -> None:
    src = Path(sys.argv[1]).resolve()
    text = src.read_text(encoding="utf-8")
    body = markdown.markdown(
        text,
        extensions=[
            "tables",
            "fenced_code",
            "pymdownx.tilde",      # ~~strike~~ / sub
            "sane_lists",
            "attr_list",
        ],
    )
    out = src.with_suffix(".html")
    out.write_text(
        HTML.format(title=src.stem, css=CSS, body=body), encoding="utf-8"
    )
    print(out)


if __name__ == "__main__":
    main()
