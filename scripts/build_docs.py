"""Regenerate the PDF copies in docs/ from their sources.

    python scripts/build_docs.py

- docs/NN-name.md  -> docs/NN-name.pdf   (Markdown -> styled HTML -> PDF)
- docs/NN-name.svg -> docs/NN-name.pdf   (diagram, one page sized to the drawing)

The .md and .svg files are the sources of truth; the PDFs are for reading and sharing.
Printing uses Microsoft Edge or Google Chrome in headless mode (installed on Windows 11).
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
BROWSERS = [
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
]

CSS = """
@page { size: A4; margin: 16mm 15mm 16mm 15mm; }
body { font-family: "Segoe UI", Arial, sans-serif; font-size: 10.5pt; line-height: 1.5; color: #1A1F29; }
h1 { font-size: 20pt; color: #1F3D6B; border-bottom: 2px solid #1F3D6B; padding-bottom: 6px; margin: 0 0 12px; }
h2 { font-size: 14pt; color: #1F3D6B; border-bottom: 1px solid #D0D7E2; padding-bottom: 4px; margin: 22px 0 10px; }
h3 { font-size: 12pt; color: #1F3D6B; margin: 16px 0 8px; }
h2, h3 { break-after: avoid; }
a { color: #2B5797; text-decoration: none; }
code { font-family: Consolas, "Cascadia Mono", monospace; font-size: 9pt; background: #F1F3F6; padding: 0 3px;
  border-radius: 3px; }
pre { background: #F6F8FA; border: 1px solid #E1E5EB; border-radius: 6px; padding: 8px 10px; font-size: 8.5pt;
  line-height: 1.35; white-space: pre-wrap; break-inside: avoid; }
pre code { background: none; padding: 0; }
table { border-collapse: collapse; width: 100%; margin: 8px 0 12px; font-size: 9pt; }
th { background: #E6EEF9; color: #1A1F29; text-align: left; font-weight: 600; }
th, td { border: 1px solid #D0D7E2; padding: 5px 7px; vertical-align: top; }
tr { break-inside: avoid; }
hr { border: none; border-top: 1px solid #D0D7E2; margin: 18px 0; }
ul, ol { padding-left: 22px; } li { margin: 2px 0; }
"""


def browser() -> Path:
    for b in BROWSERS:
        if b.is_file():
            return b
    raise SystemExit("Neither Microsoft Edge nor Google Chrome was found")


def print_pdf(html: str, out: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        page = Path(tmp) / "page.html"
        page.write_text(html, encoding="utf-8")
        subprocess.run([str(browser()), "--headless", "--disable-gpu", "--no-pdf-header-footer",
                        f"--user-data-dir={tmp}\\profile", f"--print-to-pdf={out}", page.as_uri()],
                       check=True, capture_output=True, timeout=120)
    print(f"  {out.name}")


_LIST_ITEM = re.compile(r"^\s*(?:[-*+]|\d+\.)\s")


def _widen_indents(text: str) -> str:
    """Adapt GitHub-style Markdown to python-markdown, outside fenced code blocks:
    - nested lists are indented by 2 or 4 spaces in the docs; python-markdown needs 4 per
      level, so each line's indent is rewritten from its nesting level (parents' indents);
    - a list may follow a line such as "**Tasks**" directly; python-markdown needs a blank line."""
    out, fenced = [], False
    parents: list[int] = []  # original indents of the open list items
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            fenced = not fenced
        if fenced or not line.strip():
            out.append(line)
            continue
        n = len(line) - len(line.lstrip(" "))
        if _LIST_ITEM.match(line):
            while parents and parents[-1] >= n:
                parents.pop()
            level = len(parents)
            parents.append(n)
        elif n == 0:
            parents, level = [], 0  # a plain paragraph or heading ends the list
        else:
            level = sum(1 for p in parents if p < n)  # text belonging to an item
        line = " " * (4 * level) + line.lstrip(" ")
        n = 4 * level
        prev = out[-1] if out else ""
        # A list item right after a non-list line (a paragraph, or a paragraph indented inside the
        # previous item) needs a blank line, or python-markdown folds it into that paragraph.
        if n == 0 and _LIST_ITEM.match(line) and prev.strip() and not _LIST_ITEM.match(prev) \
                and not prev.lstrip().startswith("|"):
            out.append("")
        out.append(line)
    return "\n".join(out)


def md_to_pdf(src: Path) -> None:
    text = _widen_indents(src.read_text(encoding="utf-8"))
    # Links between the docs point at the .md files; in the PDFs they point at the PDF copies.
    text = re.sub(r"\]\((0\d-[\w-]+)\.md(#[^)]*)?\)", r"](\1.pdf)", text)
    body = markdown.markdown(text, extensions=["tables", "fenced_code", "sane_lists", "toc"])
    title = re.search(r"^# (.+)$", text, re.M)
    print_pdf(f"<!doctype html><html><head><meta charset='utf-8'><title>{title.group(1) if title else src.stem}"
              f"</title><style>{CSS}</style></head><body>{body}</body></html>", src.with_suffix(".pdf"))


def svg_to_pdf(src: Path) -> None:
    svg = src.read_text(encoding="utf-8")
    w, h = (float(x) for x in re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', svg).groups())
    pw, ph = w * 0.75, h * 0.75  # CSS px -> pt, as in the previous copies
    print_pdf(f"<!doctype html><html><head><meta charset='utf-8'><style>@page {{ size: {pw}pt {ph}pt; margin: 0; }}"
              f"html, body {{ margin: 0; }} svg {{ display: block; width: {pw}pt; height: {ph}pt; }}</style></head>"
              f"<body>{svg}</body></html>", src.with_suffix(".pdf"))


def main() -> int:
    print("Building PDF copies in docs/:")
    for src in sorted(DOCS.glob("0*.md")):
        md_to_pdf(src)
    for src in sorted(DOCS.glob("0*.svg")):
        svg_to_pdf(src)
    return 0


if __name__ == "__main__":
    sys.exit(main())
