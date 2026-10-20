"""Convert REPORT.md to REPORT.pdf using the markdown package and headless Edge/Chrome.

Usage: python scripts/export_report.py   (needs: pip install markdown)
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
BROWSERS = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
]
CSS = """
@page { size: A4; margin: 18mm; }
body { font-family: Segoe UI, Arial, sans-serif; font-size: 11pt; line-height: 1.45; color: #111; }
h1 { font-size: 20pt; border-bottom: 2px solid #333; padding-bottom: 4px; }
h2 { font-size: 14pt; margin-top: 22px; border-bottom: 1px solid #bbb; }
table { border-collapse: collapse; width: 100%; font-size: 9.5pt; margin: 8px 0; }
th, td { border: 1px solid #999; padding: 4px 6px; text-align: left; }
th { background: #eee; }
img { max-width: 100%; max-height: 230mm; display: block; margin: 8px auto; }
code { background: #f2f2f2; padding: 1px 4px; font-size: 9.5pt; }
pre { background: #f2f2f2; padding: 8px; font-size: 9pt; overflow-x: auto; }
h2, img { page-break-inside: avoid; }
"""


def main():
    browser = next((b for b in BROWSERS if Path(b).exists()), None)
    if not browser:
        sys.exit("Edge or Chrome not found")
    body = markdown.markdown(
        (ROOT / "REPORT.md").read_text(encoding="utf-8"),
        extensions=["tables", "fenced_code"])
    html = (f"<html><head><meta charset='utf-8'><style>{CSS}</style></head>"
            f"<body>{body}</body></html>")
    with tempfile.TemporaryDirectory() as tmp:
        page = ROOT / "_report_tmp.html"  # same folder so relative image paths resolve
        page.write_text(html, encoding="utf-8")
        try:
            subprocess.run([
                browser, "--headless", "--disable-gpu", f"--user-data-dir={tmp}",
                "--no-pdf-header-footer", f"--print-to-pdf={ROOT / 'REPORT.pdf'}",
                page.as_uri(),
            ], check=True, timeout=120)
        finally:
            page.unlink(missing_ok=True)
    print("written", ROOT / "REPORT.pdf")


if __name__ == "__main__":
    main()
