"""Write report/index.html: summary table, how-to-read note, links to the case and all-sample pages."""
import argparse
import html as H_
from pathlib import Path

from theme import ROOT_CSS

HOW_TO_READ = """<ul>
<li>Top-down charts: <b style="color:var(--foreground)">driven path = white</b>, <b style="color:var(--chart-1)">Alpamayo = coloured</b>, <b style="color:var(--chart-3)">openpilot plan = dashed amber</b>. Car at the bottom, driving up, left is left.</li>
<li>Big dots follow the video time (clip runs from t0 - 2 s to t0 + 6.4 s); small dots mark 1, 3 and 6.4 s into the future.</li>
<li><b>ADE</b> = mean distance between the predicted and the driven path over the horizon (1, 3 or 6.4 s). <b>Lateral</b> = mean sideways error over 6.4 s. All values in metres, lower is better.</li>
<li>Green = the better of the pair (summary) or the best Alpamayo variant (case pages).</li>
</ul>"""


def between(text, start, end):
    i = text.index(start) + len(start)
    return text[i : text.index(end, i)]


def main():
    parser = argparse.ArgumentParser(description="Generate report/index.html from the report directory.")
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--title", default="Alpamayo 2 Edge vs openpilot")
    args = parser.parse_args()

    rd = Path(args.report_dir)
    summary = (rd / "summary.html").read_text()
    # reuse the summary page's table and its CSS (which starts with ROOT_CSS)
    css, body = between(summary, "<style>", "</style>"), between(summary, "<body>", "</body>")

    def links(prefix, label):
        pages = sorted(rd.glob(f"{prefix}_*.html"), key=lambda p: int(p.stem.rsplit("_", 1)[1]))
        return "".join(f'<a href="{p.name}">{label} {int(p.stem.rsplit("_", 1)[1])}</a>' for p in pages)

    cases, every = links("cases", "page"), links("all", "page")
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>{H_.escape(args.title)}</title><style>
{ROOT_CSS}{css.replace(ROOT_CSS, "")}
.wrap{{max-width:1100px;margin:0 auto}}
h1{{font-size:18px;margin:4px 0 10px}} h2{{font-size:14px;margin:18px 0 6px}}
ul{{margin:4px 0;padding-left:18px;line-height:1.6}}
.links a{{display:inline-block;margin:0 8px 6px 0;padding:3px 10px;border:1px solid var(--border);border-radius:var(--radius);color:var(--chart-1);text-decoration:none}}
.links a:hover{{border-color:var(--chart-1)}}
</style></head><body><div class="wrap">
<h1>{H_.escape(args.title)}</h1>
<h2>How to read</h2>
{HOW_TO_READ}
<h2>Summary</h2>
{body}
<h2>Picked cases (3 best and 3 worst vs openpilot, 3 worst absolute)</h2>
<div class="links">{cases or "none"}</div>
<h2>All samples</h2>
<div class="links">{every or "none"}</div>
</div></body></html>"""
    (rd / "index.html").write_text(html)
    print(f"{rd / 'index.html'} {len(html)} bytes")


if __name__ == "__main__":
    main()
