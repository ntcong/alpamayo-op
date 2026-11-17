"""Multi-variant report pages: video + synced top-down chart + table per sample."""
import argparse
import base64
import html as H_
import json
import math
import sys
from pathlib import Path

import numpy as np
from theme import ROOT_CSS

W, H = 240, 300
COLORS = ["--chart-1", "--chart-2", "--chart-4", "--chart-5"]
PAGE_BYTES, PAGE_CARDS = 490_000, 3


def ade(a, b, sec):
    n = int(sec * 10)
    return float(np.linalg.norm(a[:n, :2] - b[:n, :2], axis=1).mean())


def chart(paths, styles):
    """paths: list of (64,3) arrays, left positive; styles: list of (css var, dash)."""
    a1 = max(40, math.ceil(max(float(p[:, 0].max()) for p in paths) / 10) * 10 + 4)
    half = max(4, math.ceil(max(float(np.abs(p[:, 1]).max()) for p in paths)) + 1)

    def px(p):
        return (W / 2 - p[1] * W / (2 * half), H - 10 - p[0] * (H - 20) / a1)

    step_a = 10 if a1 <= 60 else 25 if a1 <= 130 else 50
    g = ""
    for m in range(0, a1 + 1, step_a):
        y = px((m, 0))[1]
        g += f'<line x1="0" x2="{W}" y1="{y:.1f}" y2="{y:.1f}" stroke="var(--border)"/><text x="2" y="{y - 3:.1f}" font-size="10" fill="var(--muted-foreground)">{m} m</text>'
    step_l = 2 if half <= 6 else 5 if half <= 15 else 10
    for m in [k for k in range(-half, half + 1) if k and k % step_l == 0]:
        x = px((0, m))[0]
        g += f'<line x1="{x:.1f}" x2="{x:.1f}" y1="0" y2="{H}" stroke="var(--border)" stroke-dasharray="2 3"/>'
        g += f'<text x="{x + 2:.1f}" y="{H - 2}" font-size="9" fill="var(--muted-foreground)">{"L" if m > 0 else "R"}{abs(m)}</text>'
    o = px((0, 0))
    g += f'<line x1="{o[0]:.1f}" x2="{o[0]:.1f}" y1="0" y2="{H}" stroke="var(--border)"/>'
    for p, (c, dash) in zip(paths, styles):
        pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in map(px, p))
        d = f' stroke-dasharray="{dash}"' if dash else ""
        g += f'<polyline points="{pts}" fill="none" stroke="var({c})" stroke-width="2.2"{d}/>'
        g += "".join(f'<circle cx="{px(p[i])[0]:.1f}" cy="{px(p[i])[1]:.1f}" r="2.5" fill="var({c})"/>' for i in (9, 29, 63))
    g += "".join(
        f'<circle class="mk" r="5.5" cx="{o[0]:.1f}" cy="{o[1]:.1f}" fill="var({c})" stroke="var(--background)" stroke-width="1.5"/>'
        for c, _ in styles
    )
    stretch = (W / (2 * half)) / ((H - 20) / a1)
    g += f'<text class="tl" x="{W - 4}" y="14" text-anchor="end" font-size="12" fill="var(--foreground)">t0 -2.0 s</text>'
    g += f'<text x="{W - 4}" y="27" text-anchor="end" font-size="10" fill="var(--muted-foreground)">lateral x{stretch:.0f}</text>'
    data = json.dumps([[[round(float(v), 1) for v in px(q)] for q in p] for p in paths])
    return f'<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}">{g}</svg>', json.dumps([round(o[0], 1), round(o[1], 1)]), data


def card(stem, variants, samples_dir, clips_dir):
    seg, _ = stem.split("__")
    z = np.load(samples_dir / f"{stem}.npz")
    gt, gtd = z["gt_axle_xyz"], z["gt_dev_xyz"]
    op = z["op_plan_xyz"].copy()
    op[:, 1] *= -1  # device frame is right-positive
    preds, secs, cots = [], [], []
    for name, rdir in variants:
        r = json.loads((rdir / f"{stem}.json").read_text())
        pred_arr = np.asarray(r["pred_xyz"])
        pred_0 = pred_arr[0] if pred_arr.ndim == 3 else pred_arr
        preds.append(pred_0)
        secs.append(float(r["seconds"]))
        cot_val = r.get("cot", "<NOMINAL>")
        cots.append((name, str(cot_val[0]) if isinstance(cot_val, list) else str(cot_val)))
    styles = [("--foreground", ""), ("--chart-3", "6 4")] + [(COLORS[i % len(COLORS)], "") for i in range(len(variants))]
    svg, o, data = chart([gt, op] + preds, styles)
    best = min(range(len(preds)), key=lambda i: ade(preds[i], gt, 6.4))
    rows = [("Driven", "--foreground", gt, None)]
    rows += [(f"Alpamayo {n}", COLORS[i % len(COLORS)], preds[i], i) for i, (n, _) in enumerate(variants)]
    rows += [("openpilot", "--chart-3", op, "op")]
    tr = ""
    for name, c, p, k in rows:
        end = f"{p[63, 0]:.1f} / {p[63, 1] + 0.0:+.1f}".replace("-0.0", "0.0")
        if k is None:
            e = "<td>-</td>" * 4
        elif k == "op":
            e = "".join(f"<td>{ade(z['op_plan_xyz'], gtd, s):.2f}</td>" for s in (1, 3, 6.4)) + f"<td>{np.abs(z['op_plan_xyz'][:, 1] - gtd[:, 1]).mean():.2f}</td>"
        else:
            e = "".join(f"<td>{ade(p, gt, s):.2f}</td>" for s in (1, 3)) + f"<td{' class=b' if k == best else ''}>{ade(p, gt, 6.4):.2f}</td><td>{np.abs(p[:, 1] - gt[:, 1]).mean():.2f}</td>"
        tr += f'<tr><td><i class="sw" style="background:var({c})"></i>{name}</td><td>{end}</td>{e}</tr>'
    cot = "".join(f"<div><b>{n}</b>: <code>{H_.escape(c)}</code></div>" for n, c in cots if c != "<NOMINAL>") or "<div>all <code>&lt;NOMINAL&gt;</code></div>"
    t = int(float(z["route_s"]))
    mode = "engaged" if bool(z["lat_active"]) else "manual"

    clip_candidates = [clips_dir / f"c_{stem}.mp4", clips_dir / f"{stem}.mp4"]
    clip_file = None
    for cand in clip_candidates:
        if cand.exists():
            clip_file = cand
            break
    if clip_file is None:
        raise FileNotFoundError(f"Clip not found for stem {stem} in {clips_dir}")

    video = base64.b64encode(clip_file.read_bytes()).decode()
    return f"""
<div class="card" data-o="{o}" data-paths="{data}">
 <div class="hd"><b>{seg.rsplit('--', 1)[0]}</b> route s <b>{t}</b> &middot; {mode} &middot; {z['v_ego'] * 3.6:.0f} km/h &middot; nav <code>{z['nav_text'].item()}</code> &middot; ~{np.mean(secs):.0f} s/run</div>
 <div class="row"><video controls muted loop playsinline preload="auto" src="data:video/mp4;base64,{video}"></video>{svg}</div>
 <table><tr><th></th><th>End 6.4 s (ahead / left)</th><th>ADE 1 s</th><th>ADE 3 s</th><th>ADE 6.4 s</th><th>Lat</th></tr>{tr}</table>
 <div class="cot">Reasoning (non-nominal): {cot}</div>
</div>"""


SCRIPT = """<script>
document.querySelectorAll('.card').forEach(c => {
  const v = c.querySelector('video'), P = JSON.parse(c.dataset.paths), o = JSON.parse(c.dataset.o);
  const mk = [...c.querySelectorAll('.mk')], tl = c.querySelector('.tl');
  const at = (p, f) => {
    if (f <= -1) return o;
    if (f < 0) return [o[0] + (p[0][0] - o[0]) * (f + 1), o[1] + (p[0][1] - o[1]) * (f + 1)];
    if (f >= 63) return p[63];
    const i = Math.floor(f), w = f - i;
    return [p[i][0] + (p[i + 1][0] - p[i][0]) * w, p[i][1] + (p[i + 1][1] - p[i][1]) * w];
  };
  const upd = () => {
    const tau = v.currentTime - 2;
    mk.forEach((e, j) => { const q = at(P[j], tau * 10 - 1); e.setAttribute('cx', q[0]); e.setAttribute('cy', q[1]); });
    tl.textContent = 't0 ' + (tau < 0 ? '' : '+') + tau.toFixed(1) + ' s';
  };
  const loop = () => { upd(); if (!v.paused) requestAnimationFrame(loop); };
  v.addEventListener('play', loop); v.addEventListener('seeked', upd); v.addEventListener('timeupdate', upd);
  upd();
});
</script>"""


def page(title, variants, cards, n, total):
    legend = '<span><i class="sw" style="background:var(--foreground)"></i>Driven</span>'
    legend += "".join(f'<span><i class="sw" style="background:var({COLORS[i % len(COLORS)]})"></i>Alpamayo {n_}</span>' for i, (n_, _) in enumerate(variants))
    legend += '<span><i class="sw" style="border-top:3px dashed var(--chart-3);height:0"></i>openpilot</span>'
    return f"""<!doctype html><html><head><meta charset="utf-8"><style>
{ROOT_CSS}body{{font-family:var(--font-sans);font-size:12px;margin:0}}
.card{{border:1px solid var(--border);border-radius:var(--radius);padding:12px;margin-bottom:10px}}
.hd{{margin-bottom:6px}}
.row{{display:flex;flex-wrap:wrap;gap:10px;align-items:flex-start}}
.row video{{flex:1 1 300px;max-width:100%;min-width:0;border-radius:var(--radius)}}
table{{border-collapse:collapse;margin-top:6px;width:100%}}
th,td{{padding:2px 5px;text-align:right;border-bottom:1px solid var(--border);font-variant-numeric:tabular-nums}}
th:first-child,td:first-child{{text-align:left}} th{{color:var(--muted-foreground);font-weight:500}}
td.b{{font-weight:700;color:var(--success)}}
.cot{{margin-top:4px;color:var(--muted-foreground)}}
.legend span{{margin-right:12px}} .sw{{display:inline-block;width:16px;height:3px;vertical-align:middle;margin-right:4px}}
</style></head><body>
<div class="legend" style="margin-bottom:6px"><b>{H_.escape(title)}</b> &middot; page {n}/{total} &middot; {legend}</div>
<div style="margin-bottom:8px;color:var(--muted-foreground)">Clip t0 - 2 s to t0 + 6.4 s (10 fps). Big dots follow the video time; small dots at 1, 3, 6.4 s. Metres, left positive. Green = best Alpamayo ADE 6.4 s.</div>
{''.join(cards)}
{SCRIPT}
</body></html>"""


def main():
    parser = argparse.ArgumentParser(description="Generate visual report pages.")
    parser.add_argument("--samples", required=True, help="Directory containing sample .npz files")
    parser.add_argument("--results", required=True, help="Base results dir or variant dir")
    parser.add_argument("--clips", required=True, help="Directory containing clip .mp4 files")
    parser.add_argument("--variants", required=True, help="Comma-separated name=path or variant name")
    parser.add_argument("--out-prefix", required=True, help="Output file prefix (e.g. report/cases)")
    parser.add_argument("--title", default="Alpamayo Evaluation", help="Page title")
    parser.add_argument("stems", nargs="*", help="Optional list of stems to include")
    args = parser.parse_args()

    samples_dir = Path(args.samples)
    results_base = Path(args.results)
    clips_dir = Path(args.clips)

    variants = []
    for item in args.variants.split(","):
        if "=" in item:
            name, path_str = item.split("=", 1)
            variants.append((name, Path(path_str)))
        else:
            # If variant name only, look under results_base or use as relative path
            path = results_base / item if (results_base / item).is_dir() else results_base
            variants.append((item, path))

    stems = args.stems
    if not stems:
        first_rdir = variants[0][1]
        stems = sorted(p.stem for p in first_rdir.glob("*.json"))

    if not stems:
        print("report: no stems to process.", file=sys.stderr)
        return

    cards = [card(s, variants, samples_dir, clips_dir) for s in stems]
    groups, cur, size = [], [], 0
    for c in cards:
        if cur and (size + len(c) > PAGE_BYTES or len(cur) == PAGE_CARDS):
            groups.append(cur)
            cur, size = [], 0
        cur.append(c)
        size += len(c)
    if cur:
        groups.append(cur)

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    for n, g in enumerate(groups, 1):
        html = page(args.title, variants, g, n, len(groups))
        if len(html) >= 512_000:
            print(f"WARNING: {out_prefix}_{n}.html size is {len(html)} bytes (>= 512 KB)", file=sys.stderr)
        out_file = Path(f"{args.out_prefix}_{n}.html")
        out_file.write_text(html)
        print(f"{out_file} {len(html)} bytes ({len(g)} cards)")


if __name__ == "__main__":
    main()
