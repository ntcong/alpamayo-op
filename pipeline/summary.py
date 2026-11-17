"""Breakdown of Alpamayo results vs openpilot; writes summary HTML and prints picked case stems."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from theme import ROOT_CSS


def ade(a, b, n=64):
    return float(np.linalg.norm(a[:n, :2] - b[:n, :2], axis=1).mean())


def main():
    parser = argparse.ArgumentParser(description="Generate summary breakdown table and pick cases.")
    parser.add_argument("--samples", required=True, help="Directory containing sample .npz files")
    parser.add_argument("--results", required=True, help="Directory containing result .json files")
    parser.add_argument("--out", required=True, help="Path for output summary HTML")
    parser.add_argument("--title", default="Alpamayo 2 Edge vs openpilot", help="Report title")
    args = parser.parse_args()

    samples_dir = Path(args.samples)
    results_dir = Path(args.results)

    json_files = sorted(results_dir.glob("*.json"))
    if not json_files:
        print(f"summary error: no JSON files found in {results_dir}", file=sys.stderr)
        sys.exit(1)

    rows = []
    max_k = 1
    for f in json_files:
        s = f.stem
        sample_path = samples_dir / f"{s}.npz"
        if not sample_path.exists():
            continue
        z = np.load(sample_path)
        gt, gd, op = z["gt_axle_xyz"], z["gt_dev_xyz"], z["op_plan_xyz"]
        res = json.loads(f.read_text())
        pred_all = np.asarray(res["pred_xyz"])
        pred_0 = pred_all[0] if pred_all.ndim == 3 else pred_all
        k = len(pred_all) if pred_all.ndim == 3 else 1
        if k > max_k:
            max_k = k

        min_ade64 = min(ade(pred_all[i], gt, 64) for i in range(k)) if k > 1 else ade(pred_0, gt, 64)
        kmh = float(z["v_ego"]) * 3.6
        nav_raw = str(z["nav_text"].item()) if "nav_text" in z else ""
        nav_word = nav_raw.split()[0] if nav_raw else ""

        rows.append(
            dict(
                stem=s,
                eng=bool(z["lat_active"]),
                kmh=kmh,
                turn=abs(float(gt[63, 1])) > 3,
                nav=nav_word,
                a=[np.abs(pred_0[:, 1] - gt[:, 1]).mean(), ade(pred_0, gt, 10), ade(pred_0, gt, 30), ade(pred_0, gt, 64)],
                o=[np.abs(op[:, 1] - gd[:, 1]).mean(), ade(op, gd, 10), ade(op, gd, 30), ade(op, gd, 64)],
                min_ade=min_ade64,
                k=k,
            )
        )

    if not rows:
        print("summary error: no valid sample-result pairs found.", file=sys.stderr)
        sys.exit(1)

    groups = [
        ("all", lambda r: True),
        ("engaged", lambda r: r["eng"]),
        ("manual", lambda r: not r["eng"]),
        ("driven straight (|lat| at 6.4 s <= 3 m)", lambda r: not r["turn"]),
        ("driven turn/curve (|lat| > 3 m)", lambda r: r["turn"]),
        ("nav says Turn", lambda r: r["nav"] == "Turn"),
        ("< 15 km/h", lambda r: r["kmh"] < 15),
        ("15-30 km/h", lambda r: 15 <= r["kmh"] < 30),
        ("30-50 km/h", lambda r: 30 <= r["kmh"] < 50),
        (">= 50 km/h", lambda r: r["kmh"] >= 50),
    ]

    has_k = max_k > 1
    tr = ""
    for name, flt in groups:
        g = [r for r in rows if flt(r)]
        if not g:
            continue
        a = np.array([r["a"] for r in g])
        o = np.array([r["o"] for r in g])
        min_ades = np.array([r["min_ade"] for r in g])
        win = float((a[:, 3] < o[:, 3]).mean() * 100)
        cells = ""
        for k_idx in (0, 1, 2, 3):
            am, om = a[:, k_idx].mean(), o[:, k_idx].mean()
            cells += f'<td class="{"w" if am < om else ""}">{am:.2f}</td><td class="{"w" if om < am else ""}">{om:.2f}</td>'
        cells += f"<td>{np.median(a[:, 3]):.2f} / {np.median(o[:, 3]):.2f}</td><td>{win:.0f} %</td>"
        if has_k:
            cells += f"<td>{min_ades.mean():.2f}</td>"
        tr += f"<tr><td>{name}</td><td>{len(g)}</td>{cells}</tr>"
        k_str = f" minADE {min_ades.mean():.2f}" if has_k else ""
        print(f"{name:42s} n={len(g):3d} lat {a[:,0].mean():.2f}/{o[:,0].mean():.2f} ADE6.4 {a[:,3].mean():.2f}/{o[:,3].mean():.2f} win {win:.0f}%{k_str}")

    extra_th = f'<th rowspan="2">minADE 6.4 s<br>(K={max_k})</th>' if has_k else ""
    note_k = f" minADE is minimum over K={max_k} trajectories." if has_k else ""

    html = f"""<!doctype html><html><head><meta charset="utf-8"><style>
{ROOT_CSS}body{{font-family:var(--font-sans);font-size:12px;margin:0}}
table{{border-collapse:collapse;width:100%}}
th,td{{padding:3px 5px;text-align:right;border-bottom:1px solid var(--border);font-variant-numeric:tabular-nums}}
th:first-child,td:first-child{{text-align:left}} th{{color:var(--muted-foreground);font-weight:500}}
td.w{{color:var(--success);font-weight:700}}
.n{{color:var(--muted-foreground);margin-top:8px}}
</style></head><body>
<div style="margin-bottom:6px"><b>{args.title}</b> &middot; {len(rows)} samples</div>
<table><tr><th rowspan="2"></th><th rowspan="2">n</th><th colspan="2">Lateral</th><th colspan="2">ADE 1 s</th><th colspan="2">ADE 3 s</th><th colspan="2">ADE 6.4 s</th><th rowspan="2">Median ADE 6.4 s<br>Alp / op</th><th rowspan="2">Alp wins<br>(ADE 6.4 s)</th>{extra_th}</tr>
<tr><th>Alp</th><th>op</th><th>Alp</th><th>op</th><th>Alp</th><th>op</th><th>Alp</th><th>op</th></tr>{tr}</table>
<div class="n">Metres. Green = lower of the pair. Alpamayo scored against the driven path in the rear-axle frame, openpilot in the device frame.{note_k}</div>
</body></html>"""

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html)

    # Pick cases: 3 best vs op, 3 worst vs op, 3 worst absolute
    d = sorted(rows, key=lambda r: r["a"][3] - r["o"][3])
    worst = sorted(rows, key=lambda r: -r["a"][3])
    n_pick = min(3, len(rows))
    picked = [r["stem"] for r in d[:n_pick]]
    for r in d[-n_pick:][::-1]:
        if r["stem"] not in picked:
            picked.append(r["stem"])
    for r in worst:
        if r["stem"] not in picked:
            picked.append(r["stem"])
        if len(picked) >= 9:
            break
    print("PICK", " ".join(picked))


if __name__ == "__main__":
    main()
