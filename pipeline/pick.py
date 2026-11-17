"""Pick N segments from scan stats: all mostly-engaged, top manual turns, rest spread by speed."""
import argparse
import csv
import os
import sys
from pathlib import Path

ROOT = os.environ.get("ALPAMAYO_ROOT", str(Path(__file__).resolve().parent.parent / "data"))


def main():
    parser = argparse.ArgumentParser(description="Pick N evaluation segments.")
    parser.add_argument("-n", "--count", type=int, default=60, help="Number of segments to pick")
    parser.add_argument("--scan-csv", default=f"{ROOT}/scan.csv", help="Path to scan.csv")
    parser.add_argument("--ondevice", default=f"{ROOT}/ondevice.txt", help="Path to ondevice.txt")
    parser.add_argument("--full-dir", default=f"{ROOT}/full", help="Directory of existing full segments")
    parser.add_argument("--out", help="Output file for segment list")
    parser.add_argument("--force", action="store_true", help="Overwrite output file if it exists")
    args = parser.parse_args()

    if args.out and Path(args.out).exists() and not args.force:
        if Path(args.out).stat().st_size > 0:
            print(f"pick: {args.out} already exists and is non-empty, refusing to overwrite.", file=sys.stderr)
            return

    dev = None
    if Path(args.ondevice).exists():
        dev = set(Path(args.ondevice).read_text().split())

    have = set(os.listdir(args.full_dir)) if Path(args.full_dir).exists() else set()

    if not Path(args.scan_csv).exists():
        print(f"pick error: scan CSV {args.scan_csv} not found.", file=sys.stderr)
        sys.exit(1)

    with open(args.scan_csv, mode="r", newline="") as f:
        rows = list(csv.reader(f))

    R = []
    cols = "seg vm vx st lat lon steer yaw hr".split()
    for r in rows:
        if not r or len(r) < len(cols):
            continue
        seg = r[0].strip()
        if dev is not None and seg not in dev:
            continue
        if seg in have:
            continue
        try:
            entry = dict(zip(cols, [seg] + [int(x) for x in r[1 : len(cols)]]))
            R.append(entry)
        except ValueError:
            continue

    R = [r for r in R if r["vm"] >= 10]
    if not R:
        print("pick error: no candidate segments matching criteria.", file=sys.stderr)
        sys.exit(1)

    target_n = min(args.count, len(R))
    eng = [r for r in R if r["lat"] > 30]
    man = [r for r in R if r["lat"] <= 30]

    if target_n <= len(eng):
        eng = eng[:target_n]
        turns = []
        spread = []
    else:
        needed = target_n - len(eng)
        turn_quota = min(len(man), max(1, round(target_n * 14 / 60))) if target_n >= 5 else min(len(man), needed // 2)
        turn_quota = min(turn_quota, needed)
        turns = sorted(man, key=lambda r: -r["yaw"])[:turn_quota]
        rest = sorted([r for r in man if r not in turns], key=lambda r: r["vm"])
        n_spread = target_n - len(eng) - len(turns)
        if n_spread > 0 and rest:
            if len(rest) <= n_spread:
                spread = rest
            elif n_spread == 1:
                spread = [rest[len(rest) // 2]]
            else:
                spread = [rest[round(i * (len(rest) - 1) / (n_spread - 1))] for i in range(n_spread)]
        else:
            spread = []

    pick = eng + turns + spread
    # Deduplicate while preserving order if any overlap
    seen = set()
    dedup = []
    for r in pick:
        if r["seg"] not in seen:
            seen.add(r["seg"])
            dedup.append(r)
    pick = dedup

    print(f"Picked {len(pick)} segments: engaged {len(eng)} turns {len(turns)} spread {len(spread)}")
    sorted_segs = sorted(r["seg"] for r in pick)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text("\n".join(sorted_segs) + "\n")
        print(f"Wrote {len(sorted_segs)} segments to {args.out}")
    else:
        for s in sorted_segs:
            print(s)


if __name__ == "__main__":
    main()
