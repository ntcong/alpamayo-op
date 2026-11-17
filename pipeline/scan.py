"""Per-segment stats from qlog.zst: appends new stats to scan.csv without rescanning."""
import argparse
import csv
import glob
import os
import sys
from multiprocessing import Pool
from pathlib import Path
from unittest.mock import MagicMock

sys.modules.setdefault("msgq.ipc_pyx", MagicMock())
sys.modules.setdefault("openpilot.common.params_pyx", MagicMock())
import numpy as np
from openpilot.tools.lib.logreader import LogReader

ROOT = os.environ.get("ALPAMAYO_ROOT", str(Path(__file__).resolve().parent.parent / "data"))


def scan_segment(path):
    seg = Path(path).parent.name
    v, steer, lat_on, long_on, yaw, hour = [], [], [], [], [], -1
    for m in LogReader(path):
        w = m.which()
        if w == "carState":
            v.append(m.carState.vEgo)
            steer.append(m.carState.steeringAngleDeg)
        elif w == "carControl":
            lat_on.append(m.carControl.latActive)
            long_on.append(m.carControl.longActive)
        elif w == "deviceMotion":
            yaw.append(m.deviceMotion.orientationNED.z)
        elif w == "gpsLocationExternal" and hour < 0 and m.gpsLocationExternal.unixTimestampMillis > 0:
            hour = int((m.gpsLocationExternal.unixTimestampMillis / 1000 / 3600 + 7) % 24)  # Vietnam local hour
    v = np.array(v or [0.0])
    yaw = np.unwrap(np.array(yaw or [0.0]))
    return ",".join(
        str(x)
        for x in (
            seg,
            round(float(v.mean()) * 3.6),
            round(float(v.max()) * 3.6),
            round(float((v < 0.3).mean()) * 60),
            round(float(np.mean(lat_on or [0])) * 60),
            round(float(np.mean(long_on or [0])) * 60),
            round(float(np.abs(steer or [0]).max())),
            round(float(np.degrees(yaw.max() - yaw.min()))),
            hour,
        )
    )


def main():
    parser = argparse.ArgumentParser(description="Scan qlogs and append stats to CSV.")
    parser.add_argument("--qlogs", default=f"{ROOT}/qlogs", help="Directory with segment qlogs")
    parser.add_argument("--csv", default=f"{ROOT}/scan.csv", help="Path to output CSV")
    parser.add_argument("--jobs", type=int, default=8, help="Number of parallel workers")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    existing_segs = set()
    if csv_path.exists():
        with open(csv_path, mode="r", newline="") as f:
            for row in csv.reader(f):
                if row:
                    existing_segs.add(row[0].strip())

    all_paths = sorted(glob.glob(os.path.join(args.qlogs, "*--*/qlog.zst")))
    new_paths = [p for p in all_paths if Path(p).parent.name not in existing_segs]

    if not new_paths:
        print(f"No new qlogs to scan (already have {len(existing_segs)} segments).")
        return

    print(f"Scanning {len(new_paths)} new segments ({len(existing_segs)} already recorded)...")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "a", newline="") as f, Pool(args.jobs) as pool:
        for line in pool.imap(scan_segment, new_paths):
            f.write(line + "\n")
            f.flush()
            print(line, flush=True)


if __name__ == "__main__":
    main()
