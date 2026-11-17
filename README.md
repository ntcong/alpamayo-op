# alpamayo-op

Offline evaluation of NVIDIA **Alpamayo 2 Edge** (a vision-language-action driving model) on comma/openpilot drive logs. For each sample it feeds the model 2 s of recorded camera frames plus the ego history, predicts the next 6.4 s of path, and compares that with the path the car actually drove and with openpilot's own plan from the same moment. The point is to see whether Alpamayo could replace or complement openpilot's planner, using only logs that are already on the device. Nothing runs on the car; inference is offline on a separate GPU host.

## Key findings so far

All measured, VinFast VF7 data only.

- Alpamayo takes **no camera intrinsics**. Each camera slot must match NVIDIA's rig field of view: front wide = 120 deg f-theta, front tele = 30 deg.
- The comma road camera (`fcamera`) is 61 deg (focal 1141.5 px at 1344x760). The comma wide camera (`ecamera`) is about 115 deg fisheye.
- Feeding the 61 deg image to the wide slot doubled the curve bending: lateral error 1.80 m there vs 0.46 m in the tele slot (9 samples).
- Best input: **`ecamera` in the front-wide slot + `fcamera` centre-cropped to 30 deg around the calibrated forward direction in the tele slot** (`prep.py --full`, `--slot both`).
- 300 city samples (60 segments, 4-50 km/h, 1 trajectory each, RTX 2060 Super, hybrid mode):

  | | Alpamayo | openpilot |
  |---|---|---|
  | ADE @ 6.4 s, mean (median) | 1.42 m (0.89) | 1.85 m (1.36) |
  | lateral error | 0.37 m | 0.46 m |
  | lateral, engaged only | 0.27 m | 0.16 m |

  Engaged-only favours openpilot: its plan steers the car when engaged, so the driven path follows it. Alpamayo wins 63 % of samples on ADE @ 6.4 s.
- About 45 s per sample on an 8 GB RTX 2060 Super (vision encoder runs on the CPU), so offline only.
- Highway (>= 50 km/h) is not covered yet (2 samples).

## Requirements

- Linux, `ffmpeg`, `rsync`, `ssh`/`scp`.
- Python 3 with `numpy`, `av`, `Pillow` (report stage needs only `numpy`).
- An openpilot checkout (any fork) that provides `openpilot.tools.lib.logreader` and `openpilot.common.transformations`; used to read `rlog.zst`/`qlog.zst`.
- ssh access to a comma device with drive logs in the device's standard `realdata` log folder (segments need `ecamera.hevc`, `fcamera.hevc`, `rlog.zst`, `qcamera.ts`). The device is only read from.
- A GPU host with >= 8 GB VRAM, reachable by key-based ssh, and Hugging Face access (accepted terms + read token) to `nvidia/Alpamayo2-Edge` and `nvidia/PhysicalAI-Autonomous-Vehicles`.
- The report stage alone (`report`) needs neither the device nor the GPU host, only a run directory with samples, results and clips.

## Setup

1. **Local settings.** `cp pipeline/config.example.env pipeline/config.env` and set `OPENPILOT_DIR`; optionally `PYTHON`, `ALPAMAYO_ROOT` (data root, default `data/` in this repo; a relative path is taken from the repo root), `COMMA_SSH` (ssh name of the comma device, default `comma`), `FONT` (TTF for the clip overlay). Environment variables work too; `config.env` wins when both set a value. `config.env` is gitignored.
2. **GPU host profile.** `cp pipeline/hosts/example.env pipeline/hosts/<name>.env` and edit `SSH_TARGET`, `REMOTE_BASE`, `NVIDIA_DIR`, `HOST_FLAGS`, `MIN_FREE_MIB`. Remote paths may start with `~` (expanded on the host). Select it with `--host <name>` (default `pc`). Profiles are gitignored.
3. **GPU host install** (run on the GPU host, needs `uv`, `git`, an NVIDIA driver with CUDA 12 support):
   - Cards up to ~16 GB: `bash edge-setup.sh hybrid`. Clones NVIDIA's `alpamayo2-edge` to `~/alpamayo/alpamayo2-edge`, installs the pinned environment, logs in to Hugging Face, and runs NVIDIA's sample with the vision encoder on the CPU. It uses `pacman` to install `git`/`uv` when missing; on other distros install them first. Set `HOST_FLAGS="--hybrid --num-traj 1"`.
   - Big GPUs: `bash pipeline/spark-setup.sh` (stock GPU mode, `HOST_FLAGS="--num-traj 6"`). **Not run on a DGX Spark yet.**
4. Check the geometry code: `OPENPILOT_DIR=../openpilot python -m pytest -q test_geometry.py` (tests that need openpilot are skipped when it is missing).

## Stages

Everything goes through `pipeline/run.sh <run_name> <stage>... [--host <name>] [--variant both] [--step-frames 200] [-n 60]`. A run lives in `$ALPAMAYO_ROOT/runs/<run_name>/`. All stages are idempotent and resumable.

| Stage | Command | Reads | Writes |
|---|---|---|---|
| scan | `./pipeline/run.sh r1 scan` | comma `qlog.zst` of every segment + list of complete segments | `$ROOT/qlogs/`, `$ROOT/ondevice.txt`, `$ROOT/scan.csv` (per-segment speed, engagement, turning; only new segments are scanned) |
| pick | `./pipeline/run.sh r1 pick -n 60` | `scan.csv`, `ondevice.txt` | `runs/r1/segments.txt` (all mostly-engaged segments, top manual turns, rest spread over speed; segments >= 10 km/h mean; refuses to overwrite). You can write `segments.txt` by hand instead, one segment name per line, e.g. `00000141--f1d8ad1647--13`. |
| copy | `./pipeline/run.sh r1 copy` | comma | `$ROOT/full/<seg>/{ecamera.hevc,fcamera.hevc,rlog.zst,qcamera.ts,.done}` (shared between runs) |
| prep | `./pipeline/run.sh r1 prep` | `full/<seg>/` | `runs/r1/samples/<seg>__<frame>.npz`, one every `--step-frames` frames (default 200, about 5 per segment) |
| clips | `./pipeline/run.sh r1 clips` | `samples/`, `full/<seg>/qcamera.ts` | `runs/r1/clips/c_<stem>.mp4`: 8.4 s clip (t0 - 2 s to t0 + 6.4 s) with route-second overlay |
| run | `./pipeline/run.sh r1 run --host pc` | `samples/` | rsyncs code + samples to the GPU host, runs `spark/run_edge.py --slot both`, waits, pulls `runs/r1/results/both/<stem>.json`. Refuses to start when less than `MIN_FREE_MIB` GPU memory is free. Re-running resumes. |
| report | `./pipeline/run.sh r1 report` | `samples/`, `results/both/`, `clips/` | `runs/r1/report/index.html`, `summary.html`, `cases_*.html`, `all_*.html` (each page < 512 KB) |
| all | `./pipeline/run.sh r1 all --host pc` | | copy, prep, clips, run, report in order |
| status | `./pipeline/run.sh r1 status --host pc` | | prints counts per stage and the GPU runner state |

`--variant <name>` selects `results/<name>/` (default `both`). `run` always passes `--slot both`.

**Typical session**

```sh
./pipeline/run.sh r1 scan
./pipeline/run.sh r1 pick -n 60
# detached full run (hours on an 8 GB card); follow with `tail -f` or `status`
mkdir -p data/runs/r1   # adjust if you set ALPAMAYO_ROOT
setsid nohup ./pipeline/run.sh r1 all --host pc < /dev/null >> data/runs/r1/run.log 2>&1 &
./pipeline/run.sh r1 status --host pc
```

**The report.** Open `runs/<run_name>/report/index.html` in a browser. It has the summary table, a "how to read" note and links to the case pages (3 best and 3 worst vs openpilot, 3 worst absolute) and to all-sample pages. In the charts the driven path is white, Alpamayo coloured, openpilot dashed amber; big dots follow the video time. ADE is the mean distance between predicted and driven path over the horizon; lateral is the mean sideways error; all in metres. To rebuild a report without ssh or GPU, put `samples/*.npz`, `results/both/*.json` and `clips/c_<stem>.mp4` in a run directory and run `./pipeline/run.sh <run_name> report`.

## Run directory

```
$ALPAMAYO_ROOT/
  qlogs/ ondevice.txt scan.csv      scan stage
  full/<seg>/                       copied logs and videos (shared)
  runs/<run_name>/
    segments.txt                    one segment per line
    samples/<seg>__<frame>.npz
    clips/c_<seg>__<frame>.mp4
    results/<variant>/<seg>__<frame>.json
    report/index.html summary.html cases_N.html all_N.html
    run.log                         when started detached
```

On the GPU host: `$REMOTE_BASE/alpamayo-op/` (synced code) and `$REMOTE_BASE/runs/<run_name>/{samples,results,run.log}`.

## Data format

**Sample `.npz`** (written by `prep.py`; `--full` adds `frames_wide`):

| key | content |
|---|---|
| `frames` | uint8 `[5,330,584,3]` RGB: `fcamera` centre-cropped to 30 deg (tele slot) |
| `frames_wide` | uint8 `[5,330,584,3]`: `ecamera` (front-wide slot) |
| `frame_ts_us` | camera timestamps of the 5 frames (t0 - 2 s ... t0) |
| `ego_history_xyz`, `ego_history_rot` | 20 steps (0.1 s) of history before t0, `[20,3]` / `[20,3,3]`, in the t0 rear-axle frame |
| `gt_axle_xyz` | driven path, 64 steps (0.1 ... 6.4 s), rear-axle frame (x forward, y left) |
| `gt_dev_xyz` | driven path in the openpilot device frame (y right) |
| `op_plan_xyz` | openpilot `modelV2` plan at t0, device frame, resampled to the same 64 steps |
| `segment`, `i0`, `route_s` | segment name, frame index of t0, route second |
| `v_ego`, `lat_active`, `long_active` | `carState.vEgo` at t0, engagement flags |
| `nav_text` | navigation prompt, e.g. `Turn left in 30m` |

**Result `.json`** (written by `spark/run_edge.py`): `pred_xyz` `[K,64,3]` rear-axle frame, `cot` (reasoning text per trajectory), `seconds`, `warmup`, `slot`, `nav_text`, `model_id`, `revision`, `seed`, `hybrid`.

## Coordinate frames

Alpamayo uses the rear-axle frame: x forward, y left, z up, origin at the rear axle at t0. openpilot's plan is in the device frame with y to the right. The driven path is computed from `deviceMotion` (direction, orientation) with the segment's final calibration, and speed from `carState.vEgo`; it is stored in both frames. Alpamayo is scored against `gt_axle_xyz`, openpilot against `gt_dev_xyz` (`report.py` flips the sign of openpilot's y only for drawing).

## Known limits

- Navigation text is derived from the driven future (turn within 15 s), so it is an oracle the real system would not have.
- Ground-truth speed comes from `carState.vEgo`; it can read 1-2 % above GPS at 90 km/h. openpilot's plan is in the vision speed scale, so its along-track ADE includes that scale error; the lateral number does not.
- Engaged samples favour openpilot: the car follows its plan.
- The dashboard is visible in the bottom ~30 % of `ecamera`; `ecamera` is not corrected for mount yaw.
- One trajectory per sample on 8 GB cards, so no minADE.
- Only VinFast VF7 data so far; highway is barely covered.

## Files

- `prep.py`, `geom.py`: log reading, geometry, sample building. `test_geometry.py`: frame-convention checks.
- `spark/run_edge.py`, `spark/hybrid.py`: inference on the GPU host (`hybrid.py` moves the vision encoder to the CPU).
- `pipeline/`: `run.sh` driver, `scan.py`, `pick.py`, `summary.py`, `report.py`, `index.py`, `theme.py`, setup scripts and profiles.
- `edge-setup.sh`: install + NVIDIA sample on a PC (CachyOS/pacman by default).
