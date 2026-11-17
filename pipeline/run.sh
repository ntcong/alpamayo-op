#!/usr/bin/env bash
# Pipeline driver for Alpamayo 2 Edge offline evaluation on openpilot drive logs.
#
# Usage:
#   pipeline/run.sh <run_name> <stage>... [options]
#
# Stages:
#   scan          Rsync new qlogs from comma, record ondevice list, append stats to scan.csv
#   pick [N]      Pick N evaluation segments into runs/<run_name>/segments.txt (refuses to overwrite)
#   copy          Copy missing segment logs/videos from comma to $ALPAMAYO_ROOT/full/
#   prep          Extract 5-frame samples (prep.py --full --step-frames <N>) into runs/<run_name>/samples/
#   clips         Render video clips with route-second overlay into runs/<run_name>/clips/
#   run           Sync code & samples to GPU host, run inference, wait, pull results
#   report        Build report/index.html + summary, picked-case and all-sample pages (local only: no ssh, no GPU)
#   all           Run copy -> prep -> clips -> run -> report
#   status        Print one-shot status of stages and GPU runner
#
# Options:
#   --host <name>         GPU host profile in pipeline/hosts/<name>.env (default: pc)
#   --step-frames <n>     Frames step for prep (default: 200, ~5 samples/segment)
#   --variant <name>      Variant name for results (default: both)
#   -n, --count <n>       Number of segments for pick stage (default: 60)
#
# Settings (environment or pipeline/config.env, see pipeline/config.example.env):
#   OPENPILOT_DIR  openpilot checkout providing openpilot.tools.lib.logreader (scan, prep)
#   PYTHON         python3 with numpy, av, Pillow (default: python3)
#   ALPAMAYO_ROOT  data root (default: <repo>/data)
#   COMMA_SSH      ssh name of the comma device (default: comma)
#   FONT           TTF used by ffmpeg drawtext (default: DejaVuSans)
#
# Detached execution:
#   setsid nohup ./pipeline/run.sh <run_name> all --host pc < /dev/null >> $ALPAMAYO_ROOT/runs/<run_name>/run.log 2>&1 &

set -euo pipefail

# Rule: never use /tmp
TMPDIR="/var/tmp"
export TMPDIR

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Local settings; values in config.env override the environment.
if [[ -f "$REPO_DIR/pipeline/config.env" ]]; then
  # shellcheck source=/dev/null
  source "$REPO_DIR/pipeline/config.env"
fi
OPENPILOT_DIR="${OPENPILOT_DIR:-}"
PYTHON="${PYTHON:-python3}"
ROOT="${ALPAMAYO_ROOT:-$REPO_DIR/data}"
[[ "$ROOT" = /* ]] || ROOT="$REPO_DIR/$ROOT"  # relative ALPAMAYO_ROOT is relative to the repo
COMMA_SSH="${COMMA_SSH:-comma}"
FONT="${FONT:-/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf}"
FULL_DIR="$ROOT/full"

HOST="${HOST:-pc}"
STEP_FRAMES=200
VARIANT="${VARIANT:-both}"
COUNT=60

POSITIONAL=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)
      HOST="$2"
      shift 2
      ;;
    --step-frames)
      STEP_FRAMES="$2"
      shift 2
      ;;
    --variant)
      VARIANT="$2"
      shift 2
      ;;
    -n|--count)
      COUNT="$2"
      shift 2
      ;;
    -h|--help)
      grep '^# ' "${BASH_SOURCE[0]}" | cut -c 3-
      exit 0
      ;;
    *)
      POSITIONAL+=("$1")
      shift
      ;;
  esac
done

if [[ ${#POSITIONAL[@]} -lt 1 ]]; then
  echo "Error: run_name required. See --help." >&2
  exit 1
fi

RUN_NAME="${POSITIONAL[0]}"
STAGES=("${POSITIONAL[@]:1}")

if [[ ${#STAGES[@]} -eq 0 ]]; then
  STAGES=("status")
fi

RUN_DIR="$ROOT/runs/$RUN_NAME"
SAMPLES_DIR="$RUN_DIR/samples"
RESULTS_DIR="$RUN_DIR/results/$VARIANT"
CLIPS_DIR="$RUN_DIR/clips"
REPORT_DIR="$RUN_DIR/report"
LOG_FILE="$RUN_DIR/run.log"

mkdir -p "$RUN_DIR" "$SAMPLES_DIR" "$RESULTS_DIR" "$CLIPS_DIR" "$REPORT_DIR"

log() {
  echo "$(date +%T) $*"
}

load_host_config() {
  local cfg="$REPO_DIR/pipeline/hosts/${HOST}.env"
  if [[ ! -f "$cfg" ]]; then
    echo "Host configuration $cfg not found." >&2
    exit 1
  fi
  # shellcheck source=/dev/null
  source "$cfg"
}

# Fail early when OPENPILOT_DIR cannot provide LogReader.
require_openpilot() {
  if [[ -z "$OPENPILOT_DIR" || ! -f "$OPENPILOT_DIR/openpilot/tools/lib/logreader.py" ]]; then
    echo "Error: set OPENPILOT_DIR to an openpilot checkout (needs openpilot/tools/lib/logreader.py); got '${OPENPILOT_DIR}'." >&2
    exit 1
  fi
}

# Expand a leading ~ in REMOTE_BASE / NVIDIA_DIR on the GPU host (a ~ typed in the profile reaches it unexpanded).
resolve_remote_paths() {
  REMOTE_BASE="$(ssh -o BatchMode=yes "$SSH_TARGET" "echo $REMOTE_BASE")"
  NVIDIA_DIR="$(ssh -o BatchMode=yes "$SSH_TARGET" "echo $NVIDIA_DIR")"
}

stage_scan() {
  log "--- Stage: scan ---"
  require_openpilot
  mkdir -p "$ROOT/qlogs"
  log "Rsyncing qlogs from comma (read-only)..."
  rsync -a -q -e "ssh -o BatchMode=yes -o ConnectTimeout=10" \
    --include='*/' --include='qlog.zst' --exclude='*' \
    "$COMMA_SSH:/data/media/0/realdata/" "$ROOT/qlogs/" || log "Warning: comma rsync returned non-zero"

  log "Listing on-device segments having ecamera+fcamera+rlog..."
  ssh -o BatchMode=yes -o ConnectTimeout=10 "$COMMA_SSH" \
    'for d in /data/media/0/realdata/*; do [ -f "$d/ecamera.hevc" ] && [ -f "$d/fcamera.hevc" ] && [ -f "$d/rlog.zst" ] && basename "$d"; done' \
    > "$ROOT/ondevice.txt.tmp" && mv "$ROOT/ondevice.txt.tmp" "$ROOT/ondevice.txt" || log "Warning: failed to fetch ondevice list"

  log "Scanning new qlogs into scan.csv..."
  PYTHONPATH="$OPENPILOT_DIR" "$PYTHON" "$REPO_DIR/pipeline/scan.py" --qlogs "$ROOT/qlogs" --csv "$ROOT/scan.csv"
  log "scan completed."
}

stage_pick() {
  local count="${1:-$COUNT}"
  log "--- Stage: pick (N=$count) ---"
  local seg_file="$RUN_DIR/segments.txt"
  if [[ -s "$seg_file" ]]; then
    log "pick: $seg_file already exists and is non-empty, refusing to overwrite."
    return 0
  fi

  "$PYTHON" "$REPO_DIR/pipeline/pick.py" \
    -n "$count" \
    --scan-csv "$ROOT/scan.csv" \
    --ondevice "$ROOT/ondevice.txt" \
    --full-dir "$FULL_DIR" \
    --out "$seg_file"
  log "pick completed."
}

stage_copy() {
  log "--- Stage: copy ---"
  local seg_file="$RUN_DIR/segments.txt"
  if [[ ! -f "$seg_file" ]]; then
    echo "Error: $seg_file not found. Run 'pick' or supply segments.txt first." >&2
    exit 1
  fi

  mkdir -p "$FULL_DIR"
  local count_ok=0 count_fail=0 count_skip=0

  while read -r seg || [[ -n "$seg" ]]; do
    [[ -z "$seg" ]] && continue
    local s_dir="$FULL_DIR/$seg"

    if [[ -f "$s_dir/.done" ]]; then
      count_skip=$((count_skip + 1))
      continue
    fi

    # If all 4 files already exist and non-empty, mark as .done without copying
    if [[ -s "$s_dir/ecamera.hevc" ]] && [[ -s "$s_dir/fcamera.hevc" ]] && \
       [[ -s "$s_dir/rlog.zst" ]] && [[ -s "$s_dir/qcamera.ts" ]]; then
      touch "$s_dir/.done"
      log "ok $seg (marked existing files .done)"
      count_ok=$((count_ok + 1))
      continue
    fi

    mkdir -p "$s_dir"
    log "Copying $seg from comma..."
    if timeout 900 scp -q -o BatchMode=yes -o ConnectTimeout=10 \
        "$COMMA_SSH:/data/media/0/realdata/$seg/{ecamera.hevc,fcamera.hevc,rlog.zst,qcamera.ts}" "$s_dir/"; then
      touch "$s_dir/.done"
      log "ok $seg"
      count_ok=$((count_ok + 1))
    else
      log "FAIL $seg"
      count_fail=$((count_fail + 1))
    fi
  done < "$seg_file"

  log "copy finished: $count_ok ok, $count_fail failed, $count_skip previously completed."
}

stage_prep() {
  log "--- Stage: prep (step-frames=$STEP_FRAMES) ---"
  require_openpilot
  local seg_file="$RUN_DIR/segments.txt"
  if [[ ! -f "$seg_file" ]]; then
    echo "Error: $seg_file not found." >&2
    exit 1
  fi

  # Resumable: find segments lacking samples in $SAMPLES_DIR
  local pending=()
  while read -r seg || [[ -n "$seg" ]]; do
    [[ -z "$seg" ]] && continue
    if [[ -f "$FULL_DIR/$seg/.done" ]]; then
      if ! compgen -G "$SAMPLES_DIR/${seg}__*.npz" >/dev/null; then
        pending+=("$FULL_DIR/$seg")
      fi
    fi
  done < "$seg_file"

  if [[ ${#pending[@]} -eq 0 ]]; then
    log "prep: all segments already prepped in $SAMPLES_DIR."
  else
    log "prep: running prep.py on ${#pending[@]} segments..."
    printf "%s\n" "${pending[@]}" | \
      xargs -r -P 2 -n 1 env PYTHONPATH="$OPENPILOT_DIR" "$PYTHON" "$REPO_DIR/prep.py" \
        --full --step-frames "$STEP_FRAMES" --out "$SAMPLES_DIR"
  fi
  local n_samples
  n_samples="$(compgen -G "$SAMPLES_DIR/*.npz" | wc -l || true)"
  log "prep completed: $n_samples samples in $SAMPLES_DIR."
}

stage_clips() {
  log "--- Stage: clips ---"
  if [[ ! -f "$FONT" ]]; then
    echo "Error: font $FONT not found; set FONT to a TTF file." >&2
    exit 1
  fi
  local n_done=0 n_skip=0

  shopt -s nullglob
  local sample_files=("$SAMPLES_DIR"/*.npz)
  shopt -u nullglob

  if [[ ${#sample_files[@]} -eq 0 ]]; then
    log "clips: no samples found in $SAMPLES_DIR."
    return 0
  fi

  for f in "${sample_files[@]}"; do
    local s="${f##*/}"
    s="${s%.npz}"
    local target="$CLIPS_DIR/c_${s}.mp4"
    if [[ -s "$target" ]]; then
      n_skip=$((n_skip + 1))
      continue
    fi

    local seg="${s%__*}"
    local i0="${s#*__}"
    local n="${seg##*--}"
    local t0=$((n * 60 + i0 / 20))
    local ss=$((i0 / 20 - 2))

    local qcamera="$FULL_DIR/$seg/qcamera.ts"
    if [[ ! -f "$qcamera" ]]; then
      log "Warning: $qcamera not found, skipping clip for $s"
      continue
    fi

    ffmpeg -v error -y -ss "$ss" -i "$qcamera" -t 8.4 \
      -vf "fps=10,scale=584:330,drawtext=fontfile=$FONT:text='route s %{eif\:t+$t0-2\:d}':x=8:y=8:fontsize=18:fontcolor=white:box=1:boxcolor=black@0.6,drawtext=fontfile=$FONT:text='before t0 = $t0 (model input)':enable='lt(t,2)':x=8:y=34:fontsize=15:fontcolor=white:box=1:boxcolor=black@0.6,drawtext=fontfile=$FONT:text='after t0 - the 6.4 s being predicted':enable='gte(t,2)':x=8:y=34:fontsize=15:fontcolor=yellow:box=1:boxcolor=black@0.6" \
      -c:v libx264 -crf 36 -preset slow -pix_fmt yuv420p -an -movflags +faststart "$target"
    n_done=$((n_done + 1))
  done

  log "clips finished: $n_done created, $n_skip skipped."
}

stage_run() {
  log "--- Stage: run (host=$HOST, variant=$VARIANT) ---"
  load_host_config

  log "Testing connection to $SSH_TARGET..."
  if ! ssh -o BatchMode=yes -o ConnectTimeout=10 "$SSH_TARGET" true; then
    echo "Error: SSH target $SSH_TARGET is unreachable." >&2
    exit 1
  fi
  resolve_remote_paths
  local remote_run_dir="$REMOTE_BASE/runs/$RUN_NAME"
  local remote_samples="$remote_run_dir/samples"
  local remote_results="$remote_run_dir/results/$VARIANT"
  local remote_log="$remote_run_dir/run.log"

  local local_samples
  local_samples="$(compgen -G "$SAMPLES_DIR/*.npz" | wc -l || true)"
  if [[ "$local_samples" -eq 0 ]]; then
    log "Error: no samples found in $SAMPLES_DIR. Run prep first." >&2
    exit 1
  fi

  local remote_results_cnt
  remote_results_cnt="$(ssh -o BatchMode=yes "$SSH_TARGET" "find '$remote_results' -maxdepth 1 -name '*.json' 2>/dev/null | wc -l" || echo 0)"

  local latest_done
  latest_done="$(ssh -o BatchMode=yes "$SSH_TARGET" "[ -f '$remote_log' ] && grep 'RUN_DONE' '$remote_log' | tail -n 1 || echo ''")"

  if [[ "$latest_done" == *"RUN_DONE rc=0"* && "$remote_results_cnt" -eq "$local_samples" ]]; then
    log "Remote evaluation already completed ($remote_results_cnt / $local_samples results)."
  else
    local is_running
    is_running="$(ssh -o BatchMode=yes "$SSH_TARGET" "pgrep -f '[r]un_edge.py.*$remote_samples' >/dev/null && echo yes || echo no")"
    if [[ "$is_running" == "no" ]]; then
      # Pre-flight GPU memory check
      local min_free="${MIN_FREE_MIB:-0}"
      if [[ "$min_free" -gt 0 ]]; then
        local free_mib
        free_mib="$(ssh -o BatchMode=yes "$SSH_TARGET" "nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits" 2>/dev/null | head -n 1 | tr -d ' ')"
        if [[ -n "$free_mib" && "$free_mib" =~ ^[0-9]+$ && "$free_mib" -lt "$min_free" ]]; then
          log "Pre-flight failed: free GPU memory ${free_mib} MiB < required ${min_free} MiB on $SSH_TARGET."
          log "GPU processes currently running:"
          ssh -o BatchMode=yes "$SSH_TARGET" "nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader" 2>/dev/null || true
          exit 1
        fi
      fi

      log "Syncing repo tracked files to $SSH_TARGET:$REMOTE_BASE/alpamayo-op/..."
      ssh -o BatchMode=yes "$SSH_TARGET" "mkdir -p '$REMOTE_BASE/alpamayo-op'"
      git -C "$REPO_DIR" ls-files | rsync -a -q -e "ssh -o BatchMode=yes" --files-from=- "$REPO_DIR/" "$SSH_TARGET:$REMOTE_BASE/alpamayo-op/"

      log "Syncing samples to $SSH_TARGET:$remote_samples/..."
      ssh -o BatchMode=yes "$SSH_TARGET" "mkdir -p '$remote_samples' '$remote_results'"
      rsync -a -q -e "ssh -o BatchMode=yes" "$SAMPLES_DIR/" "$SSH_TARGET:$remote_samples/"

      # Rotate old log before launch to ensure fresh log per launch
      ssh -o BatchMode=yes "$SSH_TARGET" "[ -f '$remote_log' ] && mv -f '$remote_log' '$remote_log.prev' || true"

      log "Starting run_edge on $SSH_TARGET..."
      ssh -o BatchMode=yes "$SSH_TARGET" "cd $NVIDIA_DIR && source .venv/bin/activate && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && nohup sh -c 'python \"$REMOTE_BASE/alpamayo-op/spark/run_edge.py\" --samples \"$remote_samples\" --out \"$remote_results\" --slot both $HOST_FLAGS; echo RUN_DONE rc=\$?' >> \"$remote_log\" 2>&1 &"
    else
      log "run_edge already active on $SSH_TARGET; waiting for completion..."
    fi

    # Poll until RUN_DONE or runner dies
    until ssh -o BatchMode=yes "$SSH_TARGET" "grep -q 'RUN_DONE' '$remote_log' 2>/dev/null"; do
      if ! ssh -o BatchMode=yes "$SSH_TARGET" "pgrep -f '[r]un_edge.py.*$remote_samples' >/dev/null"; then
        if ssh -o BatchMode=yes "$SSH_TARGET" "grep -q 'RUN_DONE' '$remote_log' 2>/dev/null"; then
          break
        fi
        log "Error: remote runner process exited without RUN_DONE"
        ssh -o BatchMode=yes "$SSH_TARGET" "tail -n 25 '$remote_log'" 2>/dev/null || true
        exit 1
      fi
      local cur_cnt
      cur_cnt="$(ssh -o BatchMode=yes "$SSH_TARGET" "find '$remote_results' -maxdepth 1 -name '*.json' 2>/dev/null | wc -l" || echo 0)"
      log "Progress: $cur_cnt / $local_samples done..."
      sleep 15
    done

    # Check status and results
    local final_done
    final_done="$(ssh -o BatchMode=yes "$SSH_TARGET" "grep 'RUN_DONE' '$remote_log' 2>/dev/null | tail -n 1 || echo ''")"
    remote_results_cnt="$(ssh -o BatchMode=yes "$SSH_TARGET" "find '$remote_results' -maxdepth 1 -name '*.json' 2>/dev/null | wc -l" || echo 0)"

    if [[ "$final_done" != *"RUN_DONE rc=0"* || "$remote_results_cnt" -ne "$local_samples" ]]; then
      log "Error: remote run failed ($final_done, $remote_results_cnt / $local_samples results)."
      log "Remote log tail:"
      ssh -o BatchMode=yes "$SSH_TARGET" "tail -n 25 '$remote_log'" 2>/dev/null || true
      exit 1
    fi
    log "Remote run completed successfully ($remote_results_cnt / $local_samples results)."
  fi

  log "Pulling results to $RESULTS_DIR..."
  rsync -a -q -e "ssh -o BatchMode=yes" "$SSH_TARGET:$remote_results/" "$RESULTS_DIR/"
  local final_cnt
  final_cnt="$(compgen -G "$RESULTS_DIR/*.json" | wc -l || true)"
  log "Results pulled: $final_cnt JSON files."
}

stage_report() {
  log "--- Stage: report ---"
  local summary_html="$REPORT_DIR/summary.html"

  log "Building summary page..."
  local summary_out
  summary_out="$("$PYTHON" "$REPO_DIR/pipeline/summary.py" \
    --samples "$SAMPLES_DIR" \
    --results "$RESULTS_DIR" \
    --out "$summary_html" \
    --title "$RUN_NAME ($VARIANT)")"
  printf "%s\n" "$summary_out"

  local picked_stems
  picked_stems="$(echo "$summary_out" | awk '/^PICK / {for (i=2; i<=NF; i++) printf "%s ", $i}')"

  if [[ -n "$picked_stems" ]]; then
    log "Building picked case pages..."
    # shellcheck disable=SC2086
    "$PYTHON" "$REPO_DIR/pipeline/report.py" \
      --samples "$SAMPLES_DIR" \
      --results "$RESULTS_DIR" \
      --clips "$CLIPS_DIR" \
      --variants "$VARIANT=$RESULTS_DIR" \
      --out-prefix "$REPORT_DIR/cases" \
      --title "$RUN_NAME picked cases" \
      $picked_stems
  fi

  log "Building all-sample pages..."
  "$PYTHON" "$REPO_DIR/pipeline/report.py" \
    --samples "$SAMPLES_DIR" \
    --results "$RESULTS_DIR" \
    --clips "$CLIPS_DIR" \
    --variants "$VARIANT=$RESULTS_DIR" \
    --out-prefix "$REPORT_DIR/all" \
    --title "$RUN_NAME all samples"

  "$PYTHON" "$REPO_DIR/pipeline/index.py" --report-dir "$REPORT_DIR" --title "$RUN_NAME ($VARIANT)"
  log "Report generated: open $REPORT_DIR/index.html"
}

stage_status() {
  echo "=== Pipeline Status for Run: $RUN_NAME ==="
  echo "Host profile: $HOST"
  echo "Variant:      $VARIANT"
  echo "Run dir:      $RUN_DIR"

  local seg_cnt=0
  [[ -f "$RUN_DIR/segments.txt" ]] && seg_cnt="$(wc -l < "$RUN_DIR/segments.txt")"

  local full_cnt=0
  if [[ -f "$RUN_DIR/segments.txt" ]]; then
    while read -r s; do
      [[ -f "$FULL_DIR/$s/.done" ]] && full_cnt=$((full_cnt + 1))
    done < "$RUN_DIR/segments.txt"
  fi

  local sample_cnt clip_cnt result_cnt
  sample_cnt="$(compgen -G "$SAMPLES_DIR/*.npz" | wc -l || true)"
  clip_cnt="$(compgen -G "$CLIPS_DIR/*.mp4" | wc -l || true)"
  result_cnt="$(compgen -G "$RESULTS_DIR/*.json" | wc -l || true)"

  echo "Segments:     $seg_cnt ($full_cnt copied with .done)"
  echo "Samples:      $sample_cnt"
  echo "Clips:        $clip_cnt"
  echo "Results:      $result_cnt"

  echo "Reports:"
  if [[ -d "$REPORT_DIR" ]]; then
    for f in "$REPORT_DIR"/*.html; do
      [[ -f "$f" ]] && ls -lh "$f" | awk '{print "  " $9 " (" $5 ")"}'
    done
  fi

  # GPU Host check
  local cfg="$REPO_DIR/pipeline/hosts/${HOST}.env"
  if [[ -f "$cfg" ]]; then
    # shellcheck source=/dev/null
    source "$cfg"
    echo -n "GPU Host ($HOST / $SSH_TARGET): "
    if ssh -o BatchMode=yes -o ConnectTimeout=5 "$SSH_TARGET" true 2>/dev/null; then
      echo "reachable"
      resolve_remote_paths
      local remote_samples="$REMOTE_BASE/runs/$RUN_NAME/samples"
      local remote_results="$REMOTE_BASE/runs/$RUN_NAME/results/$VARIANT"
      local remote_log="$REMOTE_BASE/runs/$RUN_NAME/run.log"

      local is_running
      is_running="$(ssh -o BatchMode=yes "$SSH_TARGET" "pgrep -f '[r]un_edge.py.*$remote_samples' >/dev/null && echo alive || echo idle" 2>/dev/null || echo unknown)"
      local rem_done
      rem_done="$(ssh -o BatchMode=yes "$SSH_TARGET" "find '$remote_results' -maxdepth 1 -name '*.json' 2>/dev/null | wc -l" 2>/dev/null || echo 0)"
      local rem_total
      rem_total="$(ssh -o BatchMode=yes "$SSH_TARGET" "find '$remote_samples' -maxdepth 1 -name '*.npz' 2>/dev/null | wc -l" 2>/dev/null || echo 0)"
      echo "  Runner:     $is_running"
      echo "  Progress:   $rem_done / $rem_total"
      echo "  Recent log:"
      ssh -o BatchMode=yes "$SSH_TARGET" "[ -f '$remote_log' ] && tail -n 3 '$remote_log'" 2>/dev/null | sed 's/^/    /' || true
    else
      echo "unreachable"
    fi
  else
    echo "GPU Host ($HOST): config file missing"
  fi
}

for stage in "${STAGES[@]}"; do
  case "$stage" in
    scan)
      stage_scan
      ;;
    pick)
      # Check if next token in STAGES was a number?
      stage_pick "$COUNT"
      ;;
    copy)
      stage_copy
      ;;
    prep)
      stage_prep
      ;;
    clips)
      stage_clips
      ;;
    run)
      stage_run
      ;;
    report)
      stage_report
      ;;
    all)
      stage_copy
      stage_prep
      stage_clips
      stage_run
      stage_report
      ;;
    status)
      stage_status
      ;;
    *)
      # Could be a number passed to pick, or unknown
      if [[ "$stage" =~ ^[0-9]+$ ]]; then
        : # Already consumed if with pick, or ignore
      else
        echo "Unknown stage: $stage" >&2
        exit 1
      fi
      ;;
  esac
done
