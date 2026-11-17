import argparse
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import av
import numpy as np
from PIL import Image

from geom import REAR_AXLE_FROM_DEVICE, forward_pixel, nav_text, poses, trajectories


def read_log(segment):
    sys.modules.setdefault("msgq.ipc_pyx", MagicMock())
    sys.modules.setdefault("openpilot.common.params_pyx", MagicMock())
    from openpilot.tools.lib.logreader import LogReader

    rows = {
        key: []
        for key in ("motion", "models", "frames", "calib", "car", "control", "gps")
    }
    for msg in LogReader(str(segment / "rlog.zst")):
        kind = msg.which()
        t = msg.logMonoTime * 1e-9
        if kind == "deviceMotion":
            m = msg.deviceMotion
            rows["motion"].append(
                (
                    t,
                    [m.orientationNED.x, m.orientationNED.y, m.orientationNED.z],
                    [m.velocityDevice.x, m.velocityDevice.y, m.velocityDevice.z],
                    bool(m.orientationNED.valid and m.velocityDevice.valid),
                )
            )
        elif kind == "extrinsicsCalibration":
            c = msg.extrinsicsCalibration
            rows["calib"].append((t, list(c.rpyCalib), str(c.calStatus)))
        elif kind == "qNarrowRoadEncodeIdx":
            f = msg.qNarrowRoadEncodeIdx
            rows["frames"].append(
                (int(f.segmentId), int(f.frameId), int(f.timestampEof), t)
            )
        elif kind == "modelV2":
            m = msg.modelV2
            position = m.position
            rows["models"].append(
                (
                    int(m.frameId),
                    np.column_stack((position.x, position.y, position.z)).astype(float),
                    np.asarray(position.t, dtype=float),
                )
            )
        elif kind == "carState":
            rows["car"].append((t, float(msg.carState.vEgo)))
        elif kind == "gpsLocationExternal":
            gps = msg.gpsLocationExternal
            rows["gps"].append(
                (t, bool(gps.hasFix), float(gps.speed), float(gps.speedAccuracy))
            )
        elif kind == "carControl":
            rows["control"].append(
                (t, bool(msg.carControl.latActive), bool(msg.carControl.longActive))
            )
    return rows


def decode(segment, video="qcamera.ts", crop_tele=False, crop_center=None):
    # loggerd/encoder/ffmpeg_encoder.cc:96-107 uses libyuv i420_scale directly, no crop.
    # Restore 1344:760 sensor aspect from the encoded 526:330 narrow frame.
    # These are openpilot's os04c10 road-camera intrinsics (common/transformations/camera.py).
    focal_px, sensor_width = 1141.5, 1344
    crop_fraction = np.tan(np.deg2rad(15)) / np.tan(np.arctan((sensor_width / 2) / focal_px))
    crop_size = (round(1344 * crop_fraction), round(760 * crop_fraction))
    output = []
    with av.open(str(segment / video)) as container:
        for frame in container.decode(video=0):
            image = frame.to_image()
            if crop_tele:
                width, height = crop_size
                if crop_center is None:
                    center_x, center_y = image.width / 2, image.height / 2
                else:
                    center_x, center_y = crop_center
                # Roll is ignored; one calibrated direction is used for the whole segment.
                left = int(round(center_x - width / 2))
                top = int(round(center_y - height / 2))
                left = min(max(left, 0), image.width - width)
                top = min(max(top, 0), image.height - height)
                image = image.crop((left, top, left + width, top + height))
            rgb = image.resize((584, 330), Image.Resampling.BICUBIC)
            output.append(np.asarray(rgb, dtype=np.uint8))
    return output


def wheel_speed_velocity(rows, times, ori, vel, forward):
    # deviceMotion's speed is the vision model's: -9 %..+1 % vs GPS over the 28 evaluation segments and it
    # drifts inside a segment, while carState.vEgo is within 0.6 % of GPS on 22 of them (2026-11-11).
    # Keep deviceMotion's direction; take the magnitude from vEgo (rear-axle speed) plus the device's
    # yaw-rate lever arm ahead of the axle.
    car_times = np.asarray([r[0] for r in rows["car"]], dtype=float)
    car_speeds = np.asarray([r[1] for r in rows["car"]], dtype=float)
    yaw_rate = np.gradient(np.unwrap(ori[:, 2]), times)
    speed = np.hypot(
        np.interp(times, car_times, car_speeds), REAR_AXLE_FROM_DEVICE[0] * yaw_rate
    )
    norm = np.linalg.norm(vel, axis=1, keepdims=True)
    # below 0.5 m/s the estimated direction is noise: use the car's forward axis
    direction = np.where(norm > 0.5, vel / np.maximum(norm, 1e-6), forward)
    return direction * speed[:, None]


def load_geometry(rows):
    times = np.asarray([r[0] for r in rows["motion"]], dtype=float)
    ori = np.asarray([r[1] for r in rows["motion"]], dtype=float)
    vel = np.asarray([r[2] for r in rows["motion"]], dtype=float)
    valid = np.asarray([r[3] for r in rows["motion"]], dtype=bool)
    calibration = [r for r in rows["calib"] if "calibrated" in r[2].lower()]
    if not calibration:
        raise ValueError("segment has no calibrated extrinsicsCalibration")
    # Verify transform direction on straight-ish high-speed observations: transpose maps device to car.
    calib = np.asarray(calibration[-1][1], dtype=float)
    from openpilot.common.transformations.orientation import rot_from_euler

    vel = wheel_speed_velocity(rows, times, ori, vel, rot_from_euler(calib)[:, 0])
    p, rd, rc, vn = poses(times, ori, vel, valid, calib, rot_from_euler)
    return times, ori, vel, valid, calib, p, rd, rc, vn


def closest(rows, t):
    return min(rows, key=lambda row: abs(row[0] - t))


def process(segment, out, step, full=False):
    rows = read_log(segment)
    times, ori, vel, valid, calib, p, rd, rc, vn = load_geometry(rows)
    frame_rows = sorted(rows["frames"])
    if full:
        from openpilot.common.transformations.orientation import rot_from_euler

        focal_px, cx, cy = 1141.5, 672.0, 380.0
        forward = rot_from_euler(calib)[:, 0]
        crop_center = forward_pixel(forward, focal_px, cx, cy)
        crop_fraction = np.tan(np.deg2rad(15)) / np.tan(np.arctan((1344 / 2) / focal_px))
        crop_width, crop_height = round(1344 * crop_fraction), round(760 * crop_fraction)
        left = min(max(round(crop_center[0] - crop_width / 2), 0), 1344 - crop_width)
        top = min(max(round(crop_center[1] - crop_height / 2), 0), 760 - crop_height)
        reported_center = (left + crop_width / 2, top + crop_height / 2)
        print(
            f"{segment.name}: calib rpy deg={np.rad2deg(calib).tolist()}, "
            f"tele crop centre px=({reported_center[0]:.2f}, {reported_center[1]:.2f})"
        )
        decoded = decode(segment, "fcamera.hevc", crop_tele=True, crop_center=crop_center)
    else:
        decoded = decode(segment)
    if len(decoded) != len(frame_rows):
        raise ValueError(
            f"decoded {len(decoded)} fcamera frames but log has {len(frame_rows)} frame indexes"
            if full
            else f"decoded {len(decoded)} frames but log has {len(frame_rows)} frame indexes"
        )
    decoded_wide = decode(segment, "ecamera.hevc") if full else None
    if full and len(decoded_wide) != len(frame_rows):
        raise ValueError(
            f"decoded {len(decoded_wide)} ecamera frames but log has {len(frame_rows)} frame indexes"
        )
    models = {fid: (xyz, mt) for fid, xyz, mt in rows["models"]}
    indices = np.asarray([r[0] for r in frame_rows], dtype=int)
    eof = np.asarray([r[2] for r in frame_rows], dtype=np.int64)
    segment_number = int(segment.name.rsplit("--", 1)[-1])
    out.mkdir(parents=True, exist_ok=True)
    kept = []
    for i0 in range(100, 1200 - 128, step):
        if i0 + 127 >= len(decoded):
            print(f"{segment.name} {i0}: skip missing video frames")
            continue
        exact = np.flatnonzero(indices == i0)
        if not len(exact):
            print(f"{segment.name} {i0}: skip missing frame index")
            continue
        fr = frame_rows[int(exact[0])]
        if fr[1] not in models:
            print(f"{segment.name} {i0}: skip missing matching modelV2 frameId {fr[1]}")
            continue
        t0 = fr[2] * 1e-9
        needed = (times >= t0 - 2.0) & (times <= t0 + 6.4)
        if not valid[needed].all():
            print(f"{segment.name} {i0}: skip invalid deviceMotion")
            continue
        hist_t = t0 + np.arange(-1.9, 0.001, 0.1)
        fut_t = t0 + np.arange(0.1, 6.401, 0.1)
        hist_xyz, hist_rot, _ = trajectories(times, p, rc, t0, hist_t)
        gt_axle, _, gt_dev = trajectories(times, p, rc, t0, fut_t)
        model_xyz, model_t = models[fr[1]]
        op_plan = np.stack(
            [
                np.interp(np.arange(0.1, 6.401, 0.1), model_t, model_xyz[:, j])
                for j in range(3)
            ],
            axis=-1,
        )
        image_indices = [i0 - 40, i0 - 20, i0 - 10, i0 - 2, i0]
        fi = np.array(image_indices, dtype=int)
        frames = np.stack([decoded[x] for x in fi])
        frames_wide = np.stack([decoded_wide[x] for x in fi]) if full else None
        ts_us = eof[fi] // 1000 - eof[0] // 1000
        car = closest(rows["car"], t0)
        control = closest(rows["control"], t0)
        path = np.zeros(len(times))
        path[1:] = np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))
        nav = nav_text(times, ori[:, 2], path, t0)
        name = f"{segment.name}__{i0}.npz"
        np.savez_compressed(
            out / name,
            frames=frames,
            **({"frames_wide": frames_wide} if full else {}),
            frame_ts_us=ts_us,
            ego_history_xyz=hist_xyz.astype(np.float32),
            ego_history_rot=hist_rot.astype(np.float32),
            gt_axle_xyz=gt_axle.astype(np.float32),
            gt_dev_xyz=gt_dev.astype(np.float32),
            op_plan_xyz=op_plan.astype(np.float32),
            segment=segment.name,
            i0=i0,
            route_s=segment_number * 60 + i0 / 20,
            v_ego=car[1],
            lat_active=control[1],
            long_active=control[2],
            nav_text=nav,
        )
        kept.append((i0, t0, fut_t, gt_axle, gt_dev, op_plan, car[1], control[1]))
    return rows, (times, ori, vel, valid, calib, p, rd, rc, vn), kept


def selfcheck(segment):
    rows, g, kept = process(segment, Path(os.environ.get("ALPAMAYO_ROOT", Path(__file__).resolve().parent / "data")) / "selfcheck", 100)
    times, _ori, vel, _valid, calib, _p, _rd, _rc, _vn = g
    from openpilot.common.transformations.orientation import rot_from_euler

    car_velocity = np.einsum("ij,nj->ni", rot_from_euler(calib).T, vel)
    # Use motion samples; carState 20 Hz corresponds closely in time.
    fast_motion = np.array([closest(rows["car"], t)[1] > 5 for t in times])
    lateral_ratio = float(
        np.median(
            np.abs(car_velocity[fast_motion, 1])
            / np.maximum(np.abs(car_velocity[fast_motion, 0]), 1e-6)
        )
    )
    vertical_ratio = float(
        np.median(
            np.abs(car_velocity[fast_motion, 2])
            / np.maximum(np.abs(car_velocity[fast_motion, 0]), 1e-6)
        )
    )
    print(
        f"{segment.name}: median |vy|/vx={lateral_ratio:.5f}, |vz|/vx={vertical_ratio:.5f}"
    )
    errors, active_errors = [], []
    failed = lateral_ratio >= 0.01 or vertical_ratio >= 0.02
    if failed:
        print(
            "SELF-CHECK FAIL: car-frame lateral/vertical velocity ratios exceed threshold"
        )
    # (b) independent of vEgo: rear-axle path length vs the integral of GPS speed over the same 6.4 s
    gps = np.asarray([(r[0], r[2]) for r in rows["gps"] if r[1] and r[3] < 0.5])
    for i0, t0, ft, axle, dev, plan, speed, lat in kept:
        window = gps[(gps[:, 0] >= t0) & (gps[:, 0] <= ft[-1])] if len(gps) else gps
        if speed > 5 and len(window) >= 20:
            query_times = np.r_[t0, ft]
            integral = np.trapezoid(
                np.interp(query_times, window[:, 0], window[:, 1]), query_times
            )
            distance = np.linalg.norm(
                np.diff(np.vstack(([0, 0, 0], axle)), axis=0), axis=1
            ).sum()
            error = abs(distance - integral) / integral
            errors.append(error)
            if error > 0.03:
                print(
                    f"{segment.name} {i0}: rear-axle path={distance:.3f}m, GPS integral={integral:.3f}m, relative error={error:.4f}"
                )
        if lat:
            # lateral only: openpilot's plan is in the vision model's speed scale, so along-track differs
            active_errors.append(float(abs(plan[19, 1] - dev[19, 1])))
    path_median = float(np.median(errors)) if errors else float("nan")
    active_median = float(np.median(active_errors)) if active_errors else float("nan")
    print(
        f"{segment.name}: GT path vs GPS integral relative error median={path_median:.4f} (n={len(errors)}); engaged 2s plan-vs-GT lateral median={active_median:.4f} m (n={len(active_errors)})"
    )
    if errors and path_median > 0.03:
        print("SELF-CHECK FAIL: median path vs GPS integral error exceeds 3%")
        failed = True
    if active_errors and active_median >= 0.3:
        print("SELF-CHECK FAIL: engaged lateral plan error >= 0.3 m")
        failed = True
    return failed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("segments", nargs="*")
    parser.add_argument("--out", default="samples")
    parser.add_argument("--step-frames", type=int, default=100)
    parser.add_argument("--full", action="store_true", help="use cropped fcamera tele and ecamera wide videos")
    parser.add_argument("--selfcheck")
    args = parser.parse_args()
    if args.selfcheck:
        if selfcheck(Path(args.selfcheck)):
            raise SystemExit(1)
        return
    for path in args.segments:
        process(Path(path), Path(args.out), args.step_frames, args.full)


if __name__ == "__main__":
    main()
