import importlib.util
import os
import sys

import numpy as np
import pytest

if os.environ.get("OPENPILOT_DIR"):
    sys.path.insert(0, os.environ["OPENPILOT_DIR"])

from geom import build_arrays, forward_pixel, nav_text, poses, trajectories


def test_forward_pixel():
    assert forward_pixel([1, 0, 0], 1141.5, 672, 380) == (672, 380)
    assert forward_pixel([1, 0.1, 0], 1141.5, 672, 380)[0] > 672
    # Device z points down, so a downward-pointing forward axis projects lower.
    assert forward_pixel([1, 0, 0.1], 1141.5, 672, 380)[1] > 380


def _has_openpilot():
    try:
        return importlib.util.find_spec("openpilot.common.transformations") is not None
    except ModuleNotFoundError:
        return False


needs_openpilot = pytest.mark.skipif(not _has_openpilot(), reason="set OPENPILOT_DIR to an openpilot checkout")


@needs_openpilot
def test_motion():
    from openpilot.common.transformations.orientation import rot_from_euler

    speed, omega = 10.0, 0.1
    t = np.arange(0, 11.01, 0.01)
    # NED yaw grows clockwise/right, so a left turn has negative yaw.
    yaw = -omega * t
    orientation = np.column_stack((np.zeros_like(t), np.zeros_like(t), yaw))
    velocity = np.tile([speed, 0.0, 0.0], (len(t), 1))
    p, _, rc, _ = poses(
        t, orientation, velocity, np.ones(len(t), bool), [0, 0, 0], rot_from_euler
    )
    hist, rot, _ = trajectories(t, p, rc, 4, 4 + np.arange(-1.9, 0.001, 0.1))
    future, _, dev = trajectories(t, p, rc, 4, 4 + np.arange(0.1, 6.401, 0.1))
    # Left turn has positive Alpamayo y and negative openpilot FRD y.
    assert future[-1, 1] > 0 and dev[-1, 1] < 0
    assert np.allclose(hist[-1], 0, atol=1e-9) and np.allclose(
        rot[-1], np.eye(3), atol=1e-9
    )
    radius = speed / omega
    theta = omega * np.arange(0.1, 6.401, 0.1)
    analytic = np.column_stack(
        (
            radius * np.sin(theta) + 1.94 * (1 - np.cos(theta)),
            radius * (1 - np.cos(theta)) - 1.94 * np.sin(theta),
            np.zeros_like(theta),
        )
    )
    assert np.max(np.linalg.norm(future - analytic, axis=1)) < 0.01


def test_arrays():
    f = np.full((5, 2, 3, 3), 7, np.uint8)
    sample = {"frames": f, "frame_ts_us": np.array([0, 1, 2, 3, 4])}
    counts = {0: 3, 1: 5, 2: 3, 3: 3, 5: 3, 6: 5}
    for slot, real in (("wide", {1}), ("tele", {6}), ("both", {1, 6})):
        imgs, cams, ts = build_arrays(sample, slot)
        assert (
            len(imgs) == 22
            and {c: int((cams == c).sum()) for c in np.unique(cams)} == counts
        )
        for camera in counts:
            values = ts[cams == camera]
            assert np.all(np.diff(values) > 0) and values[-1] == 4
            if camera in (1, 6):
                if camera in real:
                    assert np.all(imgs[cams == camera] == 7)
                else:
                    assert np.all(imgs[cams == camera] == 0)
            else:
                assert np.all(imgs[cams == camera] == 0)


def test_arrays_full():
    frames = np.full((5, 2, 3, 3), 7, np.uint8)
    frames_wide = np.full((5, 2, 3, 3), 11, np.uint8)
    sample = {
        "frames": frames,
        "frames_wide": frames_wide,
        "frame_ts_us": np.array([0, 1, 2, 3, 4]),
    }
    imgs, cams, _ = build_arrays(sample, "both")
    assert np.all(imgs[cams == 1] == 11)
    assert np.all(imgs[cams == 6] == 7)
    assert np.all(imgs[~np.isin(cams, (1, 6))] == 0)


def test_nav():
    t = np.arange(0, 8, 0.1)
    path = 10 * t
    yaw = np.where(t < 3, 0, -np.minimum(0.2 + (t - 3) * 0.3, 1.0))
    assert nav_text(t, yaw, path, 0) == "Turn left in 30m"
    # distance counts from t0, not from the start of the segment
    assert nav_text(t, yaw, path, 2) == "Turn left in 10m"
    assert nav_text(t, np.zeros_like(t), path, 0) == "Continue straight"


if __name__ == "__main__":
    test_forward_pixel()
    test_motion()
    test_arrays()
    test_nav()
    print("geometry checks passed")
