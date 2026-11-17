import numpy as np

REAR_AXLE_FROM_DEVICE = np.array([-1.94, 0.0, 0.0])
S = np.diag([1.0, -1.0, -1.0])


def forward_pixel(forward, focal, cx, cy):
    """Project the device-frame forward direction onto a pinhole image plane."""
    f = np.asarray(forward, dtype=float)
    return cx + focal * f[1] / f[0], cy + focal * f[2] / f[0]


def poses(times, orientation, velocity, valid, calib, rot_from_euler):
    """Integrate device velocity and express device/axle poses in t0 frames."""
    times = np.asarray(times, dtype=np.float64)
    orientation, velocity = np.asarray(orientation), np.asarray(velocity)
    rd = np.array([rot_from_euler(e) for e in orientation])
    rc = rd @ rot_from_euler(calib)
    vn = np.einsum("nij,nj->ni", rd, velocity)
    p = np.zeros_like(vn)
    p[1:] = np.cumsum((vn[1:] + vn[:-1]) * (np.diff(times)[:, None] / 2), axis=0)
    return p, rd, rc, vn


def trajectories(times, p, rc, t0, query):
    """Return Alpamayo axle XYZ, Alpamayo rotations, and openpilot device XYZ."""
    q = np.asarray(query, dtype=np.float64)
    p0 = np.array([np.interp(t0, times, p[:, j]) for j in range(3)])
    r0 = rc[np.argmin(np.abs(times - t0))]
    pquery = np.stack([np.interp(q, times, p[:, j]) for j in range(3)], axis=-1)
    rquery = rc[np.abs(times[:, None] - q[None, :]).argmin(axis=0)]
    axle = p + np.einsum("nij,j->ni", rc, REAR_AXLE_FROM_DEVICE)
    axleq = np.stack([np.interp(q, times, axle[:, j]) for j in range(3)], axis=-1)
    xyz = (
        S
        @ r0.T
        @ (axleq - np.array([np.interp(t0, times, axle[:, j]) for j in range(3)])).T
    ).T
    op = (r0.T @ (pquery - p0).T).T
    rot = np.einsum("ij,njk,kl->nil", S, np.einsum("ij,njk->nik", r0.T, rquery), S)
    return xyz, rot, op


def build_arrays(sample, slot):
    """Assemble camera-major numpy inputs; image shape remains NHWC here."""
    if slot not in ("wide", "tele", "both"):
        raise ValueError("slot must be wide, tele, or both")
    frames = np.asarray(sample["frames"])
    frames_wide = np.asarray(sample["frames_wide"]) if "frames_wide" in sample else frames
    offsets_front = [-2_000_000, -1_000_000, -500_000, -100_000, 0]
    offsets_other = [-1_000_000, -100_000, 0]
    counts = {0: 3, 1: 5, 2: 3, 3: 3, 5: 3, 6: 5}
    images, cameras, stamps = [], [], []
    h, w = frames.shape[1:3]
    for camera in counts:
        offsets = offsets_front if camera in (1, 6) else offsets_other
        active = (
            camera == 1
            and slot in ("wide", "both")
            or camera == 6
            and slot in ("tele", "both")
        )
        for offset in offsets:
            index = offsets_front.index(offset) if active and camera in (1, 6) else 0
            camera_frames = frames_wide if camera == 1 else frames
            image = camera_frames[index] if active else np.zeros((h, w, 3), dtype=np.uint8)
            images.append(image.transpose(2, 0, 1))
            cameras.append(camera)
            stamps.append(int(sample["frame_ts_us"][-1]) + offset)
    return (
        np.ascontiguousarray(images, dtype=np.uint8),
        np.asarray(cameras, np.int64),
        np.asarray(stamps, np.int64),
    )


def nav_text(times, yaw, path_length, t0):
    times, yaw = np.asarray(times), np.unwrap(np.asarray(yaw))
    start = np.interp(t0, times, yaw)
    mask = (times > t0) & (times <= t0 + 15)
    ids = np.flatnonzero(mask)
    if not len(ids):
        return "Continue straight"
    changes = yaw[ids] - start
    turn = np.flatnonzero(np.abs(changes) >= np.deg2rad(45))
    if not len(turn):
        return "Continue straight"
    direction = "right" if changes[turn[0]] > 0 else "left"
    onset = np.flatnonzero(np.abs(changes) > np.deg2rad(10))
    onset_time = times[ids[onset[0]]] if len(onset) else times[ids[turn[0]]]
    # path_length counts from the segment start: measure from t0
    distance = max(
        1,
        round(
            np.interp(onset_time, times, path_length)
            - np.interp(t0, times, path_length)
        ),
    )
    return f"Turn {direction} in {distance}m"
