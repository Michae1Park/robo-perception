"""Scoring against simulator ground truth."""
import numpy as np

from ppp_common import transforms as tf


def bbox_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return float(inter / union) if union > 0 else 0.0


def projected_bbox(K, T_cam_world, T_world_obj, points_obj, width, height):
    """GT 2D box: projection of the mesh points (occlusion ignored), clipped to the image."""
    uv, z = tf.project(K, T_cam_world, tf.transform_points(T_world_obj, points_obj))
    uv = uv[z > 0]
    if len(uv) == 0:
        return None
    x1, y1 = uv.min(axis=0)
    x2, y2 = uv.max(axis=0)
    return [float(np.clip(x1, 0, width)), float(np.clip(y1, 0, height)),
            float(np.clip(x2, 0, width)), float(np.clip(y2, 0, height))]


_BOX180 = [np.diag(d) for d in ([1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1])]


def rotation_error_deg(R_est, R_gt, symmetry='none'):
    """Geodesic rotation error, modulo the object's symmetry.

    cylinder_z: only the direction of the z axis is observable.
    box180:     untextured cuboid, poses related by 180 deg about any box axis are equivalent.
    """
    if symmetry == 'cylinder_z':
        return float(np.rad2deg(tf.angle_between(R_est[:, 2], R_gt[:, 2])))
    if symmetry == 'box180':
        return float(np.rad2deg(min(tf.rotation_angle(R_est, R_gt @ S) for S in _BOX180)))
    return float(np.rad2deg(tf.rotation_angle(R_est, R_gt)))


def translation_error_mm(T_est, T_gt):
    return float(np.linalg.norm(T_est[:3, 3] - T_gt[:3, 3]) * 1000.0)


def add_mm(T_est, T_gt, pts):
    a = tf.transform_points(T_est, pts)
    b = tf.transform_points(T_gt, pts)
    return float(np.linalg.norm(a - b, axis=1).mean() * 1000.0)


def adds_mm(T_est, T_gt, pts, chunk=512):
    """ADD-S: mean closest-point distance (symmetry-agnostic)."""
    a = tf.transform_points(T_est, pts)
    b = tf.transform_points(T_gt, pts)
    d = np.empty(len(b))
    for i in range(0, len(b), chunk):
        d[i:i + chunk] = np.linalg.norm(b[i:i + chunk, None, :] - a[None, :, :], axis=-1).min(axis=1)
    return float(d.mean() * 1000.0)


def tilt_deg(R):
    """Angle between the object's z axis and world z."""
    return float(np.rad2deg(tf.angle_between(R[:, 2], [0.0, 0.0, 1.0])))


def percentile(values, q):
    v = [x for x in values if x is not None]
    return float(np.percentile(v, q)) if v else None


def summarize(trials):
    """Aggregate a list of per-trial dicts (see eval_node) into rates and error statistics."""
    n = len(trials)

    def rate(key):
        return sum(1 for t in trials if t.get(key)) / n if n else 0.0

    def stat(key):
        v = [t[key] for t in trials if t.get(key) is not None]
        if not v:
            return None
        return {'mean': float(np.mean(v)), 'median': float(np.median(v)), 'p95': float(np.percentile(v, 95)), 'n': len(v)}

    stages = {}
    for t in trials:
        for name, ms in (t.get('latency_ms') or {}).items():
            stages.setdefault(name, []).append(ms)
    latency = {k: {'p50': float(np.percentile(v, 50)), 'p95': float(np.percentile(v, 95)), 'n': len(v)}
               for k, v in stages.items()}
    failures = {}
    for t in trials:
        if not t.get('task_success'):
            key = t.get('failed_stage') or 'place_check'
            failures[key] = failures.get(key, 0) + 1
    return {
        'trials': n,
        'rates': {k: rate(k) for k in ('detection_success', 'pose_success', 'grasp_success',
                                       'place_success', 'task_success', 'disturbed_scene')},
        'errors': {k: stat(k) for k in ('iou', 'trans_err_mm', 'rot_err_deg', 'add_mm', 'adds_mm',
                                        'place_err_mm')},
        'latency_ms': latency,
        'failures': failures,
    }


def to_markdown(summary):
    lines = ['| Metric | Value |', '|---|---|', '| Trials | %d |' % summary['trials']]
    for k, v in summary['rates'].items():
        lines.append('| %s | %.1f %% |' % (k.replace('_', ' '), 100.0 * v))
    lines += ['', '| Error | mean | median | p95 |', '|---|---|---|---|']
    for k, v in summary['errors'].items():
        if v:
            lines.append('| %s | %.2f | %.2f | %.2f |' % (k, v['mean'], v['median'], v['p95']))
    lines += ['', '| Stage | p50 (ms) | p95 (ms) |', '|---|---|---|']
    for k, v in summary['latency_ms'].items():
        lines.append('| %s | %.1f | %.1f |' % (k, v['p50'], v['p95']))
    if summary['failures']:
        lines += ['', '| Failure stage | count |', '|---|---|']
        for k, v in sorted(summary['failures'].items(), key=lambda kv: -kv[1]):
            lines.append('| %s | %d |' % (k, v))
    return '\n'.join(lines) + '\n'
