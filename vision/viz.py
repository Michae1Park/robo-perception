"""Overlays and plots shared by the pipeline scripts."""
import cv2
import numpy as np

from . import spatial, transforms as tf

COLORS = [(0, 255, 0), (255, 0, 255), (255, 255, 0), (0, 165, 255), (255, 0, 0), (0, 255, 255)]  # BGR


def draw_detection(img, det, color=(0, 255, 0), text=None):
    """Returns a copy of `img` with the detection's mask (tinted), box and label drawn."""
    ov = img.copy()
    ov[det['mask']] = color
    out = cv2.addWeighted(ov, 0.4, img, 0.6, 0)
    x1, y1, x2, y2 = [int(v) for v in det['bbox']]
    cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
    cv2.putText(out, text or '%s %.2f' % (det['name'], det['score']), (x1, max(12, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return out


def plot_occupancy(rgb_bgr, scene, path, title='scene'):
    """Left: scene image. Right: place-zone occupancy grid (grey unknown, green free, red occupied)."""
    import matplotlib  # lazy: not installed in the FoundationPose container
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    grid = scene['grid']
    (x0, x1), (y0, y1) = scene['zone']
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].imshow(cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB))
    axes[0].set_title(title)
    axes[0].axis('off')
    cmap = matplotlib.colors.ListedColormap(['#bbbbbb', '#2ecc71', '#e74c3c'])
    disp = np.select([grid == spatial.UNKNOWN, grid == spatial.FREE, grid == spatial.OCCUPIED], [0, 1, 2])
    axes[1].imshow(disp, cmap=cmap, origin='lower', extent=[x0, x1, y0, y1], vmin=0, vmax=2)
    axes[1].set_title('place-zone occupancy (grey=unknown, green=free, red=occupied)')
    axes[1].set_xlabel('x (pseudo-world, m)')
    axes[1].set_ylabel('y (pseudo-world, m)')
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def draw_grasps(img, K, T_world_cam, feasible, all_rows):
    """Draw grasp candidates on `img` (BGR). feasible: GraspCandidates (best first).
    all_rows: [{'T_world_tcp', 'width', 'tilt_deg'}] for every candidate (drawn red if none pass).
    Each grasp = approach arrow (TCP +z, into the object) + the two finger contact points."""
    img = img.copy()
    T_cam_world = tf.invert(T_world_cam)
    arrow_len = 0.06

    def draw(T_world_tcp, width, color):
        tip = T_world_tcp[:3, 3]
        tail = tip - T_world_tcp[:3, 2] * arrow_len
        closing = T_world_tcp[:3, 1]
        pts = tf.transform_points(T_cam_world, np.stack(
            [tail, tip, tip + closing * width / 2.0, tip - closing * width / 2.0]))
        uv, z = tf.project(K, np.eye(4), pts)
        if (z <= 0).any():
            return
        p0, p1, f0, f1 = uv.astype(int)
        # black outline first so shapes read on any background
        cv2.arrowedLine(img, tuple(p0), tuple(p1), (0, 0, 0), 6, cv2.LINE_AA, tipLength=0.35)
        cv2.arrowedLine(img, tuple(p0), tuple(p1), color, 3, cv2.LINE_AA, tipLength=0.35)
        cv2.line(img, tuple(f0), tuple(f1), (0, 0, 0), 5, cv2.LINE_AA)
        cv2.line(img, tuple(f0), tuple(f1), color, 2, cv2.LINE_AA)
        for p in (f0, f1):
            cv2.circle(img, tuple(p), 6, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(img, tuple(p), 4, color, -1, cv2.LINE_AA)

    if feasible:
        for i, c in enumerate(feasible):
            draw(c.T_world_tcp, c.width, (255, 0, 255) if i == 0 else (255, 255, 0))
        label = 'grasps: magenta=best, cyan=other feasible (%d/%d)' % (len(feasible), len(all_rows))
    else:
        for r in all_rows:
            draw(r['T_world_tcp'], r['width'], (0, 0, 255))
        label = '0/%d passed the tilt/table filter (red = rejected) -- no safe top-down approach' % len(all_rows)
    cv2.putText(img, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def grasp_rows(cands, T_world_obj):
    """Per-candidate world pose, width and approach tilt (deg), sorted by tilt."""
    down = np.array([0.0, 0.0, -1.0])
    rows = []
    for c in cands:
        T = T_world_obj @ c.T_obj_tcp
        rows.append({'face': c.face, 'width': float(c.width), 'T_world_tcp': T,
                     'tilt_deg': float(np.degrees(tf.angle_between(T[:3, 2], down)))})
    return sorted(rows, key=lambda r: r['tilt_deg'])
