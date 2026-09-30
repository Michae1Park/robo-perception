"""Stage 5: where can an object be put down? Depth + which way is up -> horizontal supports (shelf levels,
table tops) -> per-support free space + headroom -> placement candidates for a given object.

No model of the shelf: every horizontal surface the camera sees is a candidate support. Pure numpy + OpenCV.
"""
import cv2
import numpy as np

from . import spatial

FREE, OCCUPIED, UNKNOWN = spatial.FREE, spatial.OCCUPIED, spatial.UNKNOWN


def points_and_normals(depth_m, K, T_base_cam):
    """Per-pixel 3D points (H, W, 3) in the robot base frame (z up), unit normals, and a validity mask.
    Normals come from pixels `step` apart (wider = less sensitive to mm-quantised depth); depth edges are invalid."""
    step = 3
    H, W = depth_m.shape
    v, u = np.mgrid[0:H, 0:W]
    cam = np.stack([(u - K[0, 2]) * depth_m / K[0, 0], (v - K[1, 2]) * depth_m / K[1, 1], depth_m], axis=-1)
    P = cam @ T_base_cam[:3, :3].T + T_base_cam[:3, 3]
    du = np.zeros_like(P)
    dv = np.zeros_like(P)
    du[:, step:-step] = P[:, 2 * step:] - P[:, :-2 * step]
    dv[step:-step] = P[2 * step:] - P[:-2 * step]
    n = np.cross(du, dv)
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-12
    d = depth_m > 0
    ok = d.copy()
    ok[:, step:-step] &= d[:, 2 * step:] & d[:, :-2 * step]
    ok[step:-step] &= d[2 * step:] & d[:-2 * step]
    ok &= (np.linalg.norm(du, axis=-1) < 0.1) & (np.linalg.norm(dv, axis=-1) < 0.1)
    return P, n, d, ok


def find_planes(P, n, ok, cfg):
    """Sequential RANSAC on the upward-facing points: fit the biggest horizontal plane, remove its points,
    repeat. -> list of (height z, inlier pixel mask)."""
    horiz = ok & (np.abs(n[..., 2]) > np.cos(np.radians(cfg['max_tilt_deg'])))
    idx = np.flatnonzero(horiz)
    planes = []
    for k in range(cfg['max_planes']):
        if len(idx) < cfg['min_inliers']:
            break
        plane, inl = spatial.fit_plane_ransac(P.reshape(-1, 3)[idx], cfg['ransac_dist'], cfg['ransac_iters'],
                                              up=(0, 0, 1), max_tilt_deg=cfg['max_tilt_deg'],
                                              min_inliers=cfg['min_inliers'], seed=k)
        if plane is None:
            break
        mask = np.zeros(P.shape[:2], bool)
        mask.flat[idx[inl]] = True
        planes.append((float(P[mask][:, 2].mean()), mask))
        idx = idx[~inl]
    return sorted(planes, key=lambda p: p[0])


class Grid:
    """A fixed x/y grid in the base frame covering the given points."""

    def __init__(self, xy, res):
        self.res = res
        self.x0, self.y0 = xy.min(axis=0) - res
        self.shape = tuple((np.ceil((xy.max(axis=0) + res - [self.x0, self.y0]) / res).astype(int))[::-1])

    def cells(self, xy):  # -> (iy, ix), clipped
        ix = np.clip(((xy[:, 0] - self.x0) / self.res).astype(int), 0, self.shape[1] - 1)
        iy = np.clip(((xy[:, 1] - self.y0) / self.res).astype(int), 0, self.shape[0] - 1)
        return iy, ix

    def centre(self, iy, ix):
        return np.array([self.x0 + (ix + 0.5) * self.res, self.y0 + (iy + 0.5) * self.res])


def find_supports(P, planes, grid, cfg):
    """Split each plane into connected pieces on the grid (one shelf level, the table, ...) -> list of dicts:
    z, cells (seen support cells), hull (their convex hull = the support's area), pixels (inlier mask)."""
    supports = []
    for z, mask in planes:
        occ = np.zeros(grid.shape, np.uint8)
        occ[grid.cells(P[mask][:, :2])] = 1
        occ = cv2.morphologyEx(occ, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        n, lab = cv2.connectedComponents(occ, connectivity=8)
        for k in range(1, n):
            cells = lab == k
            if cells.sum() * grid.res ** 2 < cfg['min_area']:
                continue
            hull = np.zeros(grid.shape, np.uint8)
            pts = np.argwhere(cells)[:, ::-1].astype(np.int32)
            cv2.fillPoly(hull, [cv2.convexHull(pts)], 1)
            iy, ix = grid.cells(P[mask][:, :2])
            px = mask.copy()
            px[mask] = cells[iy, ix]
            supports.append({'z': z, 'cells': cells, 'hull': hull.astype(bool), 'pixels': px})
    return supports


def analyze_support(s, supports, P, valid, grid, cfg):
    """Fill in s['ceiling'] (height of the lowest support above that overlaps this one, inf = open), s['grid']
    (FREE / OCCUPIED / UNKNOWN inside the hull) and s['clearance'] (m to the nearest non-free cell).
    One ceiling per support: a board above that covers part of it is assumed to cover all of it (it may run out
    of view)."""
    above = [t['z'] for t in supports if t['z'] > s['z'] + cfg['min_gap']
             and (t['hull'] & s['hull']).sum() > 0.1 * s['hull'].sum()]
    s['ceiling_z'] = min(above, default=np.inf)
    ceiling = np.full(grid.shape, s['ceiling_z'])
    pts = P[valid]
    iy, ix = grid.cells(pts[:, :2])
    inside = s['hull'][iy, ix]
    h = pts[:, 2] - s['z']
    above = inside & (h > cfg['height_thresh']) & (pts[:, 2] < ceiling[iy, ix] - 0.01) & (h < cfg['max_height'])
    counts = np.zeros(grid.shape, int)
    np.add.at(counts, (iy[above], ix[above]), 1)
    g = np.full(grid.shape, UNKNOWN, np.int8)
    g[s['cells'] & s['hull']] = FREE
    g[(counts >= 3) & s['hull']] = OCCUPIED
    g[~s['hull']] = UNKNOWN
    free = ((g == FREE) & s['hull']).astype(np.uint8)
    clear = cv2.distanceTransform(np.pad(free, 1), cv2.DIST_L2, 5)[1:-1, 1:-1] * grid.res - grid.res / 2
    s.update(ceiling=ceiling, grid=g, clearance=np.maximum(clear, 0) * free)


SHOULDER = np.array([0.0, 0.0, 0.333])  # Franka joint 2, in the base frame


def reachable(s, grid, cfg):
    """Does any cell of the support lie within reach of the shoulder?"""
    iy, ix = np.nonzero(s['hull'])
    xy = grid.centre(iy, ix).T
    return bool((np.linalg.norm(np.c_[xy, np.full(len(xy), s['z'])] - SHOULDER, axis=1) <= cfg['max_reach']).any())


def candidates(s, grid, radius, need_headroom, cfg, shoulder=SHOULDER):
    """Greedy best-first: the free cell with the most clearance that fits the object (clearance >= radius,
    headroom >= need, within reach), then suppress everything within 2 radii, repeat."""
    headroom = s['ceiling'] - s['z']
    ok = (s['clearance'] >= radius) & (headroom >= need_headroom)
    out = []
    score = np.where(ok, s['clearance'], -1.0)
    while len(out) < cfg['per_support'] and score.max() > 0:
        iy, ix = np.unravel_index(score.argmax(), score.shape)
        xy = grid.centre(iy, ix)
        reach = np.linalg.norm(np.array([*xy, s['z']]) - shoulder)
        yy, xx = np.mgrid[0:grid.shape[0], 0:grid.shape[1]]
        score[(yy - iy) ** 2 + (xx - ix) ** 2 <= (2 * radius / grid.res) ** 2] = -1
        if reach > cfg['max_reach']:
            continue
        out.append({'xy': xy, 'z': s['z'], 'clearance': float(s['clearance'][iy, ix]),
                    'headroom': float(headroom[iy, ix]), 'reach': float(reach)})
    return out


# ---------------------------------------------------------------- how the object can be set down
def _hull2d_min_width(pts):
    """Minimum width of a 2D point set (rotating calipers over its convex hull edges)."""
    from scipy.spatial import ConvexHull
    if len(pts) < 3:
        return 0.0
    try:
        h = pts[ConvexHull(pts).vertices]
    except Exception:  # noqa: BLE001  (collinear / degenerate: a line or a point)
        return 0.0
    best = np.inf
    for i in range(len(h)):
        e = h[(i + 1) % len(h)] - h[i]
        n = np.array([-e[1], e[0]]) / (np.linalg.norm(e) + 1e-12)
        best = min(best, np.ptp(h @ n))
    return float(best)


def _margin_inside(pts, c):
    """Distance from c to the boundary of the 2D convex hull of pts (negative = outside)."""
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(pts)
    except Exception:  # noqa: BLE001
        return -np.inf
    # scipy's 2D equations: n . x + d <= 0 inside
    return float(-(hull.equations[:, :2] @ c + hull.equations[:, 2]).max())


def _rotation_to_down(n):
    """Rotation R (object -> world) that turns the object-frame direction n to world -z."""
    n = n / np.linalg.norm(n)
    down = np.array([0.0, 0.0, -1.0])
    v = np.cross(n, down)
    c = float(np.dot(n, down))
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


def rest_poses(verts, com=None, contact_tol=0.002, min_width=0.028, min_margin=0.005, max_tilt_deg=135.0,
               merge_deg=8.0):
    """How the object can stand on a flat support: one entry per resting face of its convex hull whose support
    polygon holds the centre of mass (with `min_margin`) and is at least `min_width` wide. Narrower means a curved
    surface resting on one facet of the tessellated hull: a can on its side (2.0-2.5 cm facets) would roll. Upside-down poses (the mesh's +z tilted more than `max_tilt_deg`) are left out.

    YCB meshes are modelled upright (+z up), so the tilt of the mesh's +z from vertical ranks the poses:
    upright first, the "common-sense" way to store a bottle or a can; the rest by stability margin.
    -> list of dicts: R (object -> world, yaw arbitrary), tilt_deg, height, margin, width, name."""
    from scipy.spatial import ConvexHull
    verts = np.asarray(verts, float)
    com = verts.mean(axis=0) if com is None else np.asarray(com, float)
    hull = ConvexHull(verts)
    poses = []
    for eq in hull.equations:
        n = eq[:3] / np.linalg.norm(eq[:3])                      # outward normal of a hull face: rest on it
        if any(np.degrees(np.arccos(np.clip(n @ p['n'], -1, 1))) < merge_deg for p in poses):
            continue
        R = _rotation_to_down(n)
        W = verts @ R.T
        c = R @ com
        base = W[:, 2] <= W[:, 2].min() + contact_tol
        width = _hull2d_min_width(W[base, :2])
        margin = _margin_inside(W[base, :2], c[:2]) if width > 0 else -np.inf
        tilt = float(np.degrees(np.arccos(np.clip(R[2, 2], -1, 1))))  # world z of the mesh's +z
        poses.append({'n': n, 'R': R, 'tilt_deg': tilt, 'height': float(np.ptp(W[:, 2])), 'margin': margin,
                      'width': width})
    keep = [p for p in poses if p['width'] >= min_width and p['margin'] >= min_margin and p['tilt_deg'] <= max_tilt_deg]
    keep.sort(key=lambda p: (round(p['tilt_deg'] / 30.0), -p['margin']))
    for p in keep:
        p['name'] = 'upright' if p['tilt_deg'] < 30 else 'on its side' if p['tilt_deg'] < 120 else 'tipped'
        del p['n']
    return keep
