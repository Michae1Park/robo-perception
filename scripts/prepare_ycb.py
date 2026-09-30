#!/usr/bin/env python3
"""Download the 6 YCB objects (google_16k scans) and centre their meshes.

Output per object: assets/ycb/<name>/{textured.obj, textured.mtl, texture_map.png, info.json}
The mesh is rotated about z so its footprint is axis-aligned (minimum-area bounding rectangle;
some scans are yawed) and translated so its bounding-box centre is the origin. Used by talk_and_pick.py (FoundationPose + grasp geometry).

  python3 scripts/prepare_ycb.py            # download + process
  python3 scripts/prepare_ycb.py --src DIR  # use already-extracted <name>/google_16k folders
"""
import argparse
import json
import os
import shutil
import sys
import tarfile
import tempfile
import urllib.request

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from vision import read_obj_vertices  # noqa: E402
from vision.transforms import rot_z  # noqa: E402

OBJECTS = ['004_sugar_box', '005_tomato_soup_can', '006_mustard_bottle', '010_potted_meat_can',
           '061_foam_brick', '077_rubiks_cube']

URL = 'http://ycb-benchmarks.s3-website-us-east-1.amazonaws.com/data/google/{name}_google_16k.tgz'


def download(name, dst_dir):
    url = URL.format(name=name)
    path = os.path.join(dst_dir, name + '.tgz')
    print('downloading', url)
    urllib.request.urlretrieve(url, path)
    with tarfile.open(path) as tar:
        tar.extractall(dst_dir)
    return os.path.join(dst_dir, name, 'google_16k')


def best_yaw(v, step_deg=0.25):
    """Yaw (rad) that minimises the xy bounding-rectangle area; ties -> longer side along x."""
    best = None
    for deg in np.arange(0.0, 90.0, step_deg):
        xy = v[:, :2] @ rot_z(np.deg2rad(deg))[:2, :2].T
        ext = xy.max(axis=0) - xy.min(axis=0)
        area = ext[0] * ext[1]
        if best is None or area < best[0] - 1e-9:
            best = (area, deg, ext)
    yaw = np.deg2rad(best[1])
    if best[2][1] > best[2][0]:
        yaw += np.pi / 2
    return yaw


def transform_obj(src, dst, R, offset):
    """Write the OBJ with v -> R v - offset and vn -> R vn; everything else untouched."""
    with open(src) as fi, open(dst, 'w') as fo:
        for line in fi:
            if line.startswith('v '):
                p = line.split()
                xyz = R @ np.array([float(p[1]), float(p[2]), float(p[3])]) - offset
                fo.write('v %.6f %.6f %.6f\n' % tuple(xyz))
            elif line.startswith('vn '):
                p = line.split()
                n = R @ np.array([float(p[1]), float(p[2]), float(p[3])])
                fo.write('vn %.6f %.6f %.6f\n' % tuple(n))
            else:
                fo.write(line)


def process(name, src_dir, out_dir):
    obj = os.path.join(src_dir, 'textured.obj')
    v = read_obj_vertices(obj)
    yaw = best_yaw(v)
    R = rot_z(yaw)
    vr = v @ R.T
    lo, hi = vr.min(axis=0), vr.max(axis=0)
    center = (lo + hi) / 2.0
    os.makedirs(out_dir, exist_ok=True)
    transform_obj(obj, os.path.join(out_dir, 'textured.obj'), R, center)
    for f in ('textured.mtl', 'texture_map.png'):
        if os.path.isfile(os.path.join(src_dir, f)):
            shutil.copy(os.path.join(src_dir, f), os.path.join(out_dir, f))
    ext = (hi - lo).tolist()
    info = {'name': name, 'source_yaw_deg': float(np.rad2deg(yaw)), 'source_center_offset': center.tolist(),
            'extents_m': ext,
            'num_vertices': int(len(v))}
    with open(os.path.join(out_dir, 'info.json'), 'w') as f:
        json.dump(info, f, indent=2)
    print('%-22s yaw %+6.1f deg, extents (m) x=%.3f y=%.3f z=%.3f' % (name, np.rad2deg(yaw), *ext))
    if max(ext) > 1.0:
        print('  WARNING: extents > 1 m - mesh is probably not in metres')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--src', default='', help='directory with extracted <name>/google_16k folders')
    ap.add_argument('--objects', nargs='*', help='subset of object names')
    args = ap.parse_args()
    names = args.objects or OBJECTS
    out_root = os.path.join(ROOT, 'assets', 'ycb')
    tmp = None if args.src else tempfile.mkdtemp(prefix='ycb_')
    try:
        for name in names:
            src = os.path.join(args.src, name, 'google_16k') if args.src else download(name, tmp)
            process(name, src, os.path.join(out_root, name))
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)
    print('done ->', out_root)


if __name__ == '__main__':
    main()
