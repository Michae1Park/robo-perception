#!/usr/bin/env python3
"""Stage 2 playground: RGB-D frame + object mask + mesh -> FoundationPose 6-DoF pose, top hypotheses,
timing, overlay image. Guide: docs/STAGE2_POSE.md.

  (no flags)                     the stage 1 playground's image (data/multi_object_scene): YOLOE finds the
                                 mustard bottle (visual prompt) -> mask; mesh assets/ycb/006_mustard_bottle
  --data DIR                     instead: a dataset in FoundationPose demo_data layout (e.g. data/sim/<object>
                                 from sim/scene.py)
  --frame N / --mask gt|PATH     frame + mask for --data
  --iterations 5                 refinement iterations per hypothesis
  --n-views 40 --inplane-step 60 rotation hypothesis grid

Run it from the host (.venv/bin/python): it re-runs itself inside the `foundationpose` container.
Defaults come from config.yaml `pose:`. Writes output/pose/<tag>.png + .json and, when run from a
Cursor terminal, opens the image as a tab.
"""
import argparse
import glob
import json
import os
import shlex
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from vision import load_config  # noqa: E402

CONTAINER, CONTAINER_PY = 'foundationpose', '/opt/conda/envs/my/bin/python3'
OUT = os.path.join(REPO, 'output', 'pose')
SCENE = os.path.join(REPO, 'data', 'multi_object_scene')
OBJECT = '006_mustard_bottle'  # has a visual prompt in SCENE/refs/ and a mesh in assets/ycb/


def parse_args():
    cfg = load_config()['pose']
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data', default=None, help='dataset dir: rgb/ depth/ masks/ mesh/ cam_K.txt [ob_in_cam/]')
    p.add_argument('--frame', type=int, default=None, help='frame index (default: 0)')
    p.add_argument('--mask', default=None, help='gt | path to a mask png (nonzero = object)')
    p.add_argument('--mesh', default=None, help='default: <data>/mesh/textured_simple.obj or textured.obj')
    p.add_argument('--iterations', type=int, default=cfg['iterations'])
    p.add_argument('--n-views', type=int, default=40, help='min icosphere viewpoints (rounded up to 12/42/162/...)')
    p.add_argument('--inplane-step', type=int, default=60, help='degrees between in-plane rotations')
    p.add_argument('--top', type=int, default=5, help='how many ranked hypotheses to print')
    p.add_argument('--tag', default='play')
    a = p.parse_args()
    if a.iterations < 1:
        p.error('--iterations must be >= 1 (FoundationPose crashes on 0)')
    if a.data is None:  # default: the multi-object scene; mask from YOLOE, mesh from assets/ycb
        a.data, a.frame, a.mask = SCENE, 0, 'yoloe'
        a.mesh = a.mesh or os.path.join(REPO, 'assets', 'ycb', OBJECT, 'textured.obj')
        return a
    a.data = os.path.abspath(a.data)
    if a.mask is None:
        a.mask = 'gt'
    if a.frame is None:
        a.frame = 0
    if a.mesh is None:
        a.mesh = os.path.join(a.data, 'mesh', 'textured_simple.obj')
        if not os.path.exists(a.mesh):
            a.mesh = os.path.join(a.data, 'mesh', 'textured.obj')
    return a


def run_in_container():
    """Host side: re-run this script in the container, then fix ownership and open the image."""
    subprocess.run(['docker', 'start', CONTAINER], check=True, stdout=subprocess.DEVNULL)
    cmd = 'cd %s && %s %s' % (REPO, CONTAINER_PY, shlex.join([os.path.abspath(sys.argv[0])] + sys.argv[1:]))   # absolute: the container cds to REPO
    rc = subprocess.run(['docker', 'exec', CONTAINER, 'bash', '-lc', cmd]).returncode
    subprocess.run(['docker', 'exec', CONTAINER, 'chown', '-R', '%d:%d' % (os.getuid(), os.getgid()),
                    os.path.join(REPO, 'output')], stderr=subprocess.DEVNULL)
    path = os.path.join(OUT, parse_args().tag + '.png')
    if rc == 0 and 'VSCODE_IPC_HOOK_CLI' in os.environ:  # set in Cursor / VS Code terminals
        subprocess.run(['cursor', '--reuse-window', path])
    sys.exit(rc)


def load_scene_inputs(a):
    """multi_object_scene (one BOP frame): YOLOE (same code as stage 1) picks the mask."""
    import cv2
    import numpy as np
    from vision.detect import build_visual_prompt_model, detect, detect_kwargs, load_prompt
    dcfg = load_config()['detect']
    bgr = cv2.imread(os.path.join(SCENE, 'scene_rgb.png'))
    depth_mm_per_unit = json.load(open(os.path.join(SCENE, 'meta.json')))['depth_scale_mm_per_unit']
    depth = cv2.imread(os.path.join(SCENE, 'scene_depth.png'), cv2.IMREAD_UNCHANGED) * depth_mm_per_unit / 1000.0
    K = np.loadtxt(os.path.join(SCENE, 'camera_K.txt')).reshape(3, 3)

    model = build_visual_prompt_model(dcfg['weights'], [(OBJECT, *load_prompt(
        os.path.join(SCENE, 'refs', OBJECT + '.png')))], dcfg['device'])
    dets = [d for d in detect(model, bgr, **detect_kwargs(dcfg)) if d['mask'] is not None]
    if not dets:
        raise SystemExit('YOLOE found no %s in the scene' % OBJECT)
    best = max(dets, key=lambda d: d['score'])
    print('YOLOE %s: score %.3f, bbox %s' % (OBJECT, best['score'], [int(v) for v in best['bbox']]))
    mask_path = os.path.join(OUT, a.tag + '_mask.png')
    os.makedirs(OUT, exist_ok=True)
    cv2.imwrite(mask_path, best['mask'].astype(np.uint8) * 255)
    return 0, 'scene_rgb', K, bgr[..., ::-1].copy(), depth, best['mask'], mask_path


def load_inputs(a):
    """-> frame_id, K, rgb (RGB uint8), depth (metres), mask (bool), mask description."""
    import cv2
    import numpy as np
    if a.mask == 'yoloe':
        return load_scene_inputs(a)
    rgb_path = sorted(glob.glob(os.path.join(a.data, 'rgb', '*.png')))[a.frame]
    frame_id = os.path.splitext(os.path.basename(rgb_path))[0]

    rgb = cv2.imread(rgb_path)[..., ::-1].copy()
    depth = cv2.imread(os.path.join(a.data, 'depth', frame_id + '.png'), cv2.IMREAD_UNCHANGED) / 1000.0
    depth[depth < 0.001] = 0
    K = np.loadtxt(os.path.join(a.data, 'cam_K.txt')).reshape(3, 3)

    if a.mask == 'gt':
        mask_path = os.path.join(a.data, 'masks', frame_id + '.png')
        if not os.path.exists(mask_path):
            raise SystemExit('no ground-truth mask %s' % mask_path)
    else:
        mask_path = a.mask
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    mask = (mask if mask.ndim == 2 else mask[..., 0]) > 0
    if mask.shape != depth.shape:
        mask = cv2.resize(mask.astype(np.uint8), depth.shape[::-1], interpolation=cv2.INTER_NEAREST) > 0
    return a.frame, frame_id, K, rgb, depth, mask, mask_path


def rot_deg(R):
    import numpy as np
    return np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))


def main():
    a = parse_args()
    import cv2
    import numpy as np
    from vision import pose
    frame, frame_id, K, rgb, depth, mask, mask_path = load_inputs(a)

    # 1. build: mesh + refiner/scorer networks + rotation hypothesis grid
    t0 = time.perf_counter()
    est, mesh = pose.build_estimator(a.mesh, os.path.join(OUT, a.tag + '_fp_debug'),
                                     a.n_views, a.inplane_step, quiet=True)
    t_build = time.perf_counter() - t0
    n_hyp = len(est.rot_grid)

    # 2. register: one warm-up run, then one timed run (same seed -> same result)
    kw = dict(K=K, rgb=rgb, depth=depth, ob_mask=mask, iteration=a.iterations)
    est.register(**kw)
    t0 = time.perf_counter()
    ob_in_cam = est.register(**kw).reshape(4, 4)
    t_reg = time.perf_counter() - t0
    t_guess = est.guess_translation(depth, mask, K)  # where every hypothesis starts

    # 3. report
    valid = mask & (depth >= 0.001)
    print('\nframe %d (%s) | mask %s: %d px, %d with depth' % (frame, frame_id, a.mask, mask.sum(), valid.sum()))
    print('mesh %s' % os.path.relpath(a.mesh, REPO))
    print('hypotheses %d (icosphere >= %d views x %d in-plane, clustered) | iterations=%d'
          % (n_hyp, a.n_views, len(range(0, 360, a.inplane_step)), a.iterations))
    print('build %.2f s | register %.0f ms' % (t_build, t_reg * 1000))

    t = ob_in_cam[:3, 3]
    print('\nbest pose (object in camera frame, metres):')
    print(np.array2string(ob_in_cam, precision=4, suppress_small=True))
    print('translation  x=%.3f y=%.3f z=%.3f  (start guess from mask+depth: %.3f %.3f %.3f, moved %.1f mm)'
          % (*t, *t_guess, np.linalg.norm(t - t_guess) * 1000))

    to_centered = est.get_tf_to_centered_mesh().cpu().numpy()
    poses = [p @ to_centered for p in est.poses.cpu().numpy()]
    scores = est.scores.cpu().numpy()
    print('\n%-4s %8s %12s %12s' % ('rank', 'score', 'rot vs #0', 'trans vs #0'))
    for i in range(min(a.top, len(poses))):
        dR = rot_deg(poses[i][:3, :3] @ poses[0][:3, :3].T)
        dt = np.linalg.norm(poses[i][:3, 3] - poses[0][:3, 3]) * 1000
        print('%-4d %8.2f %10.1f deg %9.1f mm' % (i, scores[i], dR, dt))
    print('score range over all %d: %.2f .. %.2f' % (len(scores), scores.min(), scores.max()))

    gt_path = os.path.join(a.data, 'ob_in_cam', frame_id + '.txt')   # written by sim/scene.py
    err = None
    if os.path.exists(gt_path):
        gt = np.loadtxt(gt_path).reshape(4, 4)
        err = {'rot_deg': float(rot_deg(ob_in_cam[:3, :3] @ gt[:3, :3].T)),
               'trans_mm': float(np.linalg.norm(ob_in_cam[:3, 3] - gt[:3, 3]) * 1000)}
        print('\nvs ground truth: rotation %.2f deg | translation %.1f mm' % (err['rot_deg'], err['trans_mm']))

    # 4. draw: pose box + axes, mask outline in cyan
    vis = pose.draw_pose(K, rgb, ob_in_cam, mesh)[..., ::-1].copy()
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, contours, -1, (255, 255, 0), 1)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, a.tag + '.png')
    cv2.imwrite(path, vis)
    with open(os.path.join(OUT, a.tag + '.json'), 'w') as f:
        json.dump({'frame_index': frame, 'frame_id': frame_id, 'mask': mask_path, 'mesh': a.mesh,
                   'iterations': a.iterations, 'n_views': a.n_views, 'inplane_step': a.inplane_step,
                   'n_hypotheses': n_hyp, 'register_ms': t_reg * 1000, 'ob_in_cam': ob_in_cam.tolist(),
                   'top_scores': scores[:a.top].tolist(), 'gt_error': err}, f, indent=2)
    print('\n-> ' + path)


if __name__ == '__main__':
    if not os.path.exists(CONTAINER_PY):
        run_in_container()
    main()
