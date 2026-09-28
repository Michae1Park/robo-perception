#!/usr/bin/env python3
"""Natural-language front end to the bare-metal vision pipeline: type a command like
"pick up mustard sauce", get detection -> (if a mesh is known for that object) pose -> spatial ->
grasp, run automatically end to end -- one command from the HOST, no manual docker exec juggling.

WHAT THIS IS NOT: an LLM or NLP model. Phrase extraction is a handful of string prefix/article
strips (see extract_object_phrase() below) and object-to-mesh matching is keyword substring
matching against this repo's 6 known assets/ycb/* objects (see MESH_KEYWORDS below) -- deliberately
minimal, matching the rest of this repo's style. The actual "understanding" is YOLOE's own
open-vocabulary TEXT-prompt detector (model.get_text_pe(), confirmed available in the installed
ultralytics version) -- it can localize arbitrary phrases in the scene with NO reference image,
unlike every other detection script in this repo so far (which all use one-shot VISUAL prompting:
a reference image + box). Detection therefore works for ANY phrase; pose/grasp only continue for
the 6 objects this repo has a 3D mesh for (assets/ycb/*) -- for anything else (the cracker box,
banana, power drill in the bundled scene) it stops cleanly after detection and says so.

Run on the HOST (this script shells into the `foundationpose` container itself for the GPU stages
-- you do not need to docker exec anything by hand):

  docker start foundationpose
  .venv/bin/python perception_demo/pipeline_demo/talk_and_pick.py --command "pick up mustard sauce"
  .venv/bin/python perception_demo/pipeline_demo/talk_and_pick.py --command "grab the tomato soup can"
  .venv/bin/python perception_demo/pipeline_demo/talk_and_pick.py --command "pick up the cheez-it box"
  .venv/bin/python perception_demo/pipeline_demo/talk_and_pick.py --repl   # keep typing commands
  docker stop foundationpose
"""
import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, 'ros2', 'ppp_common'))
sys.path.insert(0, os.path.join(REPO, 'ros2', 'ppp_perception'))
sys.path.insert(0, os.path.join(REPO, 'ros2', 'ppp_manipulation'))
from ppp_common import transforms as tf  # noqa: E402
from ppp_common.mesh import read_obj_vertices  # noqa: E402
from ppp_perception.core.plane import fit_plane_ransac  # noqa: E402
from ppp_perception.core import free_space as fs  # noqa: E402
from ppp_manipulation.core import grasp as g  # noqa: E402

SCENE = os.path.join(REPO, 'perception_demo/pipeline_demo/multi_object_scene')
OUT = os.path.join(REPO, 'perception_demo/pipeline_demo/output')
DOCS = os.path.join(REPO, 'docs/pipeline_demo/multi_object')
FP_DIR = os.path.join(REPO, 'third_party/FoundationPose')
WORKER_SRC = os.path.join(REPO, 'perception_demo/pipeline_demo/_container_worker.py')
DEPTH_SCALE_MM_PER_UNIT = 0.1  # this scene's BOP convention, see multi_object_scene/meta.json
CONTAINER = 'foundationpose'

# ---------------------------------------------------------------- phrase extraction (no NLP/LLM)
_PREFIXES = ['please pick up the ', 'please pick up ', 'please grab the ', 'please grab ',
            'please get the ', 'please get ', 'pick up the ', 'pick up ', 'grab the ', 'grab ',
            'get the ', 'get ', 'bring me the ', 'bring me ']
_ARTICLES = ('the ', 'a ', 'an ')


def extract_object_phrase(command):
    s = re.sub(r'\s+', ' ', command.strip().lower()).strip(' .!?')
    for pre in sorted(_PREFIXES, key=len, reverse=True):
        if s.startswith(pre):
            s = s[len(pre):]
            break
    for art in _ARTICLES:
        if s.startswith(art):
            s = s[len(art):]
    return s.strip()


# ------------------------------------------------------- object phrase -> known mesh (substring)
MESH_KEYWORDS = {
    '004_sugar_box': ['sugar'],
    '005_tomato_soup_can': ['tomato', 'soup'],
    '006_mustard_bottle': ['mustard'],
    '010_potted_meat_can': ['potted meat', 'spam', 'meat can'],
    '061_foam_brick': ['foam', 'brick'],
    '077_rubiks_cube': ['rubik', 'rubix', 'cube'],
}

# YOLOE's open-vocabulary text-prompt confidence is very sensitive to exact wording (CLIP-style
# text embedding, not a keyword search) -- e.g. on this scene "mustard bottle" scores 0.68 but
# "mustard sauce" scores 0.008 (verified empirically this session). So once a phrase is matched to
# a known object, detection uses a CANONICAL descriptive phrase for that object rather than the
# user's raw wording -- the match is still reported to the user for transparency (see run_command).
# For anything that doesn't match a known object, the raw extracted phrase is used as-is (it scores
# fine for names that are already close to a common object category, e.g. "cheez-it box", "banana").
CANONICAL_DETECT_PHRASE = {
    '004_sugar_box': 'sugar box',
    '005_tomato_soup_can': 'tomato soup can',
    '006_mustard_bottle': 'mustard bottle',
    '010_potted_meat_can': 'potted meat can',
    '061_foam_brick': 'foam brick',
    '077_rubiks_cube': 'rubiks cube',
}


def match_mesh(phrase):
    for name, kws in MESH_KEYWORDS.items():
        if any(kw in phrase for kw in kws):
            mesh = os.path.join(REPO, 'assets/ycb', name, 'textured.obj')
            return name, (mesh if os.path.isfile(mesh) else None)
    return None, None


# ----------------------------------------------------------------------- stage (1)+(2): container
def run_container_stages(phrase, label, mesh_path):
    os.makedirs(FP_DIR, exist_ok=True)
    shutil.copy(WORKER_SRC, os.path.join(FP_DIR, '_container_worker.py'))
    inner = ['/opt/conda/envs/my/bin/python3', '_container_worker.py', '--phrase', phrase,
            '--label', label]
    if mesh_path:
        inner += ['--mesh', mesh_path]
    inner_cmd = 'cd %s && %s' % (shlex.quote(FP_DIR), ' '.join(shlex.quote(a) for a in inner))
    r = subprocess.run(['docker', 'exec', CONTAINER, 'bash', '-lc', inner_cmd])
    if r.returncode != 0:
        raise SystemExit('container stage failed (rc=%d) -- is `docker start %s` running?' %
                         (r.returncode, CONTAINER))
    with open(os.path.join(OUT, 'talk_%s_result.json' % label)) as f:
        return json.load(f)


# --------------------------------------------------------------------------- stage (3): spatial
def depth_to_points(depth, K):
    h, w = depth.shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    us, vs = np.meshgrid(np.arange(w), np.arange(h))
    valid = depth > 0.001
    z = depth[valid]
    x = (us[valid] - cx) * z / fx
    y = (vs[valid] - cy) * z / fy
    return np.stack([x, y, z], axis=-1)


def run_spatial_stage(K, depth_m, label):
    """Same plane-aligned-pseudo-world construction as 03_spatial.py, parameterized."""
    pts_cam = depth_to_points(depth_m, K)
    plane_cam, inliers = fit_plane_ransac(pts_cam, dist_thresh=0.006, iters=300,
                                          up=(0.0, 0.0, 1.0), max_tilt_deg=180, min_inliers=500)
    if plane_cam is None:
        raise SystemExit('RANSAC found no dominant table plane')
    if plane_cam[3] < 0:
        plane_cam = -plane_cam
    n = plane_cam[:3]
    p0 = -plane_cam[3] * n
    tmp = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n, tmp); u /= np.linalg.norm(u)
    v = np.cross(n, u)
    R = np.stack([u, v, n], axis=0)
    T_world_cam = np.eye(4)
    T_world_cam[:3, :3] = R
    T_world_cam[:3, 3] = -R @ p0
    pts_world = (R @ (pts_cam - p0).T).T
    plane_world = np.array([0.0, 0.0, 1.0, 0.0])

    table_pts = pts_world[np.abs(pts_world[:, 2]) < 0.01]
    x0, x1 = np.percentile(table_pts[:, 0], [2, 98])
    y0, y1 = np.percentile(table_pts[:, 1], [2, 98])
    zone = ((float(x0), float(x1)), (float(y0), float(y1)))
    grid = fs.occupancy_grid(pts_world, plane_world, zone, resolution=0.01,
                             height_thresh=0.012, max_height=0.6)

    rgb = cv2.cvtColor(cv2.imread(os.path.join(SCENE, 'scene_rgb.png')), cv2.COLOR_BGR2RGB)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].imshow(rgb); axes[0].set_title('multi-object scene'); axes[0].axis('off')
    cmap = matplotlib.colors.ListedColormap(['#bbbbbb', '#2ecc71', '#e74c3c'])
    disp = np.select([grid == fs.UNKNOWN, grid == fs.FREE, grid == fs.OCCUPIED], [0, 1, 2])
    axes[1].imshow(disp, cmap=cmap, origin='lower', extent=[x0, x1, y0, y1], vmin=0, vmax=2)
    axes[1].set_title('place-zone occupancy (grey=unknown, green=free, red=occupied)')
    axes[1].set_xlabel('x (pseudo-world, m)'); axes[1].set_ylabel('y (pseudo-world, m)')
    fig.tight_layout()
    out_path = os.path.join(OUT, 'talk_%s_occupancy.png' % label)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print('  table plane inliers: %d / %d, grid free=%d occupied=%d unknown=%d' % (
        int(inliers.sum()), len(pts_cam), int((grid == fs.FREE).sum()),
        int((grid == fs.OCCUPIED).sum()), int((grid == fs.UNKNOWN).sum())))
    return T_world_cam, out_path


# ----------------------------------------------------------------------------- stage (4): grasp
# Candidate generation/filtering (core/grasp.py calls) happens inline in run_command() -- table_z
# = 0 by construction of run_spatial_stage's pseudo-world frame, same as 04_grasp.py. This helper
# only draws the result.
def draw_grasp_overlay(rgb_bgr, K, T_world_cam, feasible, cands, label):
    T_cam_world = tf.invert(T_world_cam)
    img = rgb_bgr.copy()
    arrow_len = 0.06

    def draw_grasp(T_world_tcp, width, color):
        # Two-finger parallel-jaw gripper: approach arrow (TCP's +z) plus the two actual finger
        # contact points, offset +/-width/2 along the TCP's closing axis (its y-axis, per
        # core/grasp.py's generate_candidates: R = [x, y, z], y = closing direction).
        tip = T_world_tcp[:3, 3]
        tail = tip - T_world_tcp[:3, 2] * arrow_len
        closing = T_world_tcp[:3, 1]
        finger0 = tip + closing * (width / 2.0)
        finger1 = tip - closing * (width / 2.0)
        pts_cam = tf.transform_points(T_cam_world, np.stack([tail, tip, finger0, finger1]))
        uv, z = tf.project(K, np.eye(4), pts_cam)
        if (z <= 0).any():
            return
        p0, p1, f0, f1 = uv.astype(int)
        cv2.arrowedLine(img, tuple(p0), tuple(p1), (0, 0, 0), 6, cv2.LINE_AA, tipLength=0.35)
        cv2.arrowedLine(img, tuple(p0), tuple(p1), color, 3, cv2.LINE_AA, tipLength=0.35)
        cv2.line(img, tuple(f0), tuple(f1), (0, 0, 0), 5, cv2.LINE_AA)
        cv2.line(img, tuple(f0), tuple(f1), color, 2, cv2.LINE_AA)
        for p in (f0, f1):
            cv2.circle(img, tuple(p), 6, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(img, tuple(p), 4, color, -1, cv2.LINE_AA)

    if feasible:
        for i, c in enumerate(feasible):
            draw_grasp(c.T_world_tcp, c.width, (255, 0, 255) if i == 0 else (255, 255, 0))
        label_txt = 'grasp candidates: magenta=best, cyan=other feasible (%d/%d)' % (
            len(feasible), len(cands))
    else:
        label_txt = '0/%d passed the tilt/table filter -- no safe top-down approach in this pose' % len(cands)
    cv2.putText(img, label_txt, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    out_path = os.path.join(OUT, 'talk_%s_grasp.png' % label)
    cv2.imwrite(out_path, img)
    return out_path


# ------------------------------------------------------------------------------------ composite
def build_composite(panels, out_path):
    """panels: list of (image_path, title, subtitle), laid out 2 per row, same visual style as
    docs/pipeline_demo/pipeline_overview.png."""
    H, pad, label_h, arrow_w = 420, 16, 54, 56
    rows = [panels[i:i + 2] for i in range(0, len(panels), 2)]

    def load(path):
        im = Image.open(path).convert('RGB')
        w = int(im.width * H / im.height)
        return im.resize((w, H))

    row_imgs = [[load(p) for p, _, _ in row] for row in rows]
    row_w = max(sum(im.width for im in imgs) + arrow_w * (len(imgs) - 1) for imgs in row_imgs)
    total_w, row_h = row_w + pad * 2, label_h + H + pad
    total_h = row_h * len(rows) + pad
    canvas = Image.new('RGB', (total_w, total_h), 'white')
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 26)
        font_sub = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 17)
    except Exception:
        font = font_sub = ImageFont.load_default()

    for ri, (row, imgs) in enumerate(zip(rows, row_imgs)):
        y0 = pad + ri * row_h
        x = pad
        for i, ((_, title, sub), im) in enumerate(zip(row, imgs)):
            canvas.paste(im, (x, y0 + label_h))
            tw = draw.textlength(title, font=font)
            draw.text((x + im.width / 2 - tw / 2, y0 + 2), title, fill='black', font=font)
            sw = draw.textlength(sub, font=font_sub)
            draw.text((x + im.width / 2 - sw / 2, y0 + 32), sub, fill='dimgray', font=font_sub)
            x += im.width
            if i < len(row) - 1:
                cy = y0 + label_h + H // 2
                draw.line([(x + 8, cy), (x + arrow_w - 14, cy)], fill='black', width=4)
                draw.polygon([(x + arrow_w - 14, cy - 11), (x + arrow_w - 14, cy + 11),
                             (x + arrow_w, cy)], fill='black')
                x += arrow_w
        if ri < len(rows) - 1:
            cx = pad + row_imgs[ri][0].width // 2
            y1 = y0 + row_h
            draw.line([(cx, y1 - pad + 6), (cx, y1 - 6)], fill='black', width=4)
            draw.polygon([(cx - 11, y1 - 6), (cx + 11, y1 - 6), (cx, y1 + 8)], fill='black')

    canvas.save(out_path)
    return out_path


# ------------------------------------------------------------------------------------- pipeline
def run_command(command, weights='yoloe-11l-seg.pt', conf=0.05):
    phrase = extract_object_phrase(command)
    label = re.sub(r'[^a-z0-9]+', '_', phrase).strip('_')
    print('command: %r -> phrase: %r' % (command, phrase))

    mesh_name, mesh_path = match_mesh(phrase)
    detect_phrase = phrase
    if mesh_name:
        detect_phrase = CANONICAL_DETECT_PHRASE[mesh_name]
        print('matched known object: %s%s -> detecting as "%s" (canonical phrase, more reliable '
              'than your exact wording for YOLOE\'s text encoder -- see module docstring)' %
              (mesh_name, '' if mesh_path else ' (mesh FILE MISSING)', detect_phrase))
    else:
        print('no known mesh matches this phrase (cracker box / banana / power drill etc. have '
              'none) -- detecting with your wording as-is: "%s"' % detect_phrase)

    os.makedirs(OUT, exist_ok=True)
    det = run_container_stages(detect_phrase, label, mesh_path)
    if not det['detected']:
        print('=> stopped: "%s" was not detected in the scene at all.' % detect_phrase)
        return

    detect_png = os.path.join(OUT, 'talk_%s_detect.png' % label)
    if not (mesh_path and det.get('pose')):
        reason = 'no 3D mesh available for this object' if not mesh_path else 'pose stage did not run'
        print('=> stopped after detection: %s. Overlay: %s' % (reason, detect_png))
        comp = build_composite([(detect_png, '① Detection', '"%s" (text prompt)' % detect_phrase)],
                               os.path.join(DOCS, 'full_pipeline_%s.png' % label))
        print('   (single-panel) composite -> %s' % comp)
        return

    print('mesh + pose available -- continuing to stages (3) and (4)')
    K = np.loadtxt(os.path.join(SCENE, 'camera_K.txt'))
    scene_bgr = cv2.imread(os.path.join(SCENE, 'scene_rgb.png'))
    depth_raw = cv2.imread(os.path.join(SCENE, 'scene_depth.png'), cv2.IMREAD_UNCHANGED)
    depth_m = depth_raw.astype(np.float32) * DEPTH_SCALE_MM_PER_UNIT / 1000.0
    T_cam_obj = np.array(det['pose']['ob_in_cam'])

    T_world_cam, occ_png = run_spatial_stage(K, depth_m, label)
    T_world_obj = T_world_cam @ T_cam_obj

    cfg = yaml.safe_load(open(os.path.join(REPO, 'ros2/ppp_bringup/config/scene.yaml')))
    robot, gcfg = cfg['robot'], cfg['grasp']
    verts = read_obj_vertices(mesh_path)
    lo, hi = verts.min(axis=0), verts.max(axis=0)
    cands = g.generate_candidates(lo, hi, robot['max_opening'], gcfg['width_margin'], robot['finger_depth'])
    feasible = g.filter_candidates(cands, T_world_obj, 0.0, gcfg['max_approach_tilt_deg'],
                                   gcfg['table_clearance'])
    print('  grasp: %d raw candidates, %d pass geometric filter' % (len(cands), len(feasible)))
    grasp_png = draw_grasp_overlay(scene_bgr, K, T_world_cam, feasible, cands, label)

    pose_png = os.path.join(OUT, 'talk_%s_pose_overlay.png' % label)
    panels = [
        (detect_png, '① Detection', '"%s" (text prompt, %.2f)' % (detect_phrase, det['score'])),
        (pose_png, '② 6-DoF Pose', 'FoundationPose, mesh=%s' % mesh_name),
        (occ_png, '③ Spatial', 'plane RANSAC + occupancy'),
        (grasp_png, '④ Grasp', '%d/%d candidates feasible' % (len(feasible), len(cands))),
    ]
    os.makedirs(DOCS, exist_ok=True)
    comp = build_composite(panels, os.path.join(DOCS, 'full_pipeline_%s.png' % label))
    print('=> full pipeline composite -> %s' % comp)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--command', help='e.g. "pick up mustard sauce"')
    p.add_argument('--repl', action='store_true', help='keep reading commands from stdin')
    p.add_argument('--weights', default='yoloe-11l-seg.pt')
    p.add_argument('--conf', type=float, default=0.05)
    args = p.parse_args()
    if not args.command and not args.repl:
        raise SystemExit('give --command "pick up X" or --repl')
    if args.repl:
        print('type a command (blank line / Ctrl-D to quit), e.g. "pick up mustard sauce"')
        while True:
            try:
                line = input('> ').strip()
            except EOFError:
                break
            if not line:
                break
            run_command(line, args.weights, args.conf)
    else:
        run_command(args.command, args.weights, args.conf)


if __name__ == '__main__':
    main()
