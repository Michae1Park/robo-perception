#!/usr/bin/env python3
"""Full pipeline from a text command: "pick up mustard sauce" -> detect -> pose -> spatial -> grasp.

Runs   : on the HOST, .venv/bin/python (it calls the `foundationpose` container itself for stages 1-2)
In     : data/multi_object_scene/ (BOP frame, 5 objects, no robot), assets/ycb/<obj>/textured.obj (meshes)
Out    : output/talk_<phrase>_{detect,pose_overlay,occupancy,grasp}.png, talk_<phrase>_result.json,
         output/full_pipeline_<phrase>.png (4-panel composite)

NOT an LLM: the phrase is extracted with a few prefix/article strips, and matched to one of the six
YCB meshes by keyword. Detection is YOLOE *text* prompting, which is very wording-sensitive
("mustard bottle" 0.68 vs "mustard sauce" 0.008), so known objects are detected via a canonical phrase.
Objects with no mesh (cracker box, banana, drill) stop cleanly after stage 1.

  .venv/bin/python pipeline/talk_and_pick.py --command "pick up mustard sauce"
  .venv/bin/python pipeline/talk_and_pick.py --repl
"""
import argparse
import json
import os
import re
import shlex
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from vision import grasp as g, load_config, read_obj_vertices, spatial, viz  # noqa: E402

SCENE = os.path.join(REPO, 'data', 'multi_object_scene')
OUT = os.path.join(REPO, 'output')
WORKER = os.path.join(REPO, 'pipeline', '_container_worker.py')
DEPTH_SCALE_MM_PER_UNIT = 0.1  # BOP convention, see data/multi_object_scene/meta.json
CONTAINER = 'foundationpose'
CONTAINER_PYTHON = '/opt/conda/envs/my/bin/python3'

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
            mesh = os.path.join(REPO, 'assets', 'ycb', name, 'textured.obj')
            return name, (mesh if os.path.isfile(mesh) else None)
    return None, None


# ---------------------------------------------------------------------- stages 1+2 (container)
def run_container_stages(phrase, label, mesh_path):
    cmd = [CONTAINER_PYTHON, WORKER, '--phrase', phrase, '--label', label]
    if mesh_path:
        cmd += ['--mesh', mesh_path]
    r = subprocess.run(['docker', 'exec', CONTAINER, 'bash', '-lc', ' '.join(shlex.quote(a) for a in cmd)])
    subprocess.run(['docker', 'exec', CONTAINER, 'chown', '-R', '%d:%d' % (os.getuid(), os.getgid()), OUT])  # root -> you
    if r.returncode != 0:
        raise SystemExit('container stage failed (rc=%d) -- is `docker start %s` running?' %
                         (r.returncode, CONTAINER))
    with open(os.path.join(OUT, 'talk_%s_result.json' % label)) as f:
        return json.load(f)


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
def run_command(command):
    cfg = load_config()
    phrase = extract_object_phrase(command)
    label = re.sub(r'[^a-z0-9]+', '_', phrase).strip('_')
    print('command: %r -> phrase: %r' % (command, phrase))

    mesh_name, mesh_path = match_mesh(phrase)
    detect_phrase = phrase
    if mesh_name:
        detect_phrase = CANONICAL_DETECT_PHRASE[mesh_name]
        print('matched known object %s%s -> detecting as "%s"' %
              (mesh_name, '' if mesh_path else ' (mesh FILE MISSING - run scripts/prepare_ycb.py)', detect_phrase))
    else:
        print('no known mesh for this phrase -- detecting with your wording as-is: "%s"' % detect_phrase)

    os.makedirs(OUT, exist_ok=True)
    det = run_container_stages(detect_phrase, label, mesh_path)
    composite = os.path.join(OUT, 'full_pipeline_%s.png' % label)
    if not det['detected']:
        print('=> stopped: "%s" was not detected in the scene.' % detect_phrase)
        return
    detect_png = os.path.join(OUT, 'talk_%s_detect.png' % label)
    panels = [(detect_png, '① Detection', '"%s" (text prompt, %.2f)' % (detect_phrase, det['score']))]
    if not (mesh_path and det.get('pose')):
        print('=> stopped after detection: no 3D mesh for this object. Overlay: %s' % detect_png)
        print('composite -> ' + build_composite(panels, composite))
        return

    K = np.loadtxt(os.path.join(SCENE, 'camera_K.txt'))
    scene_bgr = cv2.imread(os.path.join(SCENE, 'scene_rgb.png'))
    depth_raw = cv2.imread(os.path.join(SCENE, 'scene_depth.png'), cv2.IMREAD_UNCHANGED)
    depth_m = depth_raw.astype(np.float32) * DEPTH_SCALE_MM_PER_UNIT / 1000.0

    scene = spatial.analyze_scene(depth_m, K, cfg['spatial'])
    grid = scene['grid']
    print('  spatial: grid free=%d occupied=%d unknown=%d' % (
        (grid == spatial.FREE).sum(), (grid == spatial.OCCUPIED).sum(), (grid == spatial.UNKNOWN).sum()))
    occ_png = os.path.join(OUT, 'talk_%s_occupancy.png' % label)
    viz.plot_occupancy(scene_bgr, scene, occ_png, 'multi-object scene')

    T_world_cam = scene['T_world_cam']
    T_world_obj = T_world_cam @ np.array(det['pose']['ob_in_cam'])
    verts = read_obj_vertices(mesh_path)
    gripper, gcfg = cfg['gripper'], cfg['grasp']
    cands = g.generate_candidates(verts.min(axis=0), verts.max(axis=0), gripper['max_opening'],
                                  gcfg['width_margin'], gripper['finger_depth'])
    feasible = g.filter_candidates(cands, T_world_obj, 0.0, gcfg['max_approach_tilt_deg'],  # table_z = 0
                                   gcfg['table_clearance'])
    print('  grasp: %d candidates, %d feasible' % (len(cands), len(feasible)))
    grasp_png = os.path.join(OUT, 'talk_%s_grasp.png' % label)
    cv2.imwrite(grasp_png, viz.draw_grasps(scene_bgr, K, T_world_cam, feasible,
                                           viz.grasp_rows(cands, T_world_obj)))

    panels += [
        (os.path.join(OUT, 'talk_%s_pose_overlay.png' % label), '② 6-DoF Pose', 'FoundationPose, mesh=%s' % mesh_name),
        (occ_png, '③ Spatial', 'plane RANSAC + occupancy'),
        (grasp_png, '④ Grasp', '%d/%d candidates feasible' % (len(feasible), len(cands))),
    ]
    print('=> composite -> ' + build_composite(panels, composite))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--command', help='e.g. "pick up mustard sauce"')
    p.add_argument('--repl', action='store_true', help='keep reading commands from stdin')
    args = p.parse_args()
    if not args.command and not args.repl:
        raise SystemExit('give --command "pick up X" or --repl')
    if args.repl:
        print('type a command (blank line / Ctrl-D to quit)')
        while True:
            try:
                line = input('> ').strip()
            except EOFError:
                break
            if not line:
                break
            run_command(line)
    else:
        run_command(args.command)


if __name__ == '__main__':
    main()
