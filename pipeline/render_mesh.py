#!/usr/bin/env python3
"""Look at a mesh the way FoundationPose sees it (its own nvdiffrast renderer).

  python pipeline/render_mesh.py                 # rows = views; columns = texture | shape | vertices | triangles | triangles x5
  python pipeline/render_mesh.py --pose play     # mesh at the pose in output/pose/play.json, over the scene

Run from the host; it re-runs itself in the `foundationpose` container. Writes output/mesh/*.png.
"""
import argparse
import json
import os
import shlex
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = '/opt/conda/envs/my/bin/python3'  # the container's python
if not os.path.exists(PY):  # on the host: run this same command inside the container
    subprocess.run(['docker', 'start', 'foundationpose'], stdout=subprocess.DEVNULL)
    sys.exit(subprocess.run(['docker', 'exec', 'foundationpose', 'bash', '-lc', 'cd %s && %s %s && chown -R %d:%d output'
                             % (REPO, PY, shlex.join([os.path.abspath(sys.argv[0])] + sys.argv[1:]), os.getuid(), os.getgid())]).returncode)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import trimesh  # noqa: E402

sys.path.insert(0, REPO)
from vision import pose  # noqa: E402

fp = pose._fp()  # FoundationPose's Utils (renderer, axis drawing)
glctx = fp.dr.RasterizeCudaContext()
S = 320  # px per tile


def render(tensors, K, H, W, ob_in_cam, ambient=0.8, diffuse=0.5):
    """Render the mesh at one pose -> BGR image (black background)."""
    color, _, _ = fp.nvdiffrast_render(K=K, H=H, W=W, glctx=glctx, mesh_tensors=tensors, use_light=True,
                                       ob_in_cams=torch.tensor(ob_in_cam[None], device='cuda', dtype=torch.float),
                                       w_ambient=ambient, w_diffuse=diffuse)
    return cv2.cvtColor((color[0].cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def look_at(eye, up=np.array([0, 0, 1.0])):
    """Camera at `eye` looking at the origin (OpenCV: x right, y down, z forward) -> object-in-camera 4x4."""
    z = -eye / np.linalg.norm(eye)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    cam = np.eye(4)
    cam[:3, :3] = np.stack([x, np.cross(z, x), z], axis=1)
    cam[:3, 3] = eye
    return np.linalg.inv(cam)


def white(img):
    return np.where(img.any(-1, keepdims=True), img, 255).astype(np.uint8)


def label(img, text):
    cv2.putText(img, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(img, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return img


def triangles(mesh, K, T, zoom=1):
    """Edges of the triangles facing the camera (back ones hidden). zoom > 1 magnifies around the centre."""
    K = K.copy()
    K[:2, :2] *= zoom
    cam = (T[:3, :3] @ mesh.vertices.T + T[:3, 3:4]).T  # vertices in camera frame
    uv = cam @ K.T
    uv = uv[:, :2] / uv[:, 2:]
    front = (np.einsum('ij,ij->i', mesh.face_normals @ T[:3, :3].T, cam[mesh.faces].mean(1)) < 0)
    img = np.full((S, S, 3), 255, np.uint8)
    cv2.polylines(img, list(np.round(uv[mesh.faces[front]]).astype(np.int32)), True, (60, 60, 60), 1, cv2.LINE_AA)
    return img


def views(mesh, textured, shape, n, elev):
    size = np.linalg.norm(mesh.extents)
    dist, f = 3 * size, 0.8 * S * 3  # camera 3 sizes away; object fills ~80% of the tile
    K = np.array([[f, 0, S / 2], [0, f, S / 2], [0, 0, 1]])
    rows = []
    for az in np.arange(n) * 360 / n:
        e, a = np.radians(elev), np.radians(az)
        T = look_at(dist * np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)]))
        tex = white(render(textured, K, S, S, T))
        shp = white(render(shape, K, S, S, T, ambient=0.3, diffuse=0.8))
        pts = np.full((S, S, 3), 255, np.uint8)  # every vertex projected, front and back
        uv = (K @ (T[:3, :3] @ mesh.vertices.T + T[:3, 3:4])).T
        for u, v in (uv[:, :2] / uv[:, 2:]).astype(int):
            cv2.circle(pts, (u, v), 1, (40, 40, 40), -1)
        axes = fp.draw_xyz_axis(tex[..., ::-1], ob_in_cam=T, scale=size / 3, K=K, thickness=2,
                                transparency=0, is_input_rgb=True)[..., ::-1].copy()
        rows.append(np.hstack([label(axes, 'texture  az %d el %d' % (az, elev)), label(shp, 'shape'),
                               label(pts, 'vertices (%d)' % len(mesh.vertices)),
                               label(triangles(mesh, K, T), 'triangles (front)'),
                               label(triangles(mesh, K, T, zoom=5), 'triangles x5 zoom')]))
    return np.vstack(rows)


def on_scene(res, textured, shape):
    scene = os.path.join(REPO, 'data', 'multi_object_scene')
    bgr, K = cv2.imread(os.path.join(scene, 'scene_rgb.png')), np.loadtxt(os.path.join(scene, 'camera_K.txt'))
    T = np.array(res['ob_in_cam'])
    tex, shp = render(textured, K, *bgr.shape[:2], T), render(shape, K, *bgr.shape[:2], T, 0.3, 0.8)
    on = tex.any(-1)
    blend = bgr.copy()
    blend[on] = bgr[on] // 2 + tex[on] // 2
    return np.hstack([label(bgr.copy(), 'scene'), label(blend, 'texture over scene'), label(white(shp), 'shape')])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mesh', default=os.path.join(REPO, 'assets', 'ycb', '006_mustard_bottle', 'textured.obj'))
    p.add_argument('--views', type=int, default=4, help='views around the object')
    p.add_argument('--elev', type=float, default=20, help='camera height, degrees above the table')
    p.add_argument('--pose', metavar='TAG', help='overlay output/pose/<TAG>.json on the scene')
    a = p.parse_args()
    res = json.load(open(os.path.join(REPO, 'output', 'pose', a.pose + '.json'))) if a.pose else None
    mesh = trimesh.load(res['mesh'] if res else a.mesh)

    textured = fp.make_mesh_tensors(mesh)
    shape = {k: v for k, v in textured.items() if k not in ('tex', 'uv', 'uv_idx')}
    shape['vertex_color'] = torch.full_like(shape['pos'], 0.8)  # plain grey: geometry only

    print('%d vertices, %d faces, texture %s, %.0f x %.0f x %.0f mm'
          % (len(mesh.vertices), len(mesh.faces), mesh.visual.material.image.size, *mesh.extents * 1000))
    edges = mesh.edges_unique_length * 1000
    print('triangle edges: median %.1f mm (min %.2f, max %.1f) | first triangle = vertices %s:\n%s'
          % (np.median(edges), edges.min(), edges.max(), mesh.faces[0].tolist(),
             np.round(mesh.vertices[mesh.faces[0]] * 1000, 1)))
    img = on_scene(res, textured, shape) if res else views(mesh, textured, shape, a.views, a.elev)
    path = os.path.join(REPO, 'output', 'mesh', ('pose_' + a.pose if a.pose else 'views') + '.png')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img)
    print('->', path)


if __name__ == '__main__':
    main()
