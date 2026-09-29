"""Textured OBJ -> USD, written directly with pxr so the USD frame is exactly the mesh frame.

The same OBJ (assets/ycb/<name>/textured.obj) is given to FoundationPose, so the simulator's
ground-truth pose and the estimated pose refer to the same object frame (no hidden axis/unit change).
Runs inside Isaac Sim's Python (.venv-sim) or anywhere with `pip install usd-core`.

  .venv-sim/bin/python sim/ycb_usd.py 006_mustard_bottle [--rebuild]
"""
import os

from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
YCB = os.path.join(REPO, 'assets', 'ycb')


def prim_name(name):
    """USD prim names can't start with a digit (every YCB name does, e.g. "006_mustard_bottle")."""
    return name if (name[:1].isalpha() or name[:1] == '_') else '_' + name


def read_obj(path):
    """-> vertices, uvs, face vertex counts, face vertex indices, face uv indices, texture path."""
    v, vt, counts, fv, fvt, mtl = [], [], [], [], [], None
    with open(path) as f:
        for line in f:
            p = line.split()
            if not p:
                continue
            if p[0] == 'v':
                v.append(tuple(map(float, p[1:4])))
            elif p[0] == 'vt':
                vt.append(tuple(map(float, p[1:3])))
            elif p[0] == 'f':
                counts.append(len(p) - 1)
                for c in p[1:]:
                    idx = c.split('/')
                    fv.append(int(idx[0]) - 1)
                    fvt.append(int(idx[1]) - 1 if len(idx) > 1 and idx[1] else -1)
            elif p[0] == 'mtllib':
                mtl = os.path.join(os.path.dirname(path), p[1])
    texture = None
    if mtl and os.path.isfile(mtl):
        for line in open(mtl):
            p = line.split()
            if p and p[0] == 'map_Kd':
                texture = os.path.join(os.path.dirname(mtl), p[1])
    return v, vt, counts, fv, fvt, texture


def build_object_usd(obj_path, usd_path, name):
    v, vt, counts, fv, fvt, texture = read_obj(obj_path)
    stage = Usd.Stage.CreateNew(usd_path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = '/' + prim_name(name)
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, root).GetPrim())

    geom = UsdGeom.Mesh.Define(stage, root + '/geom')
    geom.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*p) for p in v]))
    geom.CreateFaceVertexCountsAttr(Vt.IntArray(counts))
    geom.CreateFaceVertexIndicesAttr(Vt.IntArray(fv))
    geom.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    geom.CreateExtentAttr(UsdGeom.PointBased.ComputeExtent(geom.GetPointsAttr().Get()))

    has_uv = bool(vt) and all(i >= 0 for i in fvt)
    if has_uv:
        st = UsdGeom.PrimvarsAPI(geom).CreatePrimvar('st', Sdf.ValueTypeNames.TexCoord2fArray,
                                                     UsdGeom.Tokens.faceVarying)
        st.Set(Vt.Vec2fArray([Gf.Vec2f(*vt[i]) for i in fvt]))

    looks = root + '/Looks/material'
    mat = UsdShade.Material.Define(stage, looks)
    shader = UsdShade.Shader.Define(stage, looks + '/surface')
    shader.CreateIdAttr('UsdPreviewSurface')
    shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(0.6)
    shader.CreateInput('metallic', Sdf.ValueTypeNames.Float).Set(0.0)
    if has_uv and texture:
        reader = UsdShade.Shader.Define(stage, looks + '/st_reader')
        reader.CreateIdAttr('UsdPrimvarReader_float2')
        reader.CreateInput('varname', Sdf.ValueTypeNames.Token).Set('st')
        tex = UsdShade.Shader.Define(stage, looks + '/texture')
        tex.CreateIdAttr('UsdUVTexture')
        rel = os.path.relpath(texture, os.path.dirname(os.path.abspath(usd_path)))
        tex.CreateInput('file', Sdf.ValueTypeNames.Asset).Set('./' + rel)
        tex.CreateInput('sourceColorSpace', Sdf.ValueTypeNames.Token).Set('sRGB')
        tex.CreateInput('st', Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), 'result')
        tex.CreateOutput('rgb', Sdf.ValueTypeNames.Float3)
        shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(), 'rgb')
    else:
        shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.6, 0.6, 0.6))
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
    UsdShade.MaterialBindingAPI.Apply(geom.GetPrim()).Bind(mat)
    stage.GetRootLayer().Save()
    return usd_path


def ensure_object_usd(name, rebuild=False):
    """assets/ycb/<name>/<name>.usd, (re)built from textured.obj when missing or older than the OBJ."""
    obj = os.path.join(YCB, name, 'textured.obj')
    usd = os.path.join(YCB, name, name + '.usd')
    if not os.path.isfile(obj):
        raise FileNotFoundError('%s missing - run scripts/prepare_ycb.py first' % obj)
    if rebuild or not os.path.isfile(usd) or os.path.getmtime(usd) < os.path.getmtime(obj):
        if os.path.isfile(usd):
            os.remove(usd)
        build_object_usd(obj, usd, name)
    return usd


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='Build assets/ycb/<name>/<name>.usd from textured.obj')
    ap.add_argument('names', nargs='+')
    ap.add_argument('--rebuild', action='store_true')
    a = ap.parse_args()
    for n in a.names:
        print(ensure_object_usd(n, a.rebuild))
