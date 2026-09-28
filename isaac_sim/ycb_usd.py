"""Textured OBJ -> USD, written directly with pxr so the USD frame is exactly the mesh frame.

The same OBJ is given to FoundationPose, so the simulator's object pose (ground truth) and the
estimated pose refer to the same frame without any hidden up-axis/unit conversion.
Runs inside Isaac Sim or with `pip install usd-core`.
"""
import os
import sys

from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'ros2', 'ppp_common'))
from ppp_common.mesh import read_obj  # noqa: E402


def build_object_usd(obj_path, usd_path, name):
    # USD prim names can't start with a digit (all YCB names do, e.g. "004_sugar_box");
    # this is purely the in-file root prim name, unrelated to the object name used elsewhere.
    name = name if (name[:1].isalpha() or name[:1] == '_') else '_' + name
    mesh = read_obj(obj_path)
    stage = Usd.Stage.CreateNew(usd_path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, '/' + name)
    stage.SetDefaultPrim(root.GetPrim())

    geom = UsdGeom.Mesh.Define(stage, '/%s/geom' % name)
    geom.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, v)) for v in mesh.vertices]))
    geom.CreateFaceVertexCountsAttr(Vt.IntArray(mesh.face_counts))
    geom.CreateFaceVertexIndicesAttr(Vt.IntArray(mesh.face_v))
    geom.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    geom.CreateDoubleSidedAttr(False)
    lo, hi = mesh.aabb
    geom.CreateExtentAttr(Vt.Vec3fArray([Gf.Vec3f(*map(float, lo)), Gf.Vec3f(*map(float, hi))]))

    has_uv = len(mesh.uvs) > 0 and all(i >= 0 for i in mesh.face_vt)
    if has_uv:
        st = UsdGeom.PrimvarsAPI(geom).CreatePrimvar('st', Sdf.ValueTypeNames.TexCoord2fArray,
                                                     UsdGeom.Tokens.faceVarying)
        st.Set(Vt.Vec2fArray([Gf.Vec2f(*map(float, mesh.uvs[i])) for i in mesh.face_vt]))

    mat = UsdShade.Material.Define(stage, '/%s/Looks/material' % name)
    shader = UsdShade.Shader.Define(stage, '/%s/Looks/material/surface' % name)
    shader.CreateIdAttr('UsdPreviewSurface')
    shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(0.7)
    shader.CreateInput('metallic', Sdf.ValueTypeNames.Float).Set(0.0)
    if has_uv and mesh.texture:
        reader = UsdShade.Shader.Define(stage, '/%s/Looks/material/st_reader' % name)
        reader.CreateIdAttr('UsdPrimvarReader_float2')
        reader.CreateInput('varname', Sdf.ValueTypeNames.Token).Set('st')
        tex = UsdShade.Shader.Define(stage, '/%s/Looks/material/texture' % name)
        tex.CreateIdAttr('UsdUVTexture')
        rel = os.path.relpath(mesh.texture, os.path.dirname(os.path.abspath(usd_path)))
        tex.CreateInput('file', Sdf.ValueTypeNames.Asset).Set(rel)
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


def ensure_object_usd(cfg, name, rebuild=False):
    usd = cfg.usd_path(name)
    obj = cfg.mesh_path(name)
    if not os.path.isfile(obj):
        raise FileNotFoundError('%s missing - run scripts/prepare_ycb.py first' % obj)
    if rebuild or not os.path.isfile(usd) or os.path.getmtime(usd) < os.path.getmtime(obj):
        if os.path.isfile(usd):
            os.remove(usd)
        build_object_usd(obj, usd, name)
    return usd


if __name__ == '__main__':
    import argparse
    from ppp_common.config import SceneConfig
    ap = argparse.ArgumentParser(description='Build USDs for all objects in scene.yaml')
    ap.add_argument('--assets', default='')
    ap.add_argument('--rebuild', action='store_true')
    a = ap.parse_args()
    cfg = SceneConfig(assets=a.assets)
    for n in cfg.object_names:
        print(ensure_object_usd(cfg, n, a.rebuild))
