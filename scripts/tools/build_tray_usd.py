# Copyright 2025 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Generate (or regenerate) the bimanual-lift tray asset as a USD file.

The repository already ships a ready-to-use ``usds/tray/tray.usda`` authored by
hand, so you normally do **not** need to run this.  Use this script only if you
want to tweak the tray geometry (board size, mass) and regenerate a
guaranteed-valid USD with the official ``pxr`` API.

Run it with the Python interpreter that ships with your Isaac Sim / Isaac Lab
install (it must have the ``pxr`` USD module available)::

    # inside the Isaac Lab environment / container
    python scripts/tools/build_tray_usd.py

Edit the constants below, then re-run; the env config points at the same file.

IMPORTANT: keep these numbers in sync with the geometry constants in
``bimanual/lift/lift_env_cfg.py`` (HALF_GRASP_Y, GRASP_Z_OFFSET, DECK_*).

The tray is a single FLAT board: its two short ends overhang the central stand,
so the grippers side-grasp the ends (one finger under the overhang) for a
form-closure lift.  No handles.
"""

from __future__ import annotations

import os

# ─────────────────────────────────────────────────────────────────────
# Geometry (tray-local frame; origin at board centre, long axis +Y).
# All values are FULL sizes in metres.
# ─────────────────────────────────────────────────────────────────────
MASS = 0.5                       # kg

DECK_SIZE = (0.36, 0.50, 0.030)  # X (depth) × Y (length) × Z (thickness, graspable)
DECK_COLOR = (0.82, 0.66, 0.36)  # warm wood

# Static/dynamic friction baked into the asset (robust grip). Restitution 0.
FRICTION_STATIC = 1.2
FRICTION_DYNAMIC = 1.0

OUT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "source", "openarm", "openarm", "tasks", "manager_based",
    "openarm_manipulation", "usds", "tray", "tray.usda",
)


# Box face topology (6 quads) with outward-facing winding; shared by all boxes.
_FACE_COUNTS = [4, 4, 4, 4, 4, 4]
_FACE_INDICES = [4, 5, 6, 7, 0, 3, 2, 1, 0, 1, 5, 4, 2, 3, 7, 6, 0, 4, 7, 3, 1, 2, 6, 5]


def _add_box(stage, parent_path, name, full_size, center, color):
    """Author one box as an explicit MESH with BOTH size and offset baked into
    the vertices (NO xformOp at all).

    The Isaac Lab USD importer drops child ``xformOp`` (scale AND translate) when
    flattening a rigid body, which silently collapses offset child colliders onto
    the body origin.  Baking everything into the points array removes that whole
    failure mode: PhysX gets exact convex-hull box colliders at the right place,
    and the visual matches 1:1.
    """
    from pxr import Gf, UsdGeom, UsdPhysics, Vt

    cx, cy, cz = center
    hx, hy, hz = (s * 0.5 for s in full_size)
    pts = [
        (cx - hx, cy - hy, cz - hz), (cx + hx, cy - hy, cz - hz),
        (cx + hx, cy + hy, cz - hz), (cx - hx, cy + hy, cz - hz),
        (cx - hx, cy - hy, cz + hz), (cx + hx, cy - hy, cz + hz),
        (cx + hx, cy + hy, cz + hz), (cx - hx, cy + hy, cz + hz),
    ]
    mesh = UsdGeom.Mesh.Define(stage, f"{parent_path}/{name}")
    mesh.CreateFaceVertexCountsAttr(_FACE_COUNTS)
    mesh.CreateFaceVertexIndicesAttr(_FACE_INDICES)
    mesh.CreatePointsAttr([Gf.Vec3f(*p) for p in pts])
    mesh.CreateExtentAttr([
        Gf.Vec3f(cx - hx, cy - hy, cz - hz), Gf.Vec3f(cx + hx, cy + hy, cz + hz)
    ])
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*color)]))
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim())
    mesh_collision.CreateApproximationAttr().Set("convexHull")
    return mesh.GetPrim()


def main() -> None:
    from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade

    stage = Usd.Stage.CreateNew(OUT_PATH) if not os.path.exists(OUT_PATH) else Usd.Stage.Open(OUT_PATH)
    stage.RemovePrim("/Tray")
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    tray = UsdGeom.Xform.Define(stage, "/Tray")
    stage.SetDefaultPrim(tray.GetPrim())

    # rigid body + mass on the root
    UsdPhysics.RigidBodyAPI.Apply(tray.GetPrim())
    mass_api = UsdPhysics.MassAPI.Apply(tray.GetPrim())
    mass_api.GetMassAttr().Set(float(MASS))

    # single flat board collider
    _add_box(stage, "/Tray", "deck", DECK_SIZE, (0.0, 0.0, 0.0), DECK_COLOR)

    # physics material (friction) bound to the whole body
    mat_path = "/Tray/PhysicsMaterial"
    material = UsdShade.Material.Define(stage, mat_path)
    phys_mat = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    phys_mat.CreateStaticFrictionAttr().Set(FRICTION_STATIC)
    phys_mat.CreateDynamicFrictionAttr().Set(FRICTION_DYNAMIC)
    phys_mat.CreateRestitutionAttr().Set(0.0)
    for name in ("deck",):
        prim = stage.GetPrimAtPath(f"/Tray/{name}")
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            material, UsdShade.Tokens.weakerThanDescendants, "physics"
        )

    stage.GetRootLayer().Save()
    print(f"[build_tray_usd] wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
