# Copyright 2025 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Generate (or regenerate) the bimanual-lift tray asset as a USD file.

The repository already ships a ready-to-use ``usds/tray/tray.usda`` authored by
hand, so you normally do **not** need to run this.  Use this script only if you
want to tweak the tray geometry (deck size, handle position/thickness, mass) and
regenerate a guaranteed-valid USD with the official ``pxr`` API.

Run it with the Python interpreter that ships with your Isaac Sim / Isaac Lab
install (it must have the ``pxr`` USD module available)::

    # inside the Isaac Lab environment / container
    python scripts/tools/build_tray_usd.py

Edit the constants below, then re-run; the env config points at the same file.

IMPORTANT: keep these numbers in sync with the geometry constants in
``bimanual/lift/lift_env_cfg.py`` (HALF_GRASP_Y, GRASP_Z_OFFSET, DECK_*).
"""

from __future__ import annotations

import os

# ─────────────────────────────────────────────────────────────────────
# Geometry (tray-local frame; origin at deck centre, long axis +Y).
# All values are FULL sizes in metres.
# ─────────────────────────────────────────────────────────────────────
MASS = 0.6                       # kg

DECK_SIZE = (0.30, 0.50, 0.025)  # X (depth) × Y (length) × Z (thickness)
DECK_COLOR = (0.82, 0.66, 0.36)  # warm wood

HANDLE_SIZE = (0.12, 0.024, 0.045)   # X (length) × Y (thin → fits gripper) × Z (height)
HANDLE_GRASP_Y = 0.22                # |Y| of each handle centre from the deck centre
HANDLE_COLOR = (0.20, 0.20, 0.23)    # dark grey

# Static/dynamic friction baked into the asset (robust grip). Restitution 0.
FRICTION_STATIC = 1.2
FRICTION_DYNAMIC = 1.0

OUT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "source", "openarm", "openarm", "tasks", "manager_based",
    "openarm_manipulation", "usds", "tray", "tray.usda",
)


def _add_box(stage, parent_path, name, full_size, translate, color):
    from pxr import Gf, UsdGeom, UsdPhysics, Vt

    path = f"{parent_path}/{name}"
    cube = UsdGeom.Cube.Define(stage, path)
    cube.GetSizeAttr().Set(2.0)  # spans [-1, 1] → scale == half-extents
    cube.CreateExtentAttr([Gf.Vec3f(-1, -1, -1), Gf.Vec3f(1, 1, 1)])
    half = [s * 0.5 for s in full_size]
    xform = UsdGeom.Xformable(cube)
    xform.AddTranslateOp().Set(Gf.Vec3d(*translate))
    xform.AddScaleOp().Set(Gf.Vec3f(*half))
    cube.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(*color)]))
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    return cube.GetPrim()


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

    # child collision boxes
    handle_z = DECK_SIZE[2] * 0.5 + HANDLE_SIZE[2] * 0.5
    _add_box(stage, "/Tray", "deck", DECK_SIZE, (0.0, 0.0, 0.0), DECK_COLOR)
    _add_box(stage, "/Tray", "handle_left", HANDLE_SIZE, (0.0, HANDLE_GRASP_Y, handle_z), HANDLE_COLOR)
    _add_box(stage, "/Tray", "handle_right", HANDLE_SIZE, (0.0, -HANDLE_GRASP_Y, handle_z), HANDLE_COLOR)

    # physics material (friction) bound to the whole body
    mat_path = "/Tray/PhysicsMaterial"
    material = UsdShade.Material.Define(stage, mat_path)
    phys_mat = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    phys_mat.CreateStaticFrictionAttr().Set(FRICTION_STATIC)
    phys_mat.CreateDynamicFrictionAttr().Set(FRICTION_DYNAMIC)
    phys_mat.CreateRestitutionAttr().Set(0.0)
    for name in ("deck", "handle_left", "handle_right"):
        prim = stage.GetPrimAtPath(f"/Tray/{name}")
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            material, UsdShade.Tokens.weakerThanDescendants, "physics"
        )

    stage.GetRootLayer().Save()
    print(f"[build_tray_usd] wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
