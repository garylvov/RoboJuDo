"""Compose an H1_2 + Psyonic Ability-hands wrapper MJCF for the Newton backend.

Inlines both generated Ability-hand MJCFs (``ability_hand_{left,right}_large.xml``)
as rigid children of the H1_2 ``{left,right}_wrist_yaw_link`` bodies so the Newton
sim2sim scene carries the finger DOFs the lift-teacher policy needs (its 24 finger
obs dims + 12-dim ``ability_fingers`` action head get real actuators, instead of
being zero-substituted as in the fingerless ``h1_2_box_feet.xml``).

Method (matches the manual wrist-mount idiom in the vendor
``ability-hand-api/.../mujoco_xml/franka_fr3/scene.xml``, done with ElementTree so
the output parses under Newton's own strict-ElementTree ``add_mjcf``):

* take each hand's ``<body name="base">`` subtree, DROP its ``<joint type="free">``
  (rigid attach) and the worldbody floor/light,
* prefix every ``name`` (body/joint/geom/site) and every mesh name/ref with
  ``lh_``/``rh_`` to avoid collisions with H1_2 and the other hand,
* merge the (prefixed) ``<asset><mesh>`` and ``<actuator><motor>`` entries into
  H1_2's blocks, rewriting all mesh ``file=`` paths to ABSOLUTE (the two source
  MJCFs use different ``meshdir`` roots; absolute paths sidestep the single
  ``<compiler meshdir>``),
* mount each hand ``base`` at the factory mount frame (per --variant, see
  MOUNT_CONFIGS), and
* inject the CAD mount-adapter ring (``assets/cad/H1_2_wrist_no_camera.STL``) as a
  VISUAL-ONLY geom on each wrist -- the USD carries it as ``wrist_mesh``; the MJCF
  must carry it too for engine parity.

Regenerate (both variants) with::

    python scripts/compose_h1_2_ability_hands.py
"""

import argparse
import math
import os
import xml.etree.ElementTree as ET

# Repo root: <repo>/third_party/RoboJuDo/scripts/<this file>. Overridable for
# out-of-tree runs (legacy behavior was a hardcoded wt_visual_s2r checkout).
WT = os.environ.get(
    "IMPRINT_REPO_ROOT",
    os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../..")),
)
H1_2_SRC = f"{WT}/third_party/ProtoMotions/protomotions/data/assets/mjcf/h1_2_box_feet.xml"
H1_2_MESHDIR = f"{WT}/third_party/ProtoMotions/protomotions/data/assets/mesh/H1_2/"
HAND_L = f"{WT}/assets/psyonic_ability_hand/mjcf/ability_hand_left_large.xml"
HAND_R = f"{WT}/assets/psyonic_ability_hand/mjcf/ability_hand_right_large.xml"
HAND_MESH_BASE = f"{WT}/assets/psyonic_ability_hand/mjcf/"  # hand file= paths are relative to this
RING_STL = f"{WT}/assets/cad/H1_2_wrist_no_camera.STL"  # CAD source, mm units, mount axis +Y
OUT_DIR = f"{WT}/third_party/RoboJuDo/assets/robots/h1_2"


def _qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


def _qrx(deg):
    h = math.radians(deg) / 2.0
    return (math.cos(h), math.sin(h), 0.0, 0.0)


def _qstr(q):
    if q[0] < 0:
        q = tuple(-v for v in q)
    q = tuple(0.0 if abs(v) < 1e-12 else v for v in q)  # snap qmul float noise
    return " ".join(f"{v:.10g}" for v in q)


# FACTORY MOUNTS (2026-07-29/31): pos/quat are the {left,right}_ability_mount_joint
# localPos0/localRot0 read straight from the r13 training USD
# (wbc_data/assets/h1_2_ability/h1_2_box_feet_ability.usd) == the factory
# ABILITY_MOUNT_ROTATION_CONFIGS["thumb_up"] seat (palms-DOWN at joint-zero).
#
# STANDOFF (2026-08-02 gap forensics): the r13 seat X (0.0645, graft-era, no vendor
# derivation) leaves a REAL +1.50mm gap between the wrist_yaw_link visual tip (63.00mm)
# and the ring/hand near face (64.50mm), both engines, both sides. The factory's own
# v1_1_4_closed standoff (imprint h1_2_usd.py ABILITY_MOUNT_STANDOFF_X_M) closes it:
# palms_in (r14 default) seats at 0.063 (flush); palms_down KEEPS the legacy 0.0645 for
# engine parity with existing r13-era checkpoints (its whole reason to exist).
MOUNT_POS = {"palms_down": "0.0645 0 0", "palms_in": "0.063 0 0"}
_PALMS_DOWN = {"lh_": (0.5, 0.5, 0.5, 0.5), "rh_": (0.5, -0.5, 0.5, -0.5)}
# palms_in (USER RULING 2026-07-30, default convention for r14+): both palm inner normals
# point INTO the robot at joint-zero (LEFT -> world -Y, RIGHT -> +Y). Derived IN CODE from
# the palms_down seats by the factory's own config delta -- ABILITY_MOUNT_ROTATION_CONFIGS
# (imprint h1_2_usd.py): palms_in - thumb_up = {left: -90, right: +90} deg about the mount
# axis (wrist-local X), pre-multiplied. Both seats collapse to (sqrt2/2, 0, sqrt2/2, 0).
_PALMS_IN_DELTA_DEG = {"lh_": -90.0, "rh_": +90.0}
MOUNT_CONFIGS = {
    "palms_down": {p: _qstr(q) for p, q in _PALMS_DOWN.items()},
    "palms_in": {
        p: _qstr(_qmul(_qrx(_PALMS_IN_DELTA_DEG[p]), _PALMS_DOWN[p])) for p in _PALMS_DOWN
    },
}
VARIANT_OUT = {
    "palms_down": "h1_2_box_feet_ability_hands.xml",
    "palms_in": "h1_2_box_feet_ability_hands_palmsin.xml",
}

# MOUNT ADAPTER RING (2026-08-02): the CAD mount (RING_STL, millimeters, mount axis +Y,
# wrist-side mating plane at y=0, fourfold-symmetric -- results/real_mount_usd/
# step1_stl_analysis.json) rendered as a VISUAL-ONLY geom on each wrist_yaw_link. The ring
# is bolted to the ARM, so it is authored on the wrist body (mount-config independent):
# pos = the mount seat localPos0 (MOUNT_POS, matching the USD wrist_mesh placement probed
# at (0.0645, 0, 0) in the wrist frame), quat = Rz(-90) mapping STL +Y -> wrist +X.
RING_QUAT = _qstr((math.sqrt(0.5), 0.0, 0.0, -math.sqrt(0.5)))  # Rz(-90): +Y_stl -> +X_wrist
RING_RGBA = "1 0.42 0 1"  # orange, matches the hand_mount render convention


def _rz180_parent_side(elem: ET.Element):
    """Parent-side Rz180 on a pos/quat-carrying element: pos (x,y,z)->(-x,-y,z),
    quat (a,b,c,d) -> (-d,-c,b,a) (exact literal permutation, no precision loss)."""

    def neg(tok: str) -> str:
        return tok[1:] if tok.startswith("-") else ("0" if float(tok) == 0.0 else "-" + tok)

    if "pos" in elem.attrib:
        x, y, z = elem.attrib["pos"].split()
        elem.attrib["pos"] = f"{neg(x)} {neg(y)} {z}"
    a, b, c, d = elem.attrib.get("quat", "1 0 0 0").split()
    q = [neg(d), neg(c), b, a]
    if q[0].startswith("-"):  # sign-normalize w >= 0
        q = [neg(t) for t in q]
    elem.attrib["quat"] = " ".join(q)


def _abs_h1_mesh(file_attr: str) -> str:
    return os.path.normpath(os.path.join(H1_2_MESHDIR, file_attr))


def _abs_hand_mesh(file_attr: str) -> str:
    return os.path.normpath(os.path.join(HAND_MESH_BASE, file_attr))


def _prefix_names(elem: ET.Element, prefix: str):
    """Prefix name/joint/mesh refs on this element and all descendants."""
    for e in elem.iter():
        for attr in ("name", "joint", "mesh", "site", "body", "target"):
            if attr in e.attrib:
                e.attrib[attr] = prefix + e.attrib[attr]


def compose(variant: str, out_dir: str) -> str:
    mount_pos = MOUNT_POS[variant]
    mounts = {
        "lh_": {"parent": "left_wrist_yaw_link", "pos": mount_pos,
                "quat": MOUNT_CONFIGS[variant]["lh_"]},
        "rh_": {"parent": "right_wrist_yaw_link", "pos": mount_pos,
                "quat": MOUNT_CONFIGS[variant]["rh_"]},
    }
    h1_tree = ET.parse(H1_2_SRC)
    h1_root = h1_tree.getroot()

    # absolute-ize H1_2 mesh paths + drop meshdir (mixed roots after merge)
    compiler = h1_root.find("compiler")
    if compiler is not None and "meshdir" in compiler.attrib:
        del compiler.attrib["meshdir"]
    h1_asset = h1_root.find("asset")
    for mesh in h1_asset.findall("mesh"):
        if "file" in mesh.attrib:
            mesh.attrib["file"] = _abs_h1_mesh(mesh.attrib["file"])

    h1_actuator = h1_root.find("actuator")

    # index H1_2 bodies by name
    bodies = {b.attrib.get("name"): b for b in h1_root.iter("body")}

    # mount adapter ring: one mesh asset (STL is mm -> scale 0.001), one visual-only
    # geom per wrist at the mount seat (see RING comment above).
    ring_mesh = ET.SubElement(h1_asset, "mesh")
    ring_mesh.attrib.update(
        {"name": "wrist_mount_ring", "file": os.path.normpath(RING_STL),
         "scale": "0.001 0.001 0.001"}
    )
    for side in ("left", "right"):
        ring = ET.SubElement(bodies[f"{side}_wrist_yaw_link"], "geom")
        ring.attrib.update(
            {"name": f"{side[0]}h_wrist_mount_ring", "type": "mesh",
             "mesh": "wrist_mount_ring", "pos": mount_pos, "quat": RING_QUAT,
             "group": "1", "contype": "0", "conaffinity": "0", "rgba": RING_RGBA}
        )

    for prefix, mount in mounts.items():
        hand_src = HAND_L if prefix == "lh_" else HAND_R
        hand_root = ET.parse(hand_src).getroot()

        # merge (prefixed, absolute) hand meshes into H1_2 asset
        hand_asset = hand_root.find("asset")
        for mesh in hand_asset.findall("mesh"):
            m = ET.SubElement(h1_asset, "mesh")
            m.attrib["name"] = prefix + mesh.attrib["name"]
            m.attrib["file"] = _abs_hand_mesh(mesh.attrib["file"])
            if "content_type" in mesh.attrib:
                m.attrib["content_type"] = mesh.attrib["content_type"]

        # grab the base body subtree, drop the free joint
        base = hand_root.find(".//body[@name='base']")
        free = base.find("joint[@type='free']")
        if free is not None:
            base.remove(free)
        _prefix_names(base, prefix)
        base.attrib["pos"] = mount["pos"]
        base.attrib["quat"] = mount["quat"]

        if prefix == "lh_":
            # LEFT CHAIN Rz180 CORRECTION (2026-07-31, ready_pose evidence
            # media/orireach/ready_pose/): at the factory mount frame the vendor left
            # chain sits 180 deg about mount-local Z vs the training USD left palm
            # subtree (measured: R_mjcf = Rz180 * R_usd * Rx180, p_mjcf = Rz180 * p_usd;
            # the child-side Rx180 is the factory's own left-joint frame convention,
            # USD lr1=(0,-1,0,0), shared by the vendor model). Apply parent-side Rz180
            # to every DIRECT child (bodies, geoms, inertial) of the left base; chain
            # internals are self-consistent and stay untouched. The correction lives in
            # the base's LOCAL frame, so it is mount-config independent (holds for both
            # palms_down and palms_in). Verified: corrected palm geom == USD
            # left_palm_coll in-palm transform to 7 decimals, and all assembled L1/L2
            # in-palm positions match the USD to 6 decimals.
            for child in list(base):
                if child.tag in ("body", "geom", "inertial", "site"):
                    _rz180_parent_side(child)

        # attach under the wrist body
        bodies[mount["parent"]].append(base)

        # merge (prefixed) hand actuators
        hand_act = hand_root.find("actuator")
        for motor in hand_act.findall("motor"):
            m = ET.SubElement(h1_actuator, "motor")
            m.attrib["name"] = prefix + motor.attrib["name"]
            m.attrib["joint"] = prefix + motor.attrib["joint"]

    out = os.path.join(out_dir, VARIANT_OUT[variant])
    os.makedirs(os.path.dirname(out), exist_ok=True)
    h1_tree.write(out, encoding="unicode")
    # count finger joints
    finger_joints = [
        j.attrib["name"]
        for j in h1_root.iter("joint")
        if j.attrib.get("name", "").startswith(("lh_", "rh_"))
    ]
    print(f"wrote {out}  (variant={variant}, mounts: "
          f"lh {mounts['lh_']['quat']} | rh {mounts['rh_']['quat']}, ring injected)")
    print(f"finger joints ({len(finger_joints)}): {finger_joints}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", choices=[*VARIANT_OUT, "all"], default="all")
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args()
    for v in (VARIANT_OUT if args.variant == "all" else [args.variant]):
        compose(v, args.out_dir)
