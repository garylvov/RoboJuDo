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
import importlib.util
import os
import sys
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
OUT_DIR = f"{WT}/third_party/RoboJuDo/assets/robots/h1_2"

# SINGLE SOURCE OF TRUTH: every mount pos/quat, the standoff, ring pose and variant table
# come from imprint's h1_2_asset_config (stdlib-only; loaded by FILE PATH so this script
# stays runnable under a bare python with no imprint install). NEVER re-declare any of
# those constants here -- 2026-08-02's left-hand flip and 1.5mm seat-gap bugs were exactly
# this script and the USD factory reading different hardcoded copies.
_CFG_PATH = os.path.join(WT, "src/imprint/integrations/unitree_lab/h1_2_asset_config.py")
_spec = importlib.util.spec_from_file_location("h1_2_asset_config", _CFG_PATH)
CFG = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = CFG  # dataclass field resolution needs the module registered
_spec.loader.exec_module(CFG)

RING_STL = os.path.join(WT, CFG.RING_STL_RELPATH)  # CAD source, mm units, mount axis +Y

# variant -> mount seat pos/quat strings, derived through the config's own helpers
# (formatting included -- part of the MJCF bit-identity contract).
MOUNT_POS = {v: CFG.seat_pos_str(v) for v in CFG.MOUNT_VARIANTS}
MOUNT_CONFIGS = {
    v: {"lh_": CFG.seat_quat_str(v, "left"), "rh_": CFG.seat_quat_str(v, "right")}
    for v in CFG.MOUNT_VARIANTS
}
VARIANT_OUT = {v: spec.mjcf_name for v, spec in CFG.MOUNT_VARIANTS.items()}

# MOUNT ADAPTER RING: CAD mount rendered as a VISUAL-ONLY geom on each wrist_yaw_link.
# The ring is bolted to the ARM, so it is authored on the wrist body: pos = the variant
# mount seat, quat = Rz(-90) mapping STL +Y -> wrist +X (config constants).
RING_QUAT = CFG.qstr(CFG.RING_QUAT_WXYZ)
RING_RGBA = CFG.RING_RGBA


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
         "scale": " ".join([f"{CFG.RING_MESH_SCALE:.10g}"] * 3)}
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
