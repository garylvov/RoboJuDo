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
  ``<compiler meshdir>``), and
* mount each hand ``base`` at a wrist offset (approximate pose -- exact palm
  alignment / contact tuning is deploy-remainder, not load-bearing for the
  harness smoke).

Regenerate (paths below are cluster-absolute) with::

    python scripts/compose_h1_2_ability_hands.py
"""

import os
import xml.etree.ElementTree as ET

WT = "/oscar/data/stellex/glvov/wt_visual_s2r"
H1_2_SRC = f"{WT}/third_party/ProtoMotions/protomotions/data/assets/mjcf/h1_2_box_feet.xml"
H1_2_MESHDIR = f"{WT}/third_party/ProtoMotions/protomotions/data/assets/mesh/H1_2/"
HAND_L = f"{WT}/assets/psyonic_ability_hand/mjcf/ability_hand_left_large.xml"
HAND_R = f"{WT}/assets/psyonic_ability_hand/mjcf/ability_hand_right_large.xml"
HAND_MESH_BASE = f"{WT}/assets/psyonic_ability_hand/mjcf/"  # hand file= paths are relative to this
OUT = f"{WT}/third_party/RoboJuDo/assets/robots/h1_2/h1_2_box_feet_ability_hands.xml"

# Approximate wrist mount: hand base sits ~8 cm past the wrist_yaw origin along +x
# (the H1_2 forearm extends +x). Left/right use mirrored yaw so the palms face
# inward toward a grasped object. Tune pos/quat for exact palm alignment (remainder).
MOUNTS = {
    "lh_": {"parent": "left_wrist_yaw_link", "pos": "0.08 0 0", "quat": "0.5 -0.5 0.5 -0.5"},
    "rh_": {"parent": "right_wrist_yaw_link", "pos": "0.08 0 0", "quat": "0.5 0.5 0.5 0.5"},
}


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


def compose():
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

    for prefix, mount in MOUNTS.items():
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

        # attach under the wrist body
        bodies[mount["parent"]].append(base)

        # merge (prefixed) hand actuators
        hand_act = hand_root.find("actuator")
        for motor in hand_act.findall("motor"):
            m = ET.SubElement(h1_actuator, "motor")
            m.attrib["name"] = prefix + motor.attrib["name"]
            m.attrib["joint"] = prefix + motor.attrib["joint"]

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    h1_tree.write(OUT, encoding="unicode")
    # count finger joints
    finger_joints = [
        j.attrib["name"]
        for j in h1_root.iter("joint")
        if j.attrib.get("name", "").startswith(("lh_", "rh_"))
    ]
    print(f"wrote {OUT}")
    print(f"finger joints ({len(finger_joints)}): {finger_joints}")


if __name__ == "__main__":
    compose()
