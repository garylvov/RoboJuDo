"""PSYONIC Ability Hand finger kinematic coupling (real-hardware mimic law).

Ground truth
------------
Vendored/verified against two independent sources that agree on the mechanism:

1. ``third_party/ability-hand-api/python/finger_4bar/abh_finger_4bar.py`` --
   the vendor's own EXACT 4-bar linkage solve (``get_abh_4bar_driven_angle``):
   given the driven proximal joint angle ``q1`` (URDF/MJCF convention), returns
   the distal joint angle ``q2`` the mechanical linkage forces. This is the
   right one to use for RoboJuDo's Newton MJCF (``h1_2_box_feet_ability_hands.xml``,
   composed by ``scripts/compose_h1_2_ability_hands.py`` directly from the
   vendor ``ability_hand_{left,right}_large.xml``, whose joints are literally
   named ``{finger}_q1``/``{finger}_q2`` -- the SAME q1/q2 convention the 4bar
   solve is defined over).
2. ``imprint_isaaclab_ext/imprint_isaaclab_ext/wbc/h1_2_ability_cfg.py`` --
   the PhysX/USD training asset's linear-fit mimic law
   (``pip = ABILITY_MIMIC_MULTIPLIER * mcp + ABILITY_PIP_OFFSET[side]``), used
   there because that asset's ``mcp``/``pip`` joint zero-references differ
   from the vendor MJCF's ``q1``/``q2`` (its PIP lower limit excludes 0), so
   its offset is NOT reusable verbatim on this MJCF -- but it corroborates the
   mechanism (linear coupling to good approximation) and supplies the
   asymmetric left/right offset intuition.

Only the 4 non-thumb fingers (index/middle/ring/pinky) have a coupled distal
joint -- ``{finger}_q1`` is the single motor-driven DOF, ``{finger}_q2`` is
NOT independently actuated (it is not present at all in the Ability Hand's
real DOF count). The thumb has 2 INDEPENDENTLY driven DOF (``thumb_q1``
rotator/CMC-like ab/adduction, ``thumb_q2`` flexor/MCP-like flexion) -- no
coupling applies there.

Newton's ``SolverMuJoCo`` does not read the vendor MJCF's <mimic>/<equality>
constraints (there are none authored in this composed asset -- confirmed by
grep), so :class:`~robojudo.environment.newton_env.NewtonEnv` must explicitly
position-drive q2 to this coupled value every step (see its ``step()``),
exactly the way ``h1_2_ability_cfg.py`` documents PhysX doing it via its own
mimic joint API.
"""

from __future__ import annotations

import numpy as np

# Coupled (non-thumb) fingers, in the order used throughout this module's
# 6-dof "driven" command convention (see NewtonEnv.HAND_JOINT_ORDER).
COUPLED_FINGERS = ("index", "middle", "ring", "pinky")

# ---------------------------------------------------------------------- #
# Ability-hand joint tables (composed h1_2_box_feet_ability_hands.xml only).
#
# These live HERE, not in newton_env, because they are pure data about the
# hand and have no Newton content: keeping them next to `import warp` made
# every consumer -- including MuJoCo-only and real-hardware ones -- require
# warp to read six lists. newton_env re-exports them for compatibility.
#
# 6 DRIVEN dof/hand (mirrors the real Ability Hand + h1_2_ability_cfg.py's
# `DRIVEN_FINGER_JOINTS_EXPR`): 4 finger MCP-equivalent joints (this MJCF's
# ``{finger}_q1``) + 2 independently-actuated thumb joints (``thumb_q1``
# rotator, ``thumb_q2`` flexor). This module's own convention (NOT verified
# against the original ProtoMotions training action order -- see module
# docstring in h1_2_lift_teacher_onnx_policy.py) fixes per-hand order as:
#   [index_q1, middle_q1, ring_q1, pinky_q1, thumb_q1, thumb_q2]
# ``hand_pose`` passed to :meth:`NewtonEnv.step` is 12-dim = concat(left(6), right(6)).
_HAND_DRIVEN_JOINTS = ["index_q1", "middle_q1", "ring_q1", "pinky_q1", "thumb_q1", "thumb_q2"]
_HAND_SIDES = ("lh_", "rh_")  # left, right (matches compose_h1_2_ability_hands.py prefixes)
# 4 non-thumb fingers: q2 (distal/PIP-equivalent) is NOT independently actuated on real
# hardware -- it is mechanically coupled to q1 by a 4-bar linkage (see below).
_HAND_COUPLED_JOINTS = [(f"{fin}_q2", f"{fin}_q1") for fin in COUPLED_FINGERS]
# Joint limits (identical across both hands -- verified against both
# ability_hand_{left,right}_large.xml source MJCFs), used to clip commands to
# something the real hand could physically reach.
_HAND_JOINT_LIMITS = {
    "index_q1": (0.0, 1.74), "middle_q1": (0.0, 1.74), "ring_q1": (0.0, 1.74), "pinky_q1": (0.0, 1.74),
    "thumb_q1": (-1.74, 0.0), "thumb_q2": (0.0, 1.74),
}
_HAND_FINGER_STIFFNESS = 8.0  # matches imprint_isaaclab_ext.wbc.h1_2_ability_cfg._FINGER_STIFFNESS
_HAND_FINGER_DAMPING = 0.3  # matches imprint_isaaclab_ext.wbc.h1_2_ability_cfg._FINGER_DAMPING

# ---------------------------------------------------------------------- #
# Vendor-exact 4-bar linkage solve (ported verbatim from
# ability-hand-api/python/finger_4bar/abh_finger_4bar.py; numpy-vectorized).
# ---------------------------------------------------------------------- #
_L1 = 38.6104
_L2 = 36.875
_L3 = 9.1241
_P3 = np.array([9.47966, -0.62133], dtype=np.float64)
_Q1_OFFSET = 0.084474  # link-frame attachment offset baked into the vendor solve


def _intersect_circles(o0: np.ndarray, r0: float, o1: np.ndarray, r1: float) -> np.ndarray:
    """Second intersection point of two circles (``o1``/``o1`` may be batched (...,2))."""
    d = np.linalg.norm(o0 - o1, axis=-1)
    a = (r0 * r0 - r1 * r1 + d * d) / (2.0 * d)
    h = np.sqrt(np.clip(r0 * r0 - a * a, 0.0, None))
    p2 = o0 + a[..., None] * (o1 - o0) / d[..., None]
    t1 = h * (o1[..., 1] - o0[..., 1]) / d
    t2 = h * (o1[..., 0] - o0[..., 0]) / d
    # sol1 (matches abh_finger_4bar.get_abh_finger_4bar's `sols[1]`)
    return np.stack([p2[..., 0] - t1, p2[..., 1] + t2], axis=-1)


def abh_finger_4bar_q2(q1) -> np.ndarray:
    """Exact Ability-Hand 4-bar solve: driven proximal ``q1`` (rad) -> distal ``q2`` (rad).

    Vectorized (accepts scalar or ndarray ``q1``); vendor-verified against
    ``finger_4bar/abh_finger_4bar.py::get_abh_4bar_driven_angle``.
    """
    q1 = np.asarray(q1, dtype=np.float64) + _Q1_OFFSET
    cq1, sq1 = np.cos(q1), np.sin(q1)
    p1 = np.stack([_L1 * cq1, _L1 * sq1], axis=-1)
    o0 = np.broadcast_to(_P3, p1.shape)
    p2 = _intersect_circles(o0, _L2, p1, _L3)
    q2pq1 = np.arctan2(p2[..., 1] - _L1 * sq1, p2[..., 0] - _L1 * cq1)
    q2 = q2pq1 - q1
    q2 = np.mod(q2 + np.pi, 2.0 * np.pi) - np.pi
    return q2
