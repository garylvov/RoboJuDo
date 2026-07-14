"""H1_2 real-hardware environment config for RoboJuDo.

Mirrors ``h1_real_env_cfg.py`` in structure. H1_2-specific facts (all
cross-checked against Unitree's SDK + this repo's own deploy code):

- ``msg_type = "hg"`` -- H1_2 uses the hg low-level message type, like G1
  (NOT H1's "go").
- IMU is torso-mounted, like H1 (see the ``h1_2`` branch added to
  ``unitree_env.py``'s ``update()``) -- so ``msg_type`` (shared with G1)
  and IMU mounting (shared with H1) are orthogonal facts.
- ``env_type = "UnitreeEnv"`` (unitree_sdk2py), NOT ``UnitreeCppEnv`` --
  the C++ env only supports G1 (``unitree_cpp_env.py`` raises
  NotImplementedError otherwise).
- ``joint2motor_idx = None`` -- H1_2's hardware motor order is identity
  (legs 0-11, waist/torso 12, arms 13-26), verified against Unitree's
  ``h1_2_low_level_example.py`` H1_2_JointIndex enum and unitree_rl_gym's
  ``h1_2.yaml``. Unlike H1 (scrambled motor order) H1_2 needs no remap --
  matches G1's own ``None``.

Gains are inherited from ``H1_2EnvCfg.dof`` (``H1_2_27DoF`` in
``h1_2_env_cfg.py``), whose stiffness/damping were copied verbatim from our
ProtoMotions-trained policy's ONNX YAML sidecar -- do NOT override them with
unitree_rl_gym's ``h1_2.yaml`` kps/kds (those target Unitree's own
pretrained example policy, a different policy entirely).
"""

from typing import Literal

from robojudo.environment.env_cfgs import UnitreeEnvCfg

from .h1_2_env_cfg import H1_2EnvCfg


class H1_2UnitreeCfg(UnitreeEnvCfg.UnitreeCfg):
    robot: Literal["h1", "g1", "h1_2"] = "h1_2"

    msg_type: Literal["hg", "go"] = "hg"
    hand_type: Literal["Dex-3", "Inspire", "NONE"] = "NONE"

    enable_odometry: bool = False


class H1_2RealEnvCfg(H1_2EnvCfg, UnitreeEnvCfg):
    env_type: str = "UnitreeEnv"
    # ====== ENV CONFIGURATION ======
    unitree: UnitreeEnvCfg.UnitreeCfg = H1_2UnitreeCfg(
        net_if="enp0s31f6",  # note: change to your network interface
    )

    odometry_type: Literal["NONE", "DUMMY", "UNITREE", "ZED"] = "DUMMY"

    joint2motor_idx: list[int] | None = None  # identity motor order (verified)

    hand_retarget: None = None  # TODO: H1_2 hand support out of scope
