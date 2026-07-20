"""Optional D435i-equivalent head camera attached to a LIVE :class:`NewtonEnv`.

Reuses the exact renderer + camera-geometry-parity machinery
``src/imprint/sim2sim/newton_visual_eval.py`` already built for Newton visual
sim2sim eval (:class:`newton.sensors.SensorTiledCamera`, the merged D435i
mount/intrinsics config, the ROS/OpenCV -> OpenGL basis flip, and the exact
424x240 -> 212x120 downsample) -- but binds it to a DYNAMIC, physics-stepped
NewtonEnv (floating-base humanoid actually walking/manipulating) instead of
that module's fixed-base, ``eval_fk``-only kinematic scene: every ``render()``
reads the CURRENT ``env.state_0.body_q`` torso pose, so frames track the sim
as the deploy harness steps it.

Requires ``newton>=1.3``'s ``newton.sensors.SensorTiledCamera`` (present since
Newton 1.4 in ``glvov-envs/newton-rtx``; NOT present in the ``newton==1.0.0``
environments used for the rest of RoboJuDo's Newton deploy path -- see
``docs/VISUAL_SIM2REAL_DEPLOY.md``'s sim2sim section for the env gap). This
module is imported lazily (only when :meth:`NewtonEnv.attach_camera` is
actually called), so envs without it still import/run the physics-only path
fine.
"""

from __future__ import annotations

import logging
import math
import os
import sys
from pathlib import Path

import numpy as np
import warp as wp

logger = logging.getLogger(__name__)

_WT_VISUAL_S2R_ROOT = Path(
    os.environ.get("IMPRINT_WT_VISUAL_S2R_ROOT", "/oscar/data/stellex/glvov/wt_visual_s2r")
)
_IMPRINT_SRC = _WT_VISUAL_S2R_ROOT / "src"
if str(_IMPRINT_SRC) not in sys.path and _IMPRINT_SRC.is_dir():
    sys.path.insert(0, str(_IMPRINT_SRC))

try:
    from newton.sensors import SensorTiledCamera
except ImportError as exc:  # pragma: no cover - only hit on newton<1.3 envs
    raise ImportError(
        "robojudo.environment.utils.newton_camera requires newton>=1.3's "
        "newton.sensors.SensorTiledCamera (not present in this newton build). "
        "Use an env with newton>=1.3 (e.g. glvov-envs/newton-rtx), or run "
        "NewtonEnv without attach_camera() for the physics-only deploy path."
    ) from exc

from imprint.integrations.visual_sim2sim.camera import (  # noqa: E402
    D435_FACTORY_FOV_DEG,
    D435_NATIVE_RESOLUTION,
    D435_SIM_RESOLUTION,
    H1_T_TORSO_CAMERA,
)

# ROS/OpenCV(optical) -> OpenGL(camera) basis flip: x same, y and z negated.
# (identical to src/imprint/sim2sim/newton_visual_eval.py's `_CV_TO_GL`)
_CV_TO_GL = np.diag([1.0, -1.0, -1.0]).astype(np.float64)
_TORSO_LINK_LABEL_SUFFIX = "torso_link"


# --------------------------------------------------------------------------- #
# transform helpers (numpy, no scipy dep -- ported from newton_visual_eval.py)
# --------------------------------------------------------------------------- #
def _quat_xyzw_to_matrix(q: np.ndarray) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n == 0.0:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _matrix_to_quat_xyzw(m: np.ndarray) -> np.ndarray:
    t = float(np.trace(m))
    if t > 0.0:
        s = math.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    else:
        i = int(np.argmax(np.diag(m)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = math.sqrt(1.0 + m[i, i] - m[j, j] - m[k, k]) * 2.0
        xyz = [0.0, 0.0, 0.0]
        xyz[i] = 0.25 * s
        xyz[j] = (m[j, i] + m[i, j]) / s
        xyz[k] = (m[k, i] + m[i, k]) / s
        w = (m[k, j] - m[j, k]) / s
        x, y, z = xyz
    q = np.array([x, y, z, w], dtype=np.float64)
    return q / np.linalg.norm(q)


def _transform_to_matrix(pos: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = _quat_xyzw_to_matrix(quat_xyzw)
    T[:3, 3] = pos
    return T


# --------------------------------------------------------------------------- #
# downsampling 424x240 -> 212x120 (exact 2x2 blocks; identical to
# newton_visual_eval.py's downsample_rgb/downsample_depth)
# --------------------------------------------------------------------------- #
def downsample_rgb(rgb: np.ndarray) -> np.ndarray:
    h, w = rgb.shape[:2]
    out = rgb.reshape(h // 2, 2, w // 2, 2, 3).astype(np.float32).mean(axis=(1, 3))
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def downsample_depth(depth: np.ndarray) -> np.ndarray:
    h, w = depth.shape
    blk = depth.reshape(h // 2, 2, w // 2, 2)
    valid = blk > 0.0
    summed = np.where(valid, blk, 0.0).sum(axis=(1, 3))
    count = valid.sum(axis=(1, 3))
    out = np.where(count > 0, summed / np.maximum(count, 1), -1.0)
    return out.astype(np.float32)


class NewtonHeadCamera:
    """D435i-equivalent head camera riding a live :class:`NewtonEnv`'s ``torso_link``."""

    def __init__(self, env, vfov_deg: float | None = None):
        self.env = env
        model = env.model
        torso_body = -1
        for i, label in enumerate(model.body_label):
            if label.split("/")[-1] == _TORSO_LINK_LABEL_SUFFIX:
                torso_body = i
                break
        if torso_body < 0:
            raise RuntimeError(
                "[NewtonHeadCamera] 'torso_link' body not found in the Newton model -- "
                "cannot attach the head camera to this MJCF"
            )
        self.torso_body = torso_body

        self.native_res = tuple(int(v) for v in D435_NATIVE_RESOLUTION)  # (424, 240)
        self.sim_res = tuple(int(v) for v in D435_SIM_RESOLUTION)  # (212, 120)
        self.vfov_deg = float(vfov_deg if vfov_deg is not None else D435_FACTORY_FOV_DEG[1])

        # API per newton==1.4's newton/examples/sensors/example_sensor_tiled_camera.py
        # (this newton build has no `SensorTiledCamera.Config` / constructor-time config --
        # NOT what src/imprint/sim2sim/newton_visual_eval.py's docstring, written against a
        # different newton version, assumes; see that module's own version-skew caveats).
        self.model = model
        self.sensor = SensorTiledCamera(model=model)
        self.sensor.default_render_config.enable_shadows = True
        self.sensor.utils.create_default_light(enable_shadows=True)
        w, h = self.native_res
        self.rays = self.sensor.utils.compute_camera_rays_pinhole(w, h, camera_fovs=math.radians(self.vfov_deg))
        self.color = self.sensor.utils.create_color_image_output(w, h, 1)
        self.depth = self.sensor.utils.create_depth_image_output(w, h, 1)
        self.T_torso_cam_cv = np.asarray(H1_T_TORSO_CAMERA, dtype=np.float64)
        logger.info(
            f"[NewtonHeadCamera] attached at torso body idx={torso_body}, "
            f"native={self.native_res}, sim={self.sim_res}, vfov={self.vfov_deg:.2f}deg"
        )

    def _camera_transform(self) -> wp.array:
        bq = self.env.state_0.body_q.numpy()[self.torso_body]
        T_world_torso = _transform_to_matrix(bq[:3], bq[3:7])
        T_world_cam_cv = T_world_torso @ self.T_torso_cam_cv
        T_world_cam_gl = T_world_cam_cv.copy()
        T_world_cam_gl[:3, :3] = T_world_cam_cv[:3, :3] @ _CV_TO_GL
        pos = T_world_cam_gl[:3, 3]
        quat = _matrix_to_quat_xyzw(T_world_cam_gl[:3, :3])
        return wp.array(
            [[wp.transformf(wp.vec3f(*pos), wp.quatf(*quat))]],
            dtype=wp.transformf,
            device=self.env.wp_device,
        )

    def render(self) -> tuple[np.ndarray, np.ndarray]:
        """Render once at the CURRENT sim state; return (rgb HxWx3 uint8, depth HxW float32 [m])."""
        # Required before update() on a DYNAMIC (physics-stepped) model -- the sensor's ray/BVH
        # intersection reads the shape BVH, which the solver does not keep refit on its own.
        self.model.bvh_refit_shapes(self.env.state_0)
        self.model.bvh_refit_particles(self.env.state_0)
        self.sensor.update(
            self.env.state_0,
            self._camera_transform(),
            self.rays,
            color_image=self.color,
            depth_image=self.depth,
        )
        wp.synchronize_device(self.env.wp_device)
        color_u32 = self.color.numpy()[0, 0]  # (h, w) uint32 0xAARRGGBB
        depth = self.depth.numpy()[0, 0].astype(np.float32)
        r = ((color_u32 >> np.uint32(16)) & np.uint32(0xFF)).astype(np.uint8)
        g = ((color_u32 >> np.uint32(8)) & np.uint32(0xFF)).astype(np.uint8)
        b = (color_u32 & np.uint32(0xFF)).astype(np.uint8)
        rgb = np.stack([r, g, b], axis=-1)
        return rgb, depth

    def save_frame(self, path: str, downsample: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """Render + save a PNG (native 424x240, or 212x120 if ``downsample``). Returns the
        (possibly downsampled) (rgb, depth) arrays."""
        from PIL import Image

        rgb, depth = self.render()
        if downsample:
            rgb = downsample_rgb(rgb)
            depth = downsample_depth(depth)
        Image.fromarray(rgb).save(path)
        return rgb, depth
