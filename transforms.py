"""
Sim <-> real coordinate-frame transforms for the diffusion policy.

The bundled checkpoint was trained on a replay buffer that had been
preprocessed with::

    python zarr_fix.py <dataset>.zarr.zip \
        --mirror-y --invert_x --invert_y \
        --gripper_flip --invert_gripper_rot

so the policy now "thinks" in the *real*-robot coordinate frame. When we
drive the policy from a simulator that still uses the original (*sim*)
frame we need to bridge the gap on both sides of ``client.predict``:

  - :func:`sim2real`  - apply the same set of transforms to the
    observation values *before* sending them to the policy server.
  - :func:`real2sim`  - apply the inverse transforms to the policy's
    action chunk *after* receiving it.

What each flag does
-------------------
Let ``M = diag(-1, -1, 1)`` (the world-axis sign flip from
``--invert_x --invert_y``). Two axes are flipped, so ``det(M) = +1``,
i.e. ``M`` is a proper 180-degree rotation about the world Z axis;
zarr_fix therefore composes rotations as ``R' = M @ R`` (not the
similarity formula it uses for an odd number of axes).

Let ``R_local = Rx(180 deg)`` (the gripper-local correction from
``--gripper_flip``, equivalent to ``--gripper_local_rot x180``). It is
its own inverse and is symmetric.

zarr_fix applies the rotation transforms in this order (see
``main()`` in zarr_fix.py)::

    R_1 = M @ R_sim         # --invert_x --invert_y
    R_2 = R_1 @ R_local     # --gripper_flip
    R_real = R_2 ** -1       # --invert_gripper_rot

For positions only the world-axis flip matters::

    pos_real = M @ pos_sim   # i.e. pos[..., 0] *= -1; pos[..., 1] *= -1

For the bbox, ``--mirror-y`` flips y in pixel coords and swaps
``y0 <-> y1`` so the box stays axis-aligned. The shape_meta on this
checkpoint declares bbox to be in pixel space already, so we skip the
``unnormalize`` step zarr_fix would do on a normalized [0, 1] dataset
bbox.

Inverse (real -> sim)
---------------------
``M`` and ``R_local`` are involutions, so inverting gives::

    pos_sim = M @ pos_real
    R_sim   = M @ R_real ** -1 @ R_local

For matrices ``R^-1 == R^T``, which is what we use when manipulating
the rotation_6d encoding.

Shape conventions (UMI)
-----------------------
``client.predict`` expects an obs_dict that matches the model's
``shape_meta`` (see ``conf/task/umi.yaml``):

  - ``camera0_rgb``                        : ``(T, 3, H, W)``  - passthrough
  - ``robot0_eef_pos``                     : ``(T, 3)``
  - ``robot0_eef_rot_axis_angle``          : ``(T, 6)``  rotation_6d
  - ``robot0_gripper_width``               : ``(T, 1)``  - passthrough
  - ``robot0_eef_rot_axis_angle_wrt_start``: ``(T, 6)``  rotation_6d
  - ``bbox`` (if present)                  : ``(T, 4)`` ``[x0, y0, x1, y1]`` in pixels

The action chunk is ``(T, 10)`` per robot:
``[pos(3), rot6d(6), gripper_width(1)]``. Multi-robot actions
(``last_dim = 10 * n_robots``) are handled by slicing each robot's 10
dims and applying the inverse independently.

Caveat about ``obs_pose_repr=relative``
---------------------------------------
zarr_fix operates on the *absolute* world-frame fields of the dataset
(``data/robot0_eef_pos``, ``data/robot0_eef_rot_axis_angle``, ...). The
relative-pose conversion that builds the obs_dict (see
``umi.real_world.real_inference_util.get_real_umi_obs_dict``) happens
*after* zarr_fix and is not invariant to the gripper-local correction,
so applying :func:`sim2real` to a *relative*-form obs_dict is only an
approximation. The conceptually clean place to apply :func:`sim2real`
is on the raw env_obs (absolute world, 3D axis-angle) before
``get_real_umi_obs_dict``. The functions here are written to be
shape-agnostic so they work in either pipeline.
"""

from typing import Dict, Optional, Tuple

import numpy as np

# diag of the world-axis sign-flip matrix M from --invert_x --invert_y.
# Public so callers can read it back for sanity checks / logging.
AXIS_SIGNS: np.ndarray = np.array([-1.0, -1.0, 1.0])

# 3x3 rotation matrix forms.
_M: np.ndarray = np.diag(AXIS_SIGNS)

# Rx(180 deg) -- the gripper local-frame correction from --gripper_flip.
# Symmetric and self-inverse.
_R_LOCAL: np.ndarray = np.array(
    [
        [1.0,  0.0,  0.0],
        [0.0, -1.0,  0.0],
        [0.0,  0.0, -1.0],
    ]
)


def _normalize(v: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.maximum(n, eps)


def _rot6d_to_mat(d6: np.ndarray) -> np.ndarray:
    """Decode rotation_6d ``(..., 6)`` to ``(..., 3, 3)`` rotation matrix.

    Mirrors ``umi.common.pose_util.rot6d_to_mat`` (Gram-Schmidt on the
    first two rows).
    """
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = _normalize(a1)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = _normalize(b2)
    b3 = np.cross(b1, b2, axis=-1)
    return np.stack((b1, b2, b3), axis=-2)


def _mat_to_rot6d(R: np.ndarray) -> np.ndarray:
    """Encode ``(..., 3, 3)`` rotation matrix as rotation_6d ``(..., 6)``.

    Mirrors ``umi.common.pose_util.mat_to_rot6d`` (first two rows of R,
    flattened).
    """
    out = R[..., :2, :].copy()
    return out.reshape(R.shape[:-2] + (6,))


def _flip_pos(pos: np.ndarray) -> np.ndarray:
    """Apply ``M @ pos`` (component-wise sign flip on x, y).

    ``M`` is its own inverse, so this is used for both directions.
    """
    out = pos.astype(np.float64, copy=True) * AXIS_SIGNS
    return out.astype(pos.dtype, copy=False)


def _rot6d_sim2real(rot6d: np.ndarray) -> np.ndarray:
    """Apply the forward rotation transform on rotation_6d values.

    Equivalent to zarr_fix's pipeline for our flag set::

        R_1     = M @ R_sim          # --invert_x --invert_y (det(M)=+1)
        R_2     = R_1 @ R_local      # --gripper_flip
        R_real  = R_2 ** -1           # --invert_gripper_rot
    """
    R = _rot6d_to_mat(rot6d.astype(np.float64))
    R = _M @ R
    R = R @ _R_LOCAL
    R = R.swapaxes(-1, -2)
    return _mat_to_rot6d(R).astype(rot6d.dtype, copy=False)


def _rot6d_real2sim(rot6d: np.ndarray) -> np.ndarray:
    """Apply the inverse of :func:`_rot6d_sim2real` on rotation_6d values.

    Undoing zarr_fix's pipeline in reverse order, using
    ``M ** -1 == M`` and ``R_local ** -1 == R_local``::

        R_2     = R_real ** -1                     # undo --invert_gripper_rot
        R_1     = R_2 @ R_local                    # undo --gripper_flip
        R_sim   = M @ R_1                          # undo --invert_x --invert_y
    """
    R = _rot6d_to_mat(rot6d.astype(np.float64))
    R = R.swapaxes(-1, -2)
    R = R @ _R_LOCAL
    R = _M @ R
    return _mat_to_rot6d(R).astype(rot6d.dtype, copy=False)


def _bbox_valid_mask(bbox: np.ndarray) -> np.ndarray:
    """Boolean mask of rows that are NOT the ``[-1,-1,-1,-1]`` sentinel.

    zarr_fix uses this sentinel to mark "frame not processed" and passes
    those rows through verbatim. We follow the same convention so the
    round-trip survives.
    """
    return ~np.all(bbox[..., :4] == -1.0, axis=-1)


def _bbox_unnormalize_and_mirror_y(
    bbox: np.ndarray, height: int, width: int,
) -> np.ndarray:
    """Forward bbox transform: ``[0, 1]`` -> pixel space, then mirror y.

    Replicates zarr_fix's ``transform_bbox`` for our flag set
    (``--mirror-y``)::

        (x, y) -> (x * (w - 1), y * (h - 1))   # unnormalize
        (y0, y1) -> ((h-1) - y1, (h-1) - y0)   # mirror y, swap y0/y1

    Operates on ``[..., 4]`` arrays of ``[x0, y0, x1, y1]``. Sentinel
    rows ``[-1,-1,-1,-1]`` pass through unchanged.
    """
    out = bbox.astype(np.float64, copy=True)
    sx = float(width - 1)
    sy = float(height - 1)
    valid = _bbox_valid_mask(out)
    if valid.any():
        x0 = out[valid, 0] * sx
        y0 = out[valid, 1] * sy
        x1 = out[valid, 2] * sx
        y1 = out[valid, 3] * sy
        # Mirror y in pixel space and swap y0/y1 so y0 <= y1.
        new_y0 = sy - y1
        new_y1 = sy - y0
        out[valid, 0] = x0
        out[valid, 1] = new_y0
        out[valid, 2] = x1
        out[valid, 3] = new_y1
    return out.astype(bbox.dtype, copy=False)


def _bbox_unmirror_y_and_normalize(
    bbox: np.ndarray, height: int, width: int,
) -> np.ndarray:
    """Inverse bbox transform: un-mirror y in pixel space, then normalize
    back to ``[0, 1]``.

    Sentinel rows ``[-1,-1,-1,-1]`` pass through unchanged.
    """
    out = bbox.astype(np.float64, copy=True)
    sx = float(width - 1)
    sy = float(height - 1)
    valid = _bbox_valid_mask(out)
    if valid.any():
        # Un-mirror y (mirror is its own inverse).
        y0 = sy - out[valid, 3]
        y1 = sy - out[valid, 1]
        out[valid, 1] = y0
        out[valid, 3] = y1
        # Normalize from pixel space back to [0, 1].
        out[valid, 0] /= sx
        out[valid, 1] /= sy
        out[valid, 2] /= sx
        out[valid, 3] /= sy
    return out.astype(bbox.dtype, copy=False)


def _infer_image_size(
    obs_dict: Dict[str, np.ndarray],
    image_size: Optional[Tuple[int, int]],
) -> Optional[Tuple[int, int]]:
    """Pick ``(H, W)`` from an explicit override or the camera array shape."""
    if image_size is not None:
        return int(image_size[0]), int(image_size[1])
    rgb = obs_dict.get('camera0_rgb')
    if rgb is None:
        return None
    arr = rgb if isinstance(rgb, np.ndarray) else np.asarray(rgb)
    # Expect (T, C, H, W).
    if arr.ndim < 3:
        return None
    return int(arr.shape[-2]), int(arr.shape[-1])


def sim2real(
    obs_dict: Dict[str, np.ndarray],
    image_size: Optional[Tuple[int, int]] = None,
) -> Dict[str, np.ndarray]:
    """Transform a sim-frame obs_dict into the real-frame obs_dict the
    policy expects.

    Equivalent to::

        zarr_fix.py ... --mirror-y --invert_x --invert_y \
            --gripper_flip --invert_gripper_rot

    applied to the obs values:

      - ``*_eef_pos``                   : sign-flip on world x/y.
      - ``*_eef_rot_axis_angle*``       : (rotation_6d) world-rotation
        composition + gripper-local correction + invert-gripper-rot.
      - ``bbox`` (assumed normalized)   : unnormalize ``[0, 1]`` to pixel
        space, then mirror y. Sentinel rows ``[-1,-1,-1,-1]`` pass
        through unchanged.
      - ``camera0_rgb``, ``robot0_gripper_width``, anything else:
        passthrough.

    Args:
        obs_dict: observation dict matching the policy's ``shape_meta``
            (typically what you would otherwise pass directly to
            ``PolicyClient.predict``). Values may be ``np.ndarray`` or
            anything ``np.asarray`` can convert. ``bbox`` values are
            assumed to be normalized ``[0, 1]`` (the sim convention);
            the policy expects pixel-space bbox, so this function
            unnormalizes on its way out.
        image_size: ``(H, W)`` of the camera image. Used only when the
            obs dict contains a ``bbox`` key. If ``None``, inferred from
            ``camera0_rgb`` (which has shape ``(T, C, H, W)``).

    Returns:
        New dict with the same keys as ``obs_dict``. Untouched keys are
        forwarded by reference; transformed keys are fresh arrays in
        the same dtype as the input.
    """
    H_W = _infer_image_size(obs_dict, image_size)

    out: Dict[str, np.ndarray] = {}
    for key, value in obs_dict.items():
        arr = value if isinstance(value, np.ndarray) else np.asarray(value)

        if key.endswith('_eef_pos'):
            out[key] = _flip_pos(arr)
        elif '_eef_rot_axis_angle' in key:
            # rotation_6d in the policy's shape_meta despite the name.
            out[key] = _rot6d_sim2real(arr)
        elif key == 'bbox':
            if H_W is None:
                raise ValueError(
                    "sim2real: obs_dict has 'bbox' but no image size "
                    "available. Pass image_size=(H, W) or include "
                    "'camera0_rgb' so (H, W) can be inferred."
                )
            H, W = H_W
            out[key] = _bbox_unnormalize_and_mirror_y(arr, H, W)
        else:
            out[key] = arr
    return out


def real2sim(action: np.ndarray) -> np.ndarray:
    """Transform the policy's real-frame action chunk back into the sim
    frame.

    Action layout (per robot, 10 dims): ``[pos(3), rot6d(6), gripper(1)]``.
    Multi-robot action chunks are handled by slicing every consecutive
    block of 10 dims.

    Args:
        action: array with last dim a multiple of 10.

    Returns:
        Same shape and dtype as ``action`` with positions, rotations
        flipped back into the sim frame; gripper width is left alone.
    """
    a = action if isinstance(action, np.ndarray) else np.asarray(action)
    if a.shape[-1] == 0 or a.shape[-1] % 10 != 0:
        raise ValueError(
            f"real2sim: action last dim ({a.shape[-1]}) must be a non-zero "
            f"multiple of 10. Expected [pos(3), rot6d(6), gripper(1)] per robot."
        )

    out = a.astype(np.float64, copy=True)
    n_robots = out.shape[-1] // 10
    for r in range(n_robots):
        s = r * 10
        out[..., s:s + 3] = _flip_pos(out[..., s:s + 3])
        out[..., s + 3:s + 9] = _rot6d_real2sim(out[..., s + 3:s + 9])
        # out[..., s + 9:s + 10] gripper width: unchanged.
    return out.astype(a.dtype, copy=False)
