"""
Example: drive the policy server with sim<->real coordinate transforms.

Mirror of ``example_run.py`` but wired through ``transforms.sim2real`` /
``transforms.real2sim`` so the same policy can be queried from a
simulator that uses the *original* coordinate frame, even though the
checkpoint was trained on data fixed up with::

    python zarr_fix.py <dataset>.zarr.zip \\
        --mirror-y --invert_x --invert_y \\
        --gripper_flip --invert_gripper_rot

Run:
    # In one terminal, start the server (from the repo root):
    python serving/policy_server.py -i /path/to/your.ckpt --port 8001

    # In another terminal:
    python serving/example_run_sim.py --server-url http://localhost:8001

Like ``example_run.py`` this is a smoke test: the obs values are dummy
(zeros for positions / gripper, identity for rotation_6d) so the
resulting action chunk is meaningless. The point is to exercise the
sim2real -> predict -> real2sim path end to end on a real checkpoint.
"""

import argparse
import logging
import time
from typing import Any, Dict

import numpy as np

from policy_client import PolicyClient
from transforms import sim2real, real2sim


# Identity rotation_6d -- first two rows of the 3x3 identity matrix,
# flattened. Used for any rotation_6d-typed obs entry so the dummy obs
# is at least geometrically consistent.
_IDENTITY_ROT6D = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)


def build_dummy_sim_obs(shape_meta: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Build a dummy sim-frame obs dict matching the server's shape_meta.

    For each entry in ``shape_meta['obs']``:
      - rgb keys                  -> zeros in [0, 1] float32, shape (T, C, H, W)
      - rotation_6d keys          -> identity rotation tiled to shape (T, 6)
      - everything else           -> zeros float32, shape (T, *feature_shape)

    The leading ``T`` is the per-key observation horizon. The
    ``PolicyClient`` adds the batch dim internally.
    """
    obs: Dict[str, np.ndarray] = {}
    for key, attr in shape_meta['obs'].items():
        feature_shape = tuple(attr['shape'])
        horizon = int(attr['horizon'])
        full_shape = (horizon,) + feature_shape

        is_rot6d = (
            attr.get('rotation_rep') == 'rotation_6d'
            or ('_eef_rot_axis_angle' in key and feature_shape == (6,))
        )
        if is_rot6d:
            obs[key] = np.broadcast_to(
                _IDENTITY_ROT6D, full_shape
            ).astype(np.float32).copy()
        else:
            obs[key] = np.zeros(full_shape, dtype=np.float32)
    return obs


def _summarize(arr: np.ndarray, n: int = 6) -> str:
    """Compact one-line array summary for logging."""
    flat = np.asarray(arr).ravel()
    head = np.array2string(
        flat[:n], precision=3, separator=', ', suppress_small=True
    )
    return f"shape={arr.shape} head={head}"


def main():
    parser = argparse.ArgumentParser(
        description="Example PolicyClient run with sim<->real transforms"
    )
    parser.add_argument(
        '--server-url',
        default='http://localhost:8001',
        help='Base URL of the running policy_server.py (default: %(default)s)',
    )
    parser.add_argument(
        '--timeout',
        type=float,
        default=60.0,
        help='Per-request timeout in seconds (default: %(default)s).',
    )
    parser.add_argument(
        '--repeats',
        type=int,
        default=3,
        help='How many /predict calls to make. The first call typically '
             'pays JIT/autotune costs; subsequent calls reflect steady-'
             'state latency.',
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("example_run_sim")

    log.info(f"Connecting to policy server at {args.server_url} ...")
    client = PolicyClient(server_url=args.server_url, timeout=args.timeout)
    if not client.health_check():
        raise SystemExit(
            f"Server at {args.server_url} is not healthy. Start it with:\n"
            f"  python serving/policy_server.py -i <ckpt> --port <port>"
        )

    info = client.get_model_info()
    log.info("Model info:")
    for k, v in info.items():
        log.info(f"  {k}: {v}")

    shape_meta = client.get_shape_meta()
    log.info("shape_meta.obs keys: %s", list(shape_meta.get('obs', {}).keys()))
    log.info("shape_meta.action:   %s", shape_meta.get('action'))

    sim_obs = build_dummy_sim_obs(shape_meta)
    log.info("Built dummy sim obs (zeros + identity rot6d):")
    for k, v in sim_obs.items():
        log.info(f"  {k}: shape={v.shape} dtype={v.dtype}")

    # sim -> real BEFORE handing the obs to the server. The server's
    # checkpoint was trained on real-frame data, so it expects its
    # inputs to live in that frame.
    real_obs = sim2real(sim_obs)
    log.info("After sim2real (showing first 6 elements per key):")
    for k in sim_obs.keys():
        if k == 'camera0_rgb':
            continue  # passthrough; nothing interesting to log
        log.info(
            f"  {k}: sim={_summarize(sim_obs[k])} -> real={_summarize(real_obs[k])}"
        )

    client.reset()

    last_action_sim: np.ndarray = np.zeros((0, 0))
    for i in range(args.repeats):
        t0 = time.perf_counter()
        action_real, _raw = client.predict_with_raw(real_obs)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        # real -> sim AFTER receiving the action chunk so it can be
        # applied directly in the simulator's coordinate frame.
        action_sim = real2sim(action_real)
        last_action_sim = action_sim
        log.info(
            f"[{i+1}/{args.repeats}] /predict ok in {dt_ms:.1f} ms "
            f"-> action(real) shape={action_real.shape} "
            f"-> action(sim) shape={action_sim.shape}"
        )

    log.info("First sim-frame action chunk:\n%s", last_action_sim)
    log.info("Done.")


if __name__ == '__main__':
    main()
