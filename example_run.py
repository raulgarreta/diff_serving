"""
Example: drive the policy server via PolicyClient.

This script connects to a running ``policy_server.py``, inspects the
model's ``shape_meta`` (so it knows exactly which obs keys / shapes the
policy expects), builds a single dummy observation that matches that
schema, and sends one /predict request. The returned action chunk has
shape ``(action_horizon, action_dim)`` -- print it and exit.

Run:
    # In one terminal, start the server (from the repo root):
    python serving/policy_server.py -i /path/to/your.ckpt --port 8001

    # In another terminal:
    python serving/example_run.py --server-url http://localhost:8001

This is intentionally a "smoke test" -- the obs values are all zeros, so
the resulting action chunk is meaningless. The point is to verify the
client/server wiring end-to-end on a real checkpoint.
"""

import argparse
import logging
import time
from typing import Any, Dict

import numpy as np

from policy_client import PolicyClient


def build_dummy_obs(shape_meta: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Build an obs dict that matches the server's shape_meta.

    For each key in ``shape_meta['obs']``:
      - ``type='rgb'``  -> zeros in [0, 1] float32, shape (T, C, H, W)
      - ``type='low_dim'`` (or omitted) -> zeros float32, shape (T, *feature_shape)

    The leading ``T`` is the observation horizon (``attr['horizon']``).
    The PolicyClient handles batching internally, so we DO NOT add a batch
    dim here -- the server prepends one in ``/predict``.
    """
    obs: Dict[str, np.ndarray] = {}
    for key, attr in shape_meta['obs'].items():
        feature_shape = tuple(attr['shape'])
        horizon = int(attr['horizon'])
        full_shape = (horizon,) + feature_shape
        obs[key] = np.zeros(full_shape, dtype=np.float32)
    return obs


def main():
    parser = argparse.ArgumentParser(description="Example PolicyClient run")
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
             'pays JIT/autotune costs; subsequent calls reflect steady-state '
             'latency.',
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("example_run")

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

    obs_dict = build_dummy_obs(shape_meta)
    log.info("Built dummy obs (zeros) with keys+shapes:")
    for k, v in obs_dict.items():
        log.info(f"  {k}: shape={v.shape} dtype={v.dtype}")

    client.reset()

    for i in range(args.repeats):
        t0 = time.perf_counter()
        action, raw_action = client.predict_with_raw(obs_dict)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        log.info(
            f"[{i+1}/{args.repeats}] /predict ok in {dt_ms:.1f} ms "
            f"-> action shape={action.shape} dtype={action.dtype}"
        )

    log.info("First action chunk:\n%s", action)
    log.info("Done.")


if __name__ == '__main__':
    main()
