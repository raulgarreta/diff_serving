# Policy Serving

Standalone serving entrypoints for a UMI diffusion policy: a FastAPI
**server** that loads a trained checkpoint and exposes `/predict`, a tiny
**client** library, and a one-shot **example** script that drives the
server end-to-end.

## Files

| File                       | What it is                                                                                          |
|----------------------------|------------------------------------------------------------------------------------------------------|
| `policy_server.py`         | FastAPI server. Loads a checkpoint via Hydra, exposes `/predict`, `/model_info`, `/shape_meta`, etc. |
| `policy_client.py`         | Thin Python client. Pure HTTP + numpy; no torch/hydra/diffusion_policy import.                       |
| `example_run.py`           | Smoke test: connects to the server, fetches `shape_meta`, builds a zero obs, calls `/predict`.       |
| `diffusion_policy/`        | Inference-only subset of the upstream `diffusion_policy` package (13 .py files, ~1.5k LoC).          |
| `requirements_server.txt`  | Full dependency set to run the **server** (torch + diffusion stack + HTTP layer).                    |
| `requirements_client.txt`  | Tiny dependency set to run the **client** (just `requests` + `numpy`).                               |

## Self-contained — no upstream `diffusion_policy/` required

The `diffusion_policy/` subfolder here is a slimmed copy that contains
only the modules needed at inference time (workspace `__init__`, the
diffusion U-Net policy, the timm obs encoder, the normalizer, etc.).
Training-only code (datasets, env_runner, wandb glue, EMAModel,
checkpoint manager, AWR/IQL training loops, …) is intentionally left out.

`policy_server.py` prepends this `serving/` directory to `sys.path` at
import time, so `hydra.utils.get_class("diffusion_policy.workspace.…")`
in your checkpoint's embedded `cfg._target_` resolves to the bundled
class — even if a full upstream `diffusion_policy/` happens to sit next
to `serving/`, it does not get loaded.

The **client** has no `diffusion_policy` dependency at all — copy
`policy_client.py` and `requirements_client.txt` anywhere and you're
done.

### Which checkpoints are supported

Any checkpoint whose embedded Hydra cfg sets:

```yaml
_target_: diffusion_policy.workspace.train_diffusion_unet_image_workspace.TrainDiffusionUnetImageWorkspace
policy:
  _target_: diffusion_policy.policy.diffusion_unet_timm_policy.DiffusionUnetTimmPolicy
  obs_encoder:
    _target_: diffusion_policy.model.vision.timm_obs_encoder.TimmObsEncoder
```

…which covers every UMI diffusion U-Net config shipped in the repo
(`train_diffusion_unet_timm_umi_*.yaml`, including the BBox and AWR
variants). Other workspace classes (e.g. the transformer policy) are not
bundled; you would need to copy the additional module(s) into
`serving/diffusion_policy/` to use them here.

## Run

### 0) Install the server deps (one-time)

```bash
pip install -r serving/requirements_server.txt
```

### 1) Start the server

```bash
# Can be run from anywhere -- policy_server.py auto-finds its bundled
# diffusion_policy package via sys.path.
python serving/policy_server.py \
    -i /path/to/your_policy.ckpt \
    --port 8001
```

Useful flags (see `--help` for the full list):

- `--accel none|compile|tensorrt` — pick eager, `torch.compile`, or
  TensorRT.
- `--dtype fp32|fp16|bf16` — inference precision (autocast for
  `none|compile`; baked into the engine for `tensorrt`).
- `--num-inference-steps 16` — DDIM denoising steps.
- `--no-fast-init` — opt out of the inference-only init shortcuts (only
  needed for debugging numerical drift).

### 2) Run the example client

```bash
# in a separate terminal (any env that has requests+numpy)
python serving/example_run.py --server-url http://localhost:8001
```

You should see something like:

```
Model info:
  model_name: vit_large_patch14_clip_224.openai
  checkpoint: latest.ckpt
  device:     cuda
  ...
[1/3] /predict ok in 312.4 ms -> action shape=(16, 10) dtype=float64
[2/3] /predict ok in 41.9 ms  -> action shape=(16, 10) dtype=float64
[3/3] /predict ok in 41.6 ms  -> action shape=(16, 10) dtype=float64
```

The action chunk is nonsense (the obs is all zeros) — this just verifies
end-to-end wiring on your actual checkpoint.

## Using the client in your own code

```python
from policy_client import PolicyClient

client = PolicyClient(server_url="http://your-gpu-box:8001", timeout=60.0)

shape_meta = client.get_shape_meta()
print(shape_meta["obs"].keys())

obs_dict = {
    # Fill in your real obs here. Each value is a numpy array shaped
    # (horizon, *feature_shape), matching shape_meta["obs"][key].
    "camera0_rgb":      np.zeros((1, 3, 224, 224), dtype=np.float32),
    "robot0_eef_pos":   np.zeros((1, 3),           dtype=np.float32),
    # ...
}
client.reset()
action, raw_action = client.predict_with_raw(obs_dict)  # (T, A)
```

## HTTP endpoints

| Method | Path           | What it does                                                       |
|--------|----------------|--------------------------------------------------------------------|
| GET    | `/`            | Liveness sanity check.                                             |
| GET    | `/health`      | Returns 503 until the model finishes loading.                      |
| GET    | `/model_info`  | Model name, dataset path, pose reps, device, dtype, accel, etc.    |
| GET    | `/shape_meta`  | The full `cfg.task.shape_meta` dict (what obs keys+shapes to send).|
| POST   | `/reset`       | Re-seed RNG, reset scheduler, empty allocator caches.              |
| POST   | `/reload`      | Strong reset: re-read the checkpoint from disk and rebuild.        |
| POST   | `/predict`     | Run one forward pass; returns the unnormalized action chunk.       |
