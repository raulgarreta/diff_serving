"""
Policy Model Server (standalone serving copy)

This script loads a trained policy model and serves it via a REST API.
It handles model inference requests and returns action predictions.

This file lives under ``serving/`` together with a SLIMMED, inference-only
copy of the ``diffusion_policy`` package (also under ``serving/``) so the
whole directory is self-contained: no upstream package required at runtime
beyond pip dependencies (torch, hydra, diffusers, timm, etc.).

When a checkpoint's embedded Hydra cfg sets
``_target_: diffusion_policy.workspace.train_diffusion_unet_image_workspace.\
TrainDiffusionUnetImageWorkspace`` (the workspace UMI uses for every
diffusion U-Net policy), ``hydra.utils.get_class(cfg._target_)`` resolves
to the bundled copy because this file prepends ``serving/`` to
``sys.path`` below.

Usage:
    python serving/policy_server.py --input <checkpoint_path> --port 8001

Example:
    python serving/policy_server.py -i data/models/cup_wild_vit_l_1img.ckpt

Acceleration:
    --accel none     : eager PyTorch (default)
    --accel compile  : torch.compile on obs_encoder + diffusion U-Net
    --accel tensorrt : torch.compile with the torch_tensorrt backend
                       (requires `torch-tensorrt` matching your torch+CUDA build)

    --dtype fp32|fp16|bf16 controls inference precision. For accel=none|compile
    this is implemented via torch.autocast in /predict. For accel=tensorrt the
    dtype is baked into the compiled engine via enabled_precisions.

    --num-inference-steps N overrides the DDIM step count (default 16). Lower
    values trade quality for latency; UMI policies often work fine at 8.
"""

# Record import-phase start time as early as possible so we can later report
# "how long did Python spend loading torch/hydra/diffusion_policy/etc. before
# any of our code ran?". This is the gap between hitting Enter on the python
# command and seeing the first INFO line. We use the stdlib `time` module
# directly (already in core) so this line itself adds no import cost.
print("Importing dependencies...")
import time as _time
_T_IMPORTS_START = _time.perf_counter()

import os
import sys
import argparse
import contextlib
import logging
from typing import Any, Dict, Optional

# Make this folder importable as a package root so the BUNDLED
# `diffusion_policy/` subpackage in `serving/` wins over any same-named
# package that happens to be on the global path (e.g. the full upstream
# `diffusion_policy/` at the repo root). The bundled copy is intentionally
# the inference-only subset; pulling in the upstream package by accident
# would drag in training-only deps (wandb, accelerate, datasets, ...).
#
# We insert at index 0 (i.e. before site-packages) and only if this file
# is actually being run from its on-disk location. When `serving/` is
# already on sys.path (e.g. user added it themselves, or the IDE picked
# it up) this is a no-op.
_SERVING_DIR = os.path.dirname(os.path.abspath(__file__))
if _SERVING_DIR not in sys.path:
    sys.path.insert(0, _SERVING_DIR)

import numpy as np
import torch
import dill
import hydra
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None  # progress bar is optional; phase timings still work

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from omegaconf import OmegaConf, open_dict

# Stop the clock once all heavy imports are done. We log this duration the
# first chance we get (after logging.basicConfig below).
_T_IMPORTS_END = _time.perf_counter()
# Alias so the rest of the file can keep using `time.*` without the underscore.
time = _time

# Configure logging with millisecond timestamps so it's easy to see which
# phase of startup is actually taking time.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

logger.info(
    f"Imports finished in {_T_IMPORTS_END - _T_IMPORTS_START:.2f}s "
    f"(Python startup + torch/hydra/diffusion_policy/fastapi/uvicorn)."
)


@contextlib.contextmanager
def _phase(description: str):
    """Log the start and end of a startup phase along with its wall-clock
    duration."""
    logger.info(f"-> {description} ...")
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        logger.info(f"<- {description} done in {dt:.2f}s")


@contextlib.contextmanager
def _skip_param_init():
    """
    Monkey-patch the standard parameter-initialization functions to no-ops
    for the duration of model construction. Saves a handful of seconds on
    large backbones by skipping the RNG fills of nn.Linear/Conv/LayerNorm
    weight tensors.

    Safe for inference-time loads only. Do NOT use this for training.
    """
    import torch.nn.init as _torch_init

    def _noop_init(tensor, *args, **kwargs):  # pragma: no cover - trivial
        return tensor

    saved = []  # list of (target_obj, attr_name, original_value)

    torch_init_names = [
        'uniform_', 'normal_', 'trunc_normal_', 'constant_',
        'zeros_', 'ones_', 'eye_', 'dirac_',
        'xavier_uniform_', 'xavier_normal_',
        'kaiming_uniform_', 'kaiming_normal_',
        'orthogonal_', 'sparse_',
    ]
    for n in torch_init_names:
        if hasattr(_torch_init, n):
            saved.append((_torch_init, n, getattr(_torch_init, n)))
            setattr(_torch_init, n, _noop_init)

    for modname in (
        'timm.layers.weight_init',
        'timm.layers',
        'timm.models.vision_transformer',
        'timm.models.swin_transformer',
        'timm.models.swin_transformer_v2',
        'timm.models.beit',
        'timm.models.eva',
    ):
        try:
            _m = __import__(modname, fromlist=['*'])
        except ImportError:
            continue
        for n in ('trunc_normal_', 'trunc_normal_tf_', 'lecun_normal_',
                  'variance_scaling_'):
            if hasattr(_m, n):
                saved.append((_m, n, getattr(_m, n)))
                setattr(_m, n, _noop_init)

    try:
        yield
    finally:
        for mod, name, fn in saved:
            setattr(mod, name, fn)


# Register OmegaConf resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)

# Global variables for model
policy = None
device = None
cfg = None
obs_pose_rep = None
action_pose_repr = None
checkpoint_path = None  # remembered so /reload can re-load from disk
checkpoint_file = None  # basename of the actual .ckpt file loaded

# Acceleration / precision settings, remembered so /reload can re-apply them.
accel_mode = "none"        # one of: "none", "compile", "tensorrt"
inference_dtype = "fp32"   # one of: "fp32", "fp16", "bf16"
num_inference_steps_cfg = 16
fast_init_cfg = True       # whether to use the inference fast-init path

_DTYPE_MAP = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


class ObservationRequest(BaseModel):
    """Request model for inference.

    The legacy single-sample diffusion path corresponds to all defaults
    (rerank_k=1, q_rerank=False, q_guidance_*=0, sample_temperature=1.0).
    The extra fields are kept on the schema for wire-compatibility with
    Q-aware clients; this minimal server ignores them.
    """
    obs_dict: Dict[str, Any]  # Observation dictionary with numpy arrays as lists
    rerank_k: int = 1
    q_rerank: bool = False
    q_guidance_eta: float = 0.0
    q_guidance_alpha: float = 1.0
    q_guidance_steps: int = 0
    sample_temperature: float = 1.0


class ActionResponse(BaseModel):
    """Response model for inference"""
    action: list  # Action array as list
    raw_action: list  # Raw action before post-processing
    q_info: Optional[Dict[str, Any]] = None


class ModelInfoResponse(BaseModel):
    """Response model for model information"""
    model_name: str
    dataset_path: str
    obs_pose_repr: str
    action_pose_repr: str
    num_inference_steps: int
    device: str
    checkpoint: str  # basename of the loaded checkpoint file
    accel: str       # "none", "compile", or "tensorrt"
    dtype: str       # "fp32", "fp16", or "bf16"


# Initialize FastAPI app
app = FastAPI(title="Policy Model Server", version="1.0.0")


def _build_dummy_obs_dict(shape_meta, batch_size: int = 1):
    """
    Build a dummy obs dict matching shape_meta, used to warm up the compiled
    modules so the first real /predict call doesn't pay the compile cost.
    """
    obs_dict = {}
    for key, attr in shape_meta['obs'].items():
        feature_shape = tuple(attr['shape'])
        horizon = int(attr['horizon'])
        full_shape = (batch_size, horizon) + feature_shape
        obs_dict[key] = torch.zeros(full_shape, dtype=torch.float32)
    return obs_dict


def _autocast_ctx():
    """Return an autocast context manager appropriate for the current device
    and requested inference dtype. For fp32 or non-CUDA devices this is a
    no-op."""
    if inference_dtype == "fp32":
        return contextlib.nullcontext()
    if device is None or device.type != "cuda":
        return contextlib.nullcontext()
    if accel_mode == "tensorrt":
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=_DTYPE_MAP[inference_dtype])


def _apply_acceleration(policy_obj, shape_meta):
    """Wrap policy_obj.obs_encoder and policy_obj.model with the requested
    accelerator, then run a warmup pass so the compile cost is paid up
    front."""
    if accel_mode == "none":
        return

    if accel_mode == "compile":
        logger.info("Compiling obs_encoder and diffusion U-Net with torch.compile (mode=reduce-overhead)...")
        policy_obj.obs_encoder = torch.compile(
            policy_obj.obs_encoder, mode="reduce-overhead", dynamic=False
        )
        policy_obj.model = torch.compile(
            policy_obj.model, mode="reduce-overhead", dynamic=False
        )
    elif accel_mode == "tensorrt":
        try:
            import torch_tensorrt  # type: ignore[import-not-found]  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "accel=tensorrt requires the `torch-tensorrt` package, which "
                "is tightly version-coupled to your torch+CUDA build. Install "
                "the matching wheel and retry."
            ) from e

        if device is None or device.type != "cuda":
            raise RuntimeError("accel=tensorrt requires a CUDA device.")

        precisions = {torch.float32}
        if inference_dtype == "fp16":
            precisions = {torch.float16, torch.float32}
        elif inference_dtype == "bf16":
            precisions = {torch.bfloat16, torch.float32}

        trt_options = {
            "enabled_precisions": precisions,
            "truncate_long_and_double": True,
        }
        logger.info(
            f"Compiling obs_encoder and diffusion U-Net with torch_tensorrt "
            f"(precisions={[str(p) for p in precisions]})..."
        )
        policy_obj.obs_encoder = torch.compile(
            policy_obj.obs_encoder, backend="torch_tensorrt", options=trt_options, dynamic=False
        )
        policy_obj.model = torch.compile(
            policy_obj.model, backend="torch_tensorrt", options=trt_options, dynamic=False
        )
    else:
        raise ValueError(f"Unknown accel mode: {accel_mode}")

    logger.info("Running warmup forward pass(es)...")
    try:
        dummy = _build_dummy_obs_dict(shape_meta, batch_size=1)
        dummy = dict_apply(dummy, lambda x: x.to(device))
        with torch.no_grad(), _autocast_ctx():
            for _ in range(2):
                policy_obj.predict_action(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        logger.info("Warmup complete.")
    except Exception as e:
        logger.warning(
            f"Warmup pass failed (compilation will happen on first /predict instead): {e}"
        )


def load_model(
    ckpt_arg: str,
    accel: str = "none",
    dtype: str = "fp32",
    num_inference_steps: int = 16,
    fast_init: bool = True,
):
    """Load the policy model from checkpoint"""
    global policy, device, cfg, obs_pose_rep, action_pose_repr, checkpoint_path, checkpoint_file
    global accel_mode, inference_dtype, num_inference_steps_cfg, fast_init_cfg

    checkpoint_path = ckpt_arg
    accel_mode = accel
    inference_dtype = dtype
    num_inference_steps_cfg = num_inference_steps
    fast_init_cfg = fast_init

    logger.info(f"Loading checkpoint from: {ckpt_arg}")

    ckpt_path = ckpt_arg
    if not ckpt_path.endswith('.ckpt'):
        ckpt_path = os.path.join(ckpt_path, 'checkpoints', 'latest.ckpt')

    checkpoint_file = os.path.basename(ckpt_path)

    t_start = time.perf_counter()

    # --- Phase 1: read checkpoint from disk -------------------------------
    file_size = os.path.getsize(ckpt_path)
    with _phase(f"Reading checkpoint ({file_size / 1e6:.1f} MB)"):
        with open(ckpt_path, 'rb') as raw_f:
            if tqdm is not None:
                with tqdm.wrapattr(
                    raw_f, "read",
                    total=file_size,
                    desc="ckpt",
                    unit='B', unit_scale=True, unit_divisor=1024,
                    leave=False,
                ) as wrapped_f:
                    payload = torch.load(
                        wrapped_f, map_location='cpu', pickle_module=dill
                    )
            else:
                payload = torch.load(
                    raw_f, map_location='cpu', pickle_module=dill
                )

    cfg = payload['cfg']
    logger.info(f"Model name: {cfg.policy.obs_encoder.model_name}")
    logger.info(f"Dataset path: {cfg.task.dataset.dataset_path}")

    # --- Phase 2: rewrite cfg for inference -------------------------------
    # Skip downloading the timm/HF pretrained backbone weights at inference
    # time. The checkpoint already contains the fully-trained obs_encoder
    # state_dict, so any pretrained init would just be overwritten a moment
    # later.
    try:
        if OmegaConf.select(cfg, "policy.obs_encoder.pretrained") is True:
            logger.info("Disabling obs_encoder.pretrained for inference load "
                        "(weights come from checkpoint).")
            with open_dict(cfg):
                cfg.policy.obs_encoder.pretrained = False
    except Exception as e:
        logger.warning(f"Could not override obs_encoder.pretrained=False: {e}")

    original_use_ema = bool(OmegaConf.select(cfg, "training.use_ema"))

    # In fast-init mode we additionally suppress the EMA deepcopy.
    if fast_init and original_use_ema:
        try:
            with open_dict(cfg):
                cfg.training.use_ema = False
        except Exception as e:
            logger.warning(
                f"fast_init: could not suppress EMA deepcopy ({e}); falling "
                f"back to standard EMA construction."
            )

    # --- Phase 3a: resolve workspace class --------------------------------
    with _phase(f"Resolving workspace class ({cfg._target_})"):
        cls = hydra.utils.get_class(cfg._target_)

    # --- Phase 3b: build workspace ----------------------------------------
    init_phase_label = (
        "Building workspace (fast-init: skip RNG, skip EMA copy)"
        if fast_init else
        "Building workspace (instantiating model graph)"
    )
    with _phase(init_phase_label):
        init_ctx = _skip_param_init() if fast_init else contextlib.nullcontext()
        with init_ctx:
            workspace = cls(cfg)
        workspace: BaseWorkspace

    # --- Phase 4: copy trained weights into the model ---------------------
    with _phase("Loading state_dict from checkpoint payload"):
        if fast_init:
            sds = payload['state_dicts']
            if original_use_ema and 'ema_model' in sds:
                src_key = 'ema_model'
            elif 'model' in sds:
                src_key = 'model'
            else:
                raise RuntimeError(
                    f"Checkpoint has no 'model' or 'ema_model' state_dict; "
                    f"got keys: {list(sds.keys())}"
                )
            logger.info(f"fast_init: loading '{src_key}' state_dict into model")
            missing, unexpected = workspace.model.load_state_dict(
                sds[src_key], strict=False,
            )
            if missing:
                logger.warning(
                    f"fast_init: {len(missing)} missing keys when loading "
                    f"'{src_key}' (e.g. {missing[:3]})."
                )
            if unexpected:
                logger.warning(
                    f"fast_init: {len(unexpected)} unexpected keys when "
                    f"loading '{src_key}' (e.g. {unexpected[:3]})."
                )
            policy = workspace.model
        else:
            workspace.load_payload(payload, exclude_keys=None, include_keys=None)
            policy = workspace.model
            if cfg.training.use_ema:
                logger.info("Using EMA model")
                policy = workspace.ema_model

    policy.num_inference_steps = num_inference_steps_cfg
    obs_pose_rep = cfg.task.pose_repr.obs_pose_repr
    action_pose_repr = cfg.task.pose_repr.action_pose_repr

    logger.info(f"obs_pose_repr: {obs_pose_rep}")
    logger.info(f"action_pose_repr: {action_pose_repr}")
    logger.info(f"num_inference_steps: {policy.num_inference_steps}")

    # --- Phase 5: pick a device -------------------------------------------
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info("Using CUDA device")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        logger.info("Using MPS device (Apple Silicon)")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU device")

    # --- Phase 6: move weights to the chosen device -----------------------
    with _phase(f"Moving model to {device}"):
        policy.eval().to(device)
        if device.type == 'cuda':
            torch.cuda.synchronize()

    # --- Phase 7: optional acceleration / warmup --------------------------
    shape_meta = OmegaConf.to_container(cfg.task.shape_meta, resolve=True)
    with _phase(f"Applying acceleration (accel={accel_mode}, dtype={inference_dtype})"):
        _apply_acceleration(policy, shape_meta)

    total_dt = time.perf_counter() - t_start
    logger.info(
        f"Model loaded successfully on device: {device} "
        f"(accel={accel_mode}, dtype={inference_dtype}, "
        f"steps={num_inference_steps_cfg}, "
        f"fast_init={'on' if fast_init else 'off'}) "
        f"in {total_dt:.2f}s total"
    )


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    logger.info("Policy Model Server starting up...")


@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Policy Model Server is running", "status": "ok"}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if policy is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model_loaded": True}


@app.get("/model_info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get model information"""
    if policy is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return ModelInfoResponse(
        model_name=cfg.policy.obs_encoder.model_name,
        dataset_path=cfg.task.dataset.dataset_path,
        obs_pose_repr=obs_pose_rep,
        action_pose_repr=action_pose_repr,
        num_inference_steps=policy.num_inference_steps,
        device=str(device),
        checkpoint=checkpoint_file or "",
        accel=accel_mode,
        dtype=inference_dtype,
    )


@app.get("/shape_meta")
async def get_shape_meta():
    """Get shape_meta configuration"""
    if policy is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    shape_meta_dict = OmegaConf.to_container(cfg.task.shape_meta, resolve=True)
    return shape_meta_dict


@app.post("/reset")
async def reset_policy():
    """Lightweight reset of policy state."""
    if policy is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        policy.reset()

        seed = 0
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        try:
            policy.noise_scheduler.set_timesteps(policy.num_inference_steps)
        except Exception as e:
            logger.warning(f"Could not reset noise scheduler timesteps: {e}")

        policy.eval()

        if device is not None and device.type == 'cuda':
            torch.cuda.empty_cache()
        elif device is not None and device.type == 'mps':
            try:
                torch.mps.empty_cache()
            except Exception:
                pass

        return {"status": "success", "message": "Policy reset successfully"}
    except Exception as e:
        logger.error(f"Error resetting policy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reload")
async def reload_policy():
    """Strong reset: fully reload the policy from the original checkpoint
    on disk."""
    if checkpoint_path is None:
        raise HTTPException(
            status_code=503,
            detail="No checkpoint path remembered; cannot reload"
        )

    try:
        logger.info("Reloading model from checkpoint (full reset)...")
        global policy
        policy = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif device is not None and device.type == 'mps':
            try:
                torch.mps.empty_cache()
            except Exception:
                pass

        load_model(
            checkpoint_path,
            accel=accel_mode,
            dtype=inference_dtype,
            num_inference_steps=num_inference_steps_cfg,
            fast_init=fast_init_cfg,
        )
        logger.info("Model reloaded successfully")
        return {"status": "success", "message": "Policy reloaded from checkpoint"}
    except Exception as e:
        logger.error(f"Error reloading policy: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", response_model=ActionResponse)
async def predict_action(request: ObservationRequest):
    """
    Predict action from observation.

    The observation dictionary should contain numpy arrays converted to lists.
    Arrays are reconstructed and processed for inference.
    """
    if policy is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Convert observation dict from lists back to numpy arrays
        obs_dict_np = {}
        for key, value in request.obs_dict.items():
            if isinstance(value, list):
                if device.type == 'mps':
                    arr = np.array(value, dtype=np.float32)
                else:
                    arr = np.array(value)
                obs_dict_np[key] = arr
            else:
                obs_dict_np[key] = value

        def to_tensor(x):
            tensor = torch.from_numpy(x)
            if device.type == 'mps' and tensor.dtype == torch.float64:
                tensor = tensor.float()
            return tensor.unsqueeze(0).to(device)

        obs_dict = dict_apply(obs_dict_np, to_tensor)

        with torch.no_grad(), _autocast_ctx():
            result = policy.predict_action(obs_dict)
        action_pred = result['action_pred']

        raw_action = action_pred[0].detach().to('cpu').numpy()
        action_list = raw_action.tolist()

        return ActionResponse(
            action=action_list,
            raw_action=action_list,
            q_info=None,
        )

    except Exception as e:
        logger.error(f"Error during inference: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def main():
    parser = argparse.ArgumentParser(description="Policy Model Server")
    parser.add_argument('--input', '-i', required=True, help='Path to checkpoint')
    parser.add_argument('--port', '-p', type=int, default=8001, help='Server port')
    parser.add_argument('--host', default='0.0.0.0', help='Server host')
    parser.add_argument(
        '--accel',
        choices=['none', 'compile', 'tensorrt'],
        default='none',
        help='Inference acceleration mode (default: none).'
    )
    parser.add_argument(
        '--dtype',
        choices=['fp32', 'fp16', 'bf16'],
        default='fp32',
        help='Inference precision (default: fp32).'
    )
    parser.add_argument(
        '--num-inference-steps',
        type=int,
        default=16,
        help='DDIM denoising steps per action chunk (default: 16).'
    )
    parser.add_argument(
        '--fast-init',
        dest='fast_init',
        action='store_true',
        default=True,
        help='[default] Skip RNG-based parameter init and the EMA deepcopy '
             'during model construction (safe because the checkpoint '
             'overwrites every parameter on load).'
    )
    parser.add_argument(
        '--no-fast-init',
        dest='fast_init',
        action='store_false',
        help='Disable fast-init: do the full random parameter init and EMA '
             'deepcopy as in training.'
    )

    args = parser.parse_args()

    load_model(
        args.input,
        accel=args.accel,
        dtype=args.dtype,
        num_inference_steps=args.num_inference_steps,
        fast_init=args.fast_init,
    )

    logger.info(f"Starting server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == '__main__':
    main()
