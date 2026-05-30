"""Inference-only subset of the ``diffusion_policy`` package.

This is NOT the full upstream package -- it contains just the modules
needed to load and run a trained UMI diffusion policy via the FastAPI
server in this folder. Training-only code (datasets, env runners,
checkpoint manager, wandb glue, AWR/IQL training pipelines, etc.) has
been intentionally left out.

Checkpoints written by ``TrainDiffusionUnetImageWorkspace`` (the only
workspace UMI uses for diffusion U-Net policies) load here byte-for-byte
because:

  * Module paths under ``diffusion_policy.*`` are preserved verbatim,
    so ``hydra.utils.get_class(cfg._target_)`` resolves the workspace
    class without any rewriting.
  * The workspace's ``__init__`` only needs the policy + obs_encoder +
    diffusion-model classes (instantiated via Hydra from the embedded
    cfg). Heavy training imports (dataset, env_runner, accelerate,
    wandb, EMAModel) are lazy-imported inside ``run()``, which the
    inference path never calls.
"""
