from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.model.vision.timm_obs_encoder import TimmObsEncoder
from diffusion_policy.common.pytorch_util import dict_apply


class DiffusionUnetTimmPolicy(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            obs_encoder: TimmObsEncoder,
            num_inference_steps=None,
            obs_as_global_cond=True,
            diffusion_step_embed_dim=256,
            down_dims=(256,512,1024),
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
            input_pertub=0.1,
            inpaint_fixed_action_prefix=False,
            train_diffusion_n_samples=1,
            # ============== AWR finetuning knobs ==============
            # Advantage-Weighted Regression: weight each sample's
            # denoising MSE by ``exp(advantage / awr_beta)``. Disabled
            # (legacy BC) when awr_beta <= 0 OR the batch lacks
            # ``advantage`` / ``has_advantage`` keys (which UmiDataset
            # adds when rl/q + rl/v are present in the source zarr).
            #
            # awr_beta:    temperature on the advantage. Lower beta
            #              sharpens the weighting (more like Q-max but
            #              unstable); higher beta softens it toward BC.
            #              A sensible default is roughly ``std(A)`` on
            #              your dataset (your IQL ckpt name encodes
            #              ``val_adv_std``, so use that as a starting
            #              point). 0.0 disables AWR entirely.
            # awr_w_max:   per-sample weight clip BEFORE batch norm.
            #              Caps the influence of a single outlier with
            #              very large advantage. Typical 10..100.
            # awr_lambda:  blend between BC weight (1.0) and AWR weight
            #              ``w``. 1.0 is pure AWR; 0.0 collapses back
            #              to BC; values in (0, 1) act as a "safety
            #              belt" toward the BC checkpoint's behavior.
            # awr_normalize_weights: if True, divide weights by their
            #              batch mean before applying. Makes the
            #              effective LR roughly invariant to beta and
            #              to which subset of the data has high
            #              advantage. Strongly recommended.
            # awr_use_ema_weights: deprecated knob slot kept here as a
            #              future hook (e.g. for a slow-moving running
            #              mean of the batch weight to stabilize early
            #              training); currently unused.
            awr_beta=0.0,
            awr_w_max=20.0,
            awr_lambda=1.0,
            awr_normalize_weights=True,
            # parameters passed to step
            **kwargs
        ):
        super().__init__()

        # parse shapes
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        action_horizon = shape_meta['action']['horizon']
        # get feature dim
        obs_feature_dim = np.prod(obs_encoder.output_shape())


        # create diffusion model
        assert obs_as_global_cond
        input_dim = action_dim
        global_cond_dim = obs_feature_dim

        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.normalizer = LinearNormalizer()
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.action_horizon = action_horizon # used for training
        self.obs_as_global_cond = obs_as_global_cond
        self.input_pertub = input_pertub
        self.inpaint_fixed_action_prefix = inpaint_fixed_action_prefix
        self.train_diffusion_n_samples = int(train_diffusion_n_samples)
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        # AWR finetuning state. ``last_awr_info`` carries per-batch
        # diagnostic scalars (weight stats, fraction of zero-advantage
        # samples, etc.) that the workspace pulls into wandb. Set to
        # None when AWR is disabled or the batch had no advantage info.
        self.awr_beta = float(awr_beta)
        self.awr_w_max = float(awr_w_max)
        self.awr_lambda = float(awr_lambda)
        self.awr_normalize_weights = bool(awr_normalize_weights)
        self.last_awr_info: Dict[str, float] = {}

    # ========= inference  ============
    def conditional_sample(self, 
            condition_data,
            condition_mask,
            local_cond=None,
            global_cond=None,
            generator=None,
            # keyword arguments to scheduler.step
            **kwargs
        ):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
    
        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(trajectory, t, 
                local_cond=local_cond, global_cond=global_cond)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample
        
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]        

        return trajectory


    def predict_action(self, obs_dict: Dict[str, torch.Tensor], fixed_action_prefix: torch.Tensor=None) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        fixed_action_prefix: unnormalized action prefix
        result: must include "action" key
        """
        assert 'past_action' not in obs_dict # not implemented yet
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        B = next(iter(nobs.values())).shape[0]

        # condition through global feature
        global_cond = self.obs_encoder(nobs)

        # empty data for action
        cond_data = torch.zeros(size=(B, self.action_horizon, self.action_dim), device=self.device, dtype=self.dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        if fixed_action_prefix is not None and self.inpaint_fixed_action_prefix:
            n_fixed_steps = fixed_action_prefix.shape[1]
            cond_data[:, :n_fixed_steps] = fixed_action_prefix
            cond_mask[:, :n_fixed_steps] = True
            cond_data = self.normalizer['action'].normalize(cond_data)


        # run sampling
        nsample = self.conditional_sample(
            condition_data=cond_data, 
            condition_mask=cond_mask,
            local_cond=None,
            global_cond=global_cond,
            **self.kwargs)
        
        # unnormalize prediction
        assert nsample.shape == (B, self.action_horizon, self.action_dim)
        action_pred = self.normalizer['action'].unnormalize(nsample)
        
        result = {
            'action': action_pred,
            'action_pred': action_pred
        }
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])

        # AWR per-sample inputs. Pulled here (BEFORE the optional
        # train_diffusion_n_samples expansion below) so we can repeat
        # them in lockstep with the action / obs duplication. Both
        # default to None when the batch doesn't carry advantage info,
        # which is also the path taken when awr_beta <= 0.
        advantage = batch.get('advantage', None)
        has_advantage = batch.get('has_advantage', None)
        awr_active = (
            self.awr_beta > 0.0
            and advantage is not None
            and has_advantage is not None
        )

        assert self.obs_as_global_cond
        global_cond = self.obs_encoder(nobs)

        # train on multiple diffusion samples per obs
        if self.train_diffusion_n_samples != 1:
            # repeat obs features and actions multiple times along the batch dimension
            # each sample will later have a different noise sample, effecty training 
            # more diffusion steps per each obs encoder forward pass
            global_cond = torch.repeat_interleave(global_cond, 
                repeats=self.train_diffusion_n_samples, dim=0)
            nactions = torch.repeat_interleave(nactions, 
                repeats=self.train_diffusion_n_samples, dim=0)
            if awr_active:
                advantage = torch.repeat_interleave(
                    advantage, repeats=self.train_diffusion_n_samples, dim=0)
                has_advantage = torch.repeat_interleave(
                    has_advantage, repeats=self.train_diffusion_n_samples, dim=0)

        trajectory = nactions
        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        # input perturbation by adding additonal noise to alleviate exposure bias
        # reference: https://github.com/forever208/DDPM-IP
        noise_new = noise + self.input_pertub * torch.randn(trajectory.shape, device=trajectory.device)

        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (nactions.shape[0],), device=trajectory.device
        ).long()

        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise_new, timesteps)
        
        # Predict the noise residual
        pred = self.model(
            noisy_trajectory,
            timesteps, 
            local_cond=None,
            global_cond=global_cond
        )

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss.type(loss.dtype)
        # Reduce to a single scalar per sample (B,). We keep this
        # un-meaned so AWR can apply its per-sample weight cleanly.
        loss_per_sample = reduce(loss, 'b ... -> b', 'mean')

        if not awr_active:
            # Legacy BC path: byte-identical to the pre-AWR loss
            # (mean over the per-sample MSE scalar is mathematically
            # equivalent to mean over the full (b, h, a) tensor).
            self.last_awr_info = {}
            return loss_per_sample.mean()

        # ===== AWR weighting =====
        # advantage / has_advantage are detached float tensors from the
        # data loader, but be paranoid about dtype/device alignment in
        # case a future loader change supplies them in fp16 / cpu.
        adv = advantage.detach().to(
            device=loss_per_sample.device, dtype=loss_per_sample.dtype)
        m = has_advantage.detach().to(
            device=loss_per_sample.device, dtype=loss_per_sample.dtype)

        # Numerically safe exp: clamp the exponent so a single sample
        # with a very large advantage can't blow up to inf. We clamp
        # in *log space* against awr_w_max so the post-clamp weight is
        # exactly the same as `clamp(exp(...), max=awr_w_max)` but we
        # never materialize the inf intermediate.
        log_w_cap = float(np.log(self.awr_w_max))
        exponent = (adv / self.awr_beta).clamp(max=log_w_cap)
        w_awr = torch.exp(exponent)              # (B,), in [0, awr_w_max]

        # Replace AWR weight with 1.0 (i.e. plain BC) for samples that
        # were flagged as not having a valid IQL anchor by the dataset.
        # This keeps gradient information flowing through those samples
        # rather than zeroing them out, which would shrink the
        # effective batch size on partially-annotated datasets.
        w = torch.where(m > 0.5, w_awr, torch.ones_like(w_awr))

        # Optional per-batch normalization. Without it, the absolute
        # magnitude of the loss (and therefore the effective learning
        # rate) drifts with awr_beta and with the empirical advantage
        # distribution of the current batch. We intentionally normalize
        # over the FULL batch (including the BC-fallback samples) so
        # the relative weighting between AWR-eligible and ineligible
        # samples is preserved.
        if self.awr_normalize_weights:
            w = w / (w.mean() + 1e-8)

        # BC↔AWR linear blend. lambda=1 = pure AWR; lambda=0 collapses
        # to plain BC (useful as a sanity baseline at config flip).
        eff_w = (1.0 - self.awr_lambda) * torch.ones_like(w) \
                + self.awr_lambda * w

        loss = (loss_per_sample * eff_w).mean()

        # Per-batch diagnostics. Stashed on the module so the workspace
        # can pull them into wandb without changing compute_loss's
        # contract (still returns a single scalar).
        with torch.no_grad():
            valid = m > 0.5
            n_valid = valid.sum().clamp_min(1)
            adv_valid_mean = (adv * valid).sum() / n_valid
            adv_valid_max = adv[valid].max() if valid.any() else torch.tensor(
                float('nan'), device=adv.device)
            adv_valid_min = adv[valid].min() if valid.any() else torch.tensor(
                float('nan'), device=adv.device)
            self.last_awr_info = {
                'awr/active': 1.0,
                'awr/beta': float(self.awr_beta),
                'awr/lambda': float(self.awr_lambda),
                'awr/w_mean': float(w.mean()),
                'awr/w_std': float(w.std()),
                'awr/w_min': float(w.min()),
                'awr/w_max_seen': float(w.max()),
                'awr/eff_w_mean': float(eff_w.mean()),
                'awr/eff_w_std': float(eff_w.std()),
                'awr/frac_clipped': float(
                    (exponent >= log_w_cap - 1e-6).float().mean()),
                'awr/frac_has_adv': float(valid.float().mean()),
                'awr/adv_valid_mean': float(adv_valid_mean),
                'awr/adv_valid_min': float(adv_valid_min),
                'awr/adv_valid_max': float(adv_valid_max),
            }

        return loss

    def forward(self, batch):
        return self.compute_loss(batch)