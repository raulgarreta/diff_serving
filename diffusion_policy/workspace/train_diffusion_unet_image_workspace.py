# Stringified annotations: lets us keep type hints like
# `self.model: DiffusionUnetImagePolicy` without paying the import cost of
# `DiffusionUnetImagePolicy` (and friends) at module load. They'll be
# imported lazily inside `run()` where the runtime values actually matter.
# Required for the lazy-import block below to be safe across the file.
from __future__ import annotations

if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
import copy
import random
import numpy as np
from diffusion_policy.workspace.base_workspace import BaseWorkspace

# NOTE: the following imports were previously at module top-level. They
# are required only by `run()` (training), not by `__init__()` or the
# inference-time accessors. Importing them at module top adds several
# seconds to cold-start of the policy server (notably `wandb` and the
# transitive `accelerate` / dataset / env_runner stack), so we defer
# them. The `if False:` block is kept to make the migration discoverable
# to static analyzers / IDEs: it documents what used to be here but is
# never executed at runtime.
if False:  # pragma: no cover - tooling hint only
    from torch.utils.data import DataLoader  # noqa: F401
    import wandb  # noqa: F401
    import pickle  # noqa: F401
    import tqdm  # noqa: F401
    import shutil  # noqa: F401
    from diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy  # noqa: F401
    from diffusion_policy.dataset.base_dataset import BaseImageDataset, BaseDataset  # noqa: F401
    from diffusion_policy.env_runner.base_image_runner import BaseImageRunner  # noqa: F401
    from diffusion_policy.common.checkpoint_util import TopKCheckpointManager  # noqa: F401
    from diffusion_policy.common.json_logger import JsonLogger  # noqa: F401
    from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to  # noqa: F401
    from diffusion_policy.model.diffusion.ema_model import EMAModel  # noqa: F401
    from diffusion_policy.model.common.lr_scheduler import get_scheduler  # noqa: F401
    from accelerate import Accelerator  # noqa: F401

OmegaConf.register_new_resolver("eval", eval, replace=True)


@torch.no_grad()
def _awr_metrics(policy) -> dict:
    """Pass-through of the policy's per-batch AWR diagnostics.

    ``DiffusionUnetTimmPolicy.compute_loss`` populates
    ``last_awr_info`` after every forward pass when AWR is active. We
    just forward those scalars to wandb, prefixed under ``awr/``. When
    AWR is off this returns an empty dict (so the metrics never appear
    on the BC training run's wandb panel and don't pollute resumes).
    """
    info = getattr(policy, 'last_awr_info', None)
    if not info:
        return {}
    return dict(info)


@torch.no_grad()
def _bbox_mask_weight_metrics(policy) -> dict:
    """Snapshot of patch_embed.proj weight norms for bbox-mask-channel rgb keys.

    Returns one block of metrics per rgb key that has a bbox mask routed to
    it (see TimmObsEncoder.bbox_mask_map). For each key we log:

      * ``bbox_mask/<rgb_key>/w_norm_mask_ch``: L2 norm of the 4th input-
        channel slice of patch_embed.proj. Exactly 0 at training step 0
        (zero-init); growing > 0 proves the optimizer is using the mask.
      * ``bbox_mask/<rgb_key>/w_norm_rgb_mean``: mean L2 norm of the three
        RGB input-channel slices. Stable reference scale to compare against.
      * ``bbox_mask/<rgb_key>/w_ratio``: ratio of the two (mask norm /
        mean rgb norm). A "healthy" trained value is roughly 0.05 - 1.0;
        values close to 0 mean the optimizer declined to use the mask
        (almost certainly because the training data didn't reward it).

    Returns an empty dict when no bbox is routed via the mask channel
    (e.g. ``bbox_mode: low_dim`` runs), so it's safe to call unconditionally.
    """
    encoder = getattr(policy, 'obs_encoder', None)
    if encoder is None:
        return {}
    mask_map = getattr(encoder, 'bbox_mask_map', None)
    if not mask_map:
        return {}

    metrics = {}
    for rgb_key in mask_map.keys():
        # nn.ModuleDict doesn't expose .get(); use __contains__ + []
        if rgb_key not in encoder.key_model_map:
            continue
        model = encoder.key_model_map[rgb_key]
        if not hasattr(model, 'patch_embed') \
                or not hasattr(model.patch_embed, 'proj'):
            continue
        weight = model.patch_embed.proj.weight  # (out_ch, in_ch=4, k, k)
        if weight.shape[1] < 4:
            continue
        rgb_norms = [weight[:, c].norm().item() for c in range(3)]
        mask_norm = weight[:, 3].norm().item()
        rgb_mean = sum(rgb_norms) / 3.0
        ratio = (mask_norm / rgb_mean) if rgb_mean > 0 else 0.0
        prefix = f'bbox_mask/{rgb_key}'
        metrics[f'{prefix}/w_norm_mask_ch'] = mask_norm
        metrics[f'{prefix}/w_norm_rgb_mean'] = rgb_mean
        metrics[f'{prefix}/w_ratio'] = ratio
    return metrics


class TrainDiffusionUnetImageWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']
    exclude_keys = tuple()

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: DiffusionUnetImagePolicy = hydra.utils.instantiate(cfg.policy)

        self.ema_model: DiffusionUnetImagePolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # configure training state
        # self.optimizer = hydra.utils.instantiate(
        #     cfg.optimizer, params=self.model.parameters())

        obs_encorder_lr = cfg.optimizer.lr
        if cfg.policy.obs_encoder.pretrained:
            obs_encorder_lr *= 0.1
            print('==> reduce pretrained obs_encorder\'s lr')
        obs_encorder_params = list()
        for param in self.model.obs_encoder.parameters():
            if param.requires_grad:
                obs_encorder_params.append(param)
        print(f'obs_encorder params: {len(obs_encorder_params)}')
        param_groups = [
            {'params': self.model.model.parameters()},
            {'params': obs_encorder_params, 'lr': obs_encorder_lr}
        ]
        # self.optimizer = hydra.utils.instantiate(
        #     cfg.optimizer, params=param_groups)
        optimizer_cfg = OmegaConf.to_container(cfg.optimizer, resolve=True)
        optimizer_cfg.pop('_target_')
        self.optimizer = torch.optim.AdamW(
            params=param_groups,
            **optimizer_cfg
        )

        # configure training state
        self.global_step = 0
        self.epoch = 0

        # do not save optimizer if resume=False
        if not cfg.training.resume:
            self.exclude_keys = ['optimizer']

    def run(self):
        # Lazy imports: these are training-only dependencies. Keeping them
        # out of module-top imports saves several seconds of cold start when
        # the workspace class is loaded purely for inference (e.g. by
        # policy_server.py -> hydra.utils.get_class). The first call to
        # run() pays the import cost; the imports are then cached in
        # sys.modules so subsequent calls are free.
        from torch.utils.data import DataLoader
        import wandb
        import pickle
        import tqdm
        import shutil
        from diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy
        from diffusion_policy.dataset.base_dataset import BaseImageDataset, BaseDataset
        from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
        from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
        from diffusion_policy.common.json_logger import JsonLogger
        from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
        from diffusion_policy.model.diffusion.ema_model import EMAModel
        from diffusion_policy.model.common.lr_scheduler import get_scheduler
        from accelerate import Accelerator, DistributedDataParallelKwargs

        cfg = copy.deepcopy(self.cfg)

        # When the visual encoder is frozen (typical for AWR finetuning),
        # its parameters don't receive gradients. The default DDP wrap
        # then fails with "Expected to have finished reduction in the
        # prior iteration..." because DDP's reducer expects every
        # registered parameter to participate in backward. The fix is
        # to construct DDP with find_unused_parameters=True so it
        # tolerates the no-grad params; we gate this on freeze_encoder
        # to avoid the (small) per-step graph-traversal overhead in
        # plain BC training where everything trains.
        ddp_kwargs = DistributedDataParallelKwargs(
            find_unused_parameters=bool(
                cfg.training.get('freeze_encoder', False)
            )
        )
        accelerator = Accelerator(
            log_with='wandb',
            kwargs_handlers=[ddp_kwargs],
        )
        wandb_cfg = OmegaConf.to_container(cfg.logging, resolve=True)
        wandb_cfg.pop('project')
        # Drop tags that resolved to empty strings (e.g. real_size/sim_size
        # when the dataset filename didn't specify them) and clamp to
        # wandb's 64-char tag limit so a long dataset-name tag doesn't
        # crash `wandb.init` with HTTP 400.
        if isinstance(wandb_cfg.get('tags'), list):
            cleaned = []
            for t in wandb_cfg['tags']:
                if not t:
                    continue
                if isinstance(t, str) and len(t) > 64:
                    accelerator.print(
                        f"[wandb] truncating tag to 64 chars: {t!r} -> {t[:64]!r}"
                    )
                    t = t[:64]
                cleaned.append(t)
            wandb_cfg['tags'] = cleaned
        accelerator.init_trackers(
            project_name=cfg.logging.project,
            config=OmegaConf.to_container(cfg, resolve=True),
            init_kwargs={"wandb": wandb_cfg}
        )

        # resume training
        if cfg.training.resume:
            # Support custom checkpoint path via config, or use latest from output_dir
            if hasattr(cfg.training, 'checkpoint_path') and cfg.training.checkpoint_path:
                ckpt_path = pathlib.Path(cfg.training.checkpoint_path)
                if ckpt_path.is_file():
                    accelerator.print(f"Resuming from checkpoint {ckpt_path}")
                    self.load_checkpoint(path=ckpt_path)
                else:
                    accelerator.print(f"Warning: Checkpoint path {ckpt_path} not found, skipping resume")
            else:
                lastest_ckpt_path = self.get_checkpoint_path()
                if lastest_ckpt_path.is_file():
                    accelerator.print(f"Resuming from checkpoint {lastest_ckpt_path}")
                    self.load_checkpoint(path=lastest_ckpt_path)
        elif (
            hasattr(cfg.training, 'init_from_ckpt')
            and cfg.training.init_from_ckpt
        ):
            # Fresh run that *initializes* model weights from another
            # checkpoint without resuming optimizer / scheduler state.
            # Used for AWR finetuning of a pre-trained BC policy:
            # we want the BC weights as a starting point but a fresh
            # optimizer (so the LR schedule restarts cleanly), fresh
            # global_step / epoch counters, and (importantly) the *new*
            # cfg's policy hyperparameters in self.model -- including
            # awr_beta etc.
            init_path = pathlib.Path(cfg.training.init_from_ckpt)
            if init_path.is_file():
                accelerator.print(
                    f"Initializing model weights from {init_path} "
                    f"(optimizer + step counters NOT loaded)"
                )
                # Skip optimizer state -- the new run uses a fresh
                # optimizer with possibly-different LR. include_keys=()
                # skips global_step / epoch pickles too. The AWR
                # config knobs are plain Python attributes (set in
                # __init__ from the new cfg, not stored in state_dict),
                # so a strict load of the BC ckpt's `model` and
                # `ema_model` state_dicts produces no key mismatch.
                self.load_checkpoint(
                    path=init_path,
                    exclude_keys=['optimizer'],
                    include_keys=(),
                )
                # Force a clean restart of the schedule even though
                # global_step/epoch were already 0 from __init__.
                self.global_step = 0
                self.epoch = 0
            else:
                accelerator.print(
                    f"Warning: init_from_ckpt={init_path} not found; "
                    f"training will start from random / pretrained weights."
                )

        # configure dataset
        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset) or isinstance(dataset, BaseDataset)
        train_dataloader = DataLoader(dataset, **cfg.dataloader)

        # compute normalizer on the main process and save to disk
        normalizer_path = os.path.join(self.output_dir, 'normalizer.pkl')
        if accelerator.is_main_process:
            normalizer = dataset.get_normalizer()
            pickle.dump(normalizer, open(normalizer_path, 'wb'))

        # load normalizer on all processes
        accelerator.wait_for_everyone()
        normalizer = pickle.load(open(normalizer_path, 'rb'))

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)
        print('train dataset:', len(dataset), 'train dataloader:', len(train_dataloader))
        print('val dataset:', len(val_dataset), 'val dataloader:', len(val_dataloader))

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # Set initial_lr for each param group if resuming (required by LR scheduler)
        if cfg.training.resume and self.global_step > 0:
            for param_group in self.optimizer.param_groups:
                if 'initial_lr' not in param_group:
                    param_group['initial_lr'] = param_group['lr']

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs) \
                    // cfg.training.gradient_accumulate_every,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step-1
        )

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)

        # configure env
        env_runner: BaseImageRunner
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=self.output_dir)
        assert isinstance(env_runner, BaseImageRunner)

        # # configure logging
        # wandb_run = wandb.init(
        #     dir=str(self.output_dir),
        #     config=OmegaConf.to_container(cfg, resolve=True),
        #     **cfg.logging
        # )
        # wandb.config.update(
        #     {
        #         "output_dir": self.output_dir,
        #     }
        # )

        # configure checkpoint
        # Build constant fields used in checkpoint filenames from the experiment
        # config (task name, model type, visual encoder, total batch size,
        # dataset file name without extension). These are optional in cfg, so
        # missing values fall back to 'na' to avoid breaking other workspaces
        # that share this manager.
        dataset_path = cfg.task.get('dataset_path', '') if hasattr(cfg, 'task') else ''
        dataset_file_name = os.path.basename(str(dataset_path))
        # Strip common archive/zarr extensions (e.g. .zarr.zip, .zarr, .zip).
        while True:
            stem, ext = os.path.splitext(dataset_file_name)
            if ext.lower() in ('.zip', '.zarr', '.gz', '.tar'):
                dataset_file_name = stem
            else:
                break
        if not dataset_file_name:
            dataset_file_name = 'na'

        total_batch = int(cfg.get('batch_size', 0)) * int(cfg.get('num_gpus', 1))

        static_format_kwargs = {
            'task_name': cfg.get('task_name', 'na'),
            'model_type': cfg.get('model_type', 'na'),
            'visual_encoder': cfg.get('visual_encoder', 'na'),
            'total_batch': total_batch,
            'dataset_file_name': dataset_file_name,
        }

        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            static_format_kwargs=static_format_kwargs,
            **cfg.checkpoint.topk
        )

        # device transfer
        # device = torch.device(cfg.training.device)
        # self.model.to(device)
        # if self.ema_model is not None:
        #     self.ema_model.to(device)
        # optimizer_to(self.optimizer, device)

        # accelerator
        train_dataloader, val_dataloader, self.model, self.optimizer, lr_scheduler = accelerator.prepare(
            train_dataloader, val_dataloader, self.model, self.optimizer, lr_scheduler
        )
        device = self.model.device
        if self.ema_model is not None:
            self.ema_model.to(device)

        # save batch for sampling
        train_sampling_batch = None

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                self.model.train()

                step_log = dict()
                # ========= train for this epoch ==========
                if cfg.training.freeze_encoder:
                    # ``self.model`` is a DDP wrapper after
                    # accelerator.prepare(), so ``obs_encoder`` lives on
                    # the inner module. Use accelerator.unwrap_model so
                    # this works in both single-GPU (no wrapper) and
                    # multi-GPU DDP. eval() + requires_grad_(False)
                    # mutate state in place; DDP forwards the
                    # frozen-grad signal through the wrapper.
                    _unwrapped = accelerator.unwrap_model(self.model)
                    _unwrapped.obs_encoder.eval()
                    _unwrapped.obs_encoder.requires_grad_(False)

                train_losses = list()
                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
                        leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        # device transfer
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        
                        # always use the latest batch
                        train_sampling_batch = batch

                        # compute loss
                        raw_loss = self.model(batch)
                        loss = raw_loss / cfg.training.gradient_accumulate_every
                        loss.backward()

                        # step optimizer
                        if self.global_step % cfg.training.gradient_accumulate_every == 0:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()
                        
                        # update ema
                        if cfg.training.use_ema:
                            ema.step(accelerator.unwrap_model(self.model))

                        # logging
                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            'train_loss': raw_loss_cpu,
                            'global_step': self.global_step,
                            'epoch': self.epoch,
                            'lr': lr_scheduler.get_last_lr()[0]
                        }

                        # Log bbox mask channel weight norms. Sample frequently
                        # at the start so the early growth (from zero) is
                        # visible on the wandb curve, then less often once
                        # training is settled. Cheap (a few tensor norms) but
                        # we still gate by step to avoid spamming.
                        if accelerator.is_main_process:
                            log_every = (10 if self.global_step < 500 else 50)
                            if (self.global_step % log_every) == 0:
                                _unwrapped = accelerator.unwrap_model(self.model)
                                step_log.update(_bbox_mask_weight_metrics(_unwrapped))
                                step_log.update(_awr_metrics(_unwrapped))

                        is_last_batch = (batch_idx == (len(train_dataloader)-1))
                        if not is_last_batch:
                            # log of last step is combined with validation and rollout
                            accelerator.log(step_log, step=self.global_step)
                            json_logger.log(step_log)
                            self.global_step += 1

                        if (cfg.training.max_train_steps is not None) \
                            and batch_idx >= (cfg.training.max_train_steps-1):
                            break

                # at the end of each epoch
                # replace train_loss with epoch average
                train_loss = np.mean(train_losses)
                step_log['train_loss'] = train_loss

                # End-of-epoch snapshot of the mask channel weights. Always
                # emitted (no throttling) so the wandb x-axis "epoch" has a
                # clean per-epoch curve to chart alongside train_loss.
                if accelerator.is_main_process:
                    _unwrapped = accelerator.unwrap_model(self.model)
                    step_log.update(_bbox_mask_weight_metrics(_unwrapped))
                    step_log.update(_awr_metrics(_unwrapped))

                # ========= eval for this epoch ==========
                policy = accelerator.unwrap_model(self.model)
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()

                # run rollout
                if (self.epoch % cfg.training.rollout_every) == 0:
                    runner_log = env_runner.run(policy)
                    # log all
                    step_log.update(runner_log)

                # run validation
                # Runs on main rank only (other ranks idle through here, then
                # rejoin at accelerator.log below). Uses the unwrapped policy
                # (EMA if enabled) so it reflects what we'd actually deploy.
                if (self.epoch % cfg.training.val_every) == 0 and len(val_dataloader) > 0 and accelerator.is_main_process:
                    with torch.no_grad():
                        val_losses = list()
                        with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}",
                                leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                                # policy is already the unwrapped (EMA if
                                # enabled) module — its forward() returns the
                                # same diffusion loss the train step uses.
                                loss = policy(batch)
                                val_losses.append(loss.item())
                                if (cfg.training.max_val_steps is not None) \
                                    and batch_idx >= (cfg.training.max_val_steps-1):
                                    break
                        if len(val_losses) > 0:
                            step_log['val_loss'] = float(np.mean(val_losses))
                
                def log_action_mse(step_log, category, pred_action, gt_action):
                    B, T, _ = pred_action.shape
                    pred_action = pred_action.view(B, T, -1, 10)
                    gt_action = gt_action.view(B, T, -1, 10)
                    step_log[f'{category}_action_mse_error'] = torch.nn.functional.mse_loss(pred_action, gt_action)
                    step_log[f'{category}_action_mse_error_pos'] = torch.nn.functional.mse_loss(pred_action[..., :3], gt_action[..., :3])
                    step_log[f'{category}_action_mse_error_rot'] = torch.nn.functional.mse_loss(pred_action[..., 3:9], gt_action[..., 3:9])
                    step_log[f'{category}_action_mse_error_width'] = torch.nn.functional.mse_loss(pred_action[..., 9], gt_action[..., 9])
                # run diffusion sampling on a training batch
                if (self.epoch % cfg.training.sample_every) == 0 and accelerator.is_main_process:
                    with torch.no_grad():
                        # sample trajectory from training set, and evaluate difference
                        batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
                        gt_action = batch['action']
                        pred_action = policy.predict_action(batch['obs'], None)['action_pred']
                        log_action_mse(step_log, 'train', pred_action, gt_action)

                        if len(val_dataloader) > 0:
                            val_sampling_batch = next(iter(val_dataloader))
                            batch = dict_apply(val_sampling_batch, lambda x: x.to(device, non_blocking=True))
                            gt_action = batch['action']
                            pred_action = policy.predict_action(batch['obs'], None)['action_pred']
                            log_action_mse(step_log, 'val', pred_action, gt_action)

                        del batch
                        del gt_action
                        del pred_action
                
                # checkpoint
                if (self.epoch % cfg.training.checkpoint_every) == 0 and accelerator.is_main_process:
                    # unwrap the model to save ckpt
                    model_ddp = self.model
                    self.model = accelerator.unwrap_model(self.model)

                    # checkpointing
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()

                    # sanitize metric names
                    metric_dict = dict()
                    for key, value in step_log.items():
                        new_key = key.replace('/', '_')
                        metric_dict[new_key] = value
                    
                    # We can't copy the last checkpoint here
                    # since save_checkpoint uses threads.
                    # therefore at this point the file might have been empty!
                    topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

                    if topk_ckpt_path is not None:
                        self.save_checkpoint(path=topk_ckpt_path)

                    # recover the DDP model
                    self.model = model_ddp
                # ========= eval end for this epoch ==========
                # end of epoch
                # log of last step is combined with validation and rollout
                accelerator.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1

        accelerator.end_training()

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainDiffusionUnetImageWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
