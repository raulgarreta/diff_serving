import copy

import timm
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import logging

from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin

from diffusion_policy.common.pytorch_util import replace_submodules

logger = logging.getLogger(__name__)


def _expand_vit_patch_embed_to_4ch(model: nn.Module) -> None:
    """Replace ``model.patch_embed.proj`` with a 4-input-channel conv.

    The first three input-channel slices are copied verbatim from the
    pretrained conv so the RGB path is bit-identical at initialization;
    the new 4th input-channel slice is zero-initialized so the modified
    network's output equals the original network's output on RGB-only
    inputs at step 0. The optimizer then grows the 4th-channel weights
    only when the mask carries gradient-reducing information.

    Implements the "zero-init weight expansion" / identity-preserving-at-init
    recipe used to bolt new modalities onto pretrained backbones.
    """
    if not (hasattr(model, "patch_embed") and hasattr(model.patch_embed, "proj")):
        raise RuntimeError(
            "bbox_mode=mask_channel requires a ViT-style encoder with "
            "`patch_embed.proj` (a Conv2d). Got a model of type "
            f"{type(model).__name__} without that structure."
        )
    old_proj: nn.Conv2d = model.patch_embed.proj
    if not isinstance(old_proj, nn.Conv2d):
        raise RuntimeError(
            f"patch_embed.proj is not a Conv2d (got {type(old_proj).__name__})."
        )
    if old_proj.in_channels != 3:
        raise RuntimeError(
            "Expected pretrained patch_embed.proj.in_channels == 3, got "
            f"{old_proj.in_channels}. Refusing to overwrite — the encoder "
            "may already have been surgically modified."
        )

    new_proj = nn.Conv2d(
        in_channels=4,
        out_channels=old_proj.out_channels,
        kernel_size=old_proj.kernel_size,
        stride=old_proj.stride,
        padding=old_proj.padding,
        dilation=old_proj.dilation,
        groups=old_proj.groups,
        bias=(old_proj.bias is not None),
    )
    with torch.no_grad():
        new_proj.weight[:, :3].copy_(old_proj.weight)
        new_proj.weight[:, 3:].zero_()
        if old_proj.bias is not None:
            new_proj.bias.copy_(old_proj.bias)
    model.patch_embed.proj = new_proj


def rasterize_bbox_to_mask(bbox: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """Rasterize pixel-space bboxes to a binary mask.

    Args:
        bbox: tensor of shape ``(..., 4)`` with ``[x0, y0, x1, y1]`` in
            pixel coordinates of the ``H x W`` image.
        H: target mask height in pixels.
        W: target mask width in pixels.

    Returns:
        Float tensor of shape ``(..., 1, H, W)`` with values in ``{0, 1}``.
        Pixels strictly inside ``[x0, x1) x [y0, y1)`` are 1.
    """
    leading_shape = bbox.shape[:-1]
    flat = bbox.reshape(-1, 4)
    N = flat.shape[0]

    device = flat.device
    # Use float32 for the comparison grid regardless of bbox dtype so we
    # don't lose precision when the encoder is running under bf16/fp16.
    xs = torch.arange(W, device=device, dtype=torch.float32).view(1, 1, 1, W)
    ys = torch.arange(H, device=device, dtype=torch.float32).view(1, 1, H, 1)

    flat_f = flat.to(torch.float32)
    x0 = flat_f[:, 0].view(N, 1, 1, 1)
    y0 = flat_f[:, 1].view(N, 1, 1, 1)
    x1 = flat_f[:, 2].view(N, 1, 1, 1)
    y1 = flat_f[:, 3].view(N, 1, 1, 1)

    in_x = (xs >= x0) & (xs < x1)
    in_y = (ys >= y0) & (ys < y1)
    mask = (in_x & in_y).to(bbox.dtype)
    return mask.reshape(*leading_shape, 1, H, W)


class MaskAwareImageTransform(nn.Module):
    """Apply geometric transforms to all channels, color transforms only to RGB.

    Used when the input image has been augmented with extra non-RGB channels
    (e.g. a binary bbox mask). RandomCrop / Resize keep the mask aligned with
    the image; ColorJitter / Normalize would corrupt the mask if applied to
    all channels, so they are restricted to the first three (RGB) channels.
    """

    def __init__(self, geometric: list, rgb_only: list, n_rgb: int = 3):
        super().__init__()
        self.geometric = nn.Sequential(*geometric) if geometric else nn.Identity()
        self.rgb_only = nn.Sequential(*rgb_only) if rgb_only else nn.Identity()
        self.n_rgb = n_rgb

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.geometric(x)
        if x.shape[1] <= self.n_rgb:
            return self.rgb_only(x)
        rgb = self.rgb_only(x[:, : self.n_rgb])
        rest = x[:, self.n_rgb:]
        return torch.cat([rgb, rest], dim=1)

class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)
    

class TimmObsEncoder(ModuleAttrMixin):
    def __init__(self,
            shape_meta: dict,
            model_name: str,
            pretrained: bool,
            frozen: bool,
            global_pool: str,
            transforms: list,
            # replace BatchNorm with GroupNorm
            use_group_norm: bool=False,
            # use single rgb model for all rgb inputs
            share_rgb_model: bool=False,
            # renormalize rgb input with imagenet normalization
            # assuming input in [0,1]
            imagenet_norm: bool=False,
            feature_aggregation: str='spatial_embedding',
            downsample_ratio: int=32,
            position_encording: str='learnable',

        ):
        """
        Assumes rgb input: B,T,C,H,W
        Assumes low_dim input: B,T,D
        """
        super().__init__()
        
        rgb_keys = list()
        low_dim_keys = list()
        key_model_map = nn.ModuleDict()
        key_transform_map = nn.ModuleDict()
        key_shape_map = dict()

        assert global_pool == ''
        model = timm.create_model(
            model_name=model_name,
            pretrained=pretrained,
            global_pool=global_pool, # '' means no pooling
            num_classes=0            # remove classification layer
        )

        if frozen:
            assert pretrained
            for param in model.parameters():
                param.requires_grad = False
        
        feature_dim = None
        if model_name.startswith('resnet'):
            # the last layer is nn.Identity() because num_classes is 0
            # second last layer is AdaptivePool2d, which is also identity because global_pool is empty
            if downsample_ratio == 32:
                modules = list(model.children())[:-2]
                model = torch.nn.Sequential(*modules)
                feature_dim = 512
            elif downsample_ratio == 16:
                modules = list(model.children())[:-3]
                model = torch.nn.Sequential(*modules)
                feature_dim = 256
            else:
                raise NotImplementedError(f"Unsupported downsample_ratio: {downsample_ratio}")
        elif model_name.startswith('convnext'):
            # the last layer is nn.Identity() because num_classes is 0
            # second last layer is AdaptivePool2d, which is also identity because global_pool is empty
            if downsample_ratio == 32:
                modules = list(model.children())[:-2]
                model = torch.nn.Sequential(*modules)
                feature_dim = 1024
            else:
                raise NotImplementedError(f"Unsupported downsample_ratio: {downsample_ratio}")

        if use_group_norm and not pretrained:
            model = replace_submodules(
                root_module=model,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=(x.num_features // 16) if (x.num_features % 16 == 0) else (x.num_features // 8), 
                    num_channels=x.num_features)
            )
        
        image_shape = None
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                assert image_shape is None or image_shape == shape[1:]
                image_shape = shape[1:]

        # ---- bbox -> mask-channel routing -----------------------------------
        # Scan low_dim obs entries for `bbox_mode == 'mask_channel'`. Each such
        # entry is rasterized into a binary mask and concatenated as a 4th
        # channel onto a target RGB key inside forward(). The bbox itself is
        # removed from the low-dim concat path so it doesn't get fed twice.
        # `bbox_mask_map`: rgb_key -> bbox_key (the bbox routed onto that rgb).
        bbox_mask_map = dict()
        bbox_mask_horizon = dict()  # rgb_key -> expected matching horizon
        for key, attr in obs_shape_meta.items():
            if attr.get('type', 'low_dim') != 'low_dim':
                continue
            if attr.get('bbox_mode', 'low_dim') != 'mask_channel':
                continue
            shape = tuple(attr['shape'])
            if shape != (4,):
                raise RuntimeError(
                    f"bbox_mode=mask_channel requires shape [4] (x0,y0,x1,y1) "
                    f"for key '{key}', got {shape}.")
            target_rgb_key = attr.get('target_rgb_key')
            if target_rgb_key is None:
                rgb_candidates = [
                    k for k, a in obs_shape_meta.items()
                    if a.get('type', 'low_dim') == 'rgb']
                if len(rgb_candidates) != 1:
                    raise RuntimeError(
                        f"bbox key '{key}' with bbox_mode=mask_channel did not "
                        f"specify `target_rgb_key` and there are "
                        f"{len(rgb_candidates)} candidate rgb keys; please set "
                        f"`target_rgb_key` explicitly in shape_meta.")
                target_rgb_key = rgb_candidates[0]
            if target_rgb_key not in obs_shape_meta or \
                    obs_shape_meta[target_rgb_key].get('type') != 'rgb':
                raise RuntimeError(
                    f"bbox key '{key}' targets rgb_key='{target_rgb_key}' "
                    f"which is not an rgb obs entry.")
            if target_rgb_key in bbox_mask_map:
                raise RuntimeError(
                    f"Multiple bbox keys are routed to rgb_key="
                    f"'{target_rgb_key}'. Only one bbox mask channel per "
                    f"image is supported.")
            rgb_horizon = int(obs_shape_meta[target_rgb_key]['horizon'])
            bbox_horizon = int(attr['horizon'])
            if bbox_horizon != rgb_horizon:
                raise RuntimeError(
                    f"bbox '{key}' (horizon={bbox_horizon}) and target image "
                    f"'{target_rgb_key}' (horizon={rgb_horizon}) must share "
                    f"the same horizon when bbox_mode=mask_channel.")
            bbox_mask_map[target_rgb_key] = key
            bbox_mask_horizon[target_rgb_key] = rgb_horizon

        if transforms is not None and not isinstance(transforms[0], torch.nn.Module):
            assert transforms[0].type == 'RandomCrop'
            ratio = transforms[0].ratio
            transforms = [
                torchvision.transforms.RandomCrop(size=int(image_shape[0] * ratio)),
                torchvision.transforms.Resize(size=image_shape[0], antialias=True)
            ] + transforms[1:]
        # The two leading transforms (RandomCrop, Resize) are geometric and
        # are safe on arbitrary channel counts; everything after them is RGB-
        # only (ColorJitter uses HSV, ImageNet Normalize assumes 3 channels).
        geometric_transforms = transforms[:2] if transforms is not None else []
        color_transforms = transforms[2:] if transforms is not None else []
        rgb_only_transform = (
            nn.Identity() if transforms is None
            else torch.nn.Sequential(*transforms)
        )

        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            type = attr.get('type', 'low_dim')
            key_shape_map[key] = shape
            if type == 'rgb':
                rgb_keys.append(key)

                this_model = model if share_rgb_model else copy.deepcopy(model)
                if key in bbox_mask_map:
                    # Expand the patch embedding to accept the mask channel.
                    # Zero-init the new slice so the model output is identical
                    # to the pretrained model at step 0 (see header docstring).
                    if share_rgb_model:
                        raise RuntimeError(
                            "bbox_mode=mask_channel is incompatible with "
                            "share_rgb_model=True (the patch_embed surgery "
                            "would be applied multiple times to the same "
                            "tensor). Disable share_rgb_model or unset the "
                            "mask routing for this key.")
                    _expand_vit_patch_embed_to_4ch(this_model)
                    this_transform = MaskAwareImageTransform(
                        geometric=geometric_transforms,
                        rgb_only=color_transforms,
                    )
                else:
                    this_transform = rgb_only_transform
                key_model_map[key] = this_model
                key_transform_map[key] = this_transform
            elif type == 'low_dim':
                if attr.get('ignore_by_policy', False):
                    continue
                if attr.get('bbox_mode', 'low_dim') == 'mask_channel':
                    # consumed via the rgb path, not as a flat feature.
                    continue
                low_dim_keys.append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")
        
        feature_map_shape = [x // downsample_ratio for x in image_shape]
            
        rgb_keys = sorted(rgb_keys)
        low_dim_keys = sorted(low_dim_keys)
        print('rgb keys:         ', rgb_keys)
        print('low_dim_keys keys:', low_dim_keys)

        self.model_name = model_name
        self.shape_meta = shape_meta
        self.key_model_map = key_model_map
        self.key_transform_map = key_transform_map
        self.share_rgb_model = share_rgb_model
        self.rgb_keys = rgb_keys
        self.low_dim_keys = low_dim_keys
        self.key_shape_map = key_shape_map
        self.feature_aggregation = feature_aggregation
        # rgb_key -> bbox_key. Empty dict means no mask routing is active and
        # forward() behaves exactly as before. Persisted on the encoder so
        # downstream code (eval, loggers, debug tools) can introspect the
        # bbox integration mode without re-parsing shape_meta.
        self.bbox_mask_map = bbox_mask_map
        if model_name.startswith('vit'):
            # assert self.feature_aggregation is None # vit uses the CLS token
            if self.feature_aggregation == 'all_tokens':
                # Use all tokens from ViT
                pass
            elif self.feature_aggregation is not None:
                logger.warn(f'vit will use the CLS token. feature_aggregation ({self.feature_aggregation}) is ignored!')
                self.feature_aggregation = None
        
        if self.feature_aggregation == 'soft_attention':
            self.attention = nn.Sequential(
                nn.Linear(feature_dim, 1, bias=False),
                nn.Softmax(dim=1)
            )
        elif self.feature_aggregation == 'spatial_embedding':
            self.spatial_embedding = torch.nn.Parameter(torch.randn(feature_map_shape[0] * feature_map_shape[1], feature_dim))
        elif self.feature_aggregation == 'transformer':
            if position_encording == 'learnable':
                self.position_embedding = torch.nn.Parameter(torch.randn(feature_map_shape[0] * feature_map_shape[1] + 1, feature_dim))
            elif position_encording == 'sinusoidal':
                num_features = feature_map_shape[0] * feature_map_shape[1] + 1
                self.position_embedding = torch.zeros(num_features, feature_dim)
                position = torch.arange(0, num_features, dtype=torch.float).unsqueeze(1)
                div_term = torch.exp(torch.arange(0, feature_dim, 2).float() * (-math.log(2 * num_features) / feature_dim))
                self.position_embedding[:, 0::2] = torch.sin(position * div_term)
                self.position_embedding[:, 1::2] = torch.cos(position * div_term)
            self.aggregation_transformer = nn.TransformerEncoder(
                encoder_layer=nn.TransformerEncoderLayer(d_model=feature_dim, nhead=4),
                num_layers=4)
        elif self.feature_aggregation == 'attention_pool_2d':
            self.attention_pool_2d = AttentionPool2d(
                spacial_dim=feature_map_shape[0],
                embed_dim=feature_dim,
                num_heads=feature_dim // 64,
                output_dim=feature_dim
            )
        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def aggregate_feature(self, feature):
        if self.model_name.startswith('vit'):
            assert self.feature_aggregation is None # vit uses the CLS token
            return feature[:, 0, :]
        
        # resnet
        assert len(feature.shape) == 4
        if self.feature_aggregation == 'attention_pool_2d':
            return self.attention_pool_2d(feature)

        feature = torch.flatten(feature, start_dim=-2) # B, 512, 7*7
        feature = torch.transpose(feature, 1, 2) # B, 7*7, 512

        if self.feature_aggregation == 'avg':
            return torch.mean(feature, dim=[1])
        elif self.feature_aggregation == 'max':
            return torch.amax(feature, dim=[1])
        elif self.feature_aggregation == 'soft_attention':
            weight = self.attention(feature)
            return torch.sum(feature * weight, dim=1)
        elif self.feature_aggregation == 'spatial_embedding':
            return torch.mean(feature * self.spatial_embedding, dim=1)
        elif self.feature_aggregation == 'transformer':
            zero_feature = torch.zeros(feature.shape[0], 1, feature.shape[-1], device=feature.device)
            if self.position_embedding.device != feature.device:
                self.position_embedding = self.position_embedding.to(feature.device)
            feature_with_pos_embedding = torch.concat([zero_feature, feature], dim=1) + self.position_embedding
            feature_output = self.aggregation_transformer(feature_with_pos_embedding)
            return feature_output[:, 0]
        else:
            assert self.feature_aggregation is None
            return feature
        
    def forward(self, obs_dict):
        features = list()
        batch_size = next(iter(obs_dict.values())).shape[0]
        
        # process rgb input
        for key in self.rgb_keys:
            img = obs_dict[key]
            B, T = img.shape[:2]
            assert B == batch_size
            assert img.shape[2:] == self.key_shape_map[key]

            if key in self.bbox_mask_map:
                # Rasterize the routed bbox into a (B, T, 1, H, W) binary
                # mask and concat as a 4th channel. Geometric transforms in
                # MaskAwareImageTransform then crop/resize image+mask
                # together, keeping the bbox geometry consistent with the
                # image content the ViT sees.
                bbox_key = self.bbox_mask_map[key]
                bbox = obs_dict[bbox_key]
                assert bbox.shape[0] == batch_size, (
                    f"bbox '{bbox_key}' batch={bbox.shape[0]} mismatches "
                    f"image '{key}' batch={batch_size}")
                assert bbox.shape[1] == T, (
                    f"bbox '{bbox_key}' horizon={bbox.shape[1]} mismatches "
                    f"image '{key}' horizon={T}")
                assert bbox.shape[2:] == (4,), (
                    f"bbox '{bbox_key}' expected shape [4], got "
                    f"{tuple(bbox.shape[2:])}")
                _, _, _, H, W = img.shape
                mask = rasterize_bbox_to_mask(bbox, H=H, W=W).to(img.dtype)
                img = torch.cat([img, mask], dim=2)

            img = img.reshape(B*T, *img.shape[2:])
            img = self.key_transform_map[key](img)
            raw_feature = self.key_model_map[key](img)
            feature = self.aggregate_feature(raw_feature)
            assert len(feature.shape) == 2 and feature.shape[0] == B * T
            features.append(feature.reshape(B, -1))

        # process lowdim input
        for key in self.low_dim_keys:
            data = obs_dict[key]
            B, T = data.shape[:2]
            assert B == batch_size
            assert data.shape[2:] == self.key_shape_map[key]
            features.append(data.reshape(B, -1))
        
        # concatenate all features
        result = torch.cat(features, dim=-1)

        return result
    

    @torch.no_grad()
    def output_shape(self):
        example_obs_dict = dict()
        obs_shape_meta = self.shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            this_obs = torch.zeros(
                (1, attr['horizon']) + shape, 
                dtype=self.dtype,
                device=self.device)
            example_obs_dict[key] = this_obs
        example_output = self.forward(example_obs_dict)
        assert len(example_output.shape) == 2
        assert example_output.shape[0] == 1
        
        return example_output.shape


if __name__=='__main__':
    timm_obs_encoder = TimmObsEncoder(
        shape_meta=None,
        model_name='resnet18.a1_in1k',
        pretrained=False,
        global_pool='',
        transforms=None
    )
