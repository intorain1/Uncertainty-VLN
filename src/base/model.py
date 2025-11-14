""" ProLIP
Copyright (c) 2024-present NAVER Cloud Corp.
MIT license

Reference code: https://github.com/mlfoundations/open_clip/blob/v2.24.0/src/open_clip/model.py
Adapted from https://github.com/openai/CLIP. Originally MIT License, Copyright (c) 2021 OpenAI.
"""
import copy
import math
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union, List

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from functools import partial

try:
    from huggingface_hub import PyTorchModelHubMixin
except ImportError:
    import warnings
    warnings.warn("Failed to import `huggingface_hub`. `ProLIPHF.from_pretrained` might not work.")

from .transformer import LayerNormFp32, LayerNorm, QuickGELU, Attention, VisionTransformer, TextTransformer, TSTransformer
from .utils import to_2tuple

@dataclass
class TScfg:
    width: int = 768
    layers: int = 4
    heads: int = 8
    mlp_ratio: float = 4.0
    ls_init_value: Optional[float] = None  # layer scale initial value
    pool_type: str = 'mean'  # 'cls', 'mean', 'max'
    act_kwargs: Optional[dict] = None
    norm_kwargs: Optional[dict] = None

@dataclass
class ProLIPVisionCfg:
    layers: Union[Tuple[int, int, int, int], int] = 12
    width: int = 768
    head_width: int = 64
    mlp_ratio: float = 4.0
    patch_size: int = 16
    image_size: Union[Tuple[int, int], int] = 224

    ls_init_value: Optional[float] = None  # layer scale initial value
    attentional_pool: bool = False  # whether to use attentional pooler in the last embedding layer (overrides pool_type)
    attn_pooler_queries: int = 256  # n_queries for attentional pooler
    attn_pooler_heads: int = 8  # n heads for attentional_pooling
    no_ln_pre: bool = False  # disable pre transformer LayerNorm
    pos_embed_type: str = 'learnable'
    pool_type: str = 'tok'
    output_tokens: bool = False
    act_kwargs: Optional[dict] = None
    norm_kwargs: Optional[dict] = None
    embed_unc: bool = False
    init_unc_bias: float = -10

    timm_model_name: Optional[str] = None  # a valid model name overrides layers, width, patch_size
    timm_model_pretrained: bool = False  # use (imagenet) pretrained weights for named model
    timm_pool: str = 'avg'  # feature pooling for timm model ('abs_attn', 'rot_attn', 'avg', '')
    timm_proj: str = 'linear'  # linear projection for timm model output ('linear', 'mlp', '')
    timm_proj_bias: bool = False  # enable bias final projection
    timm_drop: float = 0.  # head dropout
    timm_drop_path: Optional[float] = None  # backbone stochastic depth


@dataclass
class ProLIPTextCfg:
    context_length: int = 64
    vocab_size: int = 32100
    hf_tokenizer_name: Optional[str] = None
    tokenizer_kwargs: Optional[dict] = None

    width: int = 768
    heads: int = 12
    layers: int = 12
    mlp_ratio: float = 4.0
    ls_init_value: Optional[float] = None  # layer scale initial value
    embed_cls: bool = True
    embed_unc: bool = True
    pad_id: int = 0
    no_causal_mask: bool = True  # disable causal masking
    final_ln_after_pool: bool = False  # apply final LayerNorm after pooling
    pool_type: str = 'argmax'
    proj_bias: bool = True
    output_tokens: bool = False
    act_kwargs: dict = None
    norm_kwargs: dict = None
    init_unc_bias: float = -10

    # HuggingFace specific text tower config
    hf_model_name: Optional[str] = None
    hf_model_pretrained: bool = True
    hf_proj_type: str = 'mlp'
    hf_pooler_type: str = 'mean_pooler'  # attentional pooling for HF models


def get_cast_dtype(precision: str):
    cast_dtype = None
    if precision == 'bf16':
        cast_dtype = torch.bfloat16
    elif precision == 'fp16':
        cast_dtype = torch.float16
    return cast_dtype


def get_input_dtype(precision: str):
    input_dtype = None
    if precision in ('bf16', 'pure_bf16'):
        input_dtype = torch.bfloat16
    elif precision in ('fp16', 'pure_fp16'):
        input_dtype = torch.float16
    return input_dtype


def _build_vision_tower(
        embed_dim: int,
        vision_cfg: ProLIPVisionCfg,
        quick_gelu: bool = False,
        cast_dtype: Optional[torch.dtype] = None
):
    if isinstance(vision_cfg, dict):
        vision_cfg = ProLIPVisionCfg(**vision_cfg)

    # OpenAI models are pretrained w/ QuickGELU but native nn.GELU is both faster and more
    # memory efficient in recent PyTorch releases (>= 1.10).
    # NOTE: timm models always use native GELU regardless of quick_gelu flag.
    act_layer = QuickGELU if quick_gelu else nn.GELU

    vision_heads = vision_cfg.width // vision_cfg.head_width
    norm_layer = LayerNormFp32 if cast_dtype in (torch.float16, torch.bfloat16) else LayerNorm
    if vision_cfg.norm_kwargs:
        norm_layer = partial(norm_layer, **vision_cfg.norm_kwargs)
    if vision_cfg.act_kwargs is not None:
        act_layer = partial(act_layer, **vision_cfg.act_kwargs)

    visual = VisionTransformer(
        image_size=vision_cfg.image_size,
        patch_size=vision_cfg.patch_size,
        width=vision_cfg.width,
        layers=vision_cfg.layers,
        heads=vision_heads,
        mlp_ratio=vision_cfg.mlp_ratio,
        ls_init_value=vision_cfg.ls_init_value,
        attentional_pool=vision_cfg.attentional_pool,
        attn_pooler_queries=vision_cfg.attn_pooler_queries,
        attn_pooler_heads=vision_cfg.attn_pooler_heads,
        pos_embed_type=vision_cfg.pos_embed_type,
        no_ln_pre=vision_cfg.no_ln_pre,
        pool_type=vision_cfg.pool_type,
        output_tokens=vision_cfg.output_tokens,
        output_dim=embed_dim,
        act_layer=act_layer,
        norm_layer=norm_layer,
        embed_unc=vision_cfg.embed_unc,
        init_unc_bias=vision_cfg.init_unc_bias,
    )

    return visual


def _build_text_tower(
        embed_dim: int,
        text_cfg: ProLIPTextCfg,
        quick_gelu: bool = False,
        cast_dtype: Optional[torch.dtype] = None,
):
    if isinstance(text_cfg, dict):
        text_cfg = ProLIPTextCfg(**text_cfg)

    act_layer = QuickGELU if quick_gelu else nn.GELU
    norm_layer = LayerNormFp32 if cast_dtype in (torch.float16, torch.bfloat16) else LayerNorm
    if text_cfg.norm_kwargs:
        norm_layer = partial(norm_layer, **text_cfg.norm_kwargs)
    if text_cfg.act_kwargs is not None:
        act_layer = partial(act_layer, **text_cfg.act_kwargs)

    text = TextTransformer(
        context_length=text_cfg.context_length,
        vocab_size=text_cfg.vocab_size,
        width=text_cfg.width,
        heads=text_cfg.heads,
        layers=text_cfg.layers,
        mlp_ratio=text_cfg.mlp_ratio,
        ls_init_value=text_cfg.ls_init_value,
        output_dim=embed_dim,
        embed_cls=text_cfg.embed_cls,
        embed_unc=text_cfg.embed_unc,
        no_causal_mask=text_cfg.no_causal_mask,
        pad_id=text_cfg.pad_id,
        pool_type=text_cfg.pool_type,
        proj_bias=text_cfg.proj_bias,
        output_tokens=text_cfg.output_tokens,
        act_layer=act_layer,
        norm_layer=norm_layer,
        init_unc_bias=text_cfg.init_unc_bias,
    )
    return text

def _build_TS_tower(
        embeddim: int,
        ts_cfg: TScfg,
):
    if isinstance(ts_cfg, dict):
        ts_cfg = TScfg(**ts_cfg)

    norm_layer = LayerNormFp32 if ts_cfg.norm_kwargs in (torch.float16, torch.bfloat16) else LayerNorm
    if ts_cfg.norm_kwargs:
        norm_layer = partial(norm_layer, **ts_cfg.norm_kwargs)
    act_layer = QuickGELU
    if ts_cfg.act_kwargs is not None:
        act_layer = partial(act_layer, **ts_cfg.act_kwargs)

    transformer = TSTransformer(
        input_dim=embeddim,
        width=ts_cfg.width,
        layers=ts_cfg.layers,
        heads=ts_cfg.heads,
        mlp_ratio=ts_cfg.mlp_ratio,
        ls_init_value=ts_cfg.ls_init_value,
        pool_type=ts_cfg.pool_type,
        act_layer=act_layer,
        norm_layer=norm_layer,
    )
    return transformer

class ProLIP(nn.Module):
    output_dict: torch.jit.Final[bool]

    def __init__(
            self,
            embed_dim: int,
            vision_cfg: ProLIPVisionCfg,
            text_cfg: ProLIPTextCfg,
            ts_cfg: TScfg,
            quick_gelu: bool = False,
            init_logit_scale: float = np.log(1 / 0.07),
            init_logit_bias: Optional[float] = None,
            cast_dtype: Optional[torch.dtype] = None,
            output_dict: bool = True,
    ):
        super().__init__()
        self.output_dict = output_dict
        self.visual = _build_vision_tower(embed_dim, vision_cfg, quick_gelu, cast_dtype)
        self.text = _build_text_tower(embed_dim, text_cfg, quick_gelu, cast_dtype)

        self.ts_mean = _build_TS_tower(embed_dim, ts_cfg)
        self.ts_std = _build_TS_tower(embed_dim, ts_cfg)

        self.context_length = self.text.context_length
        self.vocab_size = self.text.vocab_size
        self.logit_scale = None
        self.logit_bias = None
        if init_logit_scale:
            self.logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale)
        if init_logit_bias:
            self.logit_bias = nn.Parameter(torch.ones([]) * init_logit_bias)

    def lock_image_tower(self, unlocked_groups=0, freeze_bn_stats=False):
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        self.visual.lock(unlocked_groups=unlocked_groups, freeze_bn_stats=freeze_bn_stats)

    def lock_text_tower(self, unlocked_layers: int = 0, freeze_layer_norm: bool = True):
        self.text.lock(unlocked_layers, freeze_layer_norm)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable)
        self.text.set_grad_checkpointing(enable)

    def encode_image(self, image, normalize: bool = False, image_mask_ratio: float = None):
        features = self.visual(image, image_mask_ratio=image_mask_ratio)
        return {
            "mean": F.normalize(features["mean"], dim=-1) if normalize else features["mean"],
            "std": features.get("std")
        }

    def encode_text(self, text, normalize: bool = False):
        features = self.text(text)
        return {
            "mean": F.normalize(features["mean"], dim=-1) if normalize else features["mean"],
            "std": features.get("std")
        }
    
    def encode_seq_image(self, image_seq: torch.Tensor, mask: torch.Tensor = None, normalize: bool = False, image_mask_ratio: float = None):
        batch_size, seq_len, C, H, W = image_seq.shape
        image_seq = image_seq.view(batch_size * seq_len, C, H, W)
        features = self.visual(image_seq, image_mask_ratio=image_mask_ratio)
        mean = features["mean"].view(batch_size, seq_len, -1)
        std = features['std'].view(batch_size, seq_len, -1) if 'std' in features else None
        ts_features_mean = self.ts_mean(mean, attn_mask=mask)
        ts_features_std = self.ts_std(std, attn_mask=mask) if std is not None else None

        return {
            "mean": F.normalize(ts_features_mean, dim=-1) if normalize else ts_features_mean,
            "std": ts_features_std
        }

    def get_logits(self, image_seq, text, mask):
        image_features = self.encode_seq_image(image_seq, mask=mask, normalize=True)
        text_features = self.encode_text(text, normalize=True)
        image_logits = self.logit_scale.exp() * image_features @ text_features.T
        if self.logit_bias is not None:
            image_logits += self.logit_bias
        text_logits = image_logits.T
        return image_logits, text_logits

    def forward(
            self,
            image_seq: Optional[torch.Tensor] = None,
            mask: Optional[torch.Tensor] = None,
            text: Optional[torch.Tensor] = None,
            image_mask_ratio: Optional[float] = None,
    ):
        image_features = self.encode_seq_image(image_seq, mask=mask, normalize=True, image_mask_ratio=image_mask_ratio) if image_seq is not None else None
        text_features = self.encode_text(text, normalize=True) if text is not None else None

        if self.output_dict:
            out_dict = {
                "image_features": image_features,
                "text_features": text_features,
            }
            if self.logit_scale is not None:
                out_dict["logit_scale"] = self.logit_scale.exp()
            if self.logit_bias is not None:
                out_dict['logit_bias'] = self.logit_bias
            return out_dict

        if self.logit_bias is not None:
            return image_features, text_features, self.logit_scale.exp(), self.logit_bias
        return image_features, text_features, self.logit_scale.exp()


class ProLIPHF(ProLIP, PyTorchModelHubMixin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


def convert_weights_to_lp(model: nn.Module, dtype=torch.float16):
    """Convert applicable model parameters to low-precision (bf16 or fp16)"""

    def _convert_weights(layer):
        if isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            layer.weight.data = layer.weight.data.to(dtype)
            if layer.bias is not None:
                layer.bias.data = layer.bias.data.to(dtype)

        if isinstance(layer, (nn.MultiheadAttention, Attention)):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(layer, attr)
                if tensor is not None:
                    tensor.data = tensor.data.to(dtype)

        if isinstance(layer, TextTransformer):
            # convert text nn.Parameter projections
            for attr_name in ("text_projection", "text_uncertainty_projection", "cls_emb", "unc_emb"):
                attr = getattr(layer, attr_name, None)
                if attr is not None and isinstance(attr, nn.Parameter):
                    attr.data = attr.data.to(dtype)

        if isinstance(layer, VisionTransformer):
            # convert vision nn.Parameter projections
            attr = getattr(layer, "proj", None)
            if attr is not None:
                attr.data = attr.data.to(dtype)

    model.apply(_convert_weights)


convert_weights_to_fp16 = convert_weights_to_lp  # backwards compat


def get_model_preprocess_cfg(model):
    module = getattr(model, 'visual', model)
    preprocess_cfg = getattr(module, 'preprocess_cfg', {})
    if not preprocess_cfg:
        # use separate legacy attributes if preprocess_cfg dict not found
        size = getattr(module, 'image_size')
        if size is not None:
            preprocess_cfg['size'] = size
        mean = getattr(module, 'image_mean', None)
        if mean is not None:
            preprocess_cfg['mean'] = mean
        std = getattr(module, 'image_std', None)
        if std is not None:
            preprocess_cfg['std'] = std
    return preprocess_cfg


def set_model_preprocess_cfg(model, preprocess_cfg: Dict[str, Any]):
    module = getattr(model, 'visual', model)
    module.image_mean = preprocess_cfg['mean']  # legacy attribute, keeping for bwd compat
    module.image_std = preprocess_cfg['std']  # legacy attribute, keeping for bwd compat
    module.preprocess_cfg = copy.deepcopy(preprocess_cfg)  # new attr, package all pp cfg as dict


def get_model_tokenize_cfg(model):
    module = getattr(model, 'text', model)
    cfg = {}
    context_length = getattr(module, 'context_length', None)
    if context_length is not None:
        cfg['context_length'] = context_length
    vocab_size = getattr(module, 'vocab_size', None)
    if vocab_size is not None:
        cfg['vocab_size'] = vocab_size
    return cfg


# ===================================================================== #
#                      For LongProLIP                                   #
# ===================================================================== #
def resize_prolip_pos_embed(state_dict, model, interpolation: str = 'bicubic', antialias: bool = True):
    # Rescale the grid of position embeddings when loading from state_dict
    old_pos_embed = state_dict.get('visual.positional_embedding', None)
    if old_pos_embed is None or not hasattr(model.visual, 'grid_size'):
        return
    if len(old_pos_embed.size()) == 3:
        old_pos_embed = old_pos_embed[0]
        state_dict['visual.positional_embedding'] = old_pos_embed
    grid_size = to_2tuple(model.visual.grid_size)
    # extra_tokens = 1  # FIXME detect different token configs (ie no class token, or more)
    extra_tokens = 2  # FIXME detect different token configs (ie no class token, or more)
    logging.info(f'Using {extra_tokens=} for `create_model` method. Please manually fix this if it raises an error (See src/open_clip/model.py:L594)')
    new_seq_len = grid_size[0] * grid_size[1] + extra_tokens
    if new_seq_len == old_pos_embed.shape[0]:
        return

    pos_emb_tok, pos_emb_img = model.visual.positional_embedding[:extra_tokens].to(old_pos_embed.device), old_pos_embed

    if len(old_pos_embed) == 197:
        # NOTE patch size 16 with CLS token
        pos_emb_tok[0] = old_pos_embed[0]
        pos_emb_img = old_pos_embed[1:]
    elif len(old_pos_embed) == 257:
        # NOTE patch size 14 with CLS token
        pos_emb_tok[0] = old_pos_embed[0]
        pos_emb_img = old_pos_embed[1:]
    else:
        # NOTE different det pre-trained backbone => change this..
        pos_emb_img = old_pos_embed[2:]
    old_grid_size = to_2tuple(int(math.sqrt(len(pos_emb_img))))

    if grid_size[0] != old_grid_size[0] or grid_size[1] != old_grid_size[1]:
        logging.info('Resizing position embedding grid-size from %s to %s', old_grid_size, grid_size)
        pos_emb_img = pos_emb_img.reshape(1, old_grid_size[0], old_grid_size[1], -1).permute(0, 3, 1, 2)
        pos_emb_img = F.interpolate(
            pos_emb_img,
            size=grid_size,
            mode=interpolation,
            antialias=antialias,
            align_corners=False,
        )
        pos_emb_img = pos_emb_img.permute(0, 2, 3, 1).reshape(1, grid_size[0] * grid_size[1], -1)[0]
    else:
        logging.info('same grid size (%s to %s) skip interpolation', old_grid_size, grid_size)
    if pos_emb_tok is not None:
        new_pos_embed = torch.cat([pos_emb_tok, pos_emb_img], dim=0)
    else:
        new_pos_embed = pos_emb_img
    state_dict['visual.positional_embedding'] = new_pos_embed


def resize_prolip_text_pos_embed(state_dict, model, interpolation: str = 'linear', antialias: bool = False, n_cls: int = 2):
    key = 'positional_embedding'
    old_pos_embed = state_dict.get(key, None)
    if old_pos_embed is None:
        key = 'text.positional_embedding'
        old_pos_embed = state_dict.get(key, None)
    if old_pos_embed is None:
        return
    # FIXME add support for text cls_token
    model_pos_embed = getattr(model, 'positional_embedding', None)
    if model_pos_embed is None:
        model_pos_embed = getattr(model.text, 'positional_embedding', None)

    old_num_pos = old_pos_embed.shape[0]
    old_width = old_pos_embed.shape[1]
    num_pos = model_pos_embed.shape[0]
    width = model_pos_embed.shape[1]
    assert old_width == width, 'text pos_embed width changed!'
    if old_num_pos == num_pos:
        return

    if num_pos > 127:
        logging.info('[LongCLIP-ish] Resizing text position embedding num_pos from %s to %s', old_num_pos, num_pos)
        n_fixed = 20
        if n_cls:
            logging.info(f'[LongCLIP-ish] Keep first {n_fixed} tokens and last {n_cls} tokens (CLS and UNC)')
            # shape: [1, 768, 66] (pos_len + # cls)
            old_pos_embed = old_pos_embed.reshape(1, old_num_pos, old_width).permute(0, 2, 1)
            keep_pos_embed = old_pos_embed[:, :, :n_fixed]  # First 20 tokens
            cls_embed = old_pos_embed[:, :, -n_cls:]  # Last 2 tokens
            old_pos_embed = old_pos_embed[:, :, n_fixed:-n_cls]  # Remaining dimensions

            old_pos_embed = F.interpolate(
                old_pos_embed,
                size=num_pos - n_fixed - n_cls,
                mode=interpolation,
                antialias=antialias,
                align_corners=False,
            )
            new_pos_embed = torch.cat([keep_pos_embed, old_pos_embed, cls_embed], dim=2)
            new_pos_embed = new_pos_embed.permute(0, 2, 1)[0]
        else:
            logging.info(f'[LongCLIP-ish] Keep first {n_fixed} tokens')
            # shape: [1, 768, 64] (pos_len)
            old_pos_embed = old_pos_embed.reshape(1, old_num_pos, old_width).permute(0, 2, 1)
            keep_pos_embed = old_pos_embed[:, :, :n_fixed]  # First 20 tokens
            old_pos_embed = old_pos_embed[:, :, n_fixed:]  # Remaining dimensions

            old_pos_embed = F.interpolate(
                old_pos_embed,
                size=num_pos - n_fixed,
                mode=interpolation,
                antialias=antialias,
                align_corners=False,
            )
            new_pos_embed = torch.cat([keep_pos_embed, old_pos_embed], dim=2)
            new_pos_embed = new_pos_embed.permute(0, 2, 1)[0]
    else:
        logging.info('Resizing text position embedding num_pos from %s to %s', old_num_pos, num_pos)
        old_pos_embed = old_pos_embed.reshape(1, old_num_pos, old_width).permute(0, 2, 1)
        old_pos_embed = F.interpolate(
            old_pos_embed,
            size=num_pos,
            mode=interpolation,
            antialias=antialias,
            align_corners=False,
        )
        old_pos_embed = old_pos_embed.permute(0, 2, 1)[0]
        new_pos_embed = old_pos_embed

    state_dict[key] = new_pos_embed


def resize_prolip_tok_embed(state_dict, model):
    logging.info("Resizing token embeddings")
    old_tok_embed = state_dict.get('text.token_embedding.weight', None)
    try:
        model_tok_embed = model.text.token_embedding.weight
    except AttributeError:
        model_tok_embed = None
    # model_tok_embed = getattr(model.text, 'token_embedding.weight', None)
    if old_tok_embed is None or model_tok_embed is None:
        return
    if old_tok_embed.shape[1] != model_tok_embed.shape[1]:
        raise NotImplementedError
    if old_tok_embed.shape[0] > model_tok_embed.shape[0]:
        raise NotImplementedError
    new_tok_embed = model_tok_embed.clone().to(old_tok_embed.device)
    new_tok_embed[:old_tok_embed.shape[0], :] = old_tok_embed
    state_dict['text.token_embedding.weight'] = new_tok_embed
