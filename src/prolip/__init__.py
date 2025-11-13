""" ProLIP
Copyright (c) 2024-present NAVER Cloud Corp.
MIT license
"""
from .version import __version__

from .constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
from .factory import create_model, create_model_and_transforms, get_tokenizer, create_loss
from .factory import list_models, add_model_config, get_model_config, load_checkpoint
from .clip_loss import ClipLoss
from .loss import ProLIPLoss
from .model import ProLIP, ProLIPHF, ProLIPTextCfg, ProLIPVisionCfg, \
    convert_weights_to_lp, convert_weights_to_fp16, get_cast_dtype, get_input_dtype, \
    get_model_tokenize_cfg, get_model_preprocess_cfg, set_model_preprocess_cfg
from .clip_model import CLIP, CustomTextCLIP, CLIPTextCfg, CLIPVisionCfg, trace_model
from .tokenizer import HFTokenizer, canonicalize_text
from .transform import image_transform, AugmentationCfg
from .zero_shot_classifier import build_zero_shot_classifier, build_zero_shot_classifier_with_stds
from .zero_shot_metadata import OPENAI_IMAGENET_TEMPLATES, SIMPLE_IMAGENET_TEMPLATES, IMAGENET_CLASSNAMES