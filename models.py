# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
from timm.models.registry import register_model

from visiontransformer import _create_vision_transformer


__all__ = [
    'su_vit_tiny_patch16_224', 'su_vit_small_patch16_224', 'su_vit_base_patch16_224'
    ]


# -------------------------------------------------------------
# ViT for Lifelong Learning prototype models

@register_model
def su_vit_tiny_patch16_224(pretrained=True, base_keep_rate=0.7, drop_loc=(3, 6, 9), **kwargs):
    """ ViT-Tiny (Vit-Ti/16)
    """
    keep_rate = [1] * 12
    for loc in drop_loc:
        keep_rate[loc] = base_keep_rate
    model_kwargs = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3, keep_rate=keep_rate, **kwargs)
    model_kwargs.update(kwargs)
    model = _create_vision_transformer('vit_tiny_patch16_224', pretrained=pretrained, **model_kwargs)
    return model


@register_model
def su_vit_small_patch16_224(pretrained=True, base_keep_rate=0.7, drop_loc=(3, 6, 9), **kwargs):
    """ ViT-Small (ViT-S/16)
    NOTE I've replaced my previous 'small' model definition and weights with the small variant from the DeiT paper
    """
    keep_rate = [1] * 12
    for loc in drop_loc:
        keep_rate[loc] = base_keep_rate
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, keep_rate=keep_rate, **kwargs)
    model_kwargs.update(kwargs)
    model = _create_vision_transformer('vit_small_patch16_224', pretrained=pretrained, **model_kwargs)
    return model

@register_model
def su_vit_base_patch16_224(pretrained=True, base_keep_rate=0.7, drop_loc=(3, 6, 9), **kwargs):
    """ ViT-Base (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 224x224, source https://github.com/google-research/vision_transformer.
    """
    keep_rate = [1] * 12
    for loc in drop_loc:
        keep_rate[loc] = base_keep_rate
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model_kwargs.update(kwargs)
    model = _create_vision_transformer('vit_base_patch16_224', pretrained=pretrained, keep_rate=keep_rate, **model_kwargs)
    return model