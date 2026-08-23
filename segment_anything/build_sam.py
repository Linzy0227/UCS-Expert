# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
"""Build SAM backbones at the input resolution used by UCS-Expert."""

from functools import partial
from pathlib import Path
from typing import Iterable, Optional

import torch
import torch.nn.functional as F

from .modeling import ImageEncoderViT, MaskDecoder, PromptEncoder, Sam, TwoWayTransformer


def build_sam_vit_h(image_size: int,
                    checkpoint: Optional[str] = None,
                    **_) -> Sam:
    return _build_sam(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=(7, 15, 23, 31),
        checkpoint=checkpoint,
        image_size=image_size,
    )


build_sam = build_sam_vit_h


def build_sam_vit_l(image_size: int,
                    checkpoint: Optional[str] = None,
                    **_) -> Sam:
    return _build_sam(
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=(5, 11, 17, 23),
        checkpoint=checkpoint,
        image_size=image_size,
    )


def build_sam_vit_b(image_size: int,
                    checkpoint: Optional[str] = None,
                    **_) -> Sam:
    return _build_sam(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=(2, 5, 8, 11),
        checkpoint=checkpoint,
        image_size=image_size,
    )


sam_model_registry = {
    "default": build_sam_vit_h,
    "vit_h": build_sam_vit_h,
    "vit_l": build_sam_vit_l,
    "vit_b": build_sam_vit_b,
}


def _build_sam(
    encoder_embed_dim: int,
    encoder_depth: int,
    encoder_num_heads: int,
    encoder_global_attn_indexes: Iterable[int],
    image_size: int,
    checkpoint: Optional[str] = None,
) -> Sam:
    prompt_embed_dim = 256
    patch_size = 16
    image_embedding_size = image_size // patch_size
    global_attention = tuple(encoder_global_attn_indexes)
    sam = Sam(
        image_encoder=ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=global_attention,
            window_size=14,
            out_chans=prompt_embed_dim,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    sam.eval()

    if checkpoint is not None:
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"SAM checkpoint not found: {checkpoint_path}. "
                "Download it as described in README.md.")
        state_dict = torch.load(checkpoint_path,
                                map_location="cpu",
                                weights_only=True)
        resized_state = resize_sam_state_dict(
            sam,
            state_dict,
            image_size=image_size,
            patch_size=patch_size,
            global_attention=global_attention,
        )
        sam.load_state_dict(resized_state)
    return sam


def resize_sam_state_dict(
    sam: Sam,
    state_dict,
    image_size: int,
    patch_size: int,
    global_attention: Iterable[int],
):
    """Resize SAM positional parameters when using a non-1024 input size."""
    sam_state = sam.state_dict()
    loaded = {
        key: value
        for key, value in state_dict.items() if key in sam_state
    }
    token_size = image_size // patch_size

    pos_embed = loaded["image_encoder.pos_embed"]
    if pos_embed.shape[1] != token_size:
        pos_embed = pos_embed.permute(0, 3, 1, 2)
        pos_embed = F.interpolate(
            pos_embed,
            (token_size, token_size),
            mode="bilinear",
            align_corners=False,
        )
        loaded["image_encoder.pos_embed"] = pos_embed.permute(0, 2, 3, 1)

        for block_index in global_attention:
            for axis in ("h", "w"):
                key = f"image_encoder.blocks.{block_index}.attn.rel_pos_{axis}"
                relative_position = loaded[key]
                relative_position = relative_position.transpose(0,
                                                                1).unsqueeze(0)
                relative_position = F.interpolate(
                    relative_position,
                    size=2 * token_size - 1,
                    mode="linear",
                    align_corners=False,
                )
                loaded[key] = relative_position.squeeze(0).transpose(0, 1)

    sam_state.update(loaded)
    return sam_state
