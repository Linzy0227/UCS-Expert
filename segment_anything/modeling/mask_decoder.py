# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
"""SAM mask decoder adapted to expose its upscaled image features."""

from typing import List, Tuple, Type

import torch
from torch import nn
from torch.nn import functional as F

from .common import LayerNorm2d


class MaskDecoder(nn.Module):
    """Predict masks from image and prompt embeddings."""

    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer
        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim,
                               transformer_dim // 4,
                               kernel_size=2,
                               stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(
                transformer_dim // 4,
                transformer_dim // 8,
                kernel_size=2,
                stride=2,
            ),
            activation(),
        )
        self.output_hypernetworks_mlps = nn.ModuleList([
            MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
            for _ in range(self.num_mask_tokens)
        ])
        # Kept for compatibility with the original SAM checkpoint.
        self.iou_prediction_head = MLP(
            transformer_dim,
            iou_head_hidden_dim,
            self.num_mask_tokens,
            iou_head_depth,
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        masks, upscaled_embedding = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )
        mask_slice = slice(1, None) if multimask_output else slice(0, 1)
        return masks[:, mask_slice], upscaled_embedding

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        output_tokens = torch.cat(
            [self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(
            sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        if image_embeddings.shape[0] != tokens.shape[0]:
            source = torch.repeat_interleave(image_embeddings,
                                             tokens.shape[0],
                                             dim=0)
        else:
            source = image_embeddings
        source = source + dense_prompt_embeddings
        positional_source = torch.repeat_interleave(image_pe,
                                                    tokens.shape[0],
                                                    dim=0)
        batch_size, channels, height, width = source.shape

        hidden_states, source = self.transformer(source, positional_source,
                                                 tokens)
        mask_tokens_out = hidden_states[:, 1:1 + self.num_mask_tokens]
        source = source.transpose(1, 2).view(batch_size, channels, height,
                                             width)
        upscaled_embedding = self.output_upscaling(source)

        hypernetwork_inputs: List[torch.Tensor] = []
        for index in range(self.num_mask_tokens):
            hypernetwork_inputs.append(self.output_hypernetworks_mlps[index](
                mask_tokens_out[:, index]))
        hypernetwork_inputs = torch.stack(hypernetwork_inputs, dim=1)
        batch_size, channels, height, width = upscaled_embedding.shape
        masks = (hypernetwork_inputs @ upscaled_embedding.view(
            batch_size, channels, height * width)).view(
                batch_size, -1, height, width)
        return masks, upscaled_embedding


class MLP(nn.Module):

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        hidden = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(in_features, out_features)
            for in_features, out_features in zip([input_dim] + hidden, hidden +
                                                 [output_dim]))
        self.sigmoid_output = sigmoid_output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for index, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if index < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = torch.sigmoid(x)
        return x
