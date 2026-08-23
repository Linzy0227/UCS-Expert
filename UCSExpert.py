"""UCS-Expert model components."""

from functools import partial
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from segment_anything.modeling import Sam
from segment_anything.modeling.common import LayerNorm2d


def _init_weights(module, scheme=''):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d):
        if scheme == 'normal':
            nn.init.normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'trunc_normal':
            nn.init.trunc_normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'xavier_normal':
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'kaiming_normal':
            nn.init.kaiming_normal_(module.weight,
                                    mode='fan_out',
                                    nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        else:
            # efficientnet like
            fan_out = module.kernel_size[0] * module.kernel_size[
                1] * module.out_channels
            fan_out //= module.groups
            nn.init.normal_(module.weight, 0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d) or isinstance(
            module, nn.BatchNorm3d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)


class BoundaryAwareSFEM(nn.Module):

    def __init__(self, pool_size=3):
        super().__init__()
        self.pool_size = pool_size
        self.padding = (pool_size - 1) // 2

    def compute_boundary_masks(self, mask):
        max_pool = F.max_pool2d(mask,
                                self.pool_size,
                                stride=1,
                                padding=self.padding)
        min_pool = -F.max_pool2d(-mask, self.pool_size, 1, self.padding)
        edge_mask = max_pool - min_pool
        return edge_mask

    def masked_avg_pool(self, feat, mask):
        if mask.shape[-2:] != feat.shape[-2:]:
            mask = F.interpolate(mask,
                                 size=feat.shape[-2:],
                                 mode='bilinear',
                                 align_corners=False)

        masked_feat = feat * mask
        sum_feat = torch.sum(masked_feat, dim=(2, 3), keepdim=True)
        sum_mask = torch.sum(mask, dim=(2, 3), keepdim=True) + 1e-6
        return sum_feat / sum_mask

    def cosine_similarity(self, x, y):
        x_norm = F.normalize(x, p=2, dim=1)  # [B, C, 1, 1]
        y_norm = F.normalize(y, p=2, dim=1)  # [B, C, H, W]
        similarity = torch.sum(x_norm * y_norm, dim=1,
                               keepdim=True)  # [B, 1, H, W]
        return similarity

    def forward(self, input_feature, mask):
        batch_size, _, height, width = input_feature.shape
        mask = torch.sigmoid(mask)

        edge_mask = self.compute_boundary_masks(mask)
        edge_prototype = self.masked_avg_pool(input_feature, edge_mask)
        similarity = self.cosine_similarity(edge_prototype, input_feature)
        edge_attention = F.softmax(similarity.view(batch_size, -1),
                                   dim=1).view(batch_size, 1, height, width)
        edge_feature = edge_attention * edge_prototype
        return input_feature + edge_feature


class PGModule(nn.Module):

    def __init__(self, dim, depth=3):
        super().__init__()
        self.in_dim = dim
        self.hidden_dim = dim
        self.width = depth
        self.in_conv = nn.Sequential(nn.Conv2d(dim, dim, 1, bias=False),
                                     LayerNorm2d(dim), nn.GELU())

        self.pool = nn.AvgPool2d(3, stride=1, padding=1)

        self.mid_conv = nn.ModuleList()
        self.edge_enhance = nn.ModuleList()
        for _ in range(depth - 1):
            self.mid_conv.append(
                nn.Sequential(nn.Conv2d(dim, dim, 1, bias=False),
                              LayerNorm2d(dim), nn.GELU()))
            self.edge_enhance.append(BoundaryAwareSFEM())

        self.out_conv = nn.Sequential(
            nn.Conv2d(dim * depth, dim, 1, bias=False), LayerNorm2d(dim),
            nn.GELU())
        self.init_weights('kaiming_normal')

    def init_weights(self, scheme=''):
        self.apply(partial(_init_weights, scheme=scheme))

    def forward(self, x, coarse_mask):
        out = x
        x = self.in_conv(x)
        for i in range(self.width - 1):
            mid = self.pool(x)
            mid = self.mid_conv[i](mid)
            out = torch.cat([out, self.edge_enhance[i](mid, coarse_mask)],
                            dim=1)
        out = self.out_conv(out)

        return out


class CAB(nn.Module):

    def __init__(self, in_channels, out_channels=None, ratio=4):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        if self.in_channels < ratio:
            ratio = self.in_channels
        self.reduced_channels = self.in_channels // ratio
        if self.out_channels is None:
            self.out_channels = in_channels

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.activation = nn.GELU()
        self.fc1 = nn.Conv2d(self.in_channels,
                             self.reduced_channels,
                             1,
                             bias=False)
        self.fc2 = nn.Conv2d(self.reduced_channels,
                             self.out_channels,
                             1,
                             bias=False)
        self.conv = nn.Conv2d(2, 1, 7, padding=7 // 2, bias=False)

        self.sigmoid = nn.Sigmoid()

        self.init_weights('normal')

    def init_weights(self, scheme=''):
        self.apply(partial(_init_weights, scheme=scheme))

    def forward(self, x):
        avg_pool_out = self.avg_pool(x)
        avg_out = self.fc2(self.activation(self.fc1(avg_pool_out)))

        max_pool_out = self.max_pool(x)
        max_out = self.fc2(self.activation(self.fc1(max_pool_out)))

        out = avg_out + max_out
        out = x * self.sigmoid(out)

        avg_out = torch.mean(out, dim=1, keepdim=True)
        max_out, _ = torch.max(out, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        out = out * self.sigmoid(x)

        return out


class CSAM(nn.Module):
    """Cross-stage attention module."""

    def __init__(self, vit_type="vit_b"):
        super().__init__()
        transformer_dim = 256
        vit_dim_dict = {"vit_b": 768, "vit_l": 1024, "vit_h": 1280}
        vit_dim = vit_dim_dict[vit_type]
        self.early_depth = 4

        self.compress_vit_feat_list = nn.ModuleList()
        for _ in range(self.early_depth - 1):
            self.compress_vit_feat_list.append(
                nn.Sequential(
                    nn.ConvTranspose2d(vit_dim,
                                       transformer_dim,
                                       kernel_size=2,
                                       stride=2), LayerNorm2d(transformer_dim),
                    nn.GELU(),
                    nn.ConvTranspose2d(transformer_dim,
                                       transformer_dim // 8,
                                       kernel_size=2,
                                       stride=2)))

        self.embedding_encoder = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim,
                               transformer_dim // 4,
                               kernel_size=2,
                               stride=2),
            LayerNorm2d(transformer_dim // 4),
            nn.GELU(),
            nn.ConvTranspose2d(transformer_dim // 4,
                               transformer_dim // 8,
                               kernel_size=2,
                               stride=2),
        )

        self.attn_list = nn.ModuleList()
        for _ in range(self.early_depth):
            self.attn_list.append(CAB(transformer_dim // 8))

        self.init_weights('normal')

    def init_weights(self, scheme=''):
        self.apply(partial(_init_weights, scheme=scheme))

    def forward(self, early_features):

        compress_features = []
        for i in range(len(early_features) - 1):
            compress_feature = self.compress_vit_feat_list[i](
                early_features[i])
            compress_features.append(compress_feature)

        compress_last_feature = self.embedding_encoder(early_features[-1])
        compress_features.append(compress_last_feature)

        for i in range(self.early_depth):
            x = compress_features[i]
            compress_features[i] = self.attn_list[i](x)

        return compress_features[0] + compress_features[1] + compress_features[
            2] + compress_features[3]


class PGFD(nn.Module):
    """Prompt-guided fine decoder."""

    def __init__(self, dim, vit_type):
        super().__init__()

        self.img_in_conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            LayerNorm2d(16), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(16, dim, kernel_size=3, stride=1, padding=1, bias=False),
            LayerNorm2d(dim), nn.GELU(), nn.MaxPool2d(2),
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1,
                      bias=False), LayerNorm2d(dim), nn.GELU())

        self.img_mid_conv = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 3, padding=1, bias=False),
            LayerNorm2d(dim),
            nn.GELU(),
        )
        self.feature_upsample = nn.Sequential(
            nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2),
            LayerNorm2d(dim), nn.GELU(),
            nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2),
            LayerNorm2d(dim), nn.GELU())

        self.pgm_block = PGModule(dim)
        self.csam_block = CSAM(vit_type)

        self.seghead = nn.Sequential(
            nn.Conv2d(dim, dim // 2, 3, padding=1, bias=False),
            LayerNorm2d(dim // 2), nn.GELU(), nn.Conv2d(dim // 2, 1, 1))

        self.init_weights('kaiming_normal')

    def init_weights(self, scheme=''):
        self.apply(partial(_init_weights, scheme=scheme))

    def forward(self, img, input_feature, early_features, coarse_mask):

        img_feature = self.img_in_conv(img)
        img_feature = self.pgm_block(img_feature, coarse_mask) + img_feature

        fused_feature = self.csam_block(early_features)

        combined_feature = torch.cat([input_feature, fused_feature], dim=1)
        combined_feature = self.img_mid_conv(combined_feature) + img_feature
        upsampled_combined_feature = self.feature_upsample(combined_feature)

        output = self.seghead(upsampled_combined_feature)

        return output


class UCSExpert(nn.Module):

    def __init__(self, sam: Sam, vit_type="vit_b"):
        super().__init__()

        # Fine-tune FAMA, the prompt encoder, SAM upscaling, and PGFD.
        for name, param in sam.image_encoder.named_parameters():
            param.requires_grad = "fama" in name
        for param in sam.prompt_encoder.parameters():
            param.requires_grad = True
        for name, param in sam.mask_decoder.named_parameters():
            if "output_upscaling" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.sam = sam

        self.fine_decoder = PGFD(dim=32, vit_type=vit_type)

    def forward(self, img, box_torch=None, original_size=None):
        embedding_list = self.sam.image_encoder(img)

        sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
            points=None, boxes=box_torch, masks=None)

        if original_size is None:
            original_size = img.shape[2:4]

        coarse_mask, feature = self.sam.mask_decoder(
            image_embeddings=embedding_list[-1],
            image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        fine_mask = self.fine_decoder(img, feature, embedding_list,
                                      coarse_mask)
        coarse_mask = F.interpolate(coarse_mask,
                                    original_size,
                                    mode='bilinear',
                                    align_corners=False)

        return [coarse_mask, fine_mask]
