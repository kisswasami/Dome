from typing import Sequence, Tuple, Union

import torch
import torch.nn as nn
from monai.networks.blocks import Convolution
from monai.networks.blocks.dynunet_block import (
    UnetBasicBlock,
    UnetOutBlock,
    UnetResBlock,
    get_conv_layer,
)

from networks.nest_transformer_3D import NestTransformer3D


class DecoderSkipBlock(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int],
        stride: Union[Sequence[int], int],
        upsample_kernel_size: Union[Sequence[int], int],
        norm_name: Union[Tuple, str],
        res_block: bool = False,
    ) -> None:
        super().__init__()
        self.transp_conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_kernel_size,
            conv_only=True,
            is_transposed=True,
        )

        block_cls = UnetResBlock if res_block else UnetBasicBlock
        self.conv_block = block_cls(
            spatial_dims,
            out_channels + out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            norm_name=norm_name,
        )

    def forward(self, inp, skip):
        out = self.transp_conv(inp)
        out = torch.cat((out, skip), dim=1)
        return self.conv_block(out)


class EncoderUpBlock(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        num_layer: int,
        kernel_size: Union[Sequence[int], int],
        stride: Union[Sequence[int], int],
        upsample_kernel_size: Union[Sequence[int], int],
        norm_name: Union[Tuple, str],
        conv_block: bool = False,
        res_block: bool = False,
    ) -> None:
        super().__init__()
        self.transp_conv_init = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_kernel_size,
            conv_only=True,
            is_transposed=True,
        )

        if conv_block:
            block_cls = UnetResBlock if res_block else UnetBasicBlock
            self.blocks = nn.ModuleList(
                [
                    nn.Sequential(
                        get_conv_layer(
                            spatial_dims,
                            out_channels,
                            out_channels,
                            kernel_size=upsample_kernel_size,
                            stride=upsample_kernel_size,
                            conv_only=True,
                            is_transposed=True,
                        ),
                        block_cls(
                            spatial_dims=3,
                            in_channels=out_channels,
                            out_channels=out_channels,
                            kernel_size=kernel_size,
                            stride=stride,
                            norm_name=norm_name,
                        ),
                    )
                    for _ in range(num_layer)
                ]
            )
        else:
            self.blocks = nn.ModuleList(
                [
                    get_conv_layer(
                        spatial_dims,
                        out_channels,
                        out_channels,
                        kernel_size=1,
                        stride=1,
                        conv_only=True,
                        is_transposed=True,
                    )
                    for _ in range(num_layer)
                ]
            )

    def forward(self, x):
        x = self.transp_conv_init(x)
        for block in self.blocks:
            x = block(x)
        return x


class EncoderConvBlock(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int],
        stride: Union[Sequence[int], int],
        norm_name: Union[Tuple, str],
        res_block: bool = False,
    ) -> None:
        super().__init__()
        block_cls = UnetResBlock if res_block else UnetBasicBlock
        self.layer = block_cls(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            norm_name=norm_name,
        )

    def forward(self, inp):
        return self.layer(inp)


class SingleBranchWindowFusion3D(nn.Module):
    def __init__(self, channels, num_heads=4, dropout=0.0, window_size=4):
        super().__init__()
        self.channels = channels
        self.window_size = window_size
        self.norm1 = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 4, channels),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        batch, channels, depth, height, width = x.shape
        ws = self.window_size
        assert depth % ws == 0 and height % ws == 0 and width % ws == 0, (
            "Input spatial dimensions must be divisible by window_size."
        )

        nd, nh, nw = depth // ws, height // ws, width // ws
        x_windows = x.view(batch, channels, nd, ws, nh, ws, nw, ws)
        x_windows = x_windows.permute(0, 2, 4, 6, 3, 5, 7, 1).contiguous()
        x_windows = x_windows.view(-1, ws * ws * ws, channels)

        x_norm = self.norm1(x_windows)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x_windows = x_windows + attn_out

        x_norm = self.norm2(x_windows)
        x_windows = x_windows + self.mlp(x_norm)

        x_windows = x_windows.view(batch, nd, nh, nw, ws, ws, ws, channels)
        x_windows = x_windows.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous()
        return x_windows.view(batch, channels, depth, height, width)


class CrossBranchWindowFusion3D(nn.Module):
    def __init__(self, channels, num_heads=4, dropout=0.0, window_size=4):
        super().__init__()
        self.channels = channels
        self.window_size = window_size
        self.norm1 = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(channels)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 4, channels),
            nn.Dropout(dropout),
        )

    def forward(self, seg_features, sdf_features):
        batch, channels, depth, height, width = seg_features.shape
        ws = self.window_size
        assert depth % ws == 0 and height % ws == 0 and width % ws == 0, (
            "Input spatial dimensions must be divisible by window_size."
        )

        nd, nh, nw = depth // ws, height // ws, width // ws

        seg_windows = seg_features.view(batch, channels, nd, ws, nh, ws, nw, ws)
        sdf_windows = sdf_features.view(batch, channels, nd, ws, nh, ws, nw, ws)
        seg_windows = seg_windows.permute(0, 2, 4, 6, 3, 5, 7, 1).contiguous()
        sdf_windows = sdf_windows.permute(0, 2, 4, 6, 3, 5, 7, 1).contiguous()
        seg_windows = seg_windows.view(-1, ws * ws * ws, channels)
        sdf_windows = sdf_windows.view(-1, ws * ws * ws, channels)

        seg_norm = self.norm1(seg_windows)
        sdf_norm = self.norm1(sdf_windows)
        fused_tokens = torch.cat([seg_norm, sdf_norm], dim=1)
        attn_out, _ = self.attn(fused_tokens, fused_tokens, fused_tokens)

        fused_windows = torch.cat([seg_windows, sdf_windows], dim=1) + attn_out
        fused_windows = fused_windows + self.mlp(self.norm2(fused_windows))

        seg_windows = fused_windows[:, : ws * ws * ws, :]
        sdf_windows = fused_windows[:, ws * ws * ws :, :]

        seg_out = seg_windows.view(batch, nd, nh, nw, ws, ws, ws, channels)
        sdf_out = sdf_windows.view(batch, nd, nh, nw, ws, ws, ws, channels)
        seg_out = seg_out.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous()
        sdf_out = sdf_out.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous()
        return seg_out.view(batch, channels, depth, height, width), sdf_out.view(
            batch, channels, depth, height, width
        )


class DepthwiseSeparableConv3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super().__init__()
        self.depthwise = nn.Conv3d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
        )
        self.pointwise = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class DSMSResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv3x3 = DepthwiseSeparableConv3D(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv5x5 = DepthwiseSeparableConv3D(in_channels, in_channels, kernel_size=5, padding=2)
        self.resblock_conv1 = nn.Conv3d(in_channels * 2, out_channels, kernel_size=1)
        self.resblock_bn1 = nn.BatchNorm3d(out_channels)
        self.resblock_relu = nn.ReLU(inplace=False)
        self.resblock_conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.resblock_bn2 = nn.BatchNorm3d(out_channels)
        self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.bn_shortcut = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        out = torch.cat([self.conv3x3(x), self.conv5x5(x)], dim=1)
        out = self.resblock_relu(self.resblock_bn1(self.resblock_conv1(out)))
        out = self.resblock_relu(self.resblock_bn2(self.resblock_conv2(out)))
        return out + self.bn_shortcut(self.shortcut(x))


class ValGeoFuseNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        img_size: Tuple[int, int, int] = (96, 96, 96),
        feature_size: int = 16,
        patch_size: int = 4,
        depths: Tuple[int, int, int] = (2, 2, 8),
        num_heads: Tuple[int, int, int] = (4, 8, 16),
        embed_dim: Tuple[int, int, int] = (128, 256, 512),
        window_size: Tuple[int, int, int] = (7, 7, 7),
        norm_name: Union[Tuple, str] = "instance",
        conv_block: bool = False,
        res_block: bool = True,
        dropout_rate: float = 0.0,
        topk_small_ratio: float = None,
        topk_large_ratio: float = None,
        use_cross_fusion: bool = True,
        pooling_mode: str = None,
    ) -> None:
        super().__init__()
        if not (0 <= dropout_rate <= 1):
            raise AssertionError("dropout_rate should be between 0 and 1.")

        self.use_cross_fusion = use_cross_fusion
        self.embed_dim = embed_dim

        self.nestViT = NestTransformer3D(
            img_size=img_size[0] if isinstance(img_size, tuple) else img_size,
            in_chans=in_channels,
            patch_size=patch_size,
            num_levels=3,
            embed_dims=embed_dim,
            num_heads=num_heads,
            depths=depths,
            num_classes=1000,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.5,
            norm_layer=None,
            act_layer=None,
            pad_type="",
            weight_init="",
            topk_small_ratio=topk_small_ratio,
            topk_large_ratio=topk_large_ratio,
            global_pool="avg",
            pooling_mode=pooling_mode,
        )

        self.encoder1 = EncoderConvBlock(3, in_channels, feature_size * 2, 3, 1, norm_name, res_block)
        self.encoder2 = EncoderUpBlock(
            3,
            self.embed_dim[0],
            feature_size * 4,
            num_layer=1,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=False,
            res_block=False,
        )
        self.encoder3 = EncoderConvBlock(3, self.embed_dim[0], feature_size * 8, 3, 1, norm_name, res_block)
        self.encoder4 = EncoderConvBlock(3, self.embed_dim[1], feature_size * 16, 3, 1, norm_name, res_block)

        self.decoder5 = DecoderSkipBlock(3, 2 * self.embed_dim[2], feature_size * 32, 3, 1, 2, norm_name, res_block)
        self.decoder4 = DecoderSkipBlock(3, self.embed_dim[2], feature_size * 16, 3, 1, 2, norm_name, res_block)
        self.decoder3 = DecoderSkipBlock(3, feature_size * 16, feature_size * 8, 3, 1, 2, norm_name, res_block)
        self.decoder2 = DecoderSkipBlock(3, feature_size * 8, feature_size * 4, 3, 1, 2, norm_name, res_block)
        self.decoder1 = DecoderSkipBlock(3, feature_size * 4, feature_size * 2, 3, 1, 2, norm_name, res_block)

        self.encoder10 = Convolution(
            dimensions=3,
            in_channels=feature_size * 32,
            out_channels=feature_size * 64,
            strides=2,
            adn_ordering="ADN",
            dropout=0.0,
        )

        self.out = UnetOutBlock(spatial_dims=3, in_channels=feature_size * 2, out_channels=out_channels)
        self.mapping_head = nn.Conv3d(feature_size * 2, 2, kernel_size=1)

        if self.use_cross_fusion:
            self.resblock_dec0 = CrossBranchWindowFusion3D(channels=feature_size * 2, num_heads=4, window_size=32)
            self.resblock_dec1 = SingleBranchWindowFusion3D(channels=feature_size * 2, num_heads=4, window_size=32)

        self.multi_scale_fusion4 = DSMSResBlock(512, 512)
        self.multi_scale_fusion3 = DSMSResBlock(256, 256)
        self.multi_scale_fusion2 = DSMSResBlock(128, 128)
        self.multi_scale_fusion1 = DSMSResBlock(64, 64)
        self.multi_scale_fusion0 = DSMSResBlock(32, 32)

    def forward(self, x_in):
        x, hidden_states_out = self.nestViT(x_in)
        enc0 = self.encoder1(x_in)
        enc1 = self.encoder2(hidden_states_out[0])
        enc2 = self.encoder3(hidden_states_out[1])
        enc3 = self.encoder4(hidden_states_out[2])
        enc4 = hidden_states_out[3]

        dec4 = self.encoder10(x)

        enc4 = self.multi_scale_fusion4(enc4)
        enc3 = self.multi_scale_fusion3(enc3)
        enc2 = self.multi_scale_fusion2(enc2)
        enc1 = self.multi_scale_fusion1(enc1)
        enc0 = self.multi_scale_fusion0(enc0)

        dec3 = self.decoder5(dec4, enc4)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        out = self.decoder1(dec0, enc0)

        if self.use_cross_fusion:
            seg_features = self.resblock_dec1(out)
            sdf_features = self.resblock_dec1(out)
            seg_update, sdf_update = self.resblock_dec0(seg_features, sdf_features)
            seg_features = seg_features + seg_update
            logits = self.out(seg_features)
            mapped_out = self.mapping_head(sdf_features)
        else:
            logits = self.out(out)
            mapped_out = self.mapping_head(out)

        return logits, mapped_out
