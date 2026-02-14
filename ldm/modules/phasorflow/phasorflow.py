import os
import warnings
import math
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from distutils.version import LooseVersion
import numpy as np
from functools import reduce, lru_cache
from operator import mul
from einops import rearrange
from einops.layers.torch import Rearrange
from .op.deform_attn import deform_attn, DeformAttnPack


from ldm.util import instantiate_from_config

def flow_warp(x, flow, interp_mode='bilinear', padding_mode='zeros', align_corners=True):
    """Warp an image or feature map with optical flow.

    Args:
        x (Tensor): Tensor with size (n, c, h, w).
        flow (Tensor): Tensor with size (n, h, w, 2), normal value.
        interp_mode (str): 'nearest' or 'bilinear' or 'nearest4'. Default: 'bilinear'.
        padding_mode (str): 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Before pytorch 1.3, the default value is
            align_corners=True. After pytorch 1.3, the default value is
            align_corners=False. Here, we use the True as default.


    Returns:
        Tensor: Warped image or feature map.
    """
    n, _, h, w = x.size()
    if flow.shape[1] != h or flow.shape[2] != w:
        flow = flow.view(-1, h, w, 2)

    # print(x.shape, flow.shape)
    # create mesh grid
    grid_y, grid_x = torch.meshgrid(torch.arange(0, h, dtype=x.dtype, device=x.device),
                                    torch.arange(0, w, dtype=x.dtype, device=x.device))
    grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
    grid.requires_grad = False

    vgrid = grid + flow

    # scale grid to [-1,1]
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)

    output = F.grid_sample(x, vgrid_scaled, mode=interp_mode, padding_mode=padding_mode, align_corners=align_corners)

    return output


def make_layer(block, num_blocks, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        block (nn.module): nn.module class for basic block.
        num_blocks (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_blocks):
        layers.append(block(**kwarg))
    return nn.Sequential(*layers)


class RSTBWithInputConv(nn.Module):
    """RSTB with a convolution in front.

    Args:
        in_channels (int): Number of input channels of the first conv.
        kernel_size (int): Size of kernel of the first conv.
        stride (int): Stride of the first conv.
        group (int): Group of the first conv.
        num_blocks (int): Number of residual blocks. Default: 2.
         **kwarg: Args for RSTB.
    """

    def __init__(self, in_channels=3, kernel_size=(1, 3, 3), stride=1, groups=1, num_blocks=2, **kwargs):
        super().__init__()

        main = []
        main += [Rearrange('n d c h w -> n c d h w'),
                 nn.Conv3d(in_channels,
                           kwargs['dim'],
                           kernel_size=kernel_size,
                           stride=stride,
                           padding=(kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[2] // 2),
                           groups=groups),
                 Rearrange('n c d h w -> n d h w c'),
                 nn.LayerNorm(kwargs['dim']),
                 Rearrange('n d h w c -> n c d h w')]

        # RSTB blocks
        kwargs['use_checkpoint_attn'] = kwargs.pop('use_checkpoint_attn')[0]
        kwargs['use_checkpoint_ffn'] = kwargs.pop('use_checkpoint_ffn')[0]
        main.append(make_layer(RSTB, num_blocks, **kwargs))

        main += [Rearrange('n c d h w -> n d h w c'),
                 nn.LayerNorm(kwargs['dim']),
                 Rearrange('n d h w c -> n d c h w')]

        self.main = nn.Sequential(*main)

    def forward(self, x):
        """
        Forward function for RSTBWithInputConv.

        Args:
            feat (Tensor): Input feature with shape (n, t, in_channels, h, w)

        Returns:
            Tensor: Output feature with shape (n, t, out_channels, h, w)
        """
        return self.main(x)


class GuidedDeformAttnPack(DeformAttnPack):
    """Guided deformable attention module.

    Args:
        in_channels (int): Same as nn.Conv2d.
        out_channels (int): Same as nn.Conv2d.
        attention_window (int or tuple[int]): Attention window size. Default: [3, 3].
        attention_heads (int): Attention head number.  Default: 12.
        deformable_groups (int): Deformable offset groups.  Default: 12.
        clip_size (int): clip size. Default: 2.
        max_residue_magnitude (int): The maximum magnitude of the offset residue. Default: 10.
    Ref:
        Recurrent Video Restoration Transformer with Guided Deformable Attention

    """

    def __init__(self, *args, **kwargs):
        self.max_residue_magnitude = kwargs.pop('max_residue_magnitude', 10)

        super(GuidedDeformAttnPack, self).__init__(*args, **kwargs)

        self.conv_offset = nn.Sequential(
            nn.Conv3d(self.in_channels * (1 + self.clip_size) + self.clip_size * 2, 64, kernel_size=(1, 1, 1),
                      padding=(0, 0, 0)),
            nn.LeakyReLU(negative_slope=0.1, inplace=False),
            nn.Conv3d(64, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.LeakyReLU(negative_slope=0.1, inplace=False),
            nn.Conv3d(64, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.LeakyReLU(negative_slope=0.1, inplace=False),
            nn.Conv3d(64, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.LeakyReLU(negative_slope=0.1, inplace=False),
            nn.Conv3d(64, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            nn.LeakyReLU(negative_slope=0.1, inplace=False),
            nn.Conv3d(64, self.clip_size * self.deformable_groups * self.attn_size * 2, kernel_size=(1, 1, 1),
                      padding=(0, 0, 0)),
        )
        self.init_offset()

        # proj to a higher dimension can slightly improve the performance
        self.proj_channels = int(self.in_channels * 2)
        self.proj_q = nn.Sequential(Rearrange('n d c h w -> n d h w c'),
                                    nn.Linear(self.in_channels, self.proj_channels),
                                    Rearrange('n d h w c -> n d c h w'))
        self.proj_k = nn.Sequential(Rearrange('n d c h w -> n d h w c'),
                                    nn.Linear(self.in_channels, self.proj_channels),
                                    Rearrange('n d h w c -> n d c h w'))
        self.proj_v = nn.Sequential(Rearrange('n d c h w -> n d h w c'),
                                    nn.Linear(self.in_channels, self.proj_channels),
                                    Rearrange('n d h w c -> n d c h w'))
        self.proj = nn.Sequential(Rearrange('n d c h w -> n d h w c'),
                                  nn.Linear(self.proj_channels, self.in_channels),
                                  Rearrange('n d h w c -> n d c h w'))
        self.mlp = nn.Sequential(Rearrange('n d c h w -> n d h w c'),
                                 Mlp(self.in_channels, self.in_channels * 2, self.in_channels),
                                 Rearrange('n d h w c -> n d c h w'))

    def init_offset(self):
        if hasattr(self, 'conv_offset'):
            self.conv_offset[-1].weight.data.zero_()
            self.conv_offset[-1].bias.data.zero_()

    def forward(self, q, k, v, v_prop_warped, flows, phasor, return_updateflow):
        offset1, offset2 = torch.chunk(self.max_residue_magnitude * torch.tanh(
            self.conv_offset(torch.cat([q] + v_prop_warped + flows, 2).transpose(1, 2)).transpose(1, 2)), 2, dim=2)
        offset1 = offset1 + flows[0].flip(2).repeat(1, 1, offset1.size(2) // 2, 1, 1)
        offset2 = offset2 + flows[1].flip(2).repeat(1, 1, offset2.size(2) // 2, 1, 1)
        offset = torch.cat([offset1, offset2], dim=2).flatten(0, 1)

        b, t, c, h, w = offset1.shape
        q = self.proj_q(q).view(b * t, 1, self.proj_channels, h, w)
        kv = torch.cat([self.proj_k(k), self.proj_v(v)], 2)
        v = deform_attn(q, kv, offset, phasor, self.kernel_h, self.kernel_w, self.stride, self.padding, self.dilation,
                        self.attention_heads, self.deformable_groups, self.clip_size).view(b, t, self.proj_channels, h,
                                                                                           w)
        v = self.proj(v)
        v = v + self.mlp(v)

        if return_updateflow:
            return v, offset1.view(b, t, c // 2, 2, h, w).mean(2).flip(2), offset2.view(b, t, c // 2, 2, h, w).mean(
                2).flip(2)
        else:
            return v


def window_partition(x, window_size):
    """ Partition the input into windows. Attention will be conducted within the windows.

    Args:
        x: (B, D, H, W, C)
        window_size (tuple[int]): window size

    Returns:
        windows: (B*num_windows, window_size*window_size, C)
    """
    B, D, H, W, C = x.shape
    x = x.view(B, D // window_size[0], window_size[0], H // window_size[1], window_size[1], W // window_size[2],
               window_size[2], C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, reduce(mul, window_size), C)

    return windows


def window_reverse(windows, window_size, B, D, H, W):
    """ Reverse windows back to the original input. Attention was conducted within the windows.

    Args:
        windows: (B*num_windows, window_size, window_size, C)
        window_size (tuple[int]): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, D, H, W, C)
    """
    x = windows.view(B, D // window_size[0], H // window_size[1], W // window_size[2], window_size[0], window_size[1],
                     window_size[2], -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)

    return x


def get_window_size(x_size, window_size, shift_size=None):
    """ Get the window size and the shift size """

    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0

    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)

class WindowAttention(nn.Module):
    """ Window based multi-head self attention.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The temporal length, height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=False, qk_scale=None):
        super().__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1),
                        num_heads))  # 2*Wd-1 * 2*Wh-1 * 2*Ww-1, nH
        self.register_buffer("relative_position_index", self.get_position_index(window_size))
        self.qkv_self = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, mask=None):
        """ Forward function.

        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, N, N) or None
        """

        # self attention
        B_, N, C = x.shape
        qkv = self.qkv_self(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B_, nH, N, C
        x_out = self.attention(q, k, v, mask, (B_, N, C))

        # projection
        x = self.proj(x_out)

        return x

    def attention(self, q, k, v, mask, x_shape):
        B_, N, C = x_shape
        attn = (q * self.scale) @ k.transpose(-2, -1).contiguous()

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index[:N, :N].reshape(-1)].reshape(N, N, -1)  # Wd*Wh*Ww, Wd*Wh*Ww,nH
        attn = attn + relative_position_bias.permute(2, 0, 1).unsqueeze(0)  # B_, nH, N, N

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask[:, :N, :N].unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = F.softmax(attn, -1, dtype=q.dtype)  # Don't use attn.dtype after addition!
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C).contiguous()

        return x

    def get_position_index(self, window_size):
        ''' Get pair-wise relative position index for each token inside the window. '''

        coords_d = torch.arange(window_size[0])
        coords_h = torch.arange(window_size[1])
        coords_w = torch.arange(window_size[2])
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))  # 3, Wd, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 3, Wd*Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, Wd*Wh*Ww, Wd*Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wd*Wh*Ww, Wd*Wh*Ww, 3
        relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 2] += window_size[2] - 1

        relative_coords[:, :, 0] *= (2 * window_size[1] - 1) * (2 * window_size[2] - 1)
        relative_coords[:, :, 1] *= (2 * window_size[2] - 1)
        relative_position_index = relative_coords.sum(-1)  # Wd*Wh*Ww, Wd*Wh*Ww

        return relative_position_index


class STL(nn.Module):
    """ Swin Transformer Layer (STL).

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): Window size.
        shift_size (tuple[int]): Shift size for mutual and self attention.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True.
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm.
        use_checkpoint_attn (bool): If True, use torch.checkpoint for attention modules. Default: False.
        use_checkpoint_ffn (bool): If True, use torch.checkpoint for feed-forward modules. Default: False.
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 num_heads,
                 window_size=(2, 8, 8),
                 shift_size=(0, 0, 0),
                 mlp_ratio=2.,
                 qkv_bias=True,
                 qk_scale=None,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 use_checkpoint_attn=False,
                 use_checkpoint_ffn=False
                 ):
        super().__init__()
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.use_checkpoint_attn = use_checkpoint_attn
        self.use_checkpoint_ffn = use_checkpoint_ffn

        assert 0 <= self.shift_size[0] < self.window_size[0], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[1] < self.window_size[1], "shift_size must in 0-window_size"
        assert 0 <= self.shift_size[2] < self.window_size[2], "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(dim, window_size=self.window_size, num_heads=num_heads, qkv_bias=qkv_bias,
                                    qk_scale=qk_scale)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer)

    def forward_part1(self, x, mask_matrix):
        B, D, H, W, C = x.shape
        window_size, shift_size = get_window_size((D, H, W), self.window_size, self.shift_size)

        x = self.norm1(x)

        # pad feature maps to multiples of window size
        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (window_size[0] - D % window_size[0]) % window_size[0]
        pad_b = (window_size[1] - H % window_size[1]) % window_size[1]
        pad_r = (window_size[2] - W % window_size[2]) % window_size[2]
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1), mode='constant')

        _, Dp, Hp, Wp, _ = x.shape
        # cyclic shift
        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, window_size)  # B*nW, Wd*Wh*Ww, C

        # attention / shifted attention
        attn_windows = self.attn(x_windows, mask=attn_mask)  # B*nW, Wd*Wh*Ww, C

        # merge windows
        attn_windows = attn_windows.view(-1, *(window_size + (C,)))
        shifted_x = window_reverse(attn_windows, window_size, B, Dp, Hp, Wp)  # B D' H' W' C

        # reverse cyclic shift
        if any(i > 0 for i in shift_size):
            x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
        else:
            x = shifted_x

        if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
            x = x[:, :D, :H, :W, :]

        return x

    def forward_part2(self, x):
        return self.mlp(self.norm2(x))

    def forward(self, x, mask_matrix):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, D, H, W, C).
            mask_matrix: Attention mask for cyclic shift.
        """

        # attention
        if self.use_checkpoint_attn:
            x = x + checkpoint.checkpoint(self.forward_part1, x, mask_matrix)
        else:
            x = x + self.forward_part1(x, mask_matrix)

        # feed-forward
        if self.use_checkpoint_ffn:
            x = x + checkpoint.checkpoint(self.forward_part2, x)
        else:
            x = x + self.forward_part2(x)

        return x


class STG(nn.Module):
    """ Swin Transformer Group (STG).

    Args:
        dim (int): Number of feature channels
        input_resolution (tuple[int]): Input resolution.
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (tuple[int]): Local window size. Default: (6,8,8).
        shift_size (tuple[int]): Shift size for mutual and self attention. Default: None.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 2.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        use_checkpoint_attn (bool): If True, use torch.checkpoint for attention modules. Default: False.
        use_checkpoint_ffn (bool): If True, use torch.checkpoint for feed-forward modules. Default: False.
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 num_heads,
                 window_size=[2, 8, 8],
                 shift_size=None,
                 mlp_ratio=2.,
                 qkv_bias=False,
                 qk_scale=None,
                 norm_layer=nn.LayerNorm,
                 use_checkpoint_attn=False,
                 use_checkpoint_ffn=False,
                 ):
        super().__init__()
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.shift_size = list(i // 2 for i in window_size) if shift_size is None else shift_size

        # build blocks
        self.blocks = nn.ModuleList([
            STL(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=[0, 0, 0] if i % 2 == 0 else self.shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                norm_layer=norm_layer,
                use_checkpoint_attn=use_checkpoint_attn,
                use_checkpoint_ffn=use_checkpoint_ffn
            )
            for i in range(depth)])

    def forward(self, x):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, C, D, H, W).
        """
        # calculate attention mask for attention
        B, C, D, H, W = x.shape
        window_size, shift_size = get_window_size((D, H, W), self.window_size, self.shift_size)
        x = rearrange(x, 'b c d h w -> b d h w c')
        Dp = int(np.ceil(D / window_size[0])) * window_size[0]
        Hp = int(np.ceil(H / window_size[1])) * window_size[1]
        Wp = int(np.ceil(W / window_size[2])) * window_size[2]
        attn_mask = compute_mask(Dp, Hp, Wp, window_size, shift_size, x.device)

        for blk in self.blocks:
            x = blk(x, attn_mask)

        x = x.view(B, D, H, W, -1)
        x = rearrange(x, 'b d h w c -> b c d h w')

        return x


class RSTB(nn.Module):
    """ Residual Swin Transformer Block (RSTB).

    Args:
        kwargs: Args for RSTB.
    """

    def __init__(self, **kwargs):
        super(RSTB, self).__init__()
        self.input_resolution = kwargs['input_resolution']

        self.residual_group = STG(**kwargs)
        self.linear = nn.Linear(kwargs['dim'], kwargs['dim'])

    def forward(self, x):
        return x + self.linear(self.residual_group(x).transpose(1, 4)).transpose(1, 4).contiguous()

class Upsample(nn.Sequential):
    """Upsample module for video SR.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        assert LooseVersion(torch.__version__) >= LooseVersion('1.8.1'), \
            'PyTorch version >= 1.8.1 to support 5D PixelShuffle.'

        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv3d(num_feat, 4 * num_feat, kernel_size=(1, 3, 3), padding=(0, 1, 1)))
                m.append(Rearrange('n c d h w -> n d c h w'))
                m.append(nn.PixelShuffle(2))
                m.append(Rearrange('n c d h w -> n d c h w'))
                m.append(nn.LeakyReLU(negative_slope=0.1, inplace=False))
            m.append(nn.Conv3d(num_feat, num_feat, kernel_size=(1, 3, 3), padding=(0, 1, 1)))
        elif scale == 3:
            m.append(nn.Conv3d(num_feat, 9 * num_feat, kernel_size=(1, 3, 3), padding=(0, 1, 1)))
            m.append(Rearrange('n c d h w -> n d c h w'))
            m.append(nn.PixelShuffle(3))
            m.append(Rearrange('n c d h w -> n d c h w'))
            m.append(nn.LeakyReLU(negative_slope=0.1, inplace=False))
            m.append(nn.Conv3d(num_feat, num_feat, kernel_size=(1, 3, 3), padding=(0, 1, 1)))
        else:
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)

@lru_cache()
def compute_mask(D, H, W, window_size, shift_size, device):
    """ Compute attnetion mask for input of size (D, H, W). @lru_cache caches each stage results. """

    img_mask = torch.zeros((1, D, H, W, 1), device=device)  # 1 Dp Hp Wp 1
    cnt = 0
    for d in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None):
        for h in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None):
            for w in slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2], None):
                img_mask[:, d, h, w, :] = cnt
                cnt += 1
    mask_windows = window_partition(img_mask, window_size)  # nW, ws[0]*ws[1]*ws[2], 1
    mask_windows = mask_windows.squeeze(-1)  # nW, ws[0]*ws[1]*ws[2]
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

    return attn_mask


class Mlp(nn.Module):
    """ Multilayer perceptron.

    Args:
        x: (B, D, H, W, C)

    Returns:
        x: (B, D, H, W, C)
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class PhasorFlows(pl.LightningModule):
    def __init__(self,
                 lossconfig=None,
                 flownet_config=None,
                 ckpt_path=None,
                 ignore_keys=["spynet"],
                 upscale=4,
                 clip_size=2,
                 img_size=[2, 64, 64],
                 window_size=[2, 8, 8],
                 num_blocks=[1, 2, 1],
                 depths=[2, 2, 2],
                 embed_dims=[144, 144, 144],
                 num_heads=[6, 6, 6],
                 mlp_ratio=2.,
                 qkv_bias=True,
                 qk_scale=None,
                 norm_layer=nn.LayerNorm,
                 inputconv_groups=[1, 1, 1, 1, 1, 1],
                 spynet_path=None,
                 max_residue_magnitude=10,
                 deformable_groups=12,
                 attention_heads=12,
                 attention_window=[3, 3],
                 nonblind_denoising=False,
                 use_checkpoint_attn=False,
                 use_checkpoint_ffn=False,
                 no_checkpoint_attn_blocks=[],
                 no_checkpoint_ffn_blocks=[],
                 ):

        super().__init__()
        self.upscale = upscale
        self.clip_size = clip_size
        self.nonblind_denoising = nonblind_denoising
        use_checkpoint_attns = [False if i in no_checkpoint_attn_blocks else use_checkpoint_attn for i in range(100)]
        use_checkpoint_ffns = [False if i in no_checkpoint_ffn_blocks else use_checkpoint_ffn for i in range(100)]
        self.embed_dims = embed_dims
        
            

        # optical flow
        self.instantiate_flownet_stage(flownet_config)
        self.loss = instantiate_from_config(lossconfig)
        
        
        # shallow feature extraction
        self.feat_extract = RSTBWithInputConv(in_channels=3,
                                                  kernel_size=(1, 3, 3),
                                                  groups=inputconv_groups[0],
                                                  num_blocks=num_blocks[0],
                                                  dim=embed_dims[0],
                                                  input_resolution=[1, img_size[1], img_size[2]],
                                                  depth=depths[0],
                                                  num_heads=num_heads[0],
                                                  window_size=[1, window_size[1], window_size[2]],
                                                  mlp_ratio=mlp_ratio,
                                                  qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                  norm_layer=norm_layer,
                                                  use_checkpoint_attn=[False],
                                                  use_checkpoint_ffn=[False]
                                                  )

        # check if the sequence is augmented by flipping
        self.is_mirror_extended = False

        # recurrent feature refinement
        self.backbone = nn.ModuleDict()
        self.deform_align = nn.ModuleDict()
        modules = ['backward_1', 'forward_1', 'backward_2', 'forward_2']
        for i, module in enumerate(modules):
            # deformable attention
            self.deform_align[module] = GuidedDeformAttnPack(embed_dims[1],
                                                             embed_dims[1],
                                                             attention_window=attention_window,
                                                             attention_heads=attention_heads,
                                                             deformable_groups=deformable_groups,
                                                             clip_size=clip_size,
                                                             max_residue_magnitude=max_residue_magnitude)

            # feature propagation
            self.backbone[module] = RSTBWithInputConv(
                                                     in_channels=(2 + i) * embed_dims[0],
                                                     kernel_size=(1, 3, 3),
                                                     groups=inputconv_groups[i + 1],
                                                     num_blocks=num_blocks[1],
                                                     dim=embed_dims[1],
                                                     input_resolution=img_size,
                                                     depth=depths[1],
                                                     num_heads=num_heads[1],
                                                     window_size=window_size,
                                                     mlp_ratio=mlp_ratio,
                                                     qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                     norm_layer=norm_layer,
                                                     use_checkpoint_attn=[use_checkpoint_attns[i]],
                                                     use_checkpoint_ffn=[use_checkpoint_ffns[i]]
                                                     )

        # reconstruction
        self.reconstruction = RSTBWithInputConv(
                                               in_channels=5 * embed_dims[0],
                                               kernel_size=(1, 3, 3),
                                               groups=inputconv_groups[5],
                                               num_blocks=num_blocks[2],

                                               dim=embed_dims[2],
                                               input_resolution=[1, img_size[1], img_size[2]],
                                               depth=depths[2],
                                               num_heads=num_heads[2],
                                               window_size=[1, window_size[1], window_size[2]],
                                               mlp_ratio=mlp_ratio,
                                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                                               norm_layer=norm_layer,
                                               use_checkpoint_attn=[False],
                                               use_checkpoint_ffn=[False]
                                               )
        self.conv_before_upsampler = nn.Sequential(
                                                  nn.Conv3d(embed_dims[-1], 64, kernel_size=(1, 1, 1),
                                                            padding=(0, 0, 0)),
                                                  nn.LeakyReLU(negative_slope=0.1, inplace=True)
                                                  )
        self.refine_network = nn.Sequential(
            nn.Conv2d(in_channels=292, out_channels=180, kernel_size=7, stride=1, padding=3), nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=180, out_channels=64, kernel_size=7, stride=1, padding=3), nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=7, stride=1, padding=3), nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=7, stride=1, padding=3), nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=16, out_channels=2, kernel_size=7, stride=1, padding=3))

        self.upsampler = Upsample(4, 64)
        self.conv_last = nn.Conv3d(64, 3, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        
        if ckpt_path is not None:
            missing_list = self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        else:
            missing_list = []
            
        
        
        # print('>>>>>>>>>>>>>>>>>missing>>>>>>>>>>>>>>>>>>>')
        # print(missing_list)

        
        # print('>>>>>>>>>>>>>>>>>trainable_list>>>>>>>>>>>>>>>>>>>')
        trainable_list = ["feat_extract", "backbone", "deform_align", "refine_network", "reconstruction"]
        
        for name, params in self.named_parameters():
            if any(name.startswith(layer) for layer in trainable_list):
                params.requires_grad = True
            else:
                params.requires_grad = False
        # print(trainable_list)

        # print('>>>>>>>>>>>>>>>>>Untrainable_list>>>>>>>>>>>>>>>>>>>')
        untrainable_list = []
        for name, params in self.named_parameters():
            if not params.requires_grad:
                untrainable_list.append(name)
        # print(untrainable_list)
        
    def init_from_ckpt(self, path, ignore_keys=list(), only_model=False):
        para = []
        for name, params in self.named_parameters():
            # print(name)
            para.append(name)
        
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        # sd = sd['params']
        keys = list(sd.keys())
        for k in keys:
            # if k not in para:
            #     print("extra:", k)
            # else:
            #     if self.state_dict()[k].shape != sd[k].shape:
            #         print("shape mismatch:", k, self.state_dict()[k].shape, sd[k].shape)
            # print(k, sd[k].shape)
            for ik in ignore_keys:
                if k.startswith(ik):
                    # print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
                    
        
        missing, unexpected = self.load_state_dict(sd, strict=False) if not only_model else self.model.load_state_dict(
            sd, strict=False)
        print(f"Encoder Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        # if len(missing) > 0:
            # print(f"Missing Keys: {missing}")
        # if len(unexpected) > 0:
        #     print(f"Unexpected Keys: {unexpected}")

        return missing

    def compute_flow(self, lqs):
        """Compute optical flow using SPyNet for feature alignment.

        Note that if the input is an mirror-extended sequence, 'flows_forward'
        is not needed, since it is equal to 'flows_backward.flip(1)'.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).

        Return:
            tuple(Tensor): Optical flow. 'flows_forward' corresponds to the
                flows used for forward-time propagation (current to previous).
                'flows_backward' corresponds to the flows used for
                backward-time propagation (current to next).
        """

        n, t, c, h, w = lqs.size()
        lqs_1 = lqs[:, :-1, :, :, :].reshape(-1, c, h, w)
        lqs_2 = lqs[:, 1:, :, :, :].reshape(-1, c, h, w)

        flows_backward = self.spynet(lqs_1, lqs_2).view(n, t - 1, 2, h, w)

        if self.is_mirror_extended:  # flows_forward = flows_backward.flip(1)
            flows_forward = None
        else:
            flows_forward = self.spynet(lqs_2, lqs_1).view(n, t - 1, 2, h, w)

        return flows_forward, flows_backward

    def compute_phasor(self, lqs):
        freq_idx = 1
        B, T, C, H, W = lqs.shape
        # print(lqs.shape)
        lqs = lqs[:, :, :1, :, :].squeeze(2)  # [B, T, H, W]
        lqs = lqs.permute(0, 2, 3, 1)  # [B, H, W, T]
        
        magnitudes = []
        
        for i in range(T):
            if i == T-1:
                mag = magnitudes[-1]
                magnitudes.append(mag.cuda())
                pass
            else:
                clip = lqs[:, :, :, i : (i + 2)].to('cpu')
                
                fft = torch.fft.fft(clip.float(), dim=-1)  # [B, H, W, T] -> complex
                phasor = fft[..., freq_idx]  # choose 1st harmonic
                
                magnitude = torch.abs(phasor)  # [B, H, W]
                
                magnitude = (magnitude - magnitude.view(B, -1).min(dim=1)[0][:, None, None]) / \
                            (magnitude.view(B, -1).max(dim=1)[0][:, None, None] - 
                            magnitude.view(B, -1).min(dim=1)[0][:, None, None] + 1e-8)
                            
                            
                magnitudes.append(magnitude.cuda())
                
        magnitudes = list(torch.chunk(torch.stack(magnitudes, 0), T // self.clip_size, dim=0))
        magnitudes = [mag.permute(1,0,2,3) for mag in magnitudes]
        
        # print("magnitudes length:", len(magnitudes), 'mag_size', magnitudes[0].shape)

        return magnitudes # len=T//self.clip_size, mag_size=[B, self.clip_size, H, W]

        
    def compute_phasor_seq(self, seq):
        freq_idx = 1
        B, T, C, H, W = seq.shape
        # print("seq", seq.shape)
        seq = seq[:, :, :1, :, :].squeeze(2).to('cpu')  # [B, T, H, W]
        seq = seq.permute(2, 3, 0, 1)  # [H, W, B, T]

        fft = torch.fft.fft(seq.float(), dim=-1)  # [H, W, B, T] -> complex
        phasor = fft[..., freq_idx]  # choose 1st harmonic

        magnitude = torch.abs(phasor)  # [H, W, B]
        return magnitude.permute(2, 0, 1).cuda()

    def check_if_mirror_extended(self, lqs):
        """Check whether the input is a mirror-extended sequence.

        If mirror-extended, the i-th (i=0, ..., t-1) frame is equal to the
        (t-1-i)-th frame.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).
        """

        if lqs.size(1) % 2 == 0:
            lqs_1, lqs_2 = torch.chunk(lqs, 2, dim=1)
            if torch.norm(lqs_1 - lqs_2.flip(1)) == 0:
                self.is_mirror_extended = True

    def propagate(self, lqs, feats, flows, phasors, module_name, updated_flows=None):
        """Propagate the latent clip features throughout the sequence.

        Args:
            feats dict(list[tensor]): Features from previous branches. Each
                component is a list of tensors with shape (n, clip_size, c, h, w).
            flows (tensor): Optical flows with shape (n, t - 1, 2, h, w).
            module_name (str): The name of the propgation branches. Can either
                be 'backward_1', 'forward_1', 'backward_2', 'forward_2'.
            updated_flows dict(list[tensor]): Each component is a list of updated
                optical flows with shape (n, clip_size, 2, h, w).

        Return:
            dict(list[tensor]): A dictionary containing all the propagated
                features. Each key in the dictionary corresponds to a
                propagation branch, which is represented by a list of tensors.
        """

        n, t, _, h, w = flows.size()
        if 'backward' in module_name:
            flow_idx = range(0, t + 1)[::-1]
            clip_idx = range(0, (t + 1) // self.clip_size)[::-1]
        else:
            flow_idx = range(-1, t)
            clip_idx = range(0, (t + 1) // self.clip_size)

        if '_1' in module_name:
            updated_flows[f'{module_name}_n1'] = []
            updated_flows[f'{module_name}_n2'] = []

        feat_prop = torch.zeros_like(feats['shallow'][0])

        last_key = list(feats)[-2]
        for i in range(0, len(clip_idx)):
            idx_c = clip_idx[i]
            if i > 0:
                if '_1' in module_name:
                    flow_n01 = flows[:, flow_idx[self.clip_size * i - 1], :, :, :]
                    flow_n12 = flows[:, flow_idx[self.clip_size * i], :, :, :]
                    flow_n23 = flows[:, flow_idx[self.clip_size * i + 1], :, :, :]
                    flow_n02 = flow_n12 + flow_warp(flow_n01, flow_n12.permute(0, 2, 3, 1))
                    flow_n13 = flow_n23 + flow_warp(flow_n12, flow_n23.permute(0, 2, 3, 1))
                    flow_n03 = flow_n23 + flow_warp(flow_n02, flow_n23.permute(0, 2, 3, 1))
                    flow_n1 = torch.stack([flow_n02, flow_n13], 1)
                    flow_n2 = torch.stack([flow_n12, flow_n03], 1)

                else:
                    module_name_old = module_name.replace('_2', '_1')
                    flow_n1 = updated_flows[f'{module_name_old}_n1'][i - 1]
                    flow_n2 = updated_flows[f'{module_name_old}_n2'][i - 1]
                    
                phasor_02 = self.compute_phasor_seq(lqs[:, self.clip_size * i-2 : self.clip_size * i+1, :, :, :])
                phasor_13 = self.compute_phasor_seq(lqs[:, self.clip_size * i-1 :self.clip_size * i+2, :, :, :])
                phasor_12 = self.compute_phasor_seq(lqs[:, self.clip_size * i-1 : self.clip_size * i+1, :, :, :])
                phasor_03 = self.compute_phasor_seq(lqs[:, self.clip_size * i-2 : self.clip_size * i+2, :, :, :])
                
                phasor_n1 = torch.stack([phasor_02, phasor_13], 1).unsqueeze(2).repeat(1, 1, 2, 1, 1)
                phasor_n2 = torch.stack([phasor_12, phasor_03], 1).unsqueeze(2).repeat(1, 1, 2, 1, 1)

                if 'backward' in module_name:
                    feat_q = feats[last_key][idx_c].flip(1)
                    feat_k = feats[last_key][clip_idx[i - 1]].flip(1)
                else:
                    feat_q = feats[last_key][idx_c]
                    feat_k = feats[last_key][clip_idx[i - 1]]

                feat_prop_warped1 = flow_warp(feat_prop.flatten(0, 1),
                                           (phasor_n1.permute(0, 1, 3, 4, 2) * flow_n1.permute(0, 1, 3, 4, 2)).flatten(0, 1))\
                    .view(n, feat_prop.shape[1], feat_prop.shape[2], h, w)
                feat_prop_warped2 = flow_warp(feat_prop.flip(1).flatten(0, 1),
                                           (phasor_n2.permute(0, 1, 3, 4, 2) * flow_n2.permute(0, 1, 3, 4, 2)).flatten(0, 1))\
                    .view(n, feat_prop.shape[1], feat_prop.shape[2], h, w)

                if '_1' in module_name:
                    feat_prop, flow_n1, flow_n2 = self.deform_align[module_name](feat_q, feat_k, feat_prop,
                                                                                 [feat_prop_warped1, feat_prop_warped2],
                                                                                 [flow_n1, flow_n2],
                                                                                 phasors[idx_c],
                                                                                 True)
                    updated_flows[f'{module_name}_n1'].append(flow_n1)
                    updated_flows[f'{module_name}_n2'].append(flow_n2)
                else:
                    feat_prop = self.deform_align[module_name](feat_q, feat_k, feat_prop,
                                                               [feat_prop_warped1, feat_prop_warped2],
                                                               [flow_n1, flow_n2],
                                                               phasors[idx_c],
                                                               False)

            if 'backward' in module_name:
                feat = [feats[k][idx_c].flip(1) for k in feats if k not in [module_name]] + [feat_prop]
            else:
                feat = [feats[k][idx_c] for k in feats if k not in [module_name]] + [feat_prop]


            feat_prop = feat_prop + self.backbone[module_name](torch.cat(feat, dim=2))
            feats[module_name].append(feat_prop)


        if 'backward' in module_name:
            feats[module_name] = feats[module_name][::-1]
            feats[module_name] = [f.flip(1) for f in feats[module_name]]

        

        return feats

    def refineflow(self, flow, feats, updated_flows, direction, iter_list):
        """Refine the optical flow.

        Args:
            flow (tensor): The input optical flow. n, t-1, c, h, w
            feats (tensor): The updated feature map. 
            updated_flows (tensor): The updated optical flow.
            direction (str): The direction of the flow ('forward' or 'backward').

        Returns:
            Tensor: The refined optical flow.
        """
        # for key in updated_flows.keys():
        #     print(key)
        #     print(len(updated_flows[key]), updated_flows[key][-1].shape)
            
        # print("feats:")
        # for k in feats.keys():
        #     print(k, len(feats[k]), feats[k][0].shape)
            
        if direction == 'forward':
            f = [torch.concat([feats[f'forward_{iter_list[-1]}'][c-1],
                              feats[f'forward_{iter_list[-2]}'][c],
                              updated_flows[f'forward_{iter_list[-2]}_n1'][c-1],
                              updated_flows[f'forward_{iter_list[-2]}_n2'][c-1]], dim=2) 
                              for c in range(1, len(feats['shallow'])) ]
        else:
            f = [torch.concat([feats[f'backward_{iter_list[-1]}'][c-1],
                              feats[f'backward_{iter_list[-2]}'][c],
                              updated_flows[f'backward_{iter_list[-2]}_n1'][c-1],
                              updated_flows[f'backward_{iter_list[-2]}_n2'][c-1]], dim=2) 
                              for c in range(1, len(feats['shallow'])-1) ]

        
        # print(feats[f'forward_{iter_list[-1]}'][-2].shape, updated_flows[f'forward_{iter_list[-2]}_n1'][-1].shape)
        # print("F", len(f), f[0].shape)
        
        f = torch.cat(f, dim=1)  # (b, t, c, h, w)
        b, t, c, h, w = f.shape
        # print("floooooow:")
        # print(flow.shape)
        
        off = self.refine_network(f.flatten(0, 1)).view(b, t, 2, h, w)
        # print("off shape:", off.shape)

        refined_flows = flow
        refined_flows[:, :t, :, :, :] = flow[:, :t, :, :, :] + off

        return refined_flows

    def upsample(self, lqs, feats, flow_fwd, flow_bwd):
        """Compute the output image given the features.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).
            feats (dict): The features from the propgation branches.

        Returns:
            Tensor: Output HR sequence with shape (n, t, c, 4h, 4w).

        """
        
        # print("feats:")
        # for k in feats.keys():
        #     print(k, len(feats[k]), feats[k][0].shape)

        feats['shallow'] = torch.cat(feats['shallow'], 1)
        feats['backward_1'] = torch.cat(feats['backward_1'], 1)
        feats['forward_1'] = torch.cat(feats['forward_1'], 1)
        feats['backward_2'] = torch.cat(feats['backward_2'], 1)
        feats['forward_2'] = torch.cat(feats['forward_2'], 1)


        
        hr_fwd = torch.cat([feats[k] for k in ['shallow', 'forward_1', 'forward_1', 'forward_2', 'forward_2']], dim=2)
        hr_bwd = torch.cat([feats[k] for k in ['shallow', 'backward_1', 'backward_1', 'backward_2', 'backward_2']], dim=2)
        hr_fwd = self.reconstruction(hr_fwd)
        hr_bwd = self.reconstruction(hr_bwd)
        
        # print("flow_fwd:", flow_fwd.shape)
        # print("flow_bwd:", flow_bwd.shape)

        b, t, c, h, w = hr_fwd.shape
        
        nhr_fwd = hr_fwd.clone()
        nhr_bwd = hr_bwd.clone()

        for i in range(b):
            nhr_fwd[i, 1:, :, :, :] = flow_warp(hr_fwd[i, 1:, :, :, :],flow_fwd[i])
            nhr_bwd[i, :t-1, :, :, :] = flow_warp(hr_bwd[i, :t-1, :, :, :].flip(1),flow_bwd[i])

        # print("hr_fwd:", nhr_fwd.shape)
        
        nhr_fwd = self.conv_last(self.upsampler(self.conv_before_upsampler(nhr_fwd.transpose(1, 2)))).transpose(1, 2).contiguous()
        # print("hr_fwd:", nhr_fwd.shape)

        nlqs = lqs.clone()

        # for i in range(b):
        #     nlqs[i, 1:, :, :, :] = flow_warp(lqs[i, 1:, :, :, :],flow_fwd[i])
        #     nlqs[i, :t-1, :, :, :] = flow_warp(lqs[i, :t-1, :, :, :].flip(1),flow_bwd[i])

        nlqs = torch.nn.functional.interpolate(nlqs, size=hr_fwd.shape[-3:], mode='trilinear', align_corners=False)
        

        nhr_fwd += torch.nn.functional.interpolate(nlqs, size=nhr_fwd.shape[-3:], mode='trilinear', align_corners=False)
        # print("hr_fwd:", nhr_fwd.shape)
        
        return nhr_fwd

    def forward(self, lqs):
        """Forward function for PhasorFlow.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).

        Returns:
            Tensor: Output HR sequence with shape (n, t, c, 4h, 4w).
        """

        n, t, _, h, w = lqs.size()
        
    

        # check whether the input is an extended sequence
        self.check_if_mirror_extended(lqs)

        # shallow feature extractions
        feats = {}
        
        feats['shallow'] = list(torch.chunk(self.feat_extract(lqs), t // self.clip_size, dim=1))
        # print(f"feat_extract: {feats['shallow'][0].shape}")
        flows_forward, flows_backward = self.compute_flow(lqs)
        phasors = self.compute_phasor(lqs) 

        # recurrent feature refinement
        updated_flows = {}
        iter_list = [1, 2]
        for iter_ in iter_list:
            for direction in ['backward', 'forward']:
                if direction == 'backward':
                    flows = flows_backward
                else:
                    flows = flows_forward if flows_forward is not None else flows_backward.flip(1)

                module_name = f'{direction}_{iter_}'
                feats[module_name] = []
                feats = self.propagate(lqs, feats, flows, phasors, module_name, updated_flows)

        refined_flow_fwd = self.refineflow(flows_forward, feats, updated_flows, 'forward', iter_list)
        refined_flow_bwd = self.refineflow(flows_backward, feats, updated_flows, 'backward', iter_list)

        # reconstruction
        return self.upsample(lqs[:, :, :1, :, :], feats, refined_flow_fwd, refined_flow_bwd), refined_flow_fwd, refined_flow_bwd
    
    def compute_phasorflow(self, lqs):
        n, t, _, h, w = lqs.size()

        # shallow feature extractions
        feats = {}

        feats['shallow'] = list(torch.chunk(self.feat_extract(lqs), t // self.clip_size, dim=1))
        # print(f"feat_extract: {feats['shallow'][0].shape}")
        flows_forward, flows_backward = self.compute_flow(lqs)
        phasors = self.compute_phasor(lqs) 

        # recurrent feature refinement
        updated_flows = {}
        iter_list = [1, 2]
        for iter_ in iter_list:
            for direction in ['backward', 'forward']:
                if direction == 'backward':
                    flows = flows_backward
                else:
                    flows = flows_forward if flows_forward is not None else flows_backward.flip(1)

                module_name = f'{direction}_{iter_}'
                feats[module_name] = []
                feats = self.propagate(lqs, feats, flows, phasors, module_name, updated_flows)

        refined_flow_fwd = self.refineflow(flows_forward, feats, updated_flows, 'forward', iter_list)
        refined_flow_bwd = self.refineflow(flows_backward, feats, updated_flows, 'backward', iter_list)
        
        phasors = torch.cat(phasors, dim=1).unsqueeze(2)
        
        return (refined_flow_fwd, refined_flow_bwd), phasors
    
    def instantiate_flownet_stage(self, config):
        model = instantiate_from_config(config)
        self.spynet = model
        # disable flownet training
        for param in self.spynet.parameters():
            param.requires_grad = False
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
    
    def get_input(self, batch):

        lqs = torch.stack(batch['lqs'], dim=1)
        gts = torch.stack(batch['gts'], dim=1)
        # lqs = batch['lqs']
        # gts = batch['gts']

        # print("lqs:", len(lqs), lqs[0].shape)

        lqs = lqs.to(memory_format=torch.contiguous_format).float()
        gts = gts.to(memory_format=torch.contiguous_format).float()

        gts = gts * 2.0 - 1.0
        lqs = lqs * 2.0 - 1.0
        
        return lqs, gts
    
    def training_step(self, batch, batch_idx):
        lqs, gts = self.get_input(batch)
        outputs, flows_forward, flows_backward = self(lqs)
        loss = self.loss(outputs, gts)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        lqs, gts = self.get_input(batch)
        outputs, flows_forward, flows_backward = self(lqs)
        
        # print("lqs:", lqs.shape)
        # print("outputs:", outputs.shape)
        
        lqs_np = lqs.detach().cpu().numpy()
        gts_np = gts.detach().cpu().numpy()
        outputs_np = outputs.detach().cpu().numpy()
        
        
        # print("ouput:", outputs.shape)
        loss = self.loss(outputs, gts)
        self.log('val_loss', loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        lqs, gts = self.get_input(batch)
        outputs, flows_forward, flows_backward = self(lqs)
        loss = self.loss(outputs, gts)
        self.log('test_loss', loss)
        return loss
    
 