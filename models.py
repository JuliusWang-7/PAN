'''
VoxelMorph

Original code retrieved from:
https://github.com/voxelmorph/voxelmorph

Original paper:
Balakrishnan, G., Zhao, A., Sabuncu, M. R., Guttag, J., & Dalca, A. V. (2019).
VoxelMorph: a learning framework for deformable medical image registration.
IEEE transactions on medical imaging, 38(8), 1788-1800.

Modified and tested by:
Zhuoyuan Wang
2018222020@email.szu.edu.cn
Shenzhen University
'''

import torch
import torch.nn as nn
import torch.nn.functional as nnf
import math
import numpy as np
from torch.distributions.normal import Normal
from timm.models.layers import DropPath, trunc_normal_, to_3tuple
import torch.utils.checkpoint as checkpoint


class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)


class VecInt(nn.Module):
    """
    Integrates a vector field via scaling and squaring.
    """

    def __init__(self, inshape, nsteps=7):
        super().__init__()

        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = SpatialTransformer(inshape)

    def forward(self, vec):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec


class ResizeTransform(nn.Module):
    """
    调整变换的大小，这涉及调整矢量场的大小并重新缩放它。
    """

    def __init__(self, vel_resize, ndims):
        super().__init__()
        self.factor = 1.0 / vel_resize
        self.mode = 'linear'
        if ndims == 2:
            self.mode = 'bi' + self.mode
        elif ndims == 3:
            self.mode = 'tri' + self.mode

    def forward(self, x):
        if self.factor < 1:
            # resize first to save memory
            x = nnf.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)
            x = self.factor * x

        elif self.factor > 1:
            # multiply first to save memory
            x = self.factor * x
            x = nnf.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)

        # don't do anything if resize is 1
        return x


class ConvBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, in_channels, out_channels,kernal_size=3, stride=1, padding=1, alpha=0.1):
        super().__init__()

        self.main = nn.Conv3d(in_channels, out_channels, kernal_size, stride, padding)
        self.activation = nn.LeakyReLU(alpha)

    def forward(self, x):
        out = self.main(x)
        out = self.activation(out)
        return out


class ConvInsBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, in_channels, out_channels,kernal_size=3, stride=1, padding=1, alpha=0.1):
        super().__init__()

        self.main = nn.Conv3d(in_channels, out_channels, kernal_size, stride, padding)
        self.norm = nn.InstanceNorm3d(out_channels)
        self.activation = nn.LeakyReLU(alpha)

    def forward(self, x):
        out = self.main(x)
        out = self.norm(out)
        out = self.activation(out)
        return out


def relative_pos_dis(height=32, weight=32, sita=0.9):
    coords_h = torch.arange(height)
    coords_w = torch.arange(weight)
    coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww # 0 is 32 * 32 for h, 1 is 32 * 32 for w
    coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
    dis = (relative_coords[:, :, 0].float()/height) ** 2 + (relative_coords[:, :, 1].float()/weight) ** 2
    #dis = torch.exp(-dis*(1/(2*sita**2)))
    return  dis


class SEblock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEblock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        y = x * y.expand(x.size())
        return y


class Encoder(nn.Module):
    """
    Main model
    """

    def __init__(self, in_channel=1, first_out_channel=4):
        super(Encoder, self).__init__()

        c = first_out_channel

        self.conv0 = nn.Sequential(
            ConvBlock(in_channel, c),
            ConvInsBlock(c, 2*c),
            ConvInsBlock(2*c, 2*c),
            SEblock(2*c)
        )

        self.conv1 = nn.Sequential(
            nn.AvgPool3d(2),
            ConvInsBlock(2 * c, 4 * c),
            ConvInsBlock(4 * c, 4 * c),
            SEblock(4 * c)
        )

        self.conv2 = nn.Sequential(
            nn.AvgPool3d(2),
            ConvInsBlock(4 * c, 8 * c),
            ConvInsBlock(8 * c, 8 * c),
            SEblock(8 * c)
        )

        self.conv3 = nn.Sequential(
            nn.AvgPool3d(2),
            ConvInsBlock(8 * c, 16* c),
            ConvInsBlock(16 * c, 16 * c),
            SEblock(16 * c)
        )

        self.conv4 = nn.Sequential(
            nn.AvgPool3d(2),
            ConvInsBlock(16 * c, 32 * c),
            ConvInsBlock(32 * c, 32 * c),
            SEblock(32 * c)
        )

    def forward(self, x):
        out0 = self.conv0(x)  # 1
        out1 = self.conv1(out0)  # 1/2
        out2 = self.conv2(out1)  # 1/4
        out3 = self.conv3(out2)  # 1/8
        out4 = self.conv4(out3)  # 1/8

        return out0, out1, out2, out3, out4


class PositionalEncodingLayer(nn.Module):
    def __init__(self, in_channels, dim=6, norm=nn.LayerNorm):
        super().__init__()
        self.norm = norm(dim)
        self.proj = nn.Linear(in_channels, dim)
        self.proj.weight = nn.Parameter(Normal(0, 1e-5).sample(self.proj.weight.shape))
        self.proj.bias = nn.Parameter(torch.zeros(self.proj.bias.shape))

    def forward(self, feat):
        feat = feat.permute(0, 2, 3, 4, 1)
        feat = self.norm(self.proj(feat))
        return feat


class DFIBlock(nn.Module):
    def __init__(self, in_channels, channels):
        super(DFIBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels, 3, 3, 1, 1)
        self.conv.weight = nn.Parameter(Normal(0, 1e-5).sample(self.conv.weight.shape))
        self.conv.bias = nn.Parameter(torch.zeros(self.conv.bias.shape))

    def forward(self, x):
        x = self.conv(x)
        return x


class LAT(nn.Module):
    def __init__(self, dim, num_heads, kernel_size=3, qk_scale=None, use_rpb=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // self.num_heads
        self.scale = qk_scale or self.head_dim ** -0.5
        self.kernel_size = kernel_size
        self.win_size = kernel_size // 2
        self.mid_cell = kernel_size - 1
        self.rpb_size = kernel_size
        self.use_rpb = use_rpb
        if use_rpb:
            self.rpb = nn.Parameter(torch.zeros(self.num_heads, self.rpb_size, self.rpb_size, self.rpb_size))
        vectors = [torch.arange(-s // 2 + 1, s // 2 + 1) for s in [kernel_size] * 3]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids, -1).type(torch.FloatTensor)
        v = grid.reshape(self.kernel_size ** 3, 3)
        self.register_buffer('v', v)


    def apply_pb(self, attn, N):
        # attn: B, N, self.num_heads, 1, tokens = (3x3x3)
        bias_idx = torch.arange(self.rpb_size**3).unsqueeze(-1).repeat(N, 1)
        return attn + self.rpb.flatten(1,3)[:, bias_idx].reshape(self.num_heads, N, 1, self.rpb_size**3).transpose(0,1)

    def forward(self, q, k, return_attn_matrix=False):
        B, H, W, T, C = q.shape
        N = H * W * T
        num_tokens = int(self.kernel_size ** 3)

        Q = q.reshape(B, H, W, T, self.num_heads, C // self.num_heads).permute(0, 4, 1, 2, 3, 5) * self.scale
        q = q.reshape(B, N, self.num_heads, C // self.num_heads, 1).transpose(3, 4) * self.scale  # 1, N, heads, 1, head_dim
        pd = self.kernel_size - 1  # 2
        pdr = pd // 2  # 1

        k = k.permute(0, 4, 1, 2, 3)  # C, H, W, T
        k = nnf.pad(k, (pdr, pdr, pdr, pdr, pdr, pdr))  # 1, C, H+2, W+2, T+2
        if return_attn_matrix:
            K = k.reshape(B, self.num_heads, C // self.num_heads, H + pd, W + pd, T + pd).permute(0, 1, 3, 4, 5, 2)
        k = k.flatten(0, 1)  # C, H+2, W+2, T+2
        k = k.unfold(1, self.kernel_size, 1).unfold(2, self.kernel_size, 1).unfold(3, self.kernel_size, 1).permute(0, 4, 5, 6, 1, 2, 3)  # C, 3, 3, 3, H, W, T
        k = k.reshape(B, self.num_heads, C // self.num_heads, num_tokens, N)  # memory boom
        k = k.permute(0, 4, 1, 3, 2)  # (B, N, heads, num_tokens, head_dim)

        attn = (q @ k.transpose(-2, -1))  # =>B x N x heads x 1 x num_tokens
        if self.use_rpb:
            attn = self.apply_pb(attn, N)
        attn = attn.softmax(dim=-1)
        x = (attn @ self.v)  # B x N x heads x 1 x 3
        x = x.reshape(B, H, W, T, self.num_heads * 3).permute(0, 4, 1, 2, 3)
        if return_attn_matrix:
            Q = nnf.normalize(Q.squeeze(0).flatten(1), dim=0)
            QQT = Q @ Q.transpose(0, 1)
            K = nnf.normalize(K.squeeze(0).flatten(1), dim=0)
            KKT = K @ K.transpose(0, 1)
            return x, nnf.normalize(QQT), nnf.normalize(KKT)
        else:
            return x

class DownBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.downsample1 = nn.AvgPool3d(kernel_size=2)
        self.downsample2 = nn.AvgPool3d(kernel_size=4)
        self.downsample3 = nn.AvgPool3d(kernel_size=8)
        self.downsample4 = nn.AvgPool3d(kernel_size=16)

    def forward(self, Img):

        return Img, self.downsample1(Img), self.downsample2(Img), self.downsample3(Img), self.downsample4(Img)


class PAN(nn.Module):
    def __init__(self,
                 inshape=(160, 192, 160),
                 in_channel=1,
                 channels=8,
                 head_dim=6,
                 num_heads=[8, 4, 2, 1, 1],
                 scale=1):
        super(PAN, self).__init__()
        self.channels = channels
        self.step = 7
        self.inshape = inshape

        dims = len(inshape)

        c = self.channels
        self.encoder = Encoder(in_channel=in_channel, first_out_channel=c)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample_trilin = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)#nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

        self.peblock1 = PositionalEncodingLayer(2 * c + 1, dim=head_dim*num_heads[4])
        self.flow_peblock1 = PositionalEncodingLayer(3, dim=head_dim * num_heads[4] * 27 // 2)
        self.lat1 = LAT(head_dim*num_heads[3], num_heads[4], qk_scale=scale)
        self.dfi1 = DFIBlock(3 * num_heads[4], 3 * 2 * num_heads[4])

        self.peblock2 = PositionalEncodingLayer(4 * c + 1, dim=head_dim * num_heads[3])
        self.flow_peblock2 = PositionalEncodingLayer(3, dim=head_dim * num_heads[3] * 27 // 2)
        self.lat2 = LAT(head_dim * num_heads[3], num_heads[3], qk_scale=scale)
        self.dfi2 = DFIBlock(3 * num_heads[3], 3 * 2 * num_heads[3])

        self.peblock3 = PositionalEncodingLayer(8 * c + 1, dim=head_dim * num_heads[2])
        self.flow_peblock3 = PositionalEncodingLayer(3, dim=head_dim * num_heads[2] * 27 // 2)
        self.lat3 = LAT(head_dim * num_heads[2], num_heads[2], qk_scale=scale)
        self.dfi3 = DFIBlock(3 * num_heads[2], 3 * num_heads[2] * 2)

        self.peblock4 = PositionalEncodingLayer(16 * c + 1, dim=head_dim * num_heads[1])
        self.flow_peblock4 = PositionalEncodingLayer(3, dim=head_dim * num_heads[1] * 27 // 2)
        self.lat4 = LAT(head_dim * num_heads[1], num_heads[1], qk_scale=scale)
        self.dfi4 = DFIBlock(3 * num_heads[1], 3 * num_heads[1] * 2)

        self.peblock5 = PositionalEncodingLayer(32 * c + 1, dim=head_dim * num_heads[0])
        self.lat5 = LAT(head_dim * num_heads[0], num_heads[0], qk_scale=scale)
        self.dfi5 = DFIBlock(3 * num_heads[0], 3 * num_heads[0] * 2)

        self.transformer = nn.ModuleList()
        for i in range(4):
            self.transformer.append(SpatialTransformer([s // 2**i for s in inshape]))

        self.Down = DownBlock()

    def forward(self, moving, fixed):

        # encode stage
        M1, M2, M3, M4, M5 = self.encoder(moving)
        F1, F2, F3, F4, F5 = self.encoder(fixed)

        # downsample_stage
        m1, m2, m3, m4, m5 = self.Down(moving)
        f1, f2, f3, f4, f5 = self.Down(fixed)

        fields = []

        q5, k5 = self.peblock5(torch.cat((F5, f5), dim=1)), self.peblock5(torch.cat((M5, m5), dim=1))
        w = self.lat5(q5, k5)
        w = self.dfi5(w)
        flow = self.upsample_trilin(2*w)

        M4 = self.transformer[3](torch.cat((M4, m4), dim=1), flow)
        q4, k4 = self.peblock4(torch.cat((F4, f4), dim=1)), self.peblock4(M4)
        w = self.lat4(q4, k4)
        w = self.dfi4(w)
        flow = self.upsample_trilin(2 *(self.transformer[3](flow, w)+w))

        M3 = self.transformer[2](torch.cat((M3, m3), dim=1), flow)
        q3, k3 = self.peblock3(torch.cat((F3, f3), dim=1)), self.peblock3(M3)
        w, QQT, KKT = self.lat3(q3, k3, return_attn_matrix=True)
        w = self.dfi3(w)
        flow = self.upsample_trilin(2 * (self.transformer[2](flow, w) + w))

        M2 = self.transformer[1](torch.cat((M2, m2), dim=1), flow)
        q2,k2 = self.peblock2(torch.cat((F2, f2), dim=1)), self.peblock2(M2)
        w=self.lat2(q2, k2)
        w = self.dfi2(w)
        flow = self.upsample_trilin(2 *(self.transformer[1](flow, w)+w))

        M1 = self.transformer[0](torch.cat((M1, m1), dim=1), flow)
        q1, k1 = self.peblock1(torch.cat((F1, f1), dim=1)), self.peblock1(M1)
        w = self.lat1(q1, k1)
        w = self.dfi1(w)
        flow = self.transformer[0](flow, w)+w
        # flow = w

        y_moved = self.transformer[0](moving, flow)

        return y_moved, flow, QQT, KKT

if __name__ == '__main__':
    inshape = (1, 1, 160, 160, 192)
    torch.cuda.set_device(0)
    model = PAN(inshape[2:]).cuda()
    A = torch.ones(inshape)
    B = torch.ones(inshape)
    out, flow = model(A.cuda(), B.cuda())
    print(out.shape, flow.shape)
