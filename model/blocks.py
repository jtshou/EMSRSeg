"""Shared semantic attention, BWA, and prior fusion modules."""
import math
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops import ModulatedDeformConv2d, modulated_deform_conv2d
from mmcv.cnn import constant_init
import basicsr.archs.arch_util as arch_util
from basicsr.archs.arch_util import ResidualBlockNoBN
from .model_superhuman.basic import conv2dBlock
from .model_superhuman.residual import res2dBlock_Encoder

class Upsample_seg(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Upsample_seg, self).__init__()
        self.up_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1),
            nn.LeakyReLU()
        )

    def forward(self, x):
        #[B,C,Z,H,W]->[B,C2,Z,H*2,W*2]
        B,C,Z,H,W = x.shape
        x = F.interpolate(x.permute(0,2,1,3,4).contiguous().view(-1,C,H,W),scale_factor=2, mode='bilinear',align_corners=True)
        x = self.up_conv(x).view(B,Z,-1,H*2,W*2).permute(0,2,1,3,4).contiguous()
        return x

class ConvCrossAttention2D(nn.Module):
    def __init__(self, channels, proj_channels=64, kernel_size=3, use_layernorm=False):
        super().__init__()
        self.channels = channels
        self.proj_channels = proj_channels
        self.kernel_size = kernel_size

        # projection convs
        self.query_conv = nn.Conv2d(channels, self.proj_channels, kernel_size, padding=kernel_size//2)
        self.context_conv = nn.Conv2d(channels, self.proj_channels, kernel_size, padding=kernel_size//2)
        self.bias_conv = nn.Conv2d(channels, proj_channels, kernel_size, padding=kernel_size//2)

        # optional LayerNorm
        self.use_ln = use_layernorm
        if use_layernorm:
            self.ln_q = nn.LayerNorm(self.proj_channels)
            self.ln_ctx = nn.LayerNorm(self.proj_channels)

        # output projection
        self.out_conv = nn.Conv2d(self.proj_channels, channels, 1)

    def forward(self, query, context):
        """
        query: (B,C,H,W)
        context: (B,C,H,W)
        """
        q = self.query_conv(query)      # (B, proj_channels, H, W)
        k = self.context_conv(context)  # (B, proj_channels, H, W)

        if self.use_ln:
            B,C,H,W = q.shape
            q = self.ln_q(q.permute(0,2,3,1).reshape(B*H*W,C)).reshape(B,H,W,C).permute(0,3,1,2)
            k = self.ln_ctx(k.permute(0,2,3,1).reshape(B*H*W,C)).reshape(B,H,W,C).permute(0,3,1,2)

        # element-wise attention
        attn = torch.sigmoid(q * k)  # (B, proj_channels, H, W)

        # apply attention to context
        out = attn * context + self.bias_conv(query)
        out = self.out_conv(out) + context  # back to original channels

        return out

class DeformableAlignModulewoFlow(ModulatedDeformConv2d):
    def __init__(self, *args, **kwargs):
        self.max_residue_magnitude = kwargs.pop('max_residue_magnitude', 10)

        super(DeformableAlignModulewoFlow, self).__init__(*args, **kwargs)
        self.conv_offset = nn.Sequential(
            nn.Conv2d(2 * self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.out_channels, 27 * self.deform_groups, 3, 1, 1),
        )

        self.init_offset()

    def init_offset(self):
        constant_init(self.conv_offset[-1], val=0, bias=0)

    def forward(self, ref, supp):
        cond = torch.cat((ref, supp), dim=1)
        cond = self.conv_offset(cond)
        offset1, offset2, mask = torch.chunk(cond, 3, dim=1)
        offset = self.max_residue_magnitude * torch.tanh(torch.cat((offset1, offset2), dim=1))
        mask = torch.sigmoid(mask)
        supp_align = modulated_deform_conv2d(supp, offset, mask, self.weight, self.bias,
                                             self.stride, self.padding,
                                             self.dilation, self.groups,
                                             self.deform_groups)
        return supp_align

class ThreeDA(nn.Module):
    def __init__(self, num_feat=64, num_frame=5, center_frame_idx=2):
        super(ThreeDA, self).__init__()
        self.center_frame_idx = center_frame_idx
        # temporal attention (before fusion conv)
        self.temporal_attn1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.temporal_attn2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.feat_fusion = nn.Conv2d(num_frame * num_feat, num_feat, 1, 1)

        # fusion 3D_conv
        self.conv3D_1 = nn.Conv3d(num_frame, num_frame, kernel_size=1, bias=True)
        self.conv3D_2 = nn.Conv3d(num_frame, num_frame, kernel_size=1, bias=True)
        self.conv3D_fusion_1 = nn.Conv2d(num_frame * num_feat, num_feat, kernel_size=1, bias=True)
        self.conv3D_fusion_2 = nn.Conv2d(num_frame * num_feat, num_feat, kernel_size=1, bias=True)
        self.conv2D_fusion_3 = nn.Conv2d(num_feat, num_feat, kernel_size=1, bias=True)

        # spatial attention (after fusion conv)
        self.max_pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.avg_pool = nn.AvgPool2d(3, stride=2, padding=1)
        self.spatial_attn1 = nn.Conv2d(num_frame * num_feat, num_feat, 1)
        self.spatial_attn2 = nn.Conv2d(num_feat * 2, num_feat, 1)
        self.spatial_attn3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.spatial_attn4 = nn.Conv2d(num_feat, num_feat, 1)
        self.spatial_attn5 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.spatial_attn_l1 = nn.Conv2d(num_feat, num_feat, 1)
        self.spatial_attn_l2 = nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1)
        self.spatial_attn_l3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.spatial_attn_add1 = nn.Conv2d(num_feat, num_feat, 1)
        self.spatial_attn_add2 = nn.Conv2d(num_feat, num_feat, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, aligned_feat):
        b, t, c, h, w = aligned_feat.size()
        # temporal attention
        embedding_ref = self.temporal_attn1(
            aligned_feat[:, self.center_frame_idx, :, :, :].clone())
        embedding = self.temporal_attn2(aligned_feat.view(-1, c, h, w))
        embedding = embedding.view(b, t, -1, h, w)  # (b, t, c, h, w)

        corr_l = []  # correlation list
        for i in range(t):
            emb_neighbor = embedding[:, i, :, :, :]
            corr = torch.sum(emb_neighbor * embedding_ref, 1)  # (b, h, w)
            corr_l.append(corr.unsqueeze(1))  # (b, 1, h, w)
        corr_prob = torch.sigmoid(torch.cat(corr_l, dim=1))  # (b, t, h, w)
        corr_prob = corr_prob.unsqueeze(2).expand(b, t, c, h, w)
        corr_prob = corr_prob.contiguous().view(b, -1, h, w)  # (b, t*c, h, w)
        aligned_feat = aligned_feat.view(b, -1, h, w) * corr_prob

        # fusion
        feat = self.lrelu(self.feat_fusion(aligned_feat))
        fea_3d1 = self.lrelu(self.conv3D_1(aligned_feat.view(b, t, -1, h, w)))
        fea_3d1 = self.lrelu(self.conv3D_fusion_1(fea_3d1.view(b, -1, h, w)))
        fea_3d2 = self.lrelu(self.conv3D_2(aligned_feat.view(b, t, -1, h, w)))
        fea_3d2 = self.lrelu(self.conv3D_fusion_2(fea_3d2.view(b, -1, h, w)))#B,C,H,W

        feat = feat + fea_3d1#B,C,H,W
        fea_3d3 = self.conv2D_fusion_3(feat)#B,C,H,W

        # spatial attention
        attn = self.lrelu(self.spatial_attn1(aligned_feat))
        attn_max = self.max_pool(attn)
        attn_avg = self.avg_pool(attn)
        attn = self.lrelu(
            self.spatial_attn2(torch.cat([attn_max, attn_avg], dim=1)))
        # pyramid levels
        attn_level = self.lrelu(self.spatial_attn_l1(attn))
        attn_max = self.max_pool(attn_level)
        attn_avg = self.avg_pool(attn_level)
        attn_level = self.lrelu(
            self.spatial_attn_l2(torch.cat([attn_max, attn_avg], dim=1)))
        attn_level = self.lrelu(self.spatial_attn_l3(attn_level))
        attn_level = self.upsample(attn_level)

        attn = self.lrelu(self.spatial_attn3(attn)) + attn_level
        attn = self.lrelu(self.spatial_attn4(attn))
        attn = self.upsample(attn)
        attn = self.spatial_attn5(attn)
        attn_add = self.spatial_attn_add2(
            self.lrelu(self.spatial_attn_add1(attn)))
        attn = torch.sigmoid(attn)

        feat = feat * attn * 2 + attn_add + fea_3d2 + fea_3d3
        return feat

class PriorEncoder(nn.Module):
    def __init__(self,feats = 64):
        super(PriorEncoder, self).__init__()
        self.C = nn.Sequential(
            conv2dBlock([1],[feats],kernel_size=[8],stride=[8],padding=[0], pad_mode=['zero'], bn_mode=['async'], relu_mode=['elu'], bn_momentum=0.001),
            res2dBlock_Encoder(feats, feats),
            res2dBlock_Encoder(feats, feats),
            res2dBlock_Encoder(feats, feats),
            res2dBlock_Encoder(feats, feats),
            res2dBlock_Encoder(feats, feats),
            res2dBlock_Encoder(feats, feats),
            res2dBlock_Encoder(feats, feats),
            res2dBlock_Encoder(feats, feats),
            nn.Conv2d(feats, feats*2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(feats*2),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(feats*2, feats*4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(feats*4),
            nn.LeakyReLU(0.1, True),
            nn.AdaptiveAvgPool2d(1)
        )

        self.mlp = nn.Sequential(
            nn.Linear(feats * 4, feats * 4),
            nn.LeakyReLU(0.1, True),
            nn.Linear(feats * 4, feats * 4),
            nn.LeakyReLU(0.1, True),
            nn.Linear(feats * 4, feats * 4),
            nn.LeakyReLU(0.1, True),
            nn.Linear(feats * 4, feats * 4),
            nn.LeakyReLU(0.1, True)
        )

    def forward(self, image):
        pooled_features = self.C(image).squeeze(-1).squeeze(-1)
        prior = self.mlp(pooled_features)
        return prior

class PriorModulation(nn.Module):
    def __init__(self, dim=64):
        super().__init__()

        self.kernel = nn.Sequential(
            nn.Linear(256, dim*2, bias=False),
        )

    def forward(self, x, prior):
        prior=self.kernel(prior).unsqueeze(-1).unsqueeze(-1)
        scale,shift=prior.chunk(2, dim=1)
        x = x*scale+shift
        return x

class SpatialGate(nn.Module):
    """ Spatial-Gate.
    Args:
        dim (int): Half of input channels.
    """
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim) # DW Conv

    def forward(self, x):
        '''
        [B*Z,H,W,C]
        '''
        # Split
        x1, x2 = x.chunk(2, dim = -1)
        x2 = self.conv(self.norm(x2).permute(0,3,1,2)).permute(0,2,3,1)
        return x1 * x2

class SGFN(nn.Module):
    """ Spatial-Gate Feed-Forward Network.
    Args:
        in_features (int): Number of input channels.
        hidden_features (int | None): Number of hidden channels. Default: None
        out_features (int | None): Number of output channels. Default: None
        act_layer (nn.Module): Activation layer. Default: nn.GELU
        drop (float): Dropout rate. Default: 0.0
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.sg = SpatialGate(hidden_features//2)
        self.fc2 = nn.Linear(hidden_features//2, out_features)

    def forward(self, x):
        """
        x:[B*Z,C,H,W]
        """
        x = x.permute(0,2,3,1)
        x = self.fc1(x)
        x = self.act(x)
        x = self.sg(x)
        x = self.fc2(x)
        x = x.permute(0,3,1,2)
        return x

class PriorFusion(nn.Module):
    def __init__(self, dim=64, expansion_factor=4,act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()

        self.norm1 = norm_layer(dim)

        ffn_hidden_dim = int(dim * expansion_factor)
        self.ffn = SGFN(in_features=dim, hidden_features=ffn_hidden_dim, out_features=dim, act_layer=act_layer)
        self.cdim1 = PriorModulation(dim)

    def forward(self, x, prior):
        """
        x:[B*Z,C,H,W]
        prior:[B*Z,256]
        out: [B*Z,C,H,W]
        """
        x_norm = self.norm1(x.permute(0,2,3,1)).permute(0,3,1,2)
        x = x + self.ffn(self.cdim1(x_norm, prior))

        return x
