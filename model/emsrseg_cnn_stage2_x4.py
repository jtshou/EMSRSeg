from .blocks import ConvCrossAttention2D, DeformableAlignModulewoFlow, PriorEncoder, PriorFusion, PriorModulation, SGFN, SpatialGate, ThreeDA, Upsample_seg
from emsrseg.temporal import target_sections
import torch.nn as nn
import torch.nn.functional as F
import torch
from .model_superhuman.basic import conv3dBlock, upsampleBlock, conv2dBlock
from .model_superhuman.residual import resBlock_pni, res2dBlock_Encoder
import math
from mmcv.ops import ModulatedDeformConv2d, modulated_deform_conv2d
from mmcv.cnn import constant_init
import basicsr.archs.arch_util as arch_util
from basicsr.archs.arch_util import ResidualBlockNoBN
import functools
from .model_superhuman2 import UNet_PNI_embedding as superhuman_seg_head
from .diffusion_backbone import diffusion_backbone
from .diffusion import diffusion

class EMSRSegCNNStage2(nn.Module):
    def __init__(self, in_planes=1,
                    filters=[28, 36, 48, 64, 80],    # [28, 36, 48, 64, 80], [48, 56, 64, 80, 96]
                    upsample_mode='bilinear',  # transposeS, bilinear
                    pad_mode='zero',
                    bn_mode='async',   # async or sync
                    relu_mode='elu',
                    init_mode='kaiming_normal',
                    bn_momentum=0.001,
                    do_embed=True,
                    upscale_rate=8,w_att = True,timestep_l=[200]):
        # filter_ratio: #filter_decode/#filter_encode
        super(EMSRSegCNNStage2, self).__init__()
        self.timestep_l = timestep_l
        self.w_att = w_att
        filters2 = filters[:1] + filters
        self.do_embed = do_embed
        self.depth = len(filters2) - 2
        self.upscale_rate = upscale_rate
        # 2D conv for anisotropic
        self.embed_in_sr_seg = conv3dBlock([in_planes],
                                    [filters2[0]],
                                    [(1, 5, 5)],
                                    [1],
                                    [(0, 2, 2)],
                                    [True],
                                    [pad_mode],
                                    [''],
                                    [relu_mode],
                                    init_mode,
                                    bn_momentum)
        # downsample stream, encoder
        self.conv0_sr_seg = resBlock_pni(filters2[0], filters2[1], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        self.pool0_sr_seg = nn.MaxPool3d((1, 2, 2), (1, 2, 2))
        self.conv1_sr_seg = resBlock_pni(filters2[1], filters2[2], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        self.pool1_sr_seg = nn.MaxPool3d((1, 2, 2), (1, 2, 2))
        self.conv2_sr_seg = resBlock_pni(filters2[2], filters2[3], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        self.pool2_sr_seg = nn.MaxPool3d((1, 2, 2), (1, 2, 2))
        self.conv3_sr_seg = resBlock_pni(filters2[3], filters2[4], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        self.pool3_sr_seg = nn.MaxPool3d((1, 2, 2), (1, 2, 2))
        self.center_sr_seg = resBlock_pni(filters2[4], filters2[5], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)

        # upsample stream
        self.up0_sr = upsampleBlock(filters2[5], filters2[4], (1, 2, 2), upsample_mode, init_mode=init_mode)
        self.cat0_sr = conv3dBlock([0], [filters2[4]], bn_mode=[bn_mode], relu_mode=[relu_mode],bn_momentum=bn_momentum)
        self.conv4_sr = resBlock_pni(filters2[4], filters2[4], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        self.up1_sr = upsampleBlock(filters2[4], filters2[3], (1, 2, 2), upsample_mode, init_mode=init_mode)
        self.cat1_sr = conv3dBlock([0], [filters2[3]], bn_mode=[bn_mode], relu_mode=[relu_mode],bn_momentum=bn_momentum)
        self.conv5_sr = resBlock_pni(filters2[3], filters2[3], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        self.up2_sr = upsampleBlock(filters2[3], filters2[2], (1, 2, 2), upsample_mode, init_mode=init_mode)
        self.cat2_sr = conv3dBlock([0], [filters2[2]], bn_mode=[bn_mode], relu_mode=[relu_mode],bn_momentum=bn_momentum)
        self.conv6_sr = resBlock_pni(filters2[2], filters2[2], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        self.up3_sr = upsampleBlock(filters2[2], filters2[1], (1, 2, 2), upsample_mode, init_mode=init_mode)
        self.cat3_sr = conv3dBlock([0], [filters2[1]], bn_mode=[bn_mode], relu_mode=[relu_mode],bn_momentum=bn_momentum)
        self.conv7_sr = resBlock_pni(filters2[1], filters2[1], pad_mode, bn_mode, relu_mode, init_mode, bn_momentum)
        self.conv8_sr = nn.Conv3d(filters2[1], 64, 1)
        self.sr_in = nn.Conv2d(1, 64, 3, 1, 1)

        # cross-attention fusion
        if self.w_att:
            self.cross_att_sr = ConvCrossAttention2D(64)
            self.self_att_sr = ConvCrossAttention2D(64)

        # align_DCN
        self.align_module_sr = DeformableAlignModulewoFlow(64,64,3,padding=1,deform_groups=8, max_residue_magnitude=10)

        # fusion
        self.feature_fusion_module_sr = ThreeDA(64,num_frame=3,center_frame_idx=1)

        ResidualBlock_noBN_f = functools.partial(ResidualBlockNoBN, num_feat=64)
        self.feature_extraction_sr = arch_util.make_layer(ResidualBlock_noBN_f, 5)
        self.recon_trunk_sr = arch_util.make_layer(ResidualBlock_noBN_f, 10)

        # seg head branch
        self.seg_emb = conv3dBlock([1], [64], kernel_size=[(1,3,3)], stride=[1], padding=[(0,1,1)], bn_mode=[bn_mode], relu_mode=[relu_mode],bn_momentum=bn_momentum)
        self.hr_encoder = PriorEncoder(64)
        self.prior_fusion_1 = PriorFusion(64)
        self.diff_encoder = PriorEncoder(64)
        self.diff_backbone = diffusion_backbone(64)
        self.diffusion_model = diffusion.DDPM(condition=self.diff_encoder,denoise=self.diff_backbone,feats=64,timesteps=1000,parameterization="eps",timestep_l=self.timestep_l)
        self.seg_head = superhuman_seg_head(in_planes=64)

        #upsample
        if upscale_rate==8:

            self.upconv1_sr = nn.Conv2d(64, 64 * 4, 3, 1, 1, bias=True)
            self.upconv2_sr = nn.Conv2d(64, 64 * 4, 3, 1, 1, bias=True)
            self.upconv3_sr = nn.Conv2d(64, 64 * 4, 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(2)
            self.conv_hr_sr = nn.Conv2d(64, 64, 3, 1, 1)
            self.conv_last_sr = nn.Conv2d(64, 1, 3, 1, 1)
            self.img_upsample_sr = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False)
            self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif upscale_rate==4:
            # æ³¨: Stage2 çåå²èµ° seg_head, forward ä¸­ä¸ä½¿ç¨ out_conv*;
            # x8 åæ¯å·²å°å¶æ³¨éæ, æ­¤å¤åæ ·å é¤,
            # å¦åè¿ 4 ä¸ªåæ°æ¶ä¸å°æ¢¯åº¦ -> DDP(find_unused_parameters=False) æ¥é
            self.upconv1_sr = nn.Conv2d(64, 64 * 4, 3, 1, 1, bias=True)
            self.upconv2_sr = nn.Conv2d(64, 64 * 4, 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(2)
            self.conv_hr_sr = nn.Conv2d(64, 64, 3, 1, 1)
            self.conv_last_sr = nn.Conv2d(64, 1, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, lr_context, gt=None):
        x = target_sections(lr_context)
        embed_in = self.embed_in_sr_seg(x)
        conv0 = self.conv0_sr_seg(embed_in)
        pool0 = self.pool0_sr_seg(conv0)
        conv1 = self.conv1_sr_seg(pool0)
        pool1 = self.pool1_sr_seg(conv1)
        conv2 = self.conv2_sr_seg(pool1)
        pool2 = self.pool2_sr_seg(conv2)
        conv3 = self.conv3_sr_seg(pool2)
        pool3 = self.pool3_sr_seg(conv3)
        center = self.center_sr_seg(pool3)
        # sr branch
        up0_sr = self.up0_sr(center)
        cat0_sr = self.cat0_sr(up0_sr + conv3)
        conv4_sr = self.conv4_sr(cat0_sr)
        up1_sr = self.up1_sr(conv4_sr)
        cat1_sr = self.cat1_sr(up1_sr + conv2)
        conv5_sr = self.conv5_sr(cat1_sr)
        up2_sr = self.up2_sr(conv5_sr)
        cat2_sr = self.cat2_sr(up2_sr + conv1)
        conv6_sr = self.conv6_sr(cat2_sr)
        up3_sr = self.up3_sr(conv6_sr)
        cat3_sr = self.cat3_sr(up3_sr + conv0)
        conv7_sr = self.conv7_sr(cat3_sr) #B,28,Z,H,W
        conv7_sr = self.conv8_sr(conv7_sr) #B,64,Z,H,W
        # cross-attention fusion
        B, C, Z, H, W = x.shape
        x_in = x.permute(0,2,1,3,4).contiguous().view(B*Z,C,H,W)
        conv_sr_in = self.feature_extraction_sr(self.lrelu(self.sr_in(x_in)))
        conv7_sr = conv7_sr.permute(0,2,1,3,4).contiguous().view(B*Z,-1,H,W)
        if self.w_att:
            conv_sr_att = self.cross_att_sr(query=conv7_sr,context=conv_sr_in).view(B,Z,-1,H,W) #B,Z,64,H,W
            conv_sr_in_f_0 = self.feature_extraction_sr(self.lrelu(self.sr_in(lr_context[:, :, 0])))
            conv_sr_in_b_1 = self.feature_extraction_sr(self.lrelu(self.sr_in(lr_context[:, :, -1])))
            conv_sr_att_f_0 = self.self_att_sr(query=conv_sr_in_f_0,context=conv_sr_in_f_0)
            conv_sr_att_b_1 = self.self_att_sr(query=conv_sr_in_b_1, context=conv_sr_in_b_1)
        else:
            raise ValueError
        # align_DCN
        align_feat = []
        for i in range(Z):
            if i == 0:
                forward_feat_0 = conv_sr_att_f_0
                center_feat = conv_sr_att[:, i]
                backward_feat_1 = conv_sr_att[:, i + 1]
            elif i == Z - 1:
                forward_feat_0 = conv_sr_att[:, i - 1]
                center_feat = conv_sr_att[:, i]
                backward_feat_1 = conv_sr_att_b_1
            else:
                forward_feat_0 = conv_sr_att[:, i - 1]
                center_feat = conv_sr_att[:, i]
                backward_feat_1 = conv_sr_att[:, i + 1]
            forward_feat_align_0 = self.align_module_sr(center_feat, forward_feat_0)
            backward_feat_align_1 = self.align_module_sr(center_feat, backward_feat_1)
            center_feat = self.align_module_sr(center_feat, center_feat)
            fusion_feat = self.recon_trunk_sr(self.feature_fusion_module_sr(torch.stack([forward_feat_align_0, center_feat, backward_feat_align_1],dim=1)))
            align_feat.append(fusion_feat)
        # upsample
        outputs = []
        if self.upscale_rate == 8:
            for i in range(Z):
                hr = self.lrelu(self.pixel_shuffle(self.upconv1_sr(align_feat[i])))
                hr = self.lrelu(self.pixel_shuffle(self.upconv2_sr(hr)))
                hr = self.lrelu(self.pixel_shuffle(self.upconv3_sr(hr)))
                hr = self.lrelu(self.conv_hr_sr(hr))
                hr = self.conv_last_sr(hr)
                hr += F.interpolate(x[:, :, i, :, :], scale_factor=8, mode='bilinear', align_corners=False)
                outputs.append(hr)

        elif self.upscale_rate ==4:
            for i in range(Z):
                hr = self.lrelu(self.pixel_shuffle(self.upconv1_sr(align_feat[i])))
                hr = self.lrelu(self.pixel_shuffle(self.upconv2_sr(hr)))
                hr = self.lrelu(self.conv_hr_sr(hr))
                hr = self.conv_last_sr(hr)
                hr += F.interpolate(x[:, :, i, :, :], scale_factor=4, mode='bilinear', align_corners=False)
                outputs.append(hr)
        sr = torch.stack(outputs,dim=2)
        if self.training:
            hr_prior = self.hr_encoder(gt.permute(0, 2, 1, 3, 4).contiguous().view(-1, 1, H * self.upscale_rate, W * self.upscale_rate))
        sr_for_seg = self.seg_emb(sr).permute(0, 2, 1, 3, 4).contiguous().view(B * Z, 64,H * self.upscale_rate, W * self.upscale_rate)
        structural_prior = self.diffusion_model(sr.contiguous().permute(0, 2, 1, 3, 4).contiguous().view(-1, 1, H * self.upscale_rate, W * self.upscale_rate).contiguous())
        sr_for_seg = self.prior_fusion_1(sr_for_seg, structural_prior).view(B,Z,64,H * self.upscale_rate,W * self.upscale_rate).contiguous().permute(0,2,1,3,4)
        seg_final = self.seg_head(sr_for_seg)
        if self.training:
            return seg_final, hr_prior, structural_prior
        else:
            return sr,seg_final

        return out

        return out

        return supp_align
        return out

        return out

        return fusion[:,:,0,:,:]

        return feat

        return fea

        return x

        return x1 * x2

        return x

        return x

        return bce + self.lambda_sharp * sharp

