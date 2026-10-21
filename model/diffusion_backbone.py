import torch
import torch.nn as nn


class diffusion_backbone(nn.Module):
    def __init__(self,feats = 64, max_period=1000):
        super(diffusion_backbone, self).__init__()
        self.max_period=max_period
        self.mlp=nn.Sequential(
            nn.Linear(feats*4+1, feats*4),
            nn.LeakyReLU(0.1, False),
            nn.Linear(feats*4 , feats*4 ),
            nn.LeakyReLU(0.1, False),
            nn.Linear(feats*4 , feats*4 ),
            nn.LeakyReLU(0.1, False),
            nn.Linear(feats*4 , feats*4 ),
            nn.LeakyReLU(0.1, False),
            nn.Linear(feats*4 , feats*4 ),
            nn.LeakyReLU(0.1, False),
            nn.Linear(feats*4 , feats*4 ),
            nn.LeakyReLU(0.1, False),
        )

    def forward(self, x, t):
        '''
        x:BZ*C;
        t:BZ
        out:BZ*C
        '''
        t=t.float()
        t =t/self.max_period
        t=t.unsqueeze(1)
        input_f = torch.cat([x, t],dim=1).contiguous()
        fea = self.mlp(input_f)
        return fea
