import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def gaussian_kernel_3d(kernel_size=5, sigma=1.0, device="cuda"):
    """
    Create a 3D Gaussian kernel normalized to sum=1
    """
    ax = torch.arange(kernel_size, device=device) - kernel_size // 2
    xx, yy, zz = torch.meshgrid(ax, ax, ax, indexing='ij')
    kernel = torch.exp(-(xx**2 + yy**2 + zz**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel.unsqueeze(0).unsqueeze(0)  # shape (1,1,K,K,K)


class AffinityWeightedSRLoss(nn.Module):

    def __init__(self, base_loss='l1'):
        super().__init__()
        self.base_loss = base_loss

    def compute_affinity_weight(self, A_gt, A_pred):
        """
        Args:
            A_gt:   (B, 3, Z, H, W) ground-truth affinity
            A_pred: (B, 3, Z, H, W) predicted affinity
        Returns:
            w: (B, 1, Z, H, W) weight map
        """
        
        affinity_strength = 1.0 - A_pred.mean(dim=1, keepdim=True)  # (B, 1, Z, H, W)
        #affinity_strength = A_pred.mean(dim=1, keepdim=True)  # (B, 1, Z, H, W)
        kernel = gaussian_kernel_3d()
        boundary_smooth = F.conv3d(affinity_strength, kernel, padding=5 // 2, groups=1)

        
        affinity_diff = torch.abs(A_gt - A_pred).mean(dim=1, keepdim=True)  # (B, 1, Z, H, W)

        #w = (1.0 + 2 * boundary_smooth + 5 * affinity_diff).detach()
        w = (1.0 + 2 * boundary_smooth).detach()

        return w

    def forward(self, I_sr, I_hr, A_gt, A_pred):
        """
        I_sr: (B, C, Z, H, W)  predicted super-resolved image
        I_hr: (B, C, Z, H, W)  high-resolution ground truth
        A_pred, A_gt: (B, 3, Z, H, W) ground-truth affinity
        """
        # compute affinity-based weights
        w = self.compute_affinity_weight(A_gt, A_pred)

        diff = I_sr - I_hr
        if self.base_loss == 'l1':
            loss_map = torch.abs(diff)
        elif self.base_loss == 'l2':
            loss_map = diff ** 2
        else:
            raise ValueError("base_loss must be 'l1' or 'l2'")

        loss = torch.mean(w * loss_map)
        return loss


if __name__ == "__main__":
    import cv2
    im = cv2.imread(r'C:\Users\Sjt\Desktop\000.tif',cv2.IMREAD_GRAYSCALE)
    pre = np.zeros((1,3,1,1024,1024))
    gt = np.zeros((1,3,1,1024,1024))
    pre[0,0,0] = im[0:1024,0:1024]
    pre[0,1,0] = im[0:1024, 1024:2048]
    pre[0,2,0] = im[0:1024, 2048:3072]
    gt[0, 0, 0] = im[1024:2048, 0:1024]
    gt[0, 1, 0] = im[1024:2048, 1024:2048]
    gt[0, 2, 0] = im[1024:2048, 2048:3072]
    gt = (torch.from_numpy(gt)/255).type(torch.float32)
    pre = (torch.from_numpy(pre)/255).type(torch.float32)

    def compute_affinity_weight(A_gt, A_pred):
        affinity_strength = 1.0 - A_gt.mean(dim=1, keepdim=True)  # (B, 1, Z, H, W)
        #affinity_strength = A_gt.mean(dim=1, keepdim=True)  # (B, 1, Z, H, W)
        kernel = gaussian_kernel_3d(device='cpu')
        boundary_smooth = F.conv3d(affinity_strength, kernel, padding=5 // 2, groups=1)

        
        affinity_diff = torch.abs(A_gt - A_pred).mean(dim=1, keepdim=True)  # (B, 1, Z, H, W)

        w = (1.0 + 2 * boundary_smooth ).detach() #+ 5 * affinity_diff

        return w

    w = compute_affinity_weight(gt,pre)
    w_img = w.squeeze()  # (H, W)
    
    w_img = w_img/ w_img.max()
    
    w_img_np = (w_img.cpu().numpy() * 255).astype(np.uint8)  
    
    cv2.imwrite(r'C:\Users\Sjt\Desktop\1.png',w_img_np)



