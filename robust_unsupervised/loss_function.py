
from typing import List, Optional
import torch
import torch.nn.functional as F
from torch import Tensor
from lpips import LPIPS
from pytorch_msssim import ssim
from torchvision import models

class VGGFeatureExtractor(torch.nn.Module):
    def __init__(self, layers):
        super(VGGFeatureExtractor, self).__init__()
        vgg = models.vgg19(pretrained=True).features
        self.model = torch.nn.Sequential(*[vgg[i] for i in layers]).eval().cuda()
    
    def forward(self, x):
        return self.model(x)
class MultiscaleLPIPS:
    def __init__(
        self,
        min_loss_res: int = 16,
        level_weights: List[float] = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ssim_weight: float = 0.2,
        l1_weight: float = 0.1,
        symmetry_weight: float = 0.3,
        gradient_weight: float = 0.1,
        vgg_weight: float = 0.0
    ):
        self.min_loss_res = min_loss_res
        self.weights = level_weights
        self.ssim_weight = ssim_weight
        self.l1_weight = l1_weight
        self.symmetry_weight = symmetry_weight
        self.gradient_weight = gradient_weight
        self.vgg_weight = vgg_weight

        self.lpips_network = LPIPS(net="vgg", verbose=False).cuda()
        self.feature_extractor = VGGFeatureExtractor(layers=[0, 5, 10, 19, 28])

    def measure_lpips(self, x, y, mask):
        if mask is not None:
            mask = mask.to(x.device, dtype=x.dtype)
            noise = (torch.randn_like(x) + 0.5) / 2.0
            x = x + noise * (1.0 - mask)
            y = y + noise * (1.0 - mask)
        return self.lpips_network(x, y, normalize=True).mean()

    def ssim_loss(self, pred, target):
        min_size = min(pred.size(-2), pred.size(-1))
        win_size = min(11, min_size)
        win_size = win_size - 1 if win_size % 2 == 0 else win_size
        return 1 - ssim(pred, target, data_range=1.0, size_average=True, win_size=win_size)

    def gradient_loss(self, pred, target):
        pred_dx = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        pred_dy = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        target_dx = target[:, :, 1:, :] - target[:, :, :-1, :]
        target_dy = target[:, :, :, 1:] - target[:, :, :, :-1]
        return F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)

    def symmetric_loss(self, x, x_perturbed, y, mask):
        loss_x = self.measure_lpips(x, y, mask)
        loss_x_perturbed = self.measure_lpips(x_perturbed, y, mask)
        return (loss_x + loss_x_perturbed) / 2.0

    def __call__(self, f_hat, x_clean: Tensor, y: Tensor, mask: Optional[Tensor] = None):
        x = f_hat(x_clean)

        if mask is not None:
            mask = F.interpolate(mask, size=y.shape[-2:], mode="area")

        # Perturb input
        x_perturbed = f_hat(x_clean + torch.randn_like(x_clean) * 0.01)
        x_perturbed = F.interpolate(x_perturbed, size=y.shape[-2:], mode='bilinear', align_corners=False)

        losses = []
        for weight in self.weights:
            if y.shape[-1] <= self.min_loss_res:
                break
            if weight > 0:
                symmetric_loss_value = self.symmetric_loss(x, x_perturbed, y, mask)
                losses.append(weight * self.symmetry_weight * symmetric_loss_value)

            if mask is not None:
                mask = F.avg_pool2d(mask, 2)
            x = F.avg_pool2d(x, 2)
            x_clean = F.avg_pool2d(x_clean, 2)
            y = F.avg_pool2d(y, 2)
            x_perturbed = F.avg_pool2d(x_perturbed, 2)

        total = torch.stack(losses).sum(dim=0) if len(losses) > 0 else 0.0
        l1 = self.l1_weight * F.l1_loss(x, y)
        ssim_val = self.ssim_weight * self.ssim_loss(x, y)
        gradient = self.gradient_weight * self.gradient_loss(x, y)

        vgg_loss = 0.0
        if self.vgg_weight > 0:
            feat_x, feat_y = self.feature_extractor(x), self.feature_extractor(y)
            vgg_loss = self.vgg_weight * F.l1_loss(feat_x, feat_y)

        return total + l1 + ssim_val + gradient + vgg_loss
