we have done some work and here is the respective code for changed and added degradation .........................
1. added dust_and_scratch and colorization into the original four (here is the degradation.py and task.py code respectivly )
  degradation.py ...................
    # degradations.py

import sys, os, tempfile, random, math
from typing import List

import numpy as np
import PIL
import cv2

import benchmark.config as config
from .prelude import *  # brings in torch, nn, F, etc.

# Make DiffJPEG importable
sys.path.append(os.path.join(os.path.dirname(__file__), "DiffJPEG"))
from benchmark.DiffJPEG.DiffJPEG import DiffJPEG

import torchvision.transforms.functional as TF
from PIL import JpegImagePlugin

TMP_SAVE_FILEPATH = tempfile.mkstemp()[1]

# -------------------------
# Helpers
# -------------------------
def cycle_to_file(x: torch.Tensor, save_path: str):
    """
    Save tensor -> disk -> read back, to bake quantization/clamp.
    """
    assert x.shape[0] == 1  # batching not supported here
    TF.to_pil_image(x.squeeze(0).clamp(0, 1)).save(save_path)
    return TF.to_tensor(PIL.Image.open(save_path)).unsqueeze(0).to(x.device)

# -------------------------
# Base classes
# -------------------------
class Degradation(nn.Module):
    seed = 2022
    mask = None

    def __init__(self):
        super().__init__()
        self.seed += 1

    def _true_degradation(self, ground_truth):
        raise NotImplementedError

    def degrade_prediction(self, pred):
        raise NotImplementedError

    @torch.no_grad()
    def degrade_ground_truth(self, ground_truth, save_path=None):
        """
        Applies the true (possibly non-differentiable) degradation.
        """
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)

        if save_path is None:
            save_path = TMP_SAVE_FILEPATH + ".png"

        degraded_target = self._true_degradation(ground_truth.clamp(0, 1))
        result = cycle_to_file(degraded_target, save_path)
        return result

    def forward(self, x):
        return self.degrade_prediction(x)

# -------------------------
# Original tasks
# -------------------------
class Downsample(Degradation):
    def __init__(self, downsampling_factor: int):
        super().__init__()
        self.downsampling_factor = int(downsampling_factor)
        self.filter = random.choice(
            [PIL.Image.BILINEAR, PIL.Image.BICUBIC, PIL.Image.LANCZOS]
        )

    def degrade_prediction(self, x):
        return F.avg_pool2d(x, self.downsampling_factor)

    def _true_degradation(self, x):
        assert x.shape[0] == 1, "Batching not yet supported"
        image = TF.to_pil_image(x.squeeze(0))
        res = max(1, math.floor(x.shape[-1] // self.downsampling_factor))
        image = image.resize((res, res), self.filter)
        path = TMP_SAVE_FILEPATH + ".png"
        image.save(path)
        return TF.to_tensor(PIL.Image.open(path)).unsqueeze(0).to(x.device)

class AddNoise(Degradation):
    k = 2.0
    eps = 1e-3

    def __init__(self, noise_amount: float):
        super().__init__()
        self.noise_amount = noise_amount
        self.clamp = True
        self.seed += 1

    def degrade_prediction(self, x):
        x = self.differentiable_clamp(x)
        num_photons, bernoulli_p = self.noise_amount

        # Poisson ~ Gaussian approx
        if num_photons > 0:
            noise = torch.randn(1, 3, x.shape[2], x.shape[3], device=x.device)
            lambd = x * num_photons
            mu = lambd - 0.5
            sigma = (lambd + self.eps).sqrt()
            y = (mu + sigma * noise) / num_photons
        else:
            y = x

        # Bernoulli masking
        y = y * (torch.rand_like(y)[:, 0:1] > bernoulli_p).float()
        return self.differentiable_clamp(y)

    @torch.no_grad()
    def _true_degradation(self, x):
        num_photons, bernoulli_p = self.noise_amount
        if num_photons > 0:
            y = torch.poisson(x * num_photons) / num_photons
        else:
            y = x
        y = y * (torch.rand_like(y)[:, 0:1] > bernoulli_p).float()
        return y.clamp(0.0, 1.0)

    class _ClampWithSurrogateGradient(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return x.clamp(0.0, 1.0)

        @staticmethod
        def backward(ctx, grad_y):
            (x,) = ctx.saved_tensors
            with torch.enable_grad():
                return torch.autograd.grad(
                    torch.sigmoid(AddNoise.k * (x - 0.5)), x, grad_y
                )[0], None

    differentiable_clamp = _ClampWithSurrogateGradient.apply

class CenterCrop(Degradation):
    """Zero out everything except a centered window (resolution-aware)."""
    def __init__(self, *args):
        super().__init__()

    def degrade_prediction(self, x):
        B, C, H, W = x.shape
        crop = max(1, int(0.4 * min(H, W)))
        y0 = (H - crop) // 2; y1 = y0 + crop
        x0 = (W - crop) // 2; x1 = x0 + crop
        result = torch.zeros_like(x)
        result[:, :, y0:y1, x0:x1] = x[:, :, y0:y1, x0:x1]
        return result

    def _true_degradation(self, x):
        return self.degrade_prediction(x)

class CompressJPEG(Degradation):
    k = 0.8

    def __init__(self, quality: int):
        super().__init__()
        self.quality = int(quality)

        # Probe quantization table
        x_img = TF.to_pil_image(torch.randn(3, config.resolution, config.resolution))
        path = TMP_SAVE_FILEPATH + ".jpg"
        x_img.save(path, quality=self.quality)
        compressed_image = PIL.Image.open(path)
        table = compressed_image.quantization  # type: ignore
        assert JpegImagePlugin.get_sampling(compressed_image) == 2

        # Differentiable JPEG (device-safe; move in forward)
        self.to_jpeg = DiffJPEG(self.k, differentiable=True, quantization_table=table)

    def parameters(self, recurse=False):
        return []  # don't optimize DiffJPEG internals

    def degrade_prediction(self, x):
        self.to_jpeg = self.to_jpeg.to(x.device)
        return self.to_jpeg(x)

    def _true_degradation(self, x):
        if "CHEAT_DEARTIFACT" in os.environ:
            self.to_jpeg = self.to_jpeg.to(x.device)
            return self.to_jpeg(x).detach()
        else:
            assert x.shape[0] == 1, "Batching not yet supported"
            path = TMP_SAVE_FILEPATH + ".jpg"
            TF.to_pil_image(x.squeeze(0)).save(path, quality=self.quality)
            return TF.to_tensor(PIL.Image.open(path)).unsqueeze(0).to(x.device)

class MaskRandomly(Degradation):
    def __init__(self, num_strokes: int):
        super().__init__()
        self.num_strokes = int(num_strokes)
        torch.manual_seed(self.seed)
        self.mask = self._generate_mask()  # CPU tensor [1,1,H,W]

    def _generate_mask(self):
        image_height = config.resolution * 4
        image_width = config.resolution * 4
        brush_width = int(config.resolution * 0.08) * 4

        mask = np.zeros((image_height, image_width), dtype=np.float32)

        def sample():
            w = image_width - 1
            h = image_height - 1
            return random.choice([random.randint(0, w // 3), random.randint(2 * w // 3, w)]), \
                   random.choice([random.randint(0, h // 3), random.randint(2 * h // 3, h)])

        for _ in range(self.num_strokes):
            start_x, start_y = sample()
            end_x, end_y = sample()
            mask = cv2.line(mask, (start_x, start_y), (end_x, end_y),
                            color=1.0, thickness=brush_width)
            mask = cv2.circle(mask, (start_x, start_y), int(brush_width / 2), 1.0)

        mask = cv2.pyrDown(cv2.pyrDown(mask))
        mask = 1.0 - mask
        return torch.from_numpy(mask).float()[None, None]  # CPU

    def _true_degradation(self, x):
        mask = self.mask.to(x.device)
        return x * F.interpolate(mask, x.shape[-1], mode="bicubic", align_corners=False)

    def degrade_prediction(self, x):
        return self._true_degradation(x)

class IdentityDegradation(Degradation):
    def __init__(self, *args):
        super().__init__()
    def degrade_prediction(self, x):
        return x
    def _true_degradation(self, x):
        return x

# -------------------------
# New: Colorization (remove chroma -> grayscale 3ch)
# -------------------------
class GrayscaleRemoval(Degradation):
    def __init__(self, *args, **kwargs):
        super().__init__()
    def degrade_prediction(self, x):
        # build weights on the correct device/dtype
        w = torch.tensor([0.2989, 0.5870, 0.1140], device=x.device, dtype=x.dtype).view(1,3,1,1)
        y1 = (x * w).sum(dim=1, keepdim=True)
        return y1.repeat(1, 3, 1, 1).clamp(0, 1)
    @torch.no_grad()
    def _true_degradation(self, x):
        assert x.ndim == 4 and x.shape[0] == 1, "True path expects batch size 1."
        img = TF.to_pil_image(x[0].cpu()).convert("L").convert("RGB")
        y = TF.to_tensor(img).unsqueeze(0).to(x.device)
        return y

# -------------------------
# New: Dust & Scratches (old photo artifacts)
# -------------------------
class DustAndScratches(Degradation):
    """
    Arg: (speckle_prob, scratch_count, max_thickness_px, max_len_frac, bright_frac)
    """
    def __init__(self, arg=(0.0015, 28, 2, 0.28, 0.6)):
        super().__init__()
        self.speckle_p, self.scratch_cnt, self.max_thick, self.max_len_frac, self.bright_frac = arg

    @staticmethod
    def _motion_kernel(ksize: int, angle_deg: float, thickness: float = 1.0):
        c = (ksize - 1) / 2.0
        ys, xs = torch.meshgrid(
            torch.arange(ksize), torch.arange(ksize), indexing="ij"
        )
        xs = xs - c
        ys = ys - c
        ang = math.radians(angle_deg)
        ca, sa = math.cos(ang), math.sin(ang)
        u =  xs * ca + ys * sa
        v = -xs * sa + ys * ca
        sigma_u = ksize / 6.0
        sigma_v = max(0.5, thickness)
        ker = torch.exp(-0.5*(u/sigma_u)**2 - 0.5*(v/sigma_v)**2)
        ker = ker / (ker.sum() + 1e-8)
        return ker.float()

    def degrade_prediction(self, x):
        B, C, H, W = x.shape
        device = x.device

        # ---- Dust specks (bright & dark) ----
        speck = (torch.rand(B, 1, H, W, device=device) < self.speckle_p).float()
        speck = F.avg_pool2d(speck, kernel_size=3, stride=1, padding=1)
        speck = (speck - speck.min()) / (speck.max() - speck.min() + 1e-8)
        bright_mask = (torch.rand_like(speck) < self.bright_frac).float()
        alpha_bright = speck * bright_mask
        alpha_dark   = speck * (1.0 - bright_mask)

        # ---- Hairline scratches ----
        area = float(H * W)
        impulse_prob = min(0.0002, max(1.0 / area, self.scratch_cnt / area))
        impulses = (torch.rand(B, 1, H, W, device=device) < impulse_prob).float()
        k = int(min(31, max(9, (min(H, W) // 32) * 2 + 1)))
        angle = random.random() * 180.0
        ker = self._motion_kernel(k, angle, thickness=max(1.0, float(self.max_thick))).to(device).view(1, 1, k, k)
        scratch = F.conv2d(impulses, ker, padding=k//2)
        scratch = (scratch - scratch.min()) / (scratch.max() - scratch.min() + 1e-8)
        alpha_scratch = (0.8 * scratch).clamp(0, 1)

        # Compose
        y = x
        alpha_dark_total = (alpha_dark + alpha_scratch).clamp(0, 1)
        y = y * (1.0 - alpha_dark_total)                  # dark marks
        y = y * (1.0 - alpha_bright) + alpha_bright       # bright dust
        return y.clamp(0, 1)

    @torch.no_grad()
    def _true_degradation(self, x):
        assert x.ndim == 4 and x.shape[0] == 1, "True path expects batch size 1."
        img = TF.to_pil_image(x[0].cpu())
        arr = np.array(img).astype(np.float32) / 255.0  # H×W×3
        H, W = arr.shape[:2]

        # dust specks
        num_specks = max(1, int(self.speckle_p * H * W * 0.05))
        for _ in range(num_specks):
            r = random.randint(1, 2)
            cx = random.randint(0, W-1)
            cy = random.randint(0, H-1)
            bright = (random.random() < self.bright_frac)
            color = 1.0 if bright else 0.0
            cv2.circle(arr, (cx, cy), r, (color, color, color), thickness=-1, lineType=cv2.LINE_AA)

        # scratches
        Lmax = int(self.max_len_frac * min(H, W))
        Lmin = max(5, int(0.10 * min(H, W)))
        for _ in range(int(self.scratch_cnt)):
            length = random.randint(Lmin, max(Lmin, Lmax))
            angle = random.random() * 2.0 * math.pi
            thick = random.randint(1, max(1, int(self.max_thick)))
            x0 = random.randint(0, W-1); y0 = random.randint(0, H-1)
            x1 = int(x0 + length * math.cos(angle))
            y1 = int(y0 + length * math.sin(angle))
            x1 = max(0, min(W-1, x1)); y1 = max(0, min(H-1, y1))
            bright = (random.random() < self.bright_frac)
            color = 1.0 if bright else 0.0
            cv2.line(arr, (x0, y0), (x1, y1), (color, color, color), thickness=thick, lineType=cv2.LINE_AA)

        arr = np.clip(arr, 0.0, 1.0)
        y = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(x.device).float()
        return y

# -------------------------
# Composition wrapper
# -------------------------
class ComposedDegradation(Degradation):
    def __init__(self, degradations: List[Degradation]):
        super().__init__()
        self.degradations = nn.ModuleList(degradations)

    @property
    def mask(self):
        return self.degradations[-1].mask if len(self.degradations) else None

    def parameters(self, recurse=False):
        return sum([list(deg.parameters()) for deg in self.degradations], [])

    def degrade_prediction(self, x):
        for deg in self.degradations:
            x = deg.degrade_prediction(x)
        return x

    def _true_degradation(self, x):
        for deg in self.degradations:
            x = deg._true_degradation(x)
        return x

    def degrade_ground_truth(self, x, save_path=None):
        for deg in self.degradations:
            x = deg.degrade_ground_truth(x, save_path=save_path)
        return x

class ResizePrediction(Degradation):
    def __init__(self, size: int):
        super().__init__()
        self.size = int(size)

    def degrade_prediction(self, x):
        return self._true_degradation(x)

    def _true_degradation(self, x):
        return F.interpolate(x, size=self.size, mode="area")

def adapt_to_resolution(x, res: int):
    return ComposedDegradation([ResizePrediction(res), x])


its respectivly tasks.py is as followed...............


            # tasks.py

from dataclasses import dataclass
from typing import Any, List, Type
import itertools

from . import config
from .degradations import (
    Degradation,
    ComposedDegradation,
    ResizePrediction,
    CenterCrop,
    IdentityDegradation,
    Downsample,
    MaskRandomly,
    AddNoise,
    CompressJPEG,
    GrayscaleRemoval,
    DustAndScratches,
)

# Six tasks total
task_names: List[str] = [
    "upsampling",
    "denoising",
    "deartifacting",
    "inpainting",
    "colorization",
    "dust_and_scratch",
]
task_levels: List[str] = ["XL", "L", "M", "S", "XS"]

degradation_types = {
    "upsampling":        Downsample,
    "inpainting":        MaskRandomly,
    "denoising":         AddNoise,
    "deartifacting":     CompressJPEG,
    "colorization":      GrayscaleRemoval,
    "dust_and_scratch":  DustAndScratches,
}

degradation_levels = {
    "upsampling": {
        "XL": 32, "L": 16, "M": 8, "S": 4, "XS": 2,
        2: 8, 3: 8, 4: 8, 5: 8, 6: 8,
    },
    "inpainting": {
        "XL": 17, "L": 13, "M": 9, "S": 5, "XS": 1,
        2: 9, 3: 9, 4: 9, 5: 9, 6: 9,
    },
    "denoising": {
        "XL": (6,  0.64),
        "L":  (12, 0.32),
        "M":  (24, 0.16),
        "S":  (48, 0.08),
        "XS": (96, 0.04),
        2: (24, 0.16), 3: (24, 0.16), 4: (24, 0.16), 5: (24, 0.16), 6: (24, 0.16),
    },
    "deartifacting": {
        "XL": 6, "L": 9, "M": 12, "S": 15, "XS": 18,
        2: 12, 3: 12, 4: 12, 5: 12, 6: 12,
    },

    # Colorization: parameterless (ctor takes no arg)
    "colorization": {
        "XL": None, "L": None, "M": None, "S": None, "XS": None,
        2: None, 3: None, 4: None, 5: None, 6: None,
    },

    # Dust & Scratches: (speckle_p, scratch_cnt, max_thickness_px, max_len_frac, bright_frac)
    "dust_and_scratch": {
        "XL": (0.0020, 60, 2, 0.35, 0.6),
        "L":  (0.0018, 45, 2, 0.32, 0.6),
        "M":  (0.0015, 28, 2, 0.28, 0.6),
        "S":  (0.0010, 16, 1, 0.25, 0.6),
        "XS": (0.0006,  8, 1, 0.20, 0.6),
        2: (0.0015, 24, 2, 0.28, 0.6),
        3: (0.0015, 24, 2, 0.28, 0.6),
        4: (0.0015, 24, 2, 0.28, 0.6),
        5: (0.0015, 24, 2, 0.28, 0.6),
        6: (0.0015, 24, 2, 0.28, 0.6),
    },
}

# ---------------------------

@dataclass
class Task:
    name: str
    category: str
    level: Any                    # str for singles, int for composed
    constructor: Any              # Type[Degradation] or callable builder
    arg: Any

    def init_degradation(self) -> ComposedDegradation:
        # For single tasks: ResizePrediction(res) -> Degradation(arg?)
        ctor = self.constructor
        inst = ctor(self.arg) if self.arg is not None else ctor()
        return ComposedDegradation([ResizePrediction(config.resolution), inst])


def get_task(name: str, level: str) -> Task:
    return Task(
        name=name,
        category="single_tasks",
        level=level,
        constructor=degradation_types[name],
        arg=degradation_levels[name][level],
    )

# ---------------------------

single_tasks: List[Task] = [get_task(n, l) for l in task_levels for n in task_names]

def init_composed(level: int):
    # Returns a builder: included_tasks -> ComposedDegradation([...])
    def builder(included_tasks):
        ops: List[Degradation] = []
        for name in task_names:
            if name in included_tasks:
                arg = degradation_levels[name][level]
                ctor = degradation_types[name]
                op = ctor(arg) if arg is not None else ctor()
                ops.append(op)
        return ComposedDegradation(ops)
    return builder

# Full 6-task combo (U+N+A+P+C+D) at level 6
full_composed_task = Task("UNAPCD", "composed_tasks", 6, init_composed(6), task_names)

initials = {
    "upsampling":       "U",
    "denoising":        "N",
    "deartifacting":    "A",
    "inpainting":       "P",
    "colorization":     "C",
    "dust_and_scratch": "D",
}

composed_tasks: List[Task] = []
for k in range(2, len(task_names) + 1):
    for subseq in itertools.combinations(task_names, k):
        composed_tasks.append(
            Task(
                name="".join(initials[t] for t in subseq),
                category="composed_tasks",
                level=k,
                constructor=init_composed(k),
                arg=list(subseq),
            )
        )

all_tasks: List[Task] = single_tasks + composed_tasks

extreme_tasks: List[Task] = []
extreme_tasks += [get_task(n, "XL") for n in task_names]
extreme_tasks += [full_composed_task]
extreme_tasks += [get_task(n, "XS") for n in task_names]

# Utility tasks
uncropping_task = Task("uncropping", "uncropping", 1, CenterCrop, None)
identity_task  = Task("identity",  "identity",  1, IdentityDegradation, None)

THE LOSS FUCNTION IS AS 

from .prelude import *
from lpips import LPIPS
from pytorch_msssim import ssim as ssim_fn

# --- simple loss terms ---
def charbonnier_loss(pred: Tensor, target: Tensor, epsilon: float = 1e-3) -> Tensor:
    diff = pred - target
    return torch.sqrt(diff * diff + epsilon * epsilon).mean()

def gradient_loss(pred: Tensor, target: Tensor) -> Tensor:
    # finite differences on H/W (assumes BCHW)
    dx_p = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    dy_p = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    dx_t = target[:, :, :, 1:] - target[:, :, :, :-1]
    dy_t = target[:, :, 1:, :] - target[:, :, :-1, :]
    return F.l1_loss(dx_p, dx_t) + F.l1_loss(dy_p, dy_t)

def _align_mask(mask: Optional[Tensor], ref: Tensor) -> Optional[Tensor]:
    """Make mask match ref (device/dtype/size/channels)."""
    if mask is None:
        return None
    mask = mask.to(ref.device, dtype=ref.dtype)
    if mask.dim() == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)       # 1x1xH xW
    elif mask.dim() == 3:
        mask = mask.unsqueeze(1)                     # Bx1xH xW
    if mask.shape[-2:] != ref.shape[-2:]:
        mask = F.interpolate(mask, size=ref.shape[-2:], mode='nearest')
    if mask.size(1) == 1 and ref.size(1) != 1:
        mask = mask.expand(-1, ref.size(1), -1, -1)
    if mask.max() > 1:
        mask = mask / 255.0
    return mask.clamp(0, 1)

class MultiscaleLPIPS:
    def __init__(
        self,
        min_loss_res: int = 16,
        level_weights: List[float] = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        charbonnier_weight: float = 1.0,
        ssim_weight: float = 0.5,
        gradient_weight: float = 0.1,
        noise_std: float = 0.05,        # noise in unmasked regions for LPIPS
    ):
        super().__init__()
        self.min_loss_res = min_loss_res
        self.weights = level_weights
        self.charbonnier_weight = charbonnier_weight
        self.ssim_weight = ssim_weight
        self.gradient_weight = gradient_weight
        self.noise_std = noise_std

        self.lpips_network = LPIPS(net="vgg", verbose=False).eval()
        for p in self.lpips_network.parameters():
            p.requires_grad = False

    def measure_lpips(self, x: Tensor, y: Tensor, mask: Optional[Tensor]) -> Tensor:
        # ensure LPIPS sits on the same device as inputs
        self.lpips_network.to(x.device)
        if mask is not None and self.noise_std > 0:
            mask = _align_mask(mask, x)
            noise = torch.randn_like(x) * self.noise_std
            x = x + noise * (1.0 - mask)
            y = y + noise * (1.0 - mask)
        return self.lpips_network(x, y, normalize=True).mean()

    def measure_ssim(self, x: Tensor, y: Tensor) -> Tensor:
        # SSIM loss = 1 - SSIM; use a sensible odd window size
        min_sz = min(x.size(-2), x.size(-1))
        win = max(3, min(11, min_sz))
        if win % 2 == 0:
            win -= 1
        return 1.0 - ssim_fn(x, y, data_range=1.0, size_average=True, win_size=win)

    def __call__(self, f_hat, x_clean: Tensor, y: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        device = y.device
        x = f_hat(x_clean)
        if x.shape[-2:] != y.shape[-2:]:
            x = F.interpolate(x, size=y.shape[-2:], mode='bilinear', align_corners=False)

        # align mask once at native resolution
        mask = _align_mask(mask, y) if mask is not None else None

        losses = []
        x_ms, y_ms, mask_ms = x, y, mask
        for w in self.weights:
            if min(y_ms.shape[-2], y_ms.shape[-1]) <= self.min_loss_res:
                break
            if w > 0:
                lp  = self.measure_lpips(x_ms, y_ms, mask_ms)
                ch  = charbonnier_loss(x_ms, y_ms)
                ss  = self.measure_ssim(x_ms, y_ms)
                gr  = gradient_loss(x_ms, y_ms)
                combined = lp + self.charbonnier_weight*ch + self.ssim_weight*ss + self.gradient_weight*gr
                losses.append(w * combined)

            # downscale pyramid
            if mask_ms is not None:
                mask_ms = F.avg_pool2d(mask_ms, 2)     # or max_pool2d for binary masks
            x_ms = F.avg_pool2d(x_ms, 2)
            y_ms = F.avg_pool2d(y_ms, 2)

        return torch.stack(losses).sum() if len(losses) > 0 else torch.zeros((), device=device)

Note_ we have done this changes in stylegan3 base model ..............


