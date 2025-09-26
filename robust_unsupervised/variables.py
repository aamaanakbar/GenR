from .prelude import *
# If your prelude already imports torch/nn/typing, this is harmless.
from typing import Optional, Tuple
import torch
import torch.nn as nn

# ---------------------------
# Optional: LPIPS loader
# ---------------------------
def _get_lpips(device: torch.device):
    try:
        import lpips  # pip install lpips
    except Exception as e:
        raise ImportError(
            "LPIPS is required for perceptual optimization. Install with: pip install lpips\n"
            f"Underlying error: {e}"
        )
    net = lpips.LPIPS(net='vgg').to(device)
    net.eval()
    return net


# ---------------------------
# Sampling utilities
# ---------------------------
def _to_device_like(t: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    return t.to(device=like.device, dtype=like.dtype)

def sample_truncated_z(batch_size: int, z_dim: int, trunc: float = 2.0, device=None):
    """
    Simple clamp-based truncated Gaussian sampling in Z-space.
    """
    z = torch.randn(batch_size, z_dim, device=device)
    if trunc is not None:
        z = torch.clamp(z, -trunc, trunc)
    return z

def style_mix_ws(G: nn.Module, batch_size: int, cutoff: Optional[int] = None, trunc_z: Optional[float] = None):
    """
    Style mixing: map two z's, splice their Ws at a random (or given) cutoff.
    Returns W+ of shape [B, num_ws, w_dim].
    """
    device = G.mapping.w_avg.device
    z1 = sample_truncated_z(batch_size, G.z_dim, trunc=trunc_z, device=device)
    z2 = sample_truncated_z(batch_size, G.z_dim, trunc=trunc_z, device=device)

    # StyleGAN3 mapping usually: mapping(z, c, truncation_psi=None, truncation_cutoff=None, **kw)
    # Keep compatibility with your existing signature (skip_w_avg_update).
    w1 = G.mapping(z1, None, skip_w_avg_update=True)  # [B, num_ws, w_dim]
    w2 = G.mapping(z2, None, skip_w_avg_update=True)  # [B, num_ws, w_dim]

    num_ws = w1.shape[1]
    if cutoff is None:
        cutoff = torch.randint(low=1, high=num_ws, size=()).item()

    w = w1.clone()
    w[:, cutoff:, :] = w2[:, cutoff:, :]
    return w


# ===========================================================
# Base Variable
# ===========================================================
class Variable(nn.Module):
    def __init__(self, G: networks.Generator, data: torch.Tensor):
        super().__init__()
        self.G = G
        self.data = data

    # ------------------------------------
    @staticmethod
    def sample_from(G: networks.Generator, batch_size: int = 1):
        raise NotImplementedError

    @staticmethod
    def sample_random_from(G: networks.Generator, batch_size: int = 1):
        raise NotImplementedError

    def to_input_tensor(self):
        raise NotImplementedError

    # ------------------------------------
    def parameters(self):
        # Keep only the latent tensor optimizable.
        return [self.data]

    # --------- small helper to fix __add__/__mul__/__sub__ ----------
    def from_data(self, new_data: torch.Tensor):
        """
        Return a new instance of the same subclass with provided data.
        Preserves nn.Parameter-ness if present on self.data.
        """
        if isinstance(self.data, nn.Parameter):
            new_data = nn.Parameter(new_data)
        return self.__class__(self.G, new_data)

    # ------------------------------------
    def to_image(self, noise_mode: str = "const", force_fp32: bool = True):
        return self.render_image(self.to_input_tensor(), noise_mode=noise_mode, force_fp32=force_fp32)

    def render_image(self, ws: torch.Tensor, noise_mode: str = "const", force_fp32: bool = True):
        """
        ws shape: [batch_size, num_layers, w_dim]
        Returns images in [0, 1].
        """
        imgs = self.G.synthesis(ws, noise_mode=noise_mode, force_fp32=force_fp32)
        return (imgs + 1.0) / 2.0

    def detach(self):
        data = self.data.detach().requires_grad_(self.data.requires_grad)
        data = nn.Parameter(data) if isinstance(self.data, nn.Parameter) else data
        return self.__class__(self.G, data)

    def clone(self):
        data = self.data.detach().clone().requires_grad_(self.data.requires_grad)
        data = nn.Parameter(data) if isinstance(self.data, nn.Parameter) else data
        return self.__class__(self.G, data)

    def interpolate(self, other: "Variable", alpha: float = 0.5):
        assert self.G == other.G
        return self.__class__(self.G, self.data.lerp(other.data, alpha))

    def __add__(self, other: "Variable"):
        return self.from_data(self.data + other.data)

    def __sub__(self, other: "Variable"):
        return self.from_data(self.data - other.data)

    def __mul__(self, scalar: float):
        return self.from_data(self.data * scalar)

    def unbind(self):
        """
        Splits this (batched) variable into a list of variables with batch size 1.
        """
        out = []
        for p in self.data:
            d = nn.Parameter(p.unsqueeze(0)) if isinstance(self.data, nn.Parameter) else p.unsqueeze(0)
            out.append(self.__class__(self.G, d))
        return out


# ===========================================================
# W (single vector) Variable
# ===========================================================
class WVariable(Variable):
    @staticmethod
    def sample_from(G: nn.Module, batch_size: int = 1):
        """
        Initialize from w_avg.
        """
        device = G.mapping.w_avg.device
        data = G.mapping.w_avg.reshape(1, G.w_dim).repeat(batch_size, 1).to(device)
        return WVariable(G, nn.Parameter(data))

    @staticmethod
    def sample_random_from(G: nn.Module, batch_size: int = 1, trunc_z: Optional[float] = None):
        """
        Random sample by mapping z->w and taking the first layer's style.
        """
        device = G.mapping.w_avg.device
        z = sample_truncated_z(batch_size, G.z_dim, trunc=trunc_z, device=device)
        ws = G.mapping(z, None, skip_w_avg_update=True)  # [B, num_ws, w_dim]
        data = ws[:, 0, :]  # [B, w_dim]
        return WVariable(G, nn.Parameter(data))

    def to_input_tensor(self):
        """
        Repeat W across all layers to make W+.
        """
        return self.data.unsqueeze(1).repeat(1, self.G.num_ws, 1)

    @torch.no_grad()
    def truncate(self, truncation: float = 1.0):
        """
        Truncate towards w_avg. Lower truncation -> closer to mean -> better FID.
        """
        assert 0.0 <= truncation <= 1.0
        w_avg = self.G.mapping.w_avg.reshape(1, self.G.w_dim)
        self.data.lerp_(w_avg, 1.0 - truncation)
        return self


# ===========================================================
# W+ (per-layer) Variable
# ===========================================================
class WpVariable(Variable):
    def __init__(self, G, data: torch.Tensor):
        super().__init__(G, data)

    # ---------- Random / Mixed Sampling ----------
    @staticmethod
    def sample_from(G: nn.Module, batch_size: int = 1):
        """
        Start from w_avg tiled across layers.
        """
        W = WVariable.sample_from(G, batch_size)
        return WpVariable(G, nn.Parameter(W.to_input_tensor()))

    @staticmethod
    def sample_random_from(G: nn.Module, batch_size: int = 1, trunc_z: Optional[float] = None):
        """
        Random W+ from a single z per sample.
        """
        device = G.mapping.w_avg.device
        z = sample_truncated_z(batch_size, G.z_dim, trunc=trunc_z, device=device)
        ws = G.mapping(z, None, skip_w_avg_update=True)  # [B, num_ws, w_dim]
        return WpVariable(G, nn.Parameter(ws))

    @staticmethod
    def sample_random_mixed_from(G: nn.Module, batch_size: int = 1, cutoff: Optional[int] = None, trunc_z: Optional[float] = None):
        """
        Random W+ with style mixing (recommended for better FID/diversity).
        """
        ws = style_mix_ws(G, batch_size, cutoff=cutoff, trunc_z=trunc_z)
        return WpVariable(G, nn.Parameter(ws))

    # ---------- Core ----------
    def to_input_tensor(self):
        return self.data

    def mix(self, other: "WpVariable", num_layers: int):
        """
        Take first num_layers from self, the rest from other.
        """
        assert self.G == other.G
        mixed = torch.cat((self.data[:, :num_layers, :], other.data[:, num_layers:, :]), dim=1)
        return WpVariable(self.G, mixed if not isinstance(self.data, nn.Parameter) else nn.Parameter(mixed))

    @staticmethod
    def from_W(W: WVariable):
        return WpVariable(W.G, nn.Parameter(W.to_input_tensor()))

    # ---------- Truncation ----------
    @torch.no_grad()
    def truncate(self, truncation: float = 1.0, *, layer_start: int = 0, layer_end: Optional[int] = None):
        """
        Uniform truncation towards w_avg for a layer range [layer_start:layer_end).
        """
        assert 0.0 <= truncation <= 1.0
        mu = self.G.mapping.w_avg  # [w_dim]
        target = mu.reshape(1, 1, self.G.w_dim).repeat(1, self.G.num_ws, 1)
        s, e = layer_start, layer_end
        self.data[:, s:e].lerp_(target[:, s:e], 1.0 - truncation)
        return self

    @torch.no_grad()
    def truncate_layerwise(self, psi: float = 0.7, cutoff: int = 8):
        """
        Stronger truncation on early (coarse) layers, weaker on fine layers.
        Lower psi tends toward w_avg (often better FID).
        """
        w_avg = self.G.mapping.w_avg  # [w_dim]
        for i in range(self.data.shape[1]):  # num_ws
            strength = psi if i < cutoff else 1.0
            self.data[:, i, :].lerp_(w_avg, 1.0 - strength)
        return self

    # ---------- Perceptual Optimization (LPIPS) ----------
    def optimize_latent(
        self,
        target_img: torch.Tensor,
        steps: int = 200,
        lr: float = 0.01,
        lpips_weight: float = 1.0,
        l2_weight: float = 0.0,
        wdist_weight: float = 1e-4,
        noise_mode: str = "const",
        force_fp32: bool = True,
    ):
        """
        Optimize W+ to match a target image (minimizes LPIPS; optional L2 and w-distance regularization).
        target_img: expected in [0, 1], shape [B, C, H, W], same B as self.
        """
        device = self.G.mapping.w_avg.device
        self.data.requires_grad_(True)

        # LPIPS net
        lpips_fn = _get_lpips(device) if lpips_weight > 0 else None
        opt = torch.optim.Adam([self.data], lr=lr)

        target_img = target_img.to(device)
        assert target_img.min() >= 0.0 and target_img.max() <= 1.0, "target_img must be in [0, 1]"

        w_avg = self.G.mapping.w_avg.detach()

        for _ in range(steps):
            ws = self.to_input_tensor()  # [B, num_ws, w_dim]
            gen = self.render_image(ws, noise_mode=noise_mode, force_fp32=force_fp32)  # [B, C, H, W] in [0,1]

            loss = 0.0

            if lpips_weight > 0:
                lp = lpips_fn(gen, target_img).mean()
                loss = loss + lpips_weight * lp

            if l2_weight > 0:
                l2 = (gen - target_img).pow(2).mean()
                loss = loss + l2_weight * l2

            if wdist_weight > 0:
                # Keep W+ near distribution (improves FID, avoids drifting too far)
                wd = (self.data - w_avg.view(1, 1, -1)).pow(2).mean()
                loss = loss + wdist_weight * wd

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        return self


# ===========================================================
# W++ (spatially extended per-layer) Variable
# ===========================================================
class WppVariable(Variable):
    def __init__(self, G, data: torch.Tensor, spatial_dims: Tuple[int, int] = (64, 64)):
        super().__init__(G, data)
        self.height, self.width = spatial_dims
        self.spatial_size = self.height * self.width

    @staticmethod
    def from_Wp(Wp: WpVariable, spatial_dims: Tuple[int, int] = (64, 64)):
        """
        Convert W+ [B, num_ws, w_dim] to W++ [B, num_ws*H*W, w_dim] by repeating per spatial pos.
        """
        B, num_ws, w_dim = Wp.data.shape
        H, W = spatial_dims
        S = H * W

        data = Wp.data.unsqueeze(2)            # [B, num_ws, 1, w_dim]
        data = data.repeat(1, 1, S, 1)         # [B, num_ws, S, w_dim]
        data = data.reshape(B, num_ws * S, w_dim)
        return WppVariable(Wp.G, nn.Parameter(data), spatial_dims)

    def to_input_tensor(self, pooling: str = "mean", attn: Optional[torch.Tensor] = None):
        """
        Aggregate per-pixel latents back to W+ for standard StyleGAN3 synthesis.
        pooling: "mean" | "max" | "attn"
        attn (optional): [B, 1, S, 1] or [B, S] attention weights (softmax internally).
        Returns [B, num_ws, w_dim]
        """
        B, T, w_dim = self.data.shape
        S = self.spatial_size
        assert T % S == 0, "WppVariable: total elements must be divisible by spatial size"
        num_ws = T // S

        reshaped = self.data.view(B, num_ws, S, w_dim)  # [B, num_ws, S, w_dim]

        if pooling == "mean":
            aggregated = reshaped.mean(dim=2)  # [B, num_ws, w_dim]
        elif pooling == "max":
            aggregated = reshaped.max(dim=2).values
        elif pooling == "attn":
            assert attn is not None, "Attention weights required for pooling='attn'"
            # attn shape: [B, S] or [B, 1, S, 1]
            if attn.ndim == 2:
                attn = attn.view(B, 1, S, 1)
            elif attn.ndim == 4:
                pass
            else:
                raise ValueError("attn must be [B, S] or [B, 1, S, 1]")
            # normalize over S
            attn = torch.softmax(attn, dim=2)
            aggregated = (reshaped * attn).sum(dim=2)
        else:
            raise ValueError(f"Unknown pooling mode: {pooling}")

        return aggregated

    def to_per_pixel_image(self):
        """
        Placeholder: true per-pixel latent modulation requires modifying synthesis to accept W++ directly.
        For now, use pooled W+ for standard rendering.
        """
        return self.render_image(self.to_input_tensor())
