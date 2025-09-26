# benchmark/align_ffhq.py
# Lightweight FFHQ-style face alignment with safe fallback.
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
import cv2

def _to_numpy_uint8(x: torch.Tensor) -> np.ndarray:
    # x: (1,3,H,W) or (3,H,W) in [0,1]
    if x.dim() == 4:
        x = x[0]
    x = x.clamp(0, 1).detach().cpu().numpy()
    x = np.transpose(x, (1, 2, 0))  # HWC
    x = (x * 255.0 + 0.5).astype(np.uint8)
    return x

def _to_tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
    # x: HWC uint8
    x = torch.from_numpy(x.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
    return x.clamp(0, 1)

def _detect_landmarks(img_bgr: np.ndarray) -> Optional[np.ndarray]:
    # Try face_alignment (most common on PyTorch stacks). Fallback: None.
    try:
        import face_alignment  # pip install face-alignment
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device='cuda' if torch.cuda.is_available() else 'cpu')
        lm = fa.get_landmarks(img_bgr[..., ::-1])  # expects RGB; we pass RGB
        if lm and len(lm) > 0:
            return lm[0]  # (68,2) float
        return None
    except Exception:
        return None

def _ref_points(res: int) -> np.ndarray:
    # Simple canonical points (eyes/mouth) in FFHQ-ish layout.
    # You can refine with exact StyleGAN projector refs if desired.
    eye_y = 0.35 * res
    mouth_y = 0.65 * res
    return np.float32([
        [0.30 * res, eye_y],   # left eye
        [0.70 * res, eye_y],   # right eye
        [0.50 * res, mouth_y], # mouth center
    ])

def _landmark_triplet(lm68: np.ndarray) -> np.ndarray:
    # 68-point layout: use eye centers and mouth center
    left_eye  = lm68[36:42].mean(axis=0)
    right_eye = lm68[42:48].mean(axis=0)
    mouth     = lm68[48:60].mean(axis=0)
    return np.float32([left_eye, right_eye, mouth])

def align_to_ffhq(x: torch.Tensor, resolution: int) -> torch.Tensor:
    """
    x: (1,3,H,W) in [0,1], torch float32
    returns: (1,3,res,res) aligned (or center-cropped fallback)
    """
    device = x.device
    np_img = _to_numpy_uint8(x)  # HWC RGB
    lm = _detect_landmarks(np_img)  # (68,2) or None

    if lm is not None:
        src = _landmark_triplet(lm)        # (3,2)
        dst = _ref_points(resolution)      # (3,2)
        M = cv2.getAffineTransform(src, dst)
        aligned = cv2.warpAffine(np_img, M, (resolution, resolution), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        return _to_tensor(aligned, device)

    # Fallback: center crop + resize
    H, W = np_img.shape[:2]
    s = min(H, W)
    y0 = (H - s) // 2
    x0 = (W - s) // 2
    cropped = np_img[y0:y0+s, x0:x0+s]
    aligned = cv2.resize(cropped, (resolution, resolution), interpolation=cv2.INTER_LINEAR)
    return _to_tensor(aligned, device)
