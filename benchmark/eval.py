#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Recursive evaluator for your experiment tree (W++ only).

- Discovers any experiment that contains **/inversions/** with at least one pred*.png
- **Fixed suffix** to '_W++'
- Metrics: FID (vs one or more references), PSNR, SSIM, LPIPS; optional pyiqa FR/NR
- Reference handling: accept dirs (common image extensions) or precomputed .npz stats
- FID uses random 128x128 crops by default (configurable)
- Writes a single consolidated TXT report

Examples:
  python benchmark/eval.py /path/to/timestamp_root --ref-stats benchmark/ffhq_resize35_crops128_ncrops1000.npz
  python benchmark/eval.py /path/to/timestamp_root --ref /path/to/FFHQ
  DRY_RUN=1 python benchmark/eval.py /path/to/timestamp_root --no-fid --limit 2
"""

import os
import re
import glob
import json
import shutil
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set

from joblib import Parallel, delayed

import torch
from torchvision.io import read_image, write_png
import torchvision.transforms as T

# TorchMetrics (non-deprecated imports)
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import torchmetrics.image.lpip as tm_lpips

# Optional: IQA-PyTorch (pip install pyiqa)
try:
    import pyiqa
    HAS_PYIQA = True
except Exception:
    HAS_PYIQA = False


# ---------- Global crop config (used for FID) ----------
CROP_RES = 128
CROP_NUM = 1000

def _labels() -> Tuple[str, str]:
    crop_res_label = "" if CROP_RES == 256 else str(CROP_RES)
    crop_num_label = "_ncrops" + str(CROP_NUM) if CROP_NUM != 250 else ""
    return crop_res_label, crop_num_label

CROP_RES_LABEL, CROP_NUM_LABEL = _labels()


# ---------- Utils ----------
def try_glob(pattern: str) -> List[str]:
    return glob.glob(pattern, recursive=True)

def list_images(dir_path: str, exts=("png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff")) -> List[str]:
    files: List[str] = []
    for e in exts:
        files += glob.glob(os.path.join(dir_path, f"**/*.{e}"), recursive=True)
    return sorted(files)

def save_image(x: torch.Tensor, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_png(x, path)  # CHW uint8

def _safe_random_crops(img: torch.Tensor, n: int, crop_res: int) -> List[torch.Tensor]:
    assert img.dim() == 3 and img.dtype == torch.uint8, "Expected CHW uint8"
    _, h, w = img.shape
    if h < crop_res or w < crop_res:
        img = T.Resize(size=(max(crop_res, h), max(crop_res, w)))(img)
    cropping = T.RandomCrop(crop_res)
    return [cropping(img) for _ in range(n)]

def make_crops(paths: List[str], out_path: str, n_crops: int, n_jobs: int = 16, crop_res: Optional[int] = None) -> None:
    crop_res = crop_res or CROP_RES
    print(f"Producing crops -> {out_path} (crop_res={crop_res}, crops_per_image={n_crops})")
    if os.path.exists(out_path):
        shutil.rmtree(out_path, ignore_errors=True)
    os.makedirs(out_path, exist_ok=True)

    @delayed
    def _one(i: int, pth: str):
        img = read_image(pth)
        for k, c in enumerate(_safe_random_crops(img, n_crops, crop_res)):
            save_image(c, f"{out_path}/{i:04d}/{k:04d}.png")

    Parallel(n_jobs=n_jobs, verbose=10)(_one(*it) for it in enumerate(paths))

def acronym(metric: torch.nn.Module) -> str:
    return "".join(ch for ch in metric.__class__.__name__ if not ch.islower())

def replace_str(s: str, a: str, b: str) -> str:
    if a not in s:
        raise FileNotFoundError(f"Expected '{a}' in path: {s}")
    return s.replace(a, b)

def sanitize_name(p: str) -> str:
    base = Path(p).stem if p.endswith(".npz") else Path(p).name
    return re.sub(r"[^A-Za-z0-9._-]+", "_", base)


# ---------- Discovery ----------
def discover_experiments(root: str) -> List[str]:
    """
    Any directory named 'inversions' under root is considered part of an experiment
    IFF it contains at least one 'pred*.png'. We strip '/inversions...' to get the
    experiment base path.
    """
    patterns = [os.path.join(root, "**/inversions/"), os.path.join(root, "**/inversions")]
    inv_dirs: Set[str] = set()
    for pat in patterns:
        inv_dirs.update(try_glob(pat))
    exprs: Set[str] = set()
    for inv in inv_dirs:
        if try_glob(os.path.join(inv, "**/pred*.png")):
            exprs.add(inv.split("/inversions")[0])
    expr_list = sorted(exprs)
    if not expr_list:
        raise FileNotFoundError(f"No experiments with 'inversions' and 'pred*.png' found under: {root}")
    return expr_list


# ---------- Metric builders ----------
def build_base_metrics(device: torch.device, lpips_net: str = "vgg") -> List[torch.nn.Module]:
    return [
        PeakSignalNoiseRatio(data_range=2.0).to(device).eval(),                 # PSNR for [-1,1]
        StructuralSimilarityIndexMeasure(data_range=2.0).to(device).eval(),     # SSIM for [-1,1]
        tm_lpips.LearnedPerceptualImagePatchSimilarity(net_type=lpips_net).to(device).eval(),  # LPIPS
    ]

def build_pyiqa_metrics(device: torch.device, fr_names: List[str], nr_names: List[str]) -> Tuple[Dict[str, torch.nn.Module], Dict[str, torch.nn.Module]]:
    fr, nr = {}, {}
    if (fr_names or nr_names) and not HAS_PYIQA:
        raise ImportError("pyiqa is not installed. Install with: pip install pyiqa")
    if HAS_PYIQA:
        fr = {n: pyiqa.create_metric(n, device=device).eval() for n in fr_names}
        nr = {n: pyiqa.create_metric(n, device=device).eval() for n in nr_names}
    return fr, nr


# ---------- FID reference stats ----------
def ensure_ref_stats(ref_dirs: List[str], ref_names: List[str], crop_res: int, crop_num: int, n_jobs: int) -> List[Tuple[str, str]]:
    out = []
    for i, ref in enumerate(ref_dirs):
        label = (ref_names[i] if i < len(ref_names) and ref_names[i] else sanitize_name(ref)) or f"ref{i+1}"
        crops_dir = f"benchmark/{label}_crops{crop_res}_ncrops{crop_num}"
        stats_npz = f"{crops_dir}.npz"
        if not os.path.isfile(stats_npz):
            print(f"[REF] Building crops & stats for '{label}' from: {ref}")
            imgs = list_images(ref)  # accept png/jpg/jpeg/...
            if not imgs:
                raise FileNotFoundError(f"No images under reference dir: {ref}")
            make_crops(imgs, crops_dir, crop_num, n_jobs=n_jobs, crop_res=crop_res)
            subprocess.check_call(["python", "-m", "pytorch_fid", "--batch-size", "50", "--save-stats", crops_dir, stats_npz])
        else:
            print(f"[REF] Using existing stats: {stats_npz}")
        out.append((label, stats_npz))
    return out

def collect_all_ref_stats(ref_dirs: List[str], ref_names: List[str], ref_stats_files: List[str], crop_res: int, crop_num: int, n_jobs: int) -> List[Tuple[str, str]]:
    refs: List[Tuple[str, str]] = [(sanitize_name(p), p) for p in ref_stats_files]
    if ref_dirs:
        refs.extend(ensure_ref_stats(ref_dirs, ref_names, crop_res, crop_num, n_jobs))
    if not refs:
        raise ValueError("FID requested but no references provided. Use --ref <dir> and/or --ref-stats <npz>, or pass --no-fid.")
    return refs


# ---------- Evaluation core (W++ only) ----------
FIXED_SUFFIX = "_W++"

def eval_experiment(
    expr_path: str,
    device: torch.device,
    ref_stats: List[Tuple[str, str]],
    compute_fid: bool,
    compute_distances: bool,
    n_jobs: int,
    lpips_net: str,
    iqa_fr: List[str],
    iqa_nr: List[str],
    report_lines: List[str],
) -> None:
    print(f"\n=== Evaluating: {expr_path} (suffix {FIXED_SUFFIX}) ===")

    base_metrics = build_base_metrics(device, lpips_net)
    fr_metrics, nr_metrics = build_pyiqa_metrics(device, iqa_fr or [], iqa_nr or [])

    suffix = FIXED_SUFFIX
    report_lines.append(f"\n[Experiment] {expr_path}")
    report_lines.append(f"  Suffix: {suffix}")

    # ----- FID -----
    if compute_fid and ref_stats:
        pred_paths = try_glob(f"{expr_path}/**/pred{suffix}.png")
        if not pred_paths:
            msg = f"[WARN] No preds for suffix '{suffix}' in {expr_path}; skipping FID."
            print(msg); report_lines.append("  " + msg)
        else:
            crops_out = f"{expr_path}/crops{CROP_RES_LABEL}{suffix}"
            make_crops(pred_paths, crops_out, CROP_NUM if "DRY_RUN" not in os.environ else 10, n_jobs=n_jobs, crop_res=CROP_RES)
            for ref_label, stats_npz in ref_stats:
                result = subprocess.check_output(["python", "-m", "pytorch_fid", stats_npz, crops_out]).decode().strip()
                try:
                    fid_val = float(result.split("FID:")[-1].strip())
                except Exception:
                    raise RuntimeError(f"Unexpected FID output: {result}")
                outp = f"{expr_path}/fid_{ref_label}{suffix.replace('/', '_')}{CROP_RES}{CROP_NUM_LABEL}.json"
                with open(outp, "w") as f:
                    json.dump(fid_val, f)
                print(f"[FID] {ref_label}: {fid_val:.4f} -> {outp}")
                report_lines.append(f"  FID vs {ref_label}: {fid_val:.6f}  (saved: {outp})")

    # ----- PSNR/SSIM/LPIPS (+optional pyiqa) -----
    if compute_distances:
        deg_scores: Dict[str, List[float]] = {acronym(m): [] for m in base_metrics}
        gt_scores:  Dict[str, List[float]] = {acronym(m): [] for m in base_metrics}
        deg_fr: Dict[str, List[float]] = {k: [] for k in fr_metrics}
        gt_fr:  Dict[str, List[float]] = {k: [] for k in fr_metrics}
        deg_nr: Dict[str, List[float]] = {k: [] for k in nr_metrics}
        gt_nr:  Dict[str, List[float]] = {k: [] for k in nr_metrics}

        all_preds = try_glob(f"{expr_path}/inversions/**/pred{suffix}.png")
        if not all_preds:
            msg = f"[WARN] No 'pred{suffix}.png' under {expr_path}; skipping distances."
            print(msg); report_lines.append("  " + msg)
        else:
            for im_path in all_preds:
                dp = replace_str(im_path, f"pred{suffix}", f"degraded_pred{suffix}")
                tp = replace_str(im_path, f"pred{suffix}", "target")
                gp = replace_str(im_path, f"pred{suffix}", "ground_truth")

                if not (os.path.isfile(dp) and os.path.isfile(tp) and os.path.isfile(gp)):
                    print(f"[WARN] Missing companion files for: {im_path}; skipping.")
                    continue

                def imopen(x: str) -> torch.Tensor:
                    t = read_image(x).unsqueeze(0).float() / 255.0
                    return (t * 2.0 - 1.0).to(device).clamp(-1, 1)  # [-1,1]

                pred = imopen(im_path)
                degraded_pred = imopen(dp)
                target = imopen(tp)
                gt = imopen(gp)

                with torch.no_grad():
                    for m in base_metrics:
                        name = acronym(m)
                        deg_scores[name].append(m(degraded_pred, target).item())
                        gt_scores[name].append(m(pred, gt).item())

                    for name, m in fr_metrics.items():
                        deg_fr[name].append(m(degraded_pred, target).item())
                        gt_fr[name].append(m(pred, gt).item())

                    for name, m in nr_metrics.items():
                        deg_nr[name].append(m(degraded_pred).item())
                        gt_nr[name].append(m(pred).item())

                if "DRY_RUN" in os.environ:
                    break

            def means(d: Dict[str, List[float]]) -> Dict[str, float]:
                return {k: (float(torch.tensor(v).mean().item()) if v else float("nan")) for k, v in d.items()}

            deg_means = means(deg_scores)
            gt_means  = means(gt_scores)

            deg_json = f"{expr_path}/degraded_scores{suffix.replace('/', '_')}.json"
            gt_json  = f"{expr_path}/ground_truth_scores{suffix.replace('/', '_')}.json"
            with open(deg_json, "w") as f:
                json.dump(deg_means, f, indent=2)
            with open(gt_json, "w") as f:
                json.dump(gt_means, f, indent=2)

            # Report PSNR/SSIM/LPIPS nicely
            report_lines.append("  Pred vs GT (means):")
            for k in sorted(gt_means.keys()):
                report_lines.append(f"    {k}: {gt_means[k]:.6f}  (saved: {gt_json})")
            report_lines.append("  Degraded vs Target (means):")
            for k in sorted(deg_means.keys()):
                report_lines.append(f"    {k}: {deg_means[k]:.6f}  (saved: {deg_json})")

            # pyiqa (optional)
            if fr_metrics:
                fr_deg = means(deg_fr); fr_gt = means(gt_fr)
                fr_deg_json = f"{expr_path}/degraded_iqa_fr{suffix.replace('/', '_')}.json"
                fr_gt_json  = f"{expr_path}/ground_truth_iqa_fr{suffix.replace('/', '_')}.json"
                with open(fr_deg_json, "w") as f:
                    json.dump(fr_deg, f, indent=2)
                with open(fr_gt_json, "w") as f:
                    json.dump(fr_gt, f, indent=2)
                report_lines.append("  pyiqa FR (Pred vs GT):")
                for k in sorted(fr_gt.keys()):
                    report_lines.append(f"    {k}: {fr_gt[k]:.6f}  (saved: {fr_gt_json})")
                report_lines.append("  pyiqa FR (Degraded vs Target):")
                for k in sorted(fr_deg.keys()):
                    report_lines.append(f"    {k}: {fr_deg[k]:.6f}  (saved: {fr_deg_json})")

            if nr_metrics:
                nr_deg = means(deg_nr); nr_gt = means(gt_nr)
                nr_deg_json = f"{expr_path}/degraded_iqa_nr{suffix.replace('/', '_')}.json"
                nr_gt_json  = f"{expr_path}/ground_truth_iqa_nr{suffix.replace('/', '_')}.json"
                with open(nr_deg_json, "w") as f:
                    json.dump(nr_deg, f, indent=2)
                with open(nr_gt_json, "w") as f:
                    json.dump(nr_gt, f, indent=2)
                report_lines.append("  pyiqa NR (Pred):")
                for k in sorted(nr_gt.keys()):
                    report_lines.append(f"    {k}: {nr_gt[k]:.6f}  (saved: {nr_gt_json})")
                report_lines.append("  pyiqa NR (Degraded):")
                for k in sorted(nr_deg.keys()):
                    report_lines.append(f"    {k}: {nr_deg[k]:.6f}  (saved: {nr_deg_json})")


def eval_all_experiments(
    root: str,
    device: torch.device,
    ref_stats: List[Tuple[str, str]],
    compute_fid: bool,
    compute_distances: bool,
    n_jobs: int,
    lpips_net: str,
    iqa_fr: List[str],
    iqa_nr: List[str],
    limit: Optional[int],
    report_lines: List[str],
) -> None:
    exprs = discover_experiments(root)
    if limit is not None:
        exprs = exprs[:limit]
    print(f"\nDiscovered {len(exprs)} experiment(s) under: {root}")
    report_lines.append(f"Discovered {len(exprs)} experiment(s) under: {root}")

    for expr in exprs:
        print("👉", expr)
        report_lines.append(f"--> {expr}")
        eval_experiment(
            expr_path=expr,
            device=device,
            ref_stats=ref_stats,
            compute_fid=compute_fid,
            compute_distances=compute_distances,
            n_jobs=n_jobs,
            lpips_net=lpips_net,
            iqa_fr=iqa_fr or [],
            iqa_nr=iqa_nr or [],
            report_lines=report_lines,
        )


# ---------- CLI ----------
def parse_args():
    import argparse
    ap = argparse.ArgumentParser(description="Recursive eval (W++ only): FID (+multi-ref), PSNR, SSIM, LPIPS, optional pyiqa FR/NR.")
    ap.add_argument("root", help="Top-level directory (we recursively search for **/inversions/ here).")

    # references for FID
    ap.add_argument("--ref", action="append", default=[], help="Reference image directory. Repeatable.")
    ap.add_argument("--ref-name", action="append", default=[], help="Labels for --ref, same order (optional).")
    ap.add_argument("--ref-stats", action="append", default=[], help="Precomputed FID stats .npz (repeatable).")

    # toggles
    ap.add_argument("--no-fid", action="store_true", help="Skip FID.")
    ap.add_argument("--no-distances", action="store_true", help="Skip PSNR/SSIM/LPIPS (+IQA).")

    # crops for FID
    ap.add_argument("--crop-res", type=int, default=CROP_RES, help="FID crop size (default 128).")
    ap.add_argument("--crop-num", type=int, default=CROP_NUM, help="FID crops per image (default 1000).")
    ap.add_argument("--n-jobs", type=int, default=16, help="Parallel jobs for cropping.")

    # device & LPIPS backbone
    ap.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"), help="cuda/cpu")
    ap.add_argument("--lpips-net", choices=["vgg", "alex", "squeeze"], default="vgg")

    # IQA-PyTorch choices
    ap.add_argument("--iqa-fr", action="append", default=[],
                    help="FR (pyiqa) e.g., dists, fsim, gmsd, vsi, pieapp. Repeatable.")
    ap.add_argument("--iqa-nr", action="append", default=[],
                    help="NR (pyiqa) e.g., niqe, brisque, musiq, topiq, nima, dbcnn, nrqm, piqe. Repeatable.")

    # TXT report path
    ap.add_argument("--report", default=None,
                    help="TXT output path. Default: <root>/eval_Wplusplus_report.txt")

    # debug / limits
    ap.add_argument("--limit", type=int, default=None, help="Limit number of experiments for quick tests.")
    return ap.parse_args()

def main():
    global CROP_RES, CROP_NUM, CROP_RES_LABEL, CROP_NUM_LABEL
    args = parse_args()

    # update crop globals
    CROP_RES = int(args.crop_res)
    CROP_NUM = int(args.crop_num)
    CROP_RES_LABEL, CROP_NUM_LABEL = _labels()

    device = torch.device(args.device)

    # prepare references for FID (dirs and/or .npz files)
    if not args.no_fid:
        refs = collect_all_ref_stats(args.ref, args.ref_name, args.ref_stats, CROP_RES, CROP_NUM, args.n_jobs)
    else:
        refs = []

    # set report path
    report_path = args.report or os.path.join(args.root, "eval_Wplusplus_report.txt")
    report_lines: List[str] = []
    report_lines.append("=== Evaluation Report (suffix: _W++) ===")
    report_lines.append(f"Root: {args.root}")
    report_lines.append(f"FID: {'ON' if not args.no_fid else 'OFF'}  |  Distances: {'ON' if not args.no_distances else 'OFF'}")
    if refs:
        report_lines.append("References:")
        for lbl, npz in refs:
            report_lines.append(f"  - {lbl}: {npz}")
    if args.iqa_fr:
        report_lines.append(f"pyiqa FR: {', '.join(args.iqa_fr)}")
    if args.iqa_nr:
        report_lines.append(f"pyiqa NR: {', '.join(args.iqa_nr)}")

    eval_all_experiments(
        root=args.root,
        device=device,
        ref_stats=refs,
        compute_fid=not args.no_fid,
        compute_distances=not args.no_distances,
        n_jobs=args.n_jobs,
        lpips_net=args.lpips_net,
        iqa_fr=args.iqa_fr,
        iqa_nr=args.iqa_nr,
        limit=args.limit,
        report_lines=report_lines,
    )

    # Write consolidated TXT
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines) + "\n")
    print(f"\n[REPORT] Wrote consolidated report to: {report_path}")

if __name__ == "__main__":
    main()

