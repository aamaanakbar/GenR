#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import re
import shutil
from pathlib import Path
from typing import Optional, Tuple

EXTS = ("png", "jpg", "jpeg")  # order of preference

def find_pred_wpp(sample_dir: Path) -> Optional[Tuple[Path, str]]:
    """Return (path, ext) of pred_W++ if it exists in this sample dir."""
    for ext in EXTS:
        p = sample_dir / f"pred_W++.{ext}"
        if p.is_file():
            return p, ext
    return None

def is_numeric_dir(d: Path) -> bool:
    """Accept 0000, 0001, 12, etc."""
    return d.is_dir() and re.fullmatch(r"\d+", d.name) is not None

def process_inversions_dir(inv_dir: Path, move: bool, overwrite: bool, dry_run: bool, verbose: bool) -> int:
    """Copy/move all pred_W++.* from numeric subdirs into inv_dir/W++."""
    dest = inv_dir / "W++"
    if verbose:
        print(f"[INV] {inv_dir}")
        print(f"      -> dest: {dest}")

    done = 0
    for sample in sorted(inv_dir.iterdir()):
        if sample.name == "W++":
            continue
        if not is_numeric_dir(sample):
            continue

        found = find_pred_wpp(sample)
        if not found:
            if verbose:
                print(f"  [skip] no pred_W++ in {sample}")
            continue

        src, ext = found
        out = dest / f"{sample.name}_pred_W++.{ext}"

        if out.exists() and not overwrite:
            if verbose:
                print(f"  [skip] exists: {out}")
            continue

        action = "MOVE" if move else "COPY"
        print(f"  [{action}] {src} -> {out}")
        if not dry_run:
            os.makedirs(out.parent, exist_ok=True)
            if move:
                shutil.move(str(src), str(out))
            else:
                shutil.copy2(str(src), str(out))
        done += 1

    return done

def main():
    import argparse
    ap = argparse.ArgumentParser(
        description="Collect pred_W++.* from each numeric subfolder (0000,0001,...) into each inversions/W++ across the whole tree."
    )
    ap.add_argument("root", help="Root directory (we will recursively find every 'inversions' folder)")
    ap.add_argument("--move", action="store_true", help="Move instead of copy")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite files in W++ if they already exist")
    ap.add_argument("--dry-run", action="store_true", help="Show what would happen but make no changes")
    ap.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not root.is_dir():
        print(f"ERROR: root not found: {root}", file=sys.stderr)
        sys.exit(1)

    total_inv = 0
    total_files = 0

    # Walk and process every directory literally named 'inversions'
    for dirpath, dirnames, filenames in os.walk(root):
        p = Path(dirpath)
        if p.name != "inversions":
            continue
        total_inv += 1
        total_files += process_inversions_dir(p, args.move, args.overwrite, args.dry_run, args.verbose)

    print(f"\nDone. inversions dirs processed: {total_inv}, files {'moved' if args.move else 'copied'}: {total_files}")

if __name__ == "__main__":
    main()
#this code is for copying the pred_w++ images from the inversion folders to a single folder called W++ in the inversions folder of each task.
#it will search for all the inversions folders in the root directory and its subdirectories.
#it will then look for the pred_w++ images in each numeric subfolder of the inversions folder.
#it will then copy or move the pred_w++ images to the W++ folder
#it will also rename the images to include the name of the numeric subfolder they came from.
#it will also handle different image formats (png, jpg, jpeg) and avoid overwriting
#to run this code use the command: python copy.py <root_directory>   (python datasets/copy.py /home/akbar.ali/ICASSP26/robust-unsupervised/out/restored_samples/2025-09-03T134815)