#!/usr/bin/env python3
import argparse
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MASK_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

def collect_by_stem(folder: Path, exts: set[str] | None):
    files_by_stem = {}
    for p in folder.iterdir():
        if not p.is_file():
            continue
        if exts is not None and p.suffix.lower() not in exts:
            continue
        files_by_stem.setdefault(p.stem, []).append(p)
    return files_by_stem

def main():
    ap = argparse.ArgumentParser(
        description="Delete unpaired files between images/ and masks/ by basename (stem)."
    )
    ap.add_argument("--images", type=str, default="images", help="Path to images folder")
    ap.add_argument("--masks", type=str, default="masks", help="Path to masks folder")
    ap.add_argument("--dry-run", action="store_true", help="Print actions without deleting")
    ap.add_argument("--all-exts", action="store_true",
                    help="Match all files regardless of extension (otherwise common image exts only)")
    args = ap.parse_args()

    images_dir = Path(args.images)
    masks_dir = Path(args.masks)

    if not images_dir.is_dir():
        raise SystemExit(f"Not a folder: {images_dir}")
    if not masks_dir.is_dir():
        raise SystemExit(f"Not a folder: {masks_dir}")

    img_exts = None if args.all_exts else IMG_EXTS
    mask_exts = None if args.all_exts else MASK_EXTS

    imgs = collect_by_stem(images_dir, img_exts)
    masks = collect_by_stem(masks_dir, mask_exts)

    img_stems = set(imgs.keys())
    mask_stems = set(masks.keys())

    keep = img_stems & mask_stems
    only_imgs = img_stems - mask_stems
    only_masks = mask_stems - img_stems

    to_delete = []
    for s in sorted(only_imgs):
        to_delete.extend(imgs[s])
    for s in sorted(only_masks):
        to_delete.extend(masks[s])

    print(f"Images folder: {images_dir}")
    print(f"Masks  folder: {masks_dir}")
    print(f"Paired (kept): {len(keep)} stems")
    print(f"Only in images: {len(only_imgs)} stems -> deleting {sum(len(imgs[s]) for s in only_imgs)} files")
    print(f"Only in masks : {len(only_masks)} stems -> deleting {sum(len(masks[s]) for s in only_masks)} files")
    print(f"Total files to delete: {len(to_delete)}")
    print("")

    for p in to_delete:
        if args.dry_run:
            print(f"[DRY-RUN] would delete: {p}")
        else:
            print(f"deleting: {p}")
            p.unlink()

    print("\nDone.")

if __name__ == "__main__":
    main()