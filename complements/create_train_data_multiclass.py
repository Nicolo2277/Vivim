import os
import shutil
from pathlib import Path

def find_annotated_dirs(input_root: Path):

    for dirpath, _, filenames in os.walk(input_root):
        files = set(f.lower() for f in filenames)
        if 'frame.png' in files and 'background.png' in files:
            yield Path(dirpath)

def gather_multiclass_frames(input_root: Path, output_root: Path):
    """
    Copies frame + background (required) and any optional solid/non-solid masks
    into output_root, grouped by top‐level video folder, with zero‐padded ordering.
    """
    input_root = input_root.resolve()
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    videos = {}
    for ann_dir in find_annotated_dirs(input_root):
        try:
            rel = ann_dir.relative_to(input_root)
            video_name = rel.parts[0]
        except ValueError:
            continue
        videos.setdefault(video_name, []).append(ann_dir)

    for vid, dirs in videos.items():
        dest_vid = output_root / vid
        dest_vid.mkdir(parents=True, exist_ok=True)

        dirs = sorted(dirs, key=lambda p: str(p))
        for idx, ann in enumerate(dirs):
            prefix = f"{idx:04d}_"

            for fname in ('frame.png', 'background.png'):
                src = ann / fname
                dst = dest_vid / f"{prefix}{fname}"
                shutil.copy2(src, dst)

            # optionally copy solid.png and non-solid.png if present
            for optional in ('solid.png', 'non-solid.png'):
                src = ann / optional
                if src.exists():
                    dst = dest_vid / f"{prefix}{optional}"
                    shutil.copy2(src, dst)

        print(f"[{vid}] copied {len(dirs)} clips → {dest_vid}")

if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser(
        description="Gather frames + background masks (and optional solid/non-solid) per video"
    )
    p.add_argument("input_root",  type=Path,
                   help="root folder containing video subfolders")
    p.add_argument("output_root", type=Path,
                   help="destination for gathered TrainData")
    args = p.parse_args()

    gather_multiclass_frames(args.input_root, args.output_root)
