'''Create the train dataset for binary segmnetation'''
import os
import shutil
from pathlib import Path

def is_frame_file(path: Path):

    return path.is_file() and path.name.lower() == 'frames.png'

def is_gt_file(path: Path):

    return path.is_file() and path.name.lower() == 'background.png'

def gather_annotated_frames(input_root: Path, output_root: Path):
 
 
    input_root = input_root.resolve()
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)


    annotated_by_video = {}


    for dirpath, dirnames, filenames in os.walk(input_root):
        files = set(f.lower() for f in filenames)
        if 'frame.png' in files and 'background.png' in files:
            dir_path = Path(dirpath)

            try:
                rel_parts = dir_path.relative_to(input_root).parts
                video_name = rel_parts[0]
            except Exception:
                # If dir_path == input_root, skip
                continue

            annotated_by_video.setdefault(video_name, []).append(
                (dir_path / 'frame.png', dir_path / 'background.png')
            )


    for video_name, pairs in annotated_by_video.items():
        dest_video = output_root / video_name
        dest_video.mkdir(parents=True, exist_ok=True)


        pairs.sort(key=lambda x: str(x[0]))

        for idx, (frame_path, gt_path) in enumerate(pairs):
            prefix = f"{idx:04d}_"
            shutil.copy2(frame_path, dest_video / f"{prefix}frame.png")
            shutil.copy2(gt_path, dest_video / f"{prefix}background.png")

        print(f"Collected {len(pairs)} annotated frames for video '{video_name}' into {dest_video}")

if __name__ == '__main__':
    input_root = 'Preliminary-data'
    output_root = 'TrainData'
    gather_annotated_frames(Path(input_root), Path(output_root))
