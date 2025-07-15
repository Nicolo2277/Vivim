import os
import shutil
from pathlib import Path
import re

def extract_frame_number(path):
    """Extract frame number from a path or filename."""
    # Try to find frame number in the filename
    filename = os.path.basename(str(path))
    match = re.search(r'(\d+)', filename)
    if match:
        return int(match.group(1))
    return None

def find_annotated_dirs(input_root: Path):
    """Find directories containing annotated frames (with background.png)."""
    for dirpath, _, filenames in os.walk(input_root):
        files = set(f.lower() for f in filenames)
        if 'frame.png' in files and 'background.png' in files:
            yield Path(dirpath)

def find_all_frame_dirs(input_root: Path):
    """Find all directories containing frame images."""
    for dirpath, _, filenames in os.walk(input_root):
        files = set(f.lower() for f in filenames)
        if 'frame.png' in files:
            yield Path(dirpath)

def gather_frame_sequences(input_root: Path, output_root: Path, sequence_length=5):
    """
    Creates sequences of frames where the central frame is annotated.
    Each sequence contains {sequence_length} frames with the central frame having annotations.
    
    Args:
        input_root: Root directory containing video folders with frames
        output_root: Output directory for the sequences
        sequence_length: Length of each sequence (must be odd)
    """
    if sequence_length % 2 == 0:
        raise ValueError("sequence_length must be odd")
    
    half_len = sequence_length // 2
    input_root = input_root.resolve()
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    # First, find all annotated directories
    annotated_dirs = list(find_annotated_dirs(input_root))
    all_frame_dirs = list(find_all_frame_dirs(input_root))
    
    # Group by video
    videos = {}
    for ann_dir in annotated_dirs:
        try:
            rel = ann_dir.relative_to(input_root)
            video_name = rel.parts[0]
        except ValueError:
            continue
        videos.setdefault(video_name, {}).setdefault('annotated', []).append(ann_dir)
    
    # Add all frame directories
    for frame_dir in all_frame_dirs:
        try:
            rel = frame_dir.relative_to(input_root)
            video_name = rel.parts[0]
        except ValueError:
            continue
        videos.setdefault(video_name, {}).setdefault('all', []).append(frame_dir)

    for vid, data in videos.items():
        dest_vid = output_root / vid
        dest_vid.mkdir(parents=True, exist_ok=True)
        
        annotated_dirs = data.get('annotated', [])
        all_frame_dirs = data.get('all', [])
        
        # Sort annotated directories by their frame number if possible
        annotated_dirs = sorted(annotated_dirs, key=lambda p: extract_frame_number(p))
        all_frame_dirs = sorted(all_frame_dirs, key=lambda p: extract_frame_number(p))
        
        # Create a mapping between frame numbers and directories
        all_frames_map = {}
        for dir_path in all_frame_dirs:
            frame_num = extract_frame_number(dir_path)
            if frame_num is not None:
                all_frames_map[frame_num] = dir_path
        
        sequence_count = 0
        
        # For each annotated frame, create a sequence
        for central_dir in annotated_dirs:
            central_frame_num = extract_frame_number(central_dir)
            
            if central_frame_num is None:
                # If we can't extract a frame number, skip this directory
                continue
                
            # Find frames before and after
            sequence_frames = []
            
            # Add frames before the central frame
            for offset in range(-half_len, 0):
                frame_num = central_frame_num + offset
                if frame_num in all_frames_map:
                    sequence_frames.append((frame_num, all_frames_map[frame_num]))
                else:
                    # If we can't find a frame, we'll skip the sequence
                    break
            
            # Add the central frame
            sequence_frames.append((central_frame_num, central_dir))
            
            # Add frames after the central frame
            for offset in range(1, half_len + 1):
                frame_num = central_frame_num + offset
                if frame_num in all_frames_map:
                    sequence_frames.append((frame_num, all_frames_map[frame_num]))
                else:
                    # If we can't find a frame, we'll skip the sequence
                    break
            
            # Only proceed if we have a complete sequence
            if len(sequence_frames) == sequence_length:
                # Create a new sequence directory
                sequence_dir = dest_vid / f"seq_{sequence_count:04d}"
                sequence_dir.mkdir(parents=True, exist_ok=True)
                
                # Copy frames and annotations
                for i, (frame_num, src_dir) in enumerate(sequence_frames):
                    position = i - half_len  # -2, -1, 0, 1, 2 for sequence_length=5
                    
                    # Copy frame.png
                    frame_src = src_dir / "frame.png"
                    if frame_src.exists():
                        frame_dst = sequence_dir / f"{position:+d}_frame.png"  # +/- prefix for clarity
                        shutil.copy2(frame_src, frame_dst)
                    
                    # For the central frame (position=0), copy annotations
                    if position == 0:
                        # Required: background.png
                        bg_src = src_dir / "background.png"
                        if bg_src.exists():
                            bg_dst = sequence_dir / f"{position:+d}_background.png"
                            shutil.copy2(bg_src, bg_dst)
                        
                        # Optional: solid.png and non-solid.png
                        for optional in ['solid.png', 'non-solid.png']:
                            opt_src = src_dir / optional
                            if opt_src.exists():
                                opt_dst = sequence_dir / f"{position:+d}_{optional}"
                                shutil.copy2(opt_src, opt_dst)
                    
                    # For non-central frames, copy annotations if they exist
                    else:
                        # Check if this frame has annotations
                        for mask in ['background.png', 'solid.png', 'non-solid.png']:
                            mask_src = src_dir / mask
                            if mask_src.exists():
                                mask_dst = sequence_dir / f"{position:+d}_{mask}"
                                shutil.copy2(mask_src, mask_dst)
                
                sequence_count += 1
        
        print(f"[{vid}] created {sequence_count} sequences → {dest_vid}")

if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser(
        description="Create sequences of frames where the central frame is annotated"
    )
    p.add_argument("-input_root", type=Path,
                   help="root folder containing video subfolders")
    p.add_argument("-output_root", type=Path,
                   help="destination for sequence data")
    p.add_argument("-sequence-length", type=int, default=5,
                   help="length of each sequence (must be odd, default: 5)")
    args = p.parse_args()

    if args.sequence_length % 2 == 0:
        p.error("sequence_length must be odd")

    gather_frame_sequences(args.input_root, args.output_root, args.sequence_length)