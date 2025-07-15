'''Check the multiclass dataloader'''

from pathlib import Path


import argparse
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from Multiclass_Data import MainDataset

def main(root, trainsize, clip_len, batch_size, num_workers):

    ds = MainDataset(root=root, trainsize=trainsize, clip_len=clip_len)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    clip, masks, edges = next(iter(loader))
    
    # Print shapes
    print(f"clip shape:  {clip.shape}    # expected (B, clip_len, 3, H, W)")
    print(f"masks shape: {masks.shape}   # expected (B, clip_len, C, H, W)")
    print(f"edges shape: {edges.shape}   # expected (B, clip_len, 1, H, W)")
    
    print(f"clip min/max:  {clip.min().item():.3f} / {clip.max().item():.3f}")
    print(f"masks unique: {torch.unique(masks)}         # should be 0 or 1")
    print(f"edges unique: {torch.unique(edges)}         # should be 0 or 1")

    sample_clip = clip[0]   # (clip_len, 3, H, W)
    sample_masks = masks[0] # (clip_len, C, H, W)
    sample_edges = edges[0] # (clip_len, 1, H, W)

    # Plot frame, mask and edge for center frame
    center = clip_len // 2
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    img = sample_clip[center].permute(1,2,0).numpy()
    axes[0].imshow((img - img.min()) / (img.max() - img.min()))
    axes[0].set_title("RGB Frame")
    axes[0].axis("off")

    mask = sample_masks[center].argmax(dim=0).numpy()
    axes[1].imshow(mask, cmap="tab10")
    axes[1].set_title("Mask (0:bg,1:solid,2:non-solid)")
    axes[1].axis("off")

    edge = sample_edges[center].squeeze(0).numpy()
    axes[2].imshow(edge, cmap="gray")
    axes[2].set_title("Edge Map")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check DataLoader outputs")
    parser.add_argument("--root",       type=str, default="Multiclass_TrainData_fold_0/train", help="Dataset root directory")
    parser.add_argument("--trainsize",  type=int, default=256,           help="Resize dimension")
    parser.add_argument("--clip_len",   type=int, default=3,             help="Temporal clip length (must be odd)")
    parser.add_argument("--batch_size", type=int, default=2,             help="Batch size for testing")
    parser.add_argument("--num_workers",type=int, default=4,             help="DataLoader workers")
    args = parser.parse_args()
    main(args.root, args.trainsize, args.clip_len, args.batch_size, args.num_workers)


