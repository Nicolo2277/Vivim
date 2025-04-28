# check_dataloader.py

import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from main_dataset import MainDataset  

def visualize_clip(clip_tensor, gt_tensor, edge_tensor, idx=0):
    """
    Draws the idx-th clip from the batch:
    - clip_tensor: (B, L, C, H, W)
    - gt_tensor:   (B, L, 1, H, W)
    - edge_tensor: (B, L, 1, H, W)
    """
    clip = clip_tensor[idx]    # (L, C, H, W)
    gt   = gt_tensor[idx]      # (L, 1, H, W)
    edg  = edge_tensor[idx]    # (L, 1, H, W)
    L = clip.shape[0]

    fig, axes = plt.subplots(3, L, figsize=(L * 3, 9))
    for i in range(L):
        # RGB frame
        frame = clip[i].permute(1, 2, 0).cpu().numpy()
        axes[0, i].imshow(frame)
        axes[0, i].set_title(f"Frame {i}")
        axes[0, i].axis('off')

        # GT mask
        mask = gt[i,0].cpu().numpy()
        axes[1, i].imshow(mask, cmap='gray')
        axes[1, i].set_title("GT Mask")
        axes[1, i].axis('off')

        # Edge map
        edge = edg[i,0].cpu().numpy()
        axes[2, i].imshow(edge, cmap='gray')
        axes[2, i].set_title("Edge Map")
        axes[2, i].axis('off')

    plt.tight_layout()
    plt.show()


def main():
    # --- PARAMETERS ---
    ROOT       = "./TrainData"
    TRAINSIZE  = 256
    CLIP_LEN   = 3
    BATCH_SIZE = 2
    NUM_WORKERS = 4

    # 1) instantiate dataset + loader
    dataset = MainDataset(ROOT, trainsize=TRAINSIZE, clip_len=CLIP_LEN)
    loader  = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    # 2) fetch one batch
    clip_batch, gt_batch, edge_batch = next(iter(loader))

    # 3) print shapes
    print(f"  clip_batch shape: {clip_batch.shape}  # (B, L, C, H, W)")
    print(f"     gt_batch shape: {gt_batch.shape}   # (B, L, 1, H, W)")
    print(f"   edge_batch shape: {edge_batch.shape}  # (B, L, 1, H, W)")

    # 4) visualize the first sample in the batch
    visualize_clip(clip_batch, gt_batch, edge_batch, idx=0)


if __name__ == "__main__":
    main()
