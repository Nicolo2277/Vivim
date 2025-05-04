import os
import random
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from scipy.ndimage import distance_transform_edt

def cv_random_flip(img, label):
    if random.random() < 0.5:
        img   = img.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
    return img, label

def randomRotation(image, label):
    if random.random() < 0.2:
        angle = random.uniform(-15, 15)
        image = image.rotate(angle, Image.BICUBIC)
        label = label.rotate(angle, Image.NEAREST)
    return image, label

def colorEnhance(image):
    image = ImageEnhance.Brightness(image).enhance(random.uniform(0.5, 1.5))
    image = ImageEnhance.Contrast(image).enhance(random.uniform(0.5, 1.5))
    image = ImageEnhance.Color(image).enhance(random.uniform(0.0, 2.0))
    image = ImageEnhance.Sharpness(image).enhance(random.uniform(0.0, 3.0))
    return image

def randomPeper(img):
    arr = np.array(img)
    num = int(0.0015 * arr.size)
    xs = np.random.randint(0, arr.shape[0], num)
    ys = np.random.randint(0, arr.shape[1], num)
    arr[xs, ys] = np.random.choice([0, 255], num)
    return Image.fromarray(arr)

class MainDataset(Dataset):
    def __init__(self, root, trainsize, clip_len=3):
        assert clip_len % 2 == 1, "clip_len must be odd"
        self.trainsize = trainsize
        self.clip_len = clip_len
        self.half = clip_len // 2

        self.img_transform = transforms.Compose([
            transforms.Resize((trainsize, trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],
                                 [0.229,0.224,0.225])
        ])
        self.gt_transform = transforms.Compose([
            transforms.Resize((trainsize, trainsize), interpolation=Image.NEAREST),
            transforms.ToTensor()  # will yield [0,1] per channel
        ])

        # Gather per‐video frame lists
        self.samples = []
        for vid in sorted(os.listdir(root)):
            vid_dir = os.path.join(root, vid)
            if not os.path.isdir(vid_dir):
                continue

            frames = sorted([
                f for f in os.listdir(vid_dir)
                if f.endswith('.png') and f.lower() == 'frame.png' or 'frame' in f.lower()
            ], key=lambda x: int(os.path.splitext(x)[0].split('_')[0]))

            # Slide a window of length clip_len over frames
            N = len(frames)
            if N < clip_len:
                print(N)
                continue
            for center in range(self.half, N - self.half, self.clip_len):
                clip = frames[center-self.half : center+self.half+1]
                # store full paths
                clip_paths = [os.path.join(vid_dir, f) for f in clip]
                self.samples.append(clip_paths)

        print(f"Total clips: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        clip_paths = self.samples[idx]
        imgs = [Image.open(p).convert('RGB') for p in clip_paths]

        # Load three masks per frame (background, solid, non-solid)
        all_masks = []
        for p in clip_paths:
            base = os.path.splitext(p)[0]
            masks = []
            for key in ('background', 'solid', 'non-solid'):
                mask_path = base.replace('frame', key) + '.png'
                if os.path.exists(mask_path):
                    m = Image.open(mask_path).convert('L')
                    
                else:
                    m = Image.new('L', imgs[0].size, 0) 
                masks.append(m)
            all_masks.append(masks)

        # Joint spatial aug on each frame + its masks
        for i in range(len(imgs)):
            img = imgs[i]
            masks = all_masks[i]
            for j, m in enumerate(masks):
                img, m = cv_random_flip(img, m)
                img, m = randomRotation(img, m)
                masks[j] = m
            imgs[i] = img
            all_masks[i] = masks

        # Color‐only augment on images
        imgs = [colorEnhance(im) for im in imgs]

        # Transforms + stacking
        img_ts = [self.img_transform(im) for im in imgs]
        clip_tensor = torch.stack(img_ts, dim=0)  # (clip_len, 3, H, W)

        # Build and one‐hot encode masks per frame
        mask_ts = []
        edges  = []
        for masks in all_masks:
            # each m: (L,) image converted to tensor (1,H,W)
            m_ts = [self.gt_transform(m) for m in masks]
            onehot = torch.cat(m_ts, dim=0)        # (3, H, W)
            mask_ts.append(onehot)
            edges.append(self._make_edge_map(onehot))

        masks_tensor = torch.stack(mask_ts, dim=0)  # (clip_len, 3, H, W)
        edges_tensor = torch.stack(edges,    dim=0)  # (clip_len, 1, H, W)

        return clip_tensor, masks_tensor, edges_tensor

    def _make_edge_map(self, onehot_mask, radius=2):
        """
        onehot_mask: (C, H, W), values {0,1}
        returns edge map (1, H, W)
        """
        C, H, W = onehot_mask.shape
        pad = np.pad(onehot_mask.numpy(),
                     ((0,0),(1,1),(1,1)),
                     'constant', constant_values=0)
        emap = np.zeros((H, W), dtype=np.uint8)
        for c in range(C):
            dist = distance_transform_edt(pad[c]) + distance_transform_edt(1 - pad[c])
            dist = dist[1:-1, 1:-1]
            emap += (dist <= radius).astype(np.uint8)
        return torch.from_numpy(emap[None, ...].astype(np.float32))
