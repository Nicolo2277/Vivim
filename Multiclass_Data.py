import os
import random
import numpy as np
from PIL import Image, ImageEnhance, ImageOps, ImageFilter
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from scipy.ndimage import distance_transform_edt


class MainDataset(Dataset):
    def __init__(self, root, trainsize, clip_len=3, max_num=None, augment_intensity='medium'):
        assert clip_len % 2 == 1, "clip_len must be odd"
        self.trainsize = trainsize
        self.augment_intensity = augment_intensity
        self.clip_len = clip_len
        self.max_num = max_num
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

            all_clips = []
            for center in range(self.half, N - self.half, self.clip_len):
                clip = frames[center-self.half : center+self.half+1]
                # store full paths
                clip_paths = [os.path.join(vid_dir, f) for f in clip]
                all_clips.append(clip_paths)
            
            if self.max_num is not None and len(all_clips) > self.max_num:
                selected_clips = []
                indices = np.linspace(0, len(all_clips) - 1, self.max_num, dtype=int)
                for idx in indices:
                    selected_clips.append(all_clips[idx])
                self.samples.extend(selected_clips)
            else:
                self.samples.extend(all_clips)


        print(f"Total clips: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)
    

    def cv_random_flip(self, img, label, p=0.5):
        if random.random() < p:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)
        return img, label
        
    def randomRotation(self, img, label, p=0.5, angle_range=(-15, 15)):
        if random.random() < p:
            angle = random.uniform(angle_range[0], angle_range[1])
            img = img.rotate(angle, Image.BICUBIC)
            label = label.rotate(angle, Image.NEAREST)
        return img, label
    
    def randomCrop(self, img, label, p=0.3):
        if random.random() < p:
            width, height = img.size
            crop_ratio = random.uniform(0.8, 0.95)
            crop_w, crop_h = int(width * crop_ratio), int(height * crop_ratio)
            left = random.randint(0, width - crop_w)
            top = random.randint(0, height - crop_h)
            img = img.crop((left, top, left + crop_w, top + crop_h))
            label = label.crop((left, top, left + crop_w, top + crop_h))
            img = img.resize((width, height), Image.BICUBIC)
            label = label.resize((width, height), Image.NEAREST)
        return img, label
    
    def colorEnhance(self, img, intensity='medium'):
        if intensity == 'none':
            return img
        if intensity == 'light':
            bright_range = (0.9, 1.1)
            contrast_range = (0.9, 1.1)
            color_range = (0.9, 1.1)
            sharp_range = (0.9, 1.1)
        elif intensity == 'medium':
            bright_range = (0.7, 1.3)
            contrast_range = (0.7, 1.3)
            color_range = (0.7, 1.3)
            sharp_range = (0.7, 1.3)
        elif intensity == 'heavy':
            bright_range = (0.5, 1.5)
            contrast_range = (0.5, 1.5)
            color_range = (0.5, 1.5)
            sharp_range = (0.5, 1.5)
        img = ImageEnhance.Brightness(img).enhance(random.uniform(*bright_range))
        img = ImageEnhance.Contrast(img).enhance(random.uniform(*contrast_range))
        img = ImageEnhance.Color(img).enhance(random.uniform(*color_range))
        img = ImageEnhance.Sharpness(img).enhance(random.uniform(*sharp_range))
        return img

    def randomGamma(self, img, p=0.3, gamma_range=(0.7, 1.5)):
        if random.random() < p:
            gamma = random.uniform(gamma_range[0], gamma_range[1])
            img = Image.fromarray(np.uint8(255 * np.power(np.array(img) / 255.0, gamma)))
        return img
    
    def randomPeper(self, img, p=0.3, intensity=0.0015):
        if random.random() < p:
            arr = np.array(img)
            num = int(intensity * arr.size)
            xs = np.random.randint(0, arr.shape[0], num)
            ys = np.random.randint(0, arr.shape[1], num)
            arr[xs, ys] = np.random.choice([0, 255], num)
            return Image.fromarray(arr)
        return img
    
    def randomBlur(self, img, p=0.2):
        if random.random() < p:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
        return img
    
    def apply_augmentation(self, img, mask):
        if self.augment_intensity == 'none':
            flip_p, rotate_p, crop_p, blur_p, gamma_p, peper_p = 0, 0, 0, 0, 0, 0
        elif self.augment_intensity == 'light':
            flip_p, rotate_p, crop_p, blur_p, gamma_p, peper_p = 0.5, 0.2, 0.1, 0.1, 0.1, 0.05
        elif self.augment_intensity == 'medium':
            flip_p, rotate_p, crop_p, blur_p, gamma_p, peper_p = 0.5, 0.3, 0.3, 0.2, 0.2, 0.1
        elif self.augment_intensity == 'heavy':             
            flip_p, rotate_p, crop_p, blur_p, gamma_p, peper_p = 0.5, 0.4, 0.4, 0.3, 0.3, 0.15
        
        img, mask = self.cv_random_flip(img, mask, p=flip_p)

        img, mask = self.randomRotation(img, mask, p=rotate_p, angle_range=(-20, 20) if self.augment_intensity == 'heeavy' else (-15, 15))

        img, mask = self.randomCrop(img, mask, p = crop_p)

        img = self.colorEnhance(img, intensity=self.augment_intensity)
        img = self.randomBlur(img, p=blur_p)
        img = self.randomGamma(img, p=gamma_p)
        #img = self.randomPeper(img, p=peper_p)

        return img, mask


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
                img, m = self.apply_augmentation(img, m)
                masks[j] = m
            imgs[i] = img
            all_masks[i] = masks

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




class TestDataset(Dataset):
    def __init__(self, root, testsize, clip_len=3, max_num=None):
        assert clip_len % 2 == 1, "clip_len must be odd"
        self.testsize = testsize
        self.clip_len = clip_len
        self.max_num = max_num
        self.half = clip_len // 2

        self.img_transform = transforms.Compose([
            transforms.Resize((testsize, testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],
                                 [0.229,0.224,0.225])
        ])
        self.gt_transform = transforms.Compose([
            transforms.Resize((testsize, testsize), interpolation=Image.NEAREST),
            transforms.ToTensor()  
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

            all_clips = []
            for center in range(self.half, N - self.half, self.clip_len):
                clip = frames[center-self.half : center+self.half+1]
                # store full paths
                clip_paths = [os.path.join(vid_dir, f) for f in clip]
                all_clips.append(clip_paths)
            
            if self.max_num is not None and len(all_clips) > self.max_num:
                selected_clips = []
                indices = np.linspace(0, len(all_clips) - 1, self.max_num, dtype=int)
                for idx in indices:
                    selected_clips.append(all_clips[idx])
                self.samples.extend(selected_clips)
            else:
                self.samples.extend(all_clips)


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

        return clip_tensor, masks_tensor, clip_paths

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
