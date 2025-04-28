'''Data loader for the main dataset for pretraining on binary masks'''
import os
import numpy as np
import glob
from PIL import Image, ImageEnhance, ImageOps
import random
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.ndimage.morphology import distance_transform_edt

def invert_mask(background_mask_tensor):
    return 1.0 - background_mask_tensor

def cv_random_flip(img, label):
    flip_flag = random.randint(0, 1)
    # flip_flag2= random.randint(0,1)
    #left right flip
    if flip_flag == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
    #top bottom flip
    # if flip_flag2==1:
    #     img = img.transpose(Image.FLIP_TOP_BOTTOM)
    #     label = label.transpose(Image.FLIP_TOP_BOTTOM)
    #     depth = depth.transpose(Image.FLIP_TOP_BOTTOM)
    return img, label

def randomCrop(image, label):
    border=30
    image_width = image.size[0]
    image_height = image.size[1]
    crop_win_width = np.random.randint(image_width-border , image_width)
    crop_win_height = np.random.randint(image_height-border , image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    return image.crop(random_region), label.crop(random_region)

def randomRotation(image,label):
    mode=Image.BICUBIC
    if random.random()>0.8:
        random_angle = np.random.randint(-15, 15)
        image=image.rotate(random_angle, mode)
        label=label.rotate(random_angle, mode)
    return image,label

def colorEnhance(image):
    bright_intensity=random.randint(5,15)/10.0
    image=ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity=random.randint(5,15)/10.0
    image=ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity=random.randint(0,20)/10.0
    image=ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity=random.randint(0,30)/10.0
    image=ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image

def randomGaussian(image, mean=0.1, sigma=0.35):
    def gaussianNoisy(im, mean=mean, sigma=sigma):
        for _i in range(len(im)):
            im[_i] += random.gauss(mean, sigma)
        return im
    img = np.asarray(image)
    width, height = img.shape
    img = gaussianNoisy(img[:].flatten(), mean, sigma)
    img = img.reshape([width, height])
    return Image.fromarray(np.uint8(img))

def randomPeper(img):

    img=np.array(img)
    noiseNum=int(0.0015*img.shape[0]*img.shape[1])
    for i in range(noiseNum):

        randX=random.randint(0,img.shape[0]-1)  

        randY=random.randint(0,img.shape[1]-1)  

        if random.randint(0,1)==0:  

            img[randX,randY]=0  

        else:  

            img[randX,randY]=255 
    return Image.fromarray(img)  

# dataset for training
#The current loader is not using the normalized depth maps for training and test. If you use the normalized depth maps
#(e.g., 0 represents background and 1 represents foreground.), the performance will be further improved.
class HeatmapGenerator():
    def __init__(self, output_res, num_joints, sigma=-1):
        self.output_res = output_res
        self.num_joints = num_joints
        if sigma < 0:
            sigma = self.output_res/64
        self.sigma = sigma
        size = 6*sigma + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = 3*sigma + 1, 3*sigma + 1
        self.g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    def __call__(self, joints):
        hms = np.zeros((self.num_joints, self.output_res, self.output_res),
                       dtype=np.float32)
        sigma = self.sigma
        for idx, pt in enumerate(joints):
            if pt[2] > 0:
                x, y = int(pt[0]), int(pt[1])
                if x < 0 or y < 0 or \
                    x >= self.output_res or y >= self.output_res:
                    continue

                ul = int(np.round(x - 3 * sigma - 1)), int(np.round(y - 3 * sigma - 1))
                br = int(np.round(x + 3 * sigma + 2)), int(np.round(y + 3 * sigma + 2))

                c, d = max(0, -ul[0]), min(br[0], self.output_res) - ul[0]
                a, b = max(0, -ul[1]), min(br[1], self.output_res) - ul[1]

                cc, dd = max(0, ul[0]), min(br[0], self.output_res)
                aa, bb = max(0, ul[1]), min(br[1], self.output_res)
                hms[idx, aa:bb, cc:dd] = np.maximum(
                    hms[idx, aa:bb, cc:dd], self.g[a:b, c:d])
        return hms

def augment(img_tar, img_nn, flip_h=True, rot=True):
    info_aug = {'flip_h': False, 'flip_v': False, 'trans': False}
    
    img_nn = colorEnhance(img_nn)
    if random.random() < 0.5 and flip_h:
        #img_in = ImageOps.flip(img_in)
        img_tar = ImageOps.flip(img_tar)
        img_nn = ImageOps.flip(img_nn)
        info_aug['flip_h'] = True

    if rot:
        if random.random() < 0.5:
            #img_in = ImageOps.mirror(img_in)
            img_tar = ImageOps.mirror(img_tar)
            img_nn = ImageOps.mirror(img_nn)
            info_aug['flip_v'] = True
        if random.random() < 0.5:
            #img_in = img_in.rotate(180)
            img_tar = img_tar.rotate(180)
            img_nn = img_nn.rotate(180)
            info_aug['trans'] = True
    return img_tar, img_nn

def convert_mask(mask, max_obj):
    # convert mask to one hot encoded
    oh = []
    for k in range(max_obj+1):
        oh.append(mask==k)

    if isinstance(mask, np.ndarray):
        oh = np.stack(oh, axis=0)
    else:
        oh = torch.cat(oh, dim=0).float()

    return oh


class MainDataset(Dataset):
    def __init__(self, root, trainsize, clip_len=3):
        assert clip_len % 2 == 1 #it needs to be odd
        self.trainsize = trainsize
        self.clip_len = clip_len
        half = clip_len // 2
        
        #Add the fan to the training transforms
        self.img_transform = transforms.Compose([
            transforms.Resize((trainsize, trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],
                                 [0.229,0.224,0.225])
        ])
        self.gt_transform = transforms.Compose([
            transforms.Resize((trainsize, trainsize)),
            transforms.ToTensor()
        ])

        self.samples = []

        for vid_name in sorted(os.listdir(root)):
            vid_dir = os.path.join(root, vid_name)
            if not os.path.isdir(vid_dir):
                continue

            # list and sort frame files
            frames = sorted([
                f for f in os.listdir(vid_dir)
                if f.endswith('.png') and f.lower() == 'frame.png' or 'frame' in f.lower()
            ], key=lambda x: int(os.path.splitext(x)[0].split('_')[0]))
            gts = sorted([
                f for f in os.listdir(vid_dir)
                if f.endswith('.png') and f.lower() == 'background.png' or 'background' in f.lower()
            ], key=lambda x: int(os.path.splitext(x)[0].split('_')[0]))

            N = len(frames)
            if N < clip_len:
                print(f'The video contains only {N} frames and its too short')
                continue  # skip too-short videos

            # for each valid center index, collect its window
            for i in range(half, N - half):
                clip_paths = [
                    os.path.join(vid_dir, frames[j])
                    for j in range(i - half, i + half + 1)
                ]
                
                gt_paths = [
                    os.path.join(vid_dir, gts[j])
                    for j in range(i - half, i + half + 1)
                    ]
                self.samples.append((clip_paths, gt_paths))

        self.size = len(self.samples)
        print(f"Total clips: {self.size}")

    def __len__(self):
        return self.size
    
    def __getitem__(self, idx): 
        clip_paths, gt_paths = self.samples[idx]
        #Here apply fan 
        frames = [Image.open(p).convert('RGB') for p in clip_paths]
        frames = [self.img_transform(im) for im in frames]
        clip_tensor = torch.stack(frames)  # (clip_len, C, H, W)

        gts = [Image.open(gt).convert('L') for gt in gt_paths]

        edge =  []

        for i in range(len(gts)):
            gts[i] = np.array(gts[i])
            gts[i] = Image.fromarray(gts[i])
            gts[i] = self.gt_transform(randomPeper(gts[i]))
            gts[i] = invert_mask(gts[i])
            _edgemap = gts[i].clone()
            _edgemap = convert_mask(_edgemap, 1)
            _edgemap = _edgemap.numpy()
            _edgemap = self.onehot_to_binary_edges(_edgemap, 2, num_classes=2)
            _edgemap = torch.from_numpy(_edgemap).float()
            #_edgemap = torch.from_numpy(_edgemap).unsqueeze(0).float()
            edge.append(_edgemap)

        gts_tensor = torch.stack(gts)         # (clip_len, H, W)
        #gts_tensor = gts_tensor.squeeze(1)
        #print(gts_tensor.shape)
        #print(clip_tensor.shape)

        return clip_tensor, gts_tensor, torch.stack(edge)
    
    
    def mask_to_onehot(self, mask, num_classes):
        _mask = [mask == (i) for i in range(num_classes)]
        return np.array(_mask).astype(np.uint8)
    
    def onehot_to_binary_edges(self, mask, radius, num_classes):
        mask_pad = np.pad(mask, ((0, 0), (1, 1), (1, 1)), mode='constant', constant_values=0)
        edgemap = np.zeros(mask.shape[1:])
        for i in range(num_classes):
            dist = distance_transform_edt(mask_pad[i, :]) + distance_transform_edt(1.0 - mask_pad[i, :])
            dist = dist[1:-1, 1:-1]
            dist[dist > radius] = 0
            edgemap += dist
        # edgemap = np.expand_dims(edgemap, axis=0)
        edgemap = (edgemap > 0).astype(np.uint8)
        return edgemap
    
    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts

    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt