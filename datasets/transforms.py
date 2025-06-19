from __future__ import absolute_import, division, print_function

import torch
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode

import random
from PIL import Image

class Resize:
    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.resize = T.Resize((self.height, self.width), interpolation=InterpolationMode.BILINEAR)

    def __call__(self, data):
        # Resize color original image (torch.Tensor with shape [3, H, W])
        if isinstance(data['color'], torch.Tensor):
            color_img = T.ToPILImage()(data['color'])
            resized_img = self.resize(color_img)
            data['color'] = T.ToTensor()(resized_img)

        # Resize color augmented image (torch.Tensor with shape [3, H, W])
        if isinstance(data['color_aug'], torch.Tensor):
            color_img = T.ToPILImage()(data['color_aug'])
            resized_img = self.resize(color_img)
            data['color_aug'] = T.ToTensor()(resized_img)

        # Resize depth_gt if it is a 2D tensor
        if 'depth_gt' in data and torch.is_tensor(data['depth_gt']):
            depth = data['depth_gt'].unsqueeze(0)  # [1, H, W]
            depth_resized = T.functional.resize(depth, (self.height, self.width), interpolation=InterpolationMode.NEAREST)
            data['depth_gt'] = depth_resized.squeeze(0)  # Back to [H, W]


        # Resize radar_pts_2d_t if it is a 2D tensor
        if 'radar_points_2d' in data and torch.is_tensor(data['radar_points_2d']):
            depth = data['radar_points_2d'].unsqueeze(0)  # [1, H, W]
            depth_resized = T.functional.resize(depth, (self.height, self.width), interpolation=InterpolationMode.NEAREST)
            data['radar_points_2d'] = depth_resized.squeeze(0)  # Back to [H, W]

        # Resize radar_jbf if present
        if 'radar_jbf' in data and torch.is_tensor(data['radar_jbf']):
            radar_img = T.ToPILImage()(data['radar_jbf'])
            resized_radar = self.resize(radar_img)
            data['radar_jbf'] = T.ToTensor()(resized_radar)

        return data


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        if random.random() < self.p:
            _, _, w = data['color_aug'].shape  # [C, H, W]

            # Flip image
            data['color_aug'] = torch.flip(data['color_aug'], dims=[2])  # Width flip

            # Flip depth maps (H, W)
            if 'depth_gt' in data and data['depth_gt'] is not None:
                data['depth_gt'] = torch.flip(data['depth_gt'], dims=[1])

            # Flip radar_jbf (2, H, W)
            if 'radar_jbf' in data and isinstance(data['radar_jbf'], torch.Tensor) and data['radar_jbf'].ndim == 3:
                data['radar_jbf'] = torch.flip(data['radar_jbf'], dims=[2])

            # Adjust intrinsics (cx)
            K = data['K'].clone()
            K[0, 2] = w - K[0, 2]
            data['K'] = K
            data['inv_K'] = torch.inverse(K)

        return data


class ColorJitter:
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1):
        self.transform = T.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue
        )

    def __call__(self, data):
        # Αν είναι Tensor, μετατροπή σε PIL για το ColorJitter
        img = data['color_aug']
        if isinstance(img, torch.Tensor):
            img = T.ToPILImage()(img)

        # Apply jitter
        img = self.transform(img)

        # Μετατροπή πίσω σε Tensor
        data['color_aug'] = T.ToTensor()(img)
        return data


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
        return data
