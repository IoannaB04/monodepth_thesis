from __future__ import absolute_import, division, print_function
import random

import torch
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as F_t


class Resize:
    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.resize = T.Resize((self.height, self.width), interpolation=InterpolationMode.BILINEAR)

    def __call__(self, data):
        if isinstance(data['color'], torch.Tensor):
            color_img = T.ToPILImage()(data['color'])
            resized_img = self.resize(color_img)
            data['color'] = T.ToTensor()(resized_img)

        if isinstance(data['color_aug'], torch.Tensor):
            color_img = T.ToPILImage()(data['color_aug'])
            resized_img = self.resize(color_img)
            data['color_aug'] = T.ToTensor()(resized_img)

        if 'depth_gt' in data and torch.is_tensor(data['depth_gt']):
            depth = data['depth_gt'].unsqueeze(0)
            depth_resized = F_t.resize(depth, (self.height, self.width), interpolation=InterpolationMode.NEAREST)
            data['depth_gt'] = depth_resized.squeeze(0)

        if 'radar_points_2d' in data and torch.is_tensor(data['radar_points_2d']):
            radar = data['radar_points_2d'].unsqueeze(0)
            radar_resized = F_t.resize(radar, (self.height, self.width), interpolation=InterpolationMode.NEAREST)
            data['radar_points_2d'] = radar_resized.squeeze(0)

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
            _, _, w = data['color_aug'].shape

            data['color_aug'] = torch.flip(data['color_aug'], dims=[2])

            if 'depth_gt' in data and data['depth_gt'] is not None:
                data['depth_gt'] = torch.flip(data['depth_gt'], dims=[1])

            if 'radar_jbf' in data and isinstance(data['radar_jbf'], torch.Tensor) and data['radar_jbf'].ndim == 3:
                data['radar_jbf'] = torch.flip(data['radar_jbf'], dims=[2])

            # Adjust intrinsics
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
        img = data['color_aug']
        if isinstance(img, torch.Tensor):
            img = T.ToPILImage()(img)

        img = self.transform(img)

        data['color_aug'] = T.ToTensor()(img)
        return data

class RandomBrightness:
    def __init__(self, brightness_range=[0, 1], p=0.5):
        self.brightness_min, self.brightness_max = brightness_range
        self.p = p

    def __call__(self, data):
        image = data['color_aug']  # π.χ. εφαρμογή μόνο στην color_aug εικόνα
        if torch.rand(1).item() < self.p:
            factor = torch.empty(1).uniform_(self.brightness_min, self.brightness_max).item()
            image = F_t.adjust_brightness(image, factor)
        data['color_aug'] = image
        return data

class RandomContrast:
    def __init__(self, contrast_range=[0, 1], p=0.5):
        self.contrast_min, self.contrast_max = contrast_range
        self.p = p

    def __call__(self, data):
        image = data['color_aug']
        if torch.rand(1).item() < self.p:
            factor = torch.empty(1).uniform_(self.contrast_min, self.contrast_max).item()
            image = F_t.adjust_contrast(image, factor)
        data['color_aug'] = image
        return data

class RandomSaturation:
    def __init__(self, saturation_range=[0, 1], p=0.5):
        self.saturation_min, self.saturation_max = saturation_range
        self.p = p

    def __call__(self, data):
        image = data['color_aug']
        if torch.rand(1).item() < self.p:
            factor = torch.empty(1).uniform_(self.saturation_min, self.saturation_max).item()
            image = F_t.adjust_saturation(image, factor)
        data['color_aug'] = image
        return data


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
        return data