from __future__ import absolute_import, division, print_function

import os
import numpy as np

from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud
from PIL import Image
from pyquaternion import Quaternion

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset

from datasets.transforms import Compose

def transform_points(points, translation, rotation):
    """
    Apply rotation and translation to 3D points (vectorized implementation).

    Args:
        points: (N, 3) array of 3D points
        translation: (3,) array for translation
        rotation: (4,) quaternion for rotation

    Returns:
        (N, 3) array of transformed points
    """
    q = Quaternion(rotation)
    rot_matrix = q.rotation_matrix  # Get 3x3 rotation matrix
    return (points @ rot_matrix.T) + translation  # Apply rotation and translation

def inverse_transform_points(points, translation, rotation):
    """
    Apply inverse of rotation and translation to 3D points.

    Args:
        points: (N, 3) array of 3D points
        translation: (3,) array for translation
        rotation: (4,) quaternion for rotation

    Returns:
        (N, 3) array of inverse-transformed points
    """
    q = Quaternion(rotation).inverse  # Get inverse quaternion
    rot_matrix = q.rotation_matrix
    return (points - translation) @ rot_matrix.T  # Apply inverse transformation

def project_to_image(points, intrinsic):
    """
    Project 3D camera-space points to 2D image plane.

    Args:
        points: (N, 3) array of 3D points in camera frame
        intrinsic: (3, 3) camera intrinsic matrix

    Returns:
        (N, 2) array of 2D image coordinates
    """
    points = points.T  # Transpose to (3, N)
    proj = intrinsic @ points  # Matrix multiplication
    proj[:2] /= proj[2:3]  # Perspective division
    return proj[:2].T  # Return as (N, 2)

def filter_points_in_image(points_2d, points_3d, image_width, image_height):
    """
    Filter points to keep only those within image boundaries.

    Args:
        points_2d: (N, 2) array of 2D image points
        points_3d: (N, 3) corresponding 3D points
        image_width: width of the image
        image_height: height of the image

    Returns:
        Filtered (points_2d, points_3d) containing only visible points
    """
    # Check which points are within image boundaries
    valid_x = (points_2d[:, 0] >= 0) & (points_2d[:, 0] < image_width)
    valid_y = (points_2d[:, 1] >= 0) & (points_2d[:, 1] < image_height)
    valid = valid_x & valid_y

    return points_2d[valid], points_3d[valid]

def flatten_collate_fn(batch):
    batch_data = {}
    for sample in batch:
        for key, value in sample.items():
            (datatype, scale), frame_offset = key

            if datatype in ['K', 'inv_K']:
                if frame_offset != 0: # Κρατάμε μόνο αν offset == 0
                    continue
                new_key = (datatype, scale)
            else:
                new_key = (datatype, frame_offset, scale)

            if new_key not in batch_data:
                batch_data[new_key] = []
            batch_data[new_key].append(value)

    # Stack tensors σε batch dimension
    for key in batch_data:
        batch_data[key] = torch.stack(batch_data[key], dim=0)  # dim=0 = batch size

    dict(sorted(batch_data.items(), key=lambda item: item[0]))
    return batch_data #dict(sorted(batch_data.items(), key=lambda item: item[0]))


class NuScenesDataset(Dataset):
    def __init__(self, nusc, version, split, height, width, scales, transforms):
        """
        Args:
            nusc: NuScenes object
            version: dataset version
            split: 'train', 'validation' or 'test'
            transform: optional transforms
        """
        self.nusc = nusc
        self.version = version
        self.split = split

        # Resize ratio
        self.height = height
        self.width = width
        self.scales = scales

        self.transform = transforms if transforms is not None else Compose([])
        self.to_tensor = T.ToTensor()

        self.scenes = self._get_scenes_by_split() # Get scenes for the specified split
        self.samples = self._gather_samples() # Gather all samples from these scenes

        # αγνοώ δείγματα που βρίσκονται στην αρχή ή στο τέλος της σκηνής
        # καθώς δεν έχουν προηγούμενο (-1) και επόμενο (1) frame
        self.valid_indices = [i for i in range(len(self.samples))
                              if (0 <= i - 1 < len(self.samples)) and (0 <= i + 1 < len(self.samples))]

    def _get_scenes_by_split(self):
        """Get scene names based on the split (train/validation/test)."""
        splits = create_splits_scenes()

        if self.split == 'train':
            scene_names = splits['train']
        elif self.split == 'validation':
            scene_names = splits['val']
        else:  # test
            scene_names = [scene['name'] for scene in self.nusc.scene]

        # Map scene names to scene objects
        name_to_scene = {scene['name']: scene for scene in self.nusc.scene}
        return [name_to_scene[name] for name in scene_names if name in name_to_scene]

    def _gather_samples(self):
        """Collect all samples from the selected scenes."""
        samples = []
        for scene in self.scenes:
            token = scene['first_sample_token']
            while token:
                sample = self.nusc.get('sample', token)
                samples.append(sample)
                token = sample['next'] if sample['next'] else None
        return samples

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        """Get data for previous (-1), current (0), and next (+1) samples."""
        # def safe_get(i):
        #     if 0 <= i < len(self.samples):
        #         return self._get_sample_data(self.samples[i])
        #     else:
        #         return None  # Δεν υπάρχει προηγούμενο/επόμενο
        #
        # # Get prev/current/next
        # sample_data = {
        #     offset: safe_get(idx + offset)
        #     for offset in [-1, 0, 1]
        # }

        if idx >= len(self.valid_indices):
            raise IndexError(f"Index {idx} out of range for valid_indices with length {len(self.valid_indices)}")

        real_idx = self.valid_indices[idx]
        sample_data = {
            offset: self._get_sample_data(self.samples[real_idx + offset])
            for offset in [-1, 0, 1]
        }

        # Combine them in a flat dict
        output = {}
        for offset, data in sample_data.items():
            if data is not None:
                for k, v in data.items():
                    output[(k, offset)] = v
        return output

    def _resize_data(self, data_tensor, scale):
        """Resize tensor data (C,H,W) ή (H,W) ανάλογα με το scale."""
        if data_tensor.dim() == 3:
            # image tensor (C,H,W)
            c, h, w = data_tensor.shape
            new_h, new_w = int(h * scale), int(w * scale)
            resized = F.interpolate(data_tensor.unsqueeze(0), size=(new_h, new_w), mode='bilinear', align_corners=False)
            return resized.squeeze(0)
        elif data_tensor.dim() == 2:
            # depth or radar (H,W)
            h, w = data_tensor.shape
            new_h, new_w = int(h * scale), int(w * scale)
            resized = F.interpolate(data_tensor.unsqueeze(0).unsqueeze(0), size=(new_h, new_w), mode='nearest')
            return resized.squeeze(0).squeeze(0)
        else:
            return data_tensor  # no change for other dims

    def _get_sample_data(self, sample):
        # === CAMERA ===
        cam_sd = self.nusc.get('sample_data', sample['data']['CAM_FRONT'])
        cam_path = os.path.join(self.nusc.dataroot, cam_sd['filename'])
        image = Image.open(cam_path).convert('RGB')
        image_width, image_height = image.size

        cam_cs = self.nusc.get('calibrated_sensor', cam_sd['calibrated_sensor_token'])
        cam_intrinsic = np.array(cam_cs['camera_intrinsic'])

        # === LIDAR ===
        all_lidar_points = []
        lidar_data = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        lidar_points = LidarPointCloud.from_file(os.path.join(self.nusc.dataroot, lidar_data['filename']))
        lidar_pose_rec = self.nusc.get('ego_pose', lidar_data['ego_pose_token'])
        lidar_pose = Quaternion(lidar_pose_rec['rotation']).rotation_matrix
        lidar_points.rotate(lidar_pose)
        lidar_points.translate(np.array(lidar_pose_rec['translation']))

        cam_pose_rec = self.nusc.get('ego_pose', cam_sd['ego_pose_token'])
        cam_pose = Quaternion(cam_pose_rec['rotation']).rotation_matrix
        cam_translation = np.array(cam_pose_rec['translation'])
        inv_cam_pose = np.linalg.inv(cam_pose)
        lidar_points.translate(-cam_translation)
        lidar_points.rotate(inv_cam_pose)

        ego_pose_at_cam_time = self.nusc.get('ego_pose', cam_sd['ego_pose_token'])
        ego_pose_at_lidar_time = self.nusc.get('ego_pose', lidar_data['ego_pose_token'])
        lidar_to_cam_translation = np.array(ego_pose_at_cam_time['translation']) - np.array(ego_pose_at_lidar_time['translation'])
        rotation_diff = Quaternion(ego_pose_at_cam_time['rotation']) * Quaternion(ego_pose_at_lidar_time['rotation']).inverse
        lidar_to_cam_rotation = rotation_diff.rotation_matrix

        lidar_points.rotate(lidar_to_cam_rotation.T)
        lidar_points.translate(-lidar_to_cam_translation)

        points, coloring, _ = self.nusc.explorer.map_pointcloud_to_image(lidar_data['token'], cam_sd['token'])
        valid_indices = (points[0, :] >= 0) & (points[0, :] < image.width) & (points[1, :] >= 0) & (
                points[1, :] < image.height)
        valid_points = points[:, valid_indices]
        valid_distances = coloring[valid_indices]

        all_lidar_points.extend(zip(valid_points[0], valid_points[1], valid_distances))
        all_lidar_points = np.array(all_lidar_points)

        lidar_depth = np.zeros(np.array(image).shape[:2], dtype=np.float32)
        for idx, (x, y) in enumerate(zip(all_lidar_points[:, 0], all_lidar_points[:, 1])):
            lidar_depth[int(y), int(x)] = max(lidar_depth[int(y), int(x)], all_lidar_points[idx, 2])

        # === RADAR ===
        all_radar_points = []
        radar_data = self.nusc.get('sample_data', sample['data']['RADAR_FRONT'])
        radar_points = RadarPointCloud.from_file(os.path.join(self.nusc.dataroot, radar_data['filename']))
        radar_pose_rec = self.nusc.get('ego_pose', radar_data['ego_pose_token'])
        radar_pose = Quaternion(radar_pose_rec['rotation']).rotation_matrix
        radar_points.rotate(radar_pose)
        radar_points.translate(np.array(radar_pose_rec['translation']))

        cam_pose_rec = self.nusc.get('ego_pose', cam_sd['ego_pose_token'])
        cam_pose = Quaternion(cam_pose_rec['rotation']).rotation_matrix
        cam_translation = np.array(cam_pose_rec['translation'])
        inv_cam_pose = np.linalg.inv(cam_pose)
        radar_points.translate(-cam_translation)
        radar_points.rotate(inv_cam_pose)

        ego_pose_at_cam_time = self.nusc.get('ego_pose', cam_sd['ego_pose_token'])
        ego_pose_at_radar_time = self.nusc.get('ego_pose', radar_data['ego_pose_token'])
        radar_to_cam_translation = np.array(ego_pose_at_cam_time['translation']) - np.array(ego_pose_at_radar_time['translation'])
        rotation_diff = Quaternion(ego_pose_at_cam_time['rotation']) * Quaternion(ego_pose_at_radar_time['rotation']).inverse
        radar_to_cam_rotation = rotation_diff.rotation_matrix

        radar_points.rotate(radar_to_cam_rotation.T)
        radar_points.translate(-radar_to_cam_translation)

        points, coloring, _ = self.nusc.explorer.map_pointcloud_to_image(radar_data['token'], cam_sd['token'])
        valid_indices = (points[0, :] >= 0) & (points[0, :] < image.width) & (points[1, :] >= 0) & (
                points[1, :] < image.height)
        valid_points = points[:, valid_indices]
        valid_distances = coloring[valid_indices]

        all_radar_points.extend(zip(valid_points[0], valid_points[1], valid_distances))
        all_radar_points = np.array(all_radar_points)

        radar_depth = np.zeros(np.array(image).shape[:2], dtype=np.float32)
        for idx, (x, y) in enumerate(zip(all_radar_points[:, 0], all_radar_points[:, 1])):
            radar_depth[int(y), int(x)] = max(radar_depth[int(y), int(x)], all_radar_points[idx, 2])


        # Μετατροπή numpy arrays σε τένσορες
        image_t = self.to_tensor(image).float()
        cam_intrinsic_t = torch.from_numpy(cam_intrinsic.astype(np.float32))
        lidar_pts_2d_t = torch.from_numpy(lidar_depth.astype(np.float32))
        radar_pts_2d_t = torch.from_numpy(radar_depth.astype(np.float32))

        data = {
            'color': image_t.clone(),
            'color_aug': image_t.clone(),
            'K': cam_intrinsic_t.clone(),
            'inv_K': torch.linalg.inv(cam_intrinsic_t.clone()),
            'radar_points_2d': radar_pts_2d_t.clone(),
            'depth_gt': lidar_pts_2d_t.clone()
        }

        # Εφαρμογή μετασχηματισμών **εκτός resize** (π.χ. flip, color jitter)
        if self.transform is not None:
            data = self.transform(data)


        data_at_scales = {}
        # for scale in self.scales:
        #     scale_factor = 1 / (2 ** scale)
        #     if self.split == 'train':
        #         print(f"Scale {scale}, scale_factor: {scale_factor}")
        #
        #     if scale == 0:
        #         img_s = image_t
        #         lidar_s = lidar_pts_2d_t
        #         radar_s = radar_pts_2d_t
        #         K_s = cam_intrinsic_t.clone()
        #     else:
        #         img_s = self._resize_data(image_t, scale_factor)
        #         lidar_s = self._resize_data(lidar_pts_2d_t, scale_factor)
        #         radar_s = self._resize_data(radar_pts_2d_t, scale_factor)
        #         K_s = cam_intrinsic_t.clone()
        #
        #     # Normalize intrinsics first
        #     K_s[0, :] /= image_width
        #     K_s[1, :] /= image_height
        #     # Then scale to resized image size
        #     K_s[0, :] *= image_width * scale_factor
        #     K_s[1, :] *= image_height * scale_factor
        #
        #     inv_K_s = torch.linalg.inv(K_s)
        #
        #     # Δημιουργούμε το λεξικό δεδομένων για αυτό το scale
        #     scale_data = {
        #         'color': img_s,
        #         'color_aug': img_s.clone(),  # αρχικά ίδια με color
        #         'K': K_s,
        #         'inv_K': inv_K_s,
        #         'radar_points_2d': radar_s,
        #         'depth_gt': lidar_s
        #     }
        #
        #     # if self.transform is not None:
        #     #     scale_data = self.transform(scale_data)
        #
        #     data_at_scales[('color', scale)] = scale_data['color']
        #     if self.split == 'train':
        #         data_at_scales[('color_aug', scale)] = scale_data['color_aug']
        #     data_at_scales[('K', scale)] = scale_data['K']
        #     data_at_scales[('inv_K', scale)] = scale_data['inv_K']
        #     data_at_scales[('radar_points_2d', scale)] = scale_data['radar_points_2d']
        #     data_at_scales[('depth_gt', scale)] = scale_data['depth_gt']
        for scale in self.scales:
            scale_factor = 1 / (2 ** scale)
            new_h = int(image_height * scale_factor)
            new_w = int(image_width * scale_factor)

            # Προσαρμογή intrinsics
            K_s = data['K'].clone()
            K_s[0, :] /= image_width
            K_s[1, :] /= image_height
            K_s[0, :] *= new_w
            K_s[1, :] *= new_h
            inv_K_s = torch.linalg.inv(K_s)

            # Resize κάθε tensor με την _resize_data
            img_s = self._resize_data(data['color'], scale_factor)
            img_aug_s = self._resize_data(data['color_aug'], scale_factor)
            lidar_s = self._resize_data(data['depth_gt'], scale_factor)
            radar_s = self._resize_data(data['radar_points_2d'], scale_factor)

            scale_data = {
                'color': img_s,
                'color_aug': img_aug_s,
                'K': K_s,
                'inv_K': inv_K_s,
                'radar_points_2d': radar_s,
                'depth_gt': lidar_s
            }

            data_at_scales[('color', scale)] = scale_data['color']
            if self.split == 'train':
                data_at_scales[('color_aug', scale)] = scale_data['color_aug']
            else:
                data_at_scales[('color_aug', scale)] = scale_data['color']
            data_at_scales[('K', scale)] = scale_data['K']
            data_at_scales[('inv_K', scale)] = scale_data['inv_K']
            data_at_scales[('radar_points_2d', scale)] = scale_data['radar_points_2d']
            data_at_scales[('depth_gt', scale)] = scale_data['depth_gt']

        return dict(sorted(data_at_scales.items(), key=lambda item: item[0]))
