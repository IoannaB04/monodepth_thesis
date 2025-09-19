from __future__ import absolute_import, division, print_function

import os
import numpy as np

from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.geometry_utils import transform_matrix
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud
from PIL import Image
from pyquaternion import Quaternion

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset

from datasets.transforms_fusionnet import Compose


# def flatten_collate_fn(batch):
#     batch_data = {}
#     for sample in batch:
#         for key, value in sample.items():
#             (datatype, scale), frame_offset = key

#             if datatype in ['K', 'inv_K']:
#                 if frame_offset != 0: # Κρατάμε μόνο αν offset == 0
#                     continue
#                 new_key = (datatype, scale)
#             else:
#                 new_key = (datatype, frame_offset, scale)

#             if new_key not in batch_data:
#                 batch_data[new_key] = []
#             batch_data[new_key].append(value)

#     # Stack tensors σε batch dimension
#     for key in batch_data:
#         batch_data[key] = torch.stack(batch_data[key], dim=0)  # dim=0 = batch size

#     dict(sorted(batch_data.items(), key=lambda item: item[0]))
#     return batch_data #dict(sorted(batch_data.items(), key=lambda item: item[0]))

def flatten_collate_fn(batch):
    batch_data = {}
    for sample in batch:
        for key, value in sample.items():
            try:
                (datatype, scale), frame_offset = key
            except ValueError:
                # Αν δεν είναι σε μορφή ((datatype, scale), frame_offset), αγνοούμε ή κρατάμε αυτούσιο
                batch_data.setdefault(key, []).append(value)
                continue

            if datatype in ['K', 'inv_K']:
                if frame_offset != 0:
                    continue
                new_key = (datatype, scale)
            else:
                new_key = (datatype, frame_offset, scale)

            if new_key not in batch_data:
                batch_data[new_key] = []
            batch_data[new_key].append(value)

    for key in batch_data:
        batch_data[key] = torch.stack(batch_data[key], dim=0)

    return batch_data

class NuScenesDataset(Dataset):
    def __init__(self, nusc, version, split,
                 height, width, scales,
                 radar_jbf_path,
                 transforms,
                 temp_context=[0, -1, 1]):
        """
        Initialize dataset.

        Args:
            nusc: NuScenes object
            version: dataset version
            split: 'train', 'validation' or 'test'
            radar_jbf_path: path to jbf folder
            transform: optional image transforms
        """
        self.nusc = nusc
        self.version = version
        self.split = split

        # Resize ratio
        self.height = height
        self.width = width
        self.scales = scales

        self.radar_jbf_path = radar_jbf_path

        self.transform = transforms if transforms is not None else Compose([])
        self.to_tensor = T.ToTensor()

        self.scenes = self._get_scenes_by_split()  # Get scenes for the specified split
        self.samples = self._gather_samples()  # Gather all samples from these scenes

        # αγνοώ δείγματα που βρίσκονται στην αρχή ή στο τέλος της σκηνής
        # καθώς δεν έχουν προηγούμενο (-1) και επόμενο (1) frame
        self.valid_indices = [i for i in range(len(self.samples))
                              if (0 <= i - 1 < len(self.samples)) and (0 <= i + 1 < len(self.samples))]
        
        self.temp_context = temp_context




    def _get_scenes_by_split(self):
        """Get scene names based on the split (train/validation/test)."""
        splits = create_splits_scenes()

        if self.split == 'train':
            self.scene_names = splits['train']
        elif self.split == 'validation':
            self.scene_names = splits['val']
        elif self.split == 'test':
            self.scene_names = [scene['name'] for scene in self.nusc.scene]

        # Map scene names to scene objects
        name_to_scene = {scene['name']: scene for scene in self.nusc.scene}
        return [name_to_scene[name] for name in self.scene_names if name in name_to_scene]

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

        # --- Προσθήκη pose_gt ---
        # Το reference frame είναι το 0 
        for f_id in self.temp_context[1:]: # [0, -1, 1]
            cam_data_shifted = self.get_cam_sample_data(real_idx, "CAM_FRONT", f_id)
            if cam_data_shifted is None:
                continue  # ή βάλε pose = np.eye(4)
            gt_pose = self.get_pose(real_idx, camera_sensor="CAM_FRONT", temp_shift=f_id)
            output[("pose_gt", f_id)] = torch.from_numpy(gt_pose).float()
        
        # ''' debugging '''
        # for f_id in self.temp_context:
        #     if f_id != 0:
        #         key = ("pose_gt", f_id)
        #         if key in output:
        #             print(f"pose_gt for frame {f_id} exists, shape: {output[key].shape}")
        #         else:
        #             print(f"pose_gt for frame {f_id} MISSING!")

        return output

    def get_scene_name_from_sample(self, sample_token):
        """
        Δίνει το όνομα της σκηνής (scene name) που περιέχει το συγκεκριμένο sample.

        Args:
            sample_token (str): Το token του sample.

        Returns:
            str: Το όνομα της σκηνής.
        """
        for scene in self.nusc.scene:
            first = self.nusc.get('sample', scene['first_sample_token'])['token']
            last = self.nusc.get('sample', scene['last_sample_token'])['token']
            current_token = first
            while True:
                if current_token == sample_token:
                    return scene['name']
                if current_token == last:
                    break
                current_sample = self.nusc.get('sample', current_token)
                current_token = current_sample['next']
        raise ValueError(f"Sample token {sample_token} not found in any scene")

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
        camera_data = self.nusc.get('sample_data', sample['data']['CAM_FRONT'])
        cam_path = os.path.join(self.nusc.dataroot, camera_data['filename'])
        image = Image.open(cam_path)
        image_width, image_height = image.size

        cam_cs = self.nusc.get('calibrated_sensor', camera_data['calibrated_sensor_token'])
        cam_intrinsic = np.array(cam_cs['camera_intrinsic'])

        # === LIDAR ===
        all_lidar_points = []
        lidar_data = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        lidar_points = LidarPointCloud.from_file(os.path.join(self.nusc.dataroot, lidar_data['filename']))
        lidar_pose_rec = self.nusc.get('ego_pose', lidar_data['ego_pose_token'])
        lidar_pose = Quaternion(lidar_pose_rec['rotation']).rotation_matrix
        lidar_points.rotate(lidar_pose)
        lidar_points.translate(np.array(lidar_pose_rec['translation']))

        cam_pose_rec = self.nusc.get('ego_pose', camera_data['ego_pose_token'])
        cam_pose = Quaternion(cam_pose_rec['rotation']).rotation_matrix
        cam_translation = np.array(cam_pose_rec['translation'])
        inv_cam_pose = np.linalg.inv(cam_pose)
        lidar_points.translate(-cam_translation)
        lidar_points.rotate(inv_cam_pose)

        ego_pose_at_cam_time = self.nusc.get('ego_pose', camera_data['ego_pose_token'])
        ego_pose_at_lidar_time = self.nusc.get('ego_pose', lidar_data['ego_pose_token'])
        lidar_to_cam_translation = np.array(ego_pose_at_cam_time['translation']) - np.array(
            ego_pose_at_lidar_time['translation'])
        rotation_diff = Quaternion(ego_pose_at_cam_time['rotation']) * Quaternion(
            ego_pose_at_lidar_time['rotation']).inverse
        lidar_to_cam_rotation = rotation_diff.rotation_matrix

        lidar_points.rotate(lidar_to_cam_rotation.T)
        lidar_points.translate(-lidar_to_cam_translation)

        points, coloring, _ = self.nusc.explorer.map_pointcloud_to_image(lidar_data['token'], camera_data['token'])
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
        radar_token = sample['data']['RADAR_FRONT']
        radar_data = self.nusc.get('sample_data', radar_token)
        radar_points = RadarPointCloud.from_file(os.path.join(self.nusc.dataroot, radar_data['filename']))
        radar_pose_rec = self.nusc.get('ego_pose', radar_data['ego_pose_token'])
        radar_pose = Quaternion(radar_pose_rec['rotation']).rotation_matrix
        radar_points.rotate(radar_pose)
        radar_points.translate(np.array(radar_pose_rec['translation']))

        cam_pose_rec = self.nusc.get('ego_pose', camera_data['ego_pose_token'])
        cam_pose = Quaternion(cam_pose_rec['rotation']).rotation_matrix
        cam_translation = np.array(cam_pose_rec['translation'])
        inv_cam_pose = np.linalg.inv(cam_pose)
        radar_points.translate(-cam_translation)
        radar_points.rotate(inv_cam_pose)

        ego_pose_at_cam_time = self.nusc.get('ego_pose', camera_data['ego_pose_token'])
        ego_pose_at_radar_time = self.nusc.get('ego_pose', radar_data['ego_pose_token'])
        radar_to_cam_translation = np.array(ego_pose_at_cam_time['translation']) - np.array(
            ego_pose_at_radar_time['translation'])
        rotation_diff = Quaternion(ego_pose_at_cam_time['rotation']) * Quaternion(
            ego_pose_at_radar_time['rotation']).inverse
        radar_to_cam_rotation = rotation_diff.rotation_matrix

        radar_points.rotate(radar_to_cam_rotation.T)
        radar_points.translate(-radar_to_cam_translation)

        points, coloring, _ = self.nusc.explorer.map_pointcloud_to_image(radar_data['token'], camera_data['token'])
        valid_indices = (points[0, :] >= 0) & (points[0, :] < image.width) & (points[1, :] >= 0) & (
                points[1, :] < image.height)
        valid_points = points[:, valid_indices]
        valid_distances = coloring[valid_indices]

        all_radar_points.extend(zip(valid_points[0], valid_points[1], valid_distances))
        all_radar_points = np.array(all_radar_points)

        radar_depth = np.zeros(np.array(image).shape[:2], dtype=np.float32)
        for idx, (x, y) in enumerate(zip(all_radar_points[:, 0], all_radar_points[:, 1])):
            radar_depth[int(y), int(x)] = max(radar_depth[int(y), int(x)], all_radar_points[idx, 2])

        # === RADAR JBF ===
        name = os.path.basename(cam_path)[:-4]
        scene_name = self.get_scene_name_from_sample(sample['token'])
        radar_jbf_path = os.path.join(self.radar_jbf_path, self.split, 'depth_map', scene_name, name+'.png')
        if not os.path.exists(radar_jbf_path):
            raise FileNotFoundError(f"Missing JBF radar file: {radar_jbf_path}")
        radar_jbf = np.array(Image.open(radar_jbf_path))

        confidence_jbf_path = os.path.join(self.radar_jbf_path, self.split, 'confidence_map', scene_name, name+'_confidence.npy')
        if not os.path.exists(confidence_jbf_path):
            raise FileNotFoundError(f"Missing JBF confidence file: {confidence_jbf_path}")
        confidence_jbf = np.load(confidence_jbf_path)

        assert radar_jbf.shape == confidence_jbf.shape, "Shapes must match!"
        radar_jbf = np.stack([radar_jbf, confidence_jbf], axis=0)

        # Μετατροπή numpy arrays σε τένσορες
        image_t = self.to_tensor(image).float()
        cam_intrinsic_t = torch.from_numpy(cam_intrinsic.astype(np.float32))
        lidar_pts_2d_t = torch.from_numpy(lidar_depth.astype(np.float32))
        radar_pts_2d_t = torch.from_numpy(radar_depth.astype(np.float32))
        radar_jbf_t = torch.from_numpy(radar_jbf.astype(np.float32))

        data = {
            'color': image_t.clone(),
            'color_aug': image_t.clone(),
            'K': cam_intrinsic_t.clone(),
            'inv_K': torch.linalg.inv(cam_intrinsic_t.clone()),
            'radar_points_2d': radar_pts_2d_t.clone(),
            'depth_gt': lidar_pts_2d_t.clone(),
            'radar_jbf': radar_jbf_t.clone()
        }

        # Εφαρμογή μετασχηματισμών **εκτός resize** (π.χ. flip, color jitter)
        if self.transform is not None:
            data = self.transform(data)

        data_at_scales = {}
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
            radar_jbf_s = self._resize_data(data['radar_jbf'], scale_factor)

            scale_data = {
                'color': img_s,
                'color_aug': img_aug_s,
                'K': K_s,
                'inv_K': inv_K_s,
                'radar_points_2d': radar_s,
                'depth_gt': lidar_s,
                'radar_jbf': radar_jbf_s
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
            data_at_scales[('radar_jbf', scale)] = scale_data['radar_jbf']

        return dict(sorted(data_at_scales.items(), key=lambda item: item[0]))
    
    def get_pose(self, frame_index, camera_sensor, temp_shift):
        """Υπολογίζει τη μετασχηματιστική μήτρα 4x4 από το reference frame στο shifted frame."""
        cam_data_origin = self.get_cam_sample_data(frame_index, camera_sensor, 0)
        cam_data_shifted = self.get_cam_sample_data(frame_index, camera_sensor, temp_shift)

        ego_pose_record = self.nusc.get('ego_pose', cam_data_shifted['ego_pose_token'])
        ego_pose_origin_record = self.nusc.get('ego_pose', cam_data_origin['ego_pose_token'])

        ego_to_global_transform = transform_matrix(
            translation=np.array(ego_pose_record['translation']),
            rotation=Quaternion(ego_pose_record['rotation'])
        )
        ego_origin_to_global_transform = transform_matrix(
            translation=np.array(ego_pose_origin_record['translation']),
            rotation=Quaternion(ego_pose_origin_record['rotation'])
        )

        calibrated_sensor_record = self.nusc.get('calibrated_sensor', cam_data_shifted['calibrated_sensor_token'])
        calibrated_sensor_origin_record = self.nusc.get('calibrated_sensor', cam_data_origin['calibrated_sensor_token'])

        ref_to_ego_transform = transform_matrix(
            translation=np.array(calibrated_sensor_record['translation']),
            rotation=Quaternion(calibrated_sensor_record["rotation"])
        )
        ref_to_ego_origin_transform = transform_matrix(
            translation=np.array(calibrated_sensor_origin_record['translation']),
            rotation=Quaternion(calibrated_sensor_origin_record['rotation'])
        )

        # Τελική μετασχηματιστική μήτρα
        pose = np.linalg.inv(ref_to_ego_transform) @ np.linalg.inv(ego_to_global_transform) \
               @ ego_origin_to_global_transform @ ref_to_ego_origin_transform
        return pose

    def get_cam_sample_data(self, frame_index, camera_sensor, temp_shift):
        keyframe = self.nusc.get('sample_data', self.nusc.sample[frame_index]['data'][camera_sensor])
        temp_dir = 0

        if camera_sensor in ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT"]:
            if temp_shift < 0:
                temp_dir = 'prev'
            elif temp_shift > 0:
                temp_dir = 'next'
        elif camera_sensor in ["CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]:
            if temp_shift < 0:
                temp_dir = 'next'
            elif temp_shift > 0:
                temp_dir = 'prev'

        i = 0
        while i < abs(temp_shift):
            temp_token = keyframe[temp_dir]
            if temp_token == '':
                return None
            keyframe = self.nusc.get('sample_data', temp_token)
            i += 1

        return keyframe
    
