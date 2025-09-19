import os
import cv2
import numpy as np
from PIL import Image

from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.data_classes import RadarPointCloud

import time


from utils import create_output_directories, plot_image
from jbf import jbf_method

'''
You have to download the nuscenes dataset in a folder named dataset in the main directory.
The path of the path should be at dataset_root in the bash file.

This code expands the radar data using the JBF method for both raw and filtered data.
'''

from config.config import get_cfg, get_parser
args = get_parser().parse_args()
cfg = get_cfg(args)

def run_combine_jbf(dataset_root=cfg.DATASET.DATAROOT,
                    dataset_version='v1.0-mini',
                    number_of_total_sweeps=5,
                    sigma_s = 25,
                    sigma_r = 10
                    ):
    nusc = NuScenes(version=dataset_version, dataroot=dataset_root, verbose=True)

    # Splitting the scenes to train and validation with the predefined splits of nuScenes
    splits = create_splits_scenes()
    train_scenes = splits['train']
    val_scenes = splits['val']
    print(f'Train samples: {len(train_scenes)}')
    print(f'Validation samples: {len(val_scenes)}')

    # Grouping of samples by scene
    samples_by_scene = {}
    for my_sample in nusc.sample:
        scene_token = my_sample['scene_token']
        if scene_token not in samples_by_scene:
            samples_by_scene[scene_token] = []
        samples_by_scene[scene_token].append(my_sample)



    # Define output directories to save the image paths for FusionNET
    current_dir = os.path.dirname(os.path.abspath(__file__))
    ''' the base_output_dir is the directory where the raw jbf output is saved '''
    raw_output_dir = os.path.join(current_dir, 'jbf_output_raw')
    raw_train_dir = os.path.join(raw_output_dir, 'train')
    raw_validation_dir = os.path.join(raw_output_dir, 'validation')

    ''' the base_output_dir is the directory where the filtered jbf output is saved '''
    filtered_output_dir = os.path.join(current_dir, 'jbf_output_filtered')
    filtered_train_dir = os.path.join(filtered_output_dir, 'train')
    filtered_validation_dir = os.path.join(filtered_output_dir, 'validation')

    print()

    # LUTs
    LUT_s = np.exp(-0.5 * (np.arange(500) ** 2) / sigma_s ** 2)
    LUT_r = np.exp(-0.5 * (np.arange(442) ** 2) / sigma_r ** 2)
    MAX_SHIFT = next((i for i, j in enumerate(LUT_s < 0.1) if j), None)


    # Edit samples per scene
    for scene_token, scene_samples in samples_by_scene.items():
        scene = nusc.get('scene', scene_token)
        scene_name = scene['name']

        # # If the process stops while processing a scene
        # scene_number = int(scene_name.split('-')[1])  # scene names are in the format "scene-XXXX" for nuscene dataset
        # if scene_number < 1109 or scene_name in val_scenes:
        #     continue  # Skip scenes before 1080

        total_samples = len(scene_samples)
        current_sample_counter = 0

        start_time = time.time()  # Start time for the scene processing

        for my_sample in scene_samples:
            camera_data = nusc.get('sample_data', my_sample['data']['CAM_FRONT'])
            camera_filepath = os.path.join(dataset_root, camera_data['filename'])
            camera_image = Image.open(camera_filepath)
            camera_image_np = np.array(camera_image)

            current_sample_counter += 1

            # Keeping the paths of the rbg images for FusionNET
            if scene_name in train_scenes:
                base_dir_raw = raw_train_dir
                base_dir_filtered = filtered_train_dir

            elif scene_name in val_scenes:
                base_dir_raw = raw_validation_dir
                base_dir_filtered = filtered_validation_dir

            else:
                print(f"\nScene {scene_name} not found in train or validation lists.")
                continue

            print(f"\rScene {scene_name} : {current_sample_counter}/{total_samples} \t{base_dir_raw}", end='', flush=True)

            raw_directories = create_output_directories(base_dir_raw, scene_name)
            filtered_directories = create_output_directories(base_dir_filtered, scene_name, labels=True)

            radar_token = my_sample['data']['RADAR_FRONT']
            sweeps = []

            while radar_token and len(sweeps) < number_of_total_sweeps:
                radar_data = nusc.get('sample_data', radar_token)
                sweeps.append(radar_data)
                radar_token = radar_data['prev']

            all_radar_points = []
            for radar_data in reversed(sweeps):
                radar_points = RadarPointCloud.from_file(os.path.join(dataset_root, radar_data['filename']))
                radar_pose_rec = nusc.get('ego_pose', radar_data['ego_pose_token'])
                radar_pose = Quaternion(radar_pose_rec['rotation']).rotation_matrix
                radar_points.rotate(radar_pose)
                radar_points.translate(np.array(radar_pose_rec['translation']))

                cam_pose_rec = nusc.get('ego_pose', camera_data['ego_pose_token'])
                cam_pose = Quaternion(cam_pose_rec['rotation']).rotation_matrix
                cam_translation = np.array(cam_pose_rec['translation'])
                inv_cam_pose = np.linalg.inv(cam_pose)
                radar_points.translate(-cam_translation)
                radar_points.rotate(inv_cam_pose)

                ego_pose_at_cam_time = nusc.get('ego_pose', camera_data['ego_pose_token'])
                ego_pose_at_radar_time = nusc.get('ego_pose', radar_data['ego_pose_token'])
                radar_to_cam_translation = np.array(ego_pose_at_cam_time['translation']) - np.array(
                    ego_pose_at_radar_time['translation'])
                rotation_diff = Quaternion(ego_pose_at_cam_time['rotation']) * Quaternion(
                    ego_pose_at_radar_time['rotation']).inverse
                radar_to_cam_rotation = rotation_diff.rotation_matrix

                radar_points.rotate(radar_to_cam_rotation.T)
                radar_points.translate(-radar_to_cam_translation)

                points, coloring, _ = nusc.explorer.map_pointcloud_to_image(radar_data['token'], camera_data['token'])
                valid_indices = (points[0, :] >= 0) & (points[0, :] < camera_image.width) & (points[1, :] >= 0) & (
                            points[1, :] < camera_image.height)
                valid_points = points[:, valid_indices]
                valid_distances = coloring[valid_indices]

                all_radar_points.extend(zip(valid_points[0], valid_points[1], valid_distances))

            all_radar_points = np.array(all_radar_points)

            name = os.path.basename(camera_filepath)[:-4]


            ''' ---- RAW POINTS ---- '''
            radar_depth = np.zeros(camera_image_np.shape[:2], dtype=np.float32)
            for idx, (x, y) in enumerate(zip(all_radar_points[:, 0], all_radar_points[:, 1])):
                radar_depth[int(y), int(x)] = max(radar_depth[int(y), int(x)], all_radar_points[idx, 2])

            ''' PLOT RAW RADAR POINTS ON TOP OF RGB IMAGE '''
            plot_image('radar_on_image', all_radar_points, name, raw_directories['radar_on_image'], image=camera_image_np)

            ''' APPLY JOINT BILATERAL FILTERS WITH RAW DATA '''
            # Get camera intrinsic values
            cam_sensor = nusc.get('sample_data', my_sample['data']['CAM_FRONT'])
            calibrated_sensor = nusc.get('calibrated_sensor', cam_sensor['calibrated_sensor_token'])
            camera_intrinsic = np.array(calibrated_sensor['camera_intrinsic'])
            f_u = camera_intrinsic[0, 0]
            f_v = camera_intrinsic[1, 1]

            # Apply the JBF method
            expanded_depth_map, confidence_map = jbf_method(radar_depth, camera_image_np,
                                                            f_u, f_v,
                                                            LUT_s, LUT_r, MAX_SHIFT)

            # Save confidence map as numpy array for FusionNET
            confidence_map_path = os.path.join(raw_directories['confidence_map'], name + "_confidence.npy")
            np.save(confidence_map_path, confidence_map)

            # Save confidence map as heatmap
            plot_image('confidence_map', confidence_map, name, raw_directories['confidence_map_heatmap'])
            # plot_image('confidence_map', confidence_map, name, directories['confidence_map_heatmap_colorbar'], colorbar=True)

            # Save expanded depth map
            plot_image('expanded_depth', expanded_depth_map, name, raw_directories['depth_map'])

            # Save image with depth map overlay
            plot_image('visualization', expanded_depth_map, name, raw_directories['visualization'], image=camera_image_np)

            ''' ---- FILTERED POINTS ---- '''
            # Filtering radar points using the 2D bounding boxes and tolerance d < dm + β
            # Get the boxes with filtered annotations
            _, boxes, _ = nusc.get_sample_data(my_sample['data']['CAM_FRONT'])
            anns = [nusc.get('sample_annotation', ann_token) for ann_token in my_sample['anns']]
            filtered_anns = [ann for ann in anns if ann['visibility_token'] >= '4']
            ''' 1: 0-40%   (invisible)
                2: 40-60%  (partially visible)
                3: 60-80%  (mostly visible)
                4: 80-100% (fully visible)'''
            filtered_boxes = [box for box in boxes if box.token in [ann['token'] for ann in filtered_anns]]
            del boxes, filtered_anns


            # Get camera intrinsic values
            cam_sensor = nusc.get('sample_data', my_sample['data']['CAM_FRONT'])
            calibrated_sensor = nusc.get('calibrated_sensor', cam_sensor['calibrated_sensor_token'])
            camera_intrinsic = np.array(calibrated_sensor['camera_intrinsic'])

            # Keeping only the bottom half of each box
            bounding_boxes_binary_masks_bottom_half(camera_image_np.shape, filtered_boxes, camera_intrinsic)

            filtered_radar_points = np.zeros(camera_image_np.shape[:2], dtype=np.float32)

            epsilon = 1e-6  # to avoid division by zero
            for box in filtered_boxes:
                # keeping radar points at the bottom half of the bounding box
                radar_in_box = radar_depth * box.bb_array_mask

                # calculating the weighted harmonic mean
                non_zero_elements = radar_in_box[radar_in_box != 0]
                if len(non_zero_elements) > 0:
                    min_value = non_zero_elements.min()
                    weights = epsilon + np.abs(non_zero_elements - min_value)
                    weighted_harmonic_mean = len(non_zero_elements) / np.sum(weights / non_zero_elements)

                    # applying depth threshold to address the noise in radar points
                    depth_threshold = min_value + weighted_harmonic_mean
                    mask_above_threshold = (radar_in_box > 0) & (radar_in_box <= depth_threshold)
                    filtered_radar_points[mask_above_threshold] = radar_in_box[mask_above_threshold]

            ''' PLOT FILTERED RADAR POINTS ON TOP OF RGB IMAGE '''
            plot_image('radar_on_image_filtered', filtered_radar_points, name, filtered_directories['radar_on_image'],
                       image=camera_image_np)

            ''' APPLY JOINT BILATERAL FILTERS '''
            f_u = camera_intrinsic[0, 0]
            f_v = camera_intrinsic[1, 1]

            expanded_depth_map, confidence_map, labels_number = jbf_method(filtered_radar_points, camera_image_np,
                                                                           f_u, f_v,
                                                                           LUT_s, LUT_r, MAX_SHIFT,
                                                                           creat_labels=True)

            # Save the label radar number that each expanded point corresponds to
            label_number_map_path = os.path.join(filtered_directories['depth_map_label_number'], name + ".npy")
            np.save(label_number_map_path, labels_number)

            # Save confidence map as numpy array for FusionNET
            confidence_map_path = os.path.join(filtered_directories['confidence_map'], name + "_confidence.npy")
            np.save(confidence_map_path, confidence_map)

            # Save confidence map as heatmap
            plot_image('confidence_map', confidence_map, name, filtered_directories['confidence_map_heatmap'])
            # plot_image('confidence_map', confidence_map, name, directories['confidence_map_heatmap_colorbar'], colorbar=True)

            # Save expanded depth map
            plot_image('expanded_depth', expanded_depth_map, name, filtered_directories['depth_map'])

            # Save image with expanded depth map on top
            plot_image('visualization', expanded_depth_map, name, filtered_directories['visualization'], image=camera_image_np)

        end_time = time.time()  # End time for the scene processing
        elapsed_time = end_time - start_time
        print(f"\nProcessing time for scene {scene_name}: {elapsed_time:.2f} seconds\n")




def bounding_boxes_binary_masks_bottom_half(image_shape, boxes, camera_intrinsic):
    '''
    Makes binary mask for each box in the image keeping only the bottom half
    '''

    img_height, img_width = image_shape[:2]
    bb_array = np.zeros((img_height, img_width, len(boxes)), dtype=np.uint8)

    for i, box in enumerate(boxes):
        corners_3d = box.corners()
        corners_2d = view_points(corners_3d, camera_intrinsic, normalize=True)[:2, :].T
        corners_2d = np.int32(corners_2d)

        y_up   = np.min(corners_2d[7,1])
        y_down = np.max(corners_2d[4,1])
        height = y_up - y_down

        rows_to_divide = [0, 1, 5, 4] # Top face
        corners_2d[rows_to_divide, 1] = corners_2d[rows_to_divide, 1] + height/2

        mask = np.zeros((img_height, img_width), dtype=np.uint8)
        faces = [
            corners_2d[:4],  # Front face
            corners_2d[4:],  # Back face
            corners_2d[[0, 1, 5, 4]],  # Top face
            corners_2d[[2, 3, 7, 6]],  # Down face
            corners_2d[[0, 3, 7, 4]],  # Left face
            corners_2d[[1, 2, 6, 5]]   # Right face
        ]

        # Fill the faces
        for face in faces:
            cv2.fillConvexPoly(mask, face, True)

        # Fill the interior of the bounding box
        bb_array[:, :, i] = mask

        # Store the bb_array in the corresponding bounding box
        box.bb_array_mask = bb_array[:, :, i]

    final_mask = np.sum(bb_array, axis=2)
    # plt.figure(figsize=(10, 10))
    # plt.imshow(final_mask, cmap='gray')
    # plt.title('Bottom half of bounding boxes')
    # plt.axis('off')
    # plt.show()


if __name__ == '__main__':
    run_combine_jbf(
        dataset_root=cfg.DATASET.DATAROOT,
        dataset_version='v1.0-' + cfg.DATASET.VERSION,
        number_of_total_sweeps=cfg.JBF.NUMBER_OF_TOTAL_SWEEPS,
        sigma_s=cfg.JBF.SIGMA_S,
        sigma_r=cfg.JBF.SIGMA_R
    )
