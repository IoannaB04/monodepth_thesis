from __future__ import absolute_import, division, print_function

import os
import cv2

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
import networks
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from layers import disp_to_depth
from options_ioanna import MonodepthOptions

from datasets.transforms import *

from nuscenes.nuscenes import NuScenes
from datasets.nuscenes_dataset import NuScenesDataset
from datasets.nuscenes_dataset import flatten_collate_fn

STEREO_SCALE_FACTOR = 5.4


def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths."""
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3

def batch_post_process_disparity(l_disp, r_disp):
    """Apply the disparity post-processing method as introduced in Monodepthv1."""
    _, h, w = l_disp.shape
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = (1.0 - np.clip(20 * (l - 0.05), 0, 1))[None, ...]
    r_mask = l_mask[:, :, ::-1]
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp


# def evaluate(opt):
#     """Evaluates a pretrained model using a specified test set
#     """
#     MIN_DEPTH = opt.min_depth
#     MAX_DEPTH = opt.max_depth
#
#     if opt.ext_disp_to_eval is None:
#         opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)
#         assert os.path.isdir(opt.load_weights_folder), "Cannot find a folder at {}".format(opt.load_weights_folder)
#         print("-> Loading weights from {}".format(opt.load_weights_folder))
#
#         encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
#         decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")
#
#         encoder_dict = torch.load(encoder_path)
#
#         encoder = networks.ResnetEncoder(opt.num_layers, False)
#         depth_decoder = networks.DepthDecoder(encoder.num_ch_enc)
#
#         model_dict = encoder.state_dict()
#         encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
#         depth_decoder.load_state_dict(torch.load(decoder_path))
#
#         # === Load dataset ===
#         transforms_test = Compose([
#             Resize(opt.height, opt.width)
#         ])
#         version = "v1.0-" + opt.dataset_version
#         if version == "v1.0-test":
#             nusc = NuScenes(version=version, dataroot=opt.data_path, verbose=True)
#             test_dataset = NuScenesDataset(nusc, version, split='test',
#                                             height=opt.height,
#                                             width=opt.width,
#                                             scales=opt.scales,
#                                             transforms=transforms_test)
#             test_loader = DataLoader(test_dataset,
#                                       batch_size=opt.batch_size,
#                                       shuffle=True,
#                                       num_workers=opt.num_workers,
#                                       collate_fn=flatten_collate_fn)
#             print(f"Test samples: {len(test_loader)}")
#
#         else:
#             raise ValueError("Unsupported nuScenes dataset version: {}".format(version))
#
#         # === Load model ===
#         encoder = networks.ResnetEncoder(opt.num_layers, False)
#         depth_decoder = networks.DepthDecoder(encoder.num_ch_enc)
#
#         model_dict = encoder.state_dict()
#         encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
#         depth_decoder.load_state_dict(torch.load(decoder_path))
#
#         encoder.cuda()
#         encoder.eval()
#         depth_decoder.cuda()
#         depth_decoder.eval()
#
#         pred_disps = []
#
#         print("-> Computing predictions with size {}x{}".format(
#             encoder_dict['height'], encoder_dict['width']))
#
#         with torch.no_grad():
#             for data in test_loader:
#                 input_color = data[("color", 0, 0)].cuda()
#
#                 if opt.post_process:
#                     # Post-processed results require each image to have two forward passes
#                     input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)
#
#                 output = depth_decoder(encoder(input_color))
#
#                 pred_disp, _ = disp_to_depth(output[("disp", 0)], opt.min_depth, opt.max_depth)
#                 pred_disp = pred_disp.cpu()[:, 0].numpy()
#
#                 if opt.post_process:
#                     N = pred_disp.shape[0] // 2
#                     pred_disp = batch_post_process_disparity(pred_disp[:N], pred_disp[N:, :, ::-1])
#
#                 pred_disps.append(pred_disp)
#
#         pred_disps = np.concatenate(pred_disps)
#
#     if opt.save_pred_disps:
#         output_path = os.path.join(
#             opt.load_weights_folder, "disps_{}_split.npy".format(opt.eval_split))
#         print("-> Saving predicted disparities to ", output_path)
#         np.save(output_path, pred_disps)
#
#     if opt.no_eval:
#         print("-> Evaluation disabled. Done.")
#         quit()
#
#     # gt_path = os.path.join(splits_dir, opt.eval_split, "gt_depths.npz")
#     # gt_depths = np.load(gt_path, fix_imports=True, encoding='latin1')["data"]
#
#     print("-> Evaluating")
#     print("   Mono evaluation - using median scaling")
#
#     errors = []
#     ratios = []
#
#     for i in range(pred_disps.shape[0]):
#         gt_depth = gt_depths[i]
#         gt_height, gt_width = gt_depth.shape[:2]
#
#         pred_disp = pred_disps[i]
#         pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
#         pred_depth = 1 / pred_disp
#
#         mask = gt_depth > 0
#
#         pred_depth = pred_depth[mask]
#         gt_depth = gt_depth[mask]
#
#         pred_depth *= opt.pred_depth_scale_factor
#         if not opt.disable_median_scaling:
#             ratio = np.median(gt_depth) / np.median(pred_depth)
#             ratios.append(ratio)
#             pred_depth *= ratio
#
#         pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
#         pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH
#
#         errors.append(compute_errors(gt_depth, pred_depth))
#
#     if not opt.disable_median_scaling:
#         ratios = np.array(ratios)
#         med = np.median(ratios)
#         print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))
#
#     mean_errors = np.array(errors).mean(0)
#
#     print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
#     print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
#     print("\n-> Done!")

def evaluate(opt):
    """Evaluates a pretrained model using a specified test set and ground truth from the dataloader"""
    MIN_DEPTH = opt.min_depth
    MAX_DEPTH = opt.max_depth

    opt.load_weights_folder = os.path.expanduser(opt.load_weights_folder)
    assert os.path.isdir(opt.load_weights_folder), "Cannot find a folder at {}".format(opt.load_weights_folder)
    print("-> Loading weights from {}".format(opt.load_weights_folder))

    encoder_path = os.path.join(opt.load_weights_folder, "encoder.pth")
    decoder_path = os.path.join(opt.load_weights_folder, "depth.pth")

    encoder_dict = torch.load(encoder_path)

    encoder = networks.ResnetEncoder(opt.num_layers, False)
    depth_decoder = networks.DepthDecoder(encoder.num_ch_enc)

    model_dict = encoder.state_dict()
    encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
    depth_decoder.load_state_dict(torch.load(decoder_path))

    encoder.cuda()
    encoder.eval()
    depth_decoder.cuda()
    depth_decoder.eval()

    # === Load dataset ===
    transforms_test = Compose([Resize(opt.height, opt.width)])
    version = "v1.0-" + opt.dataset_version
    if version == "v1.0-test":
        nusc = NuScenes(version=version, dataroot=opt.data_path, verbose=True)
        test_dataset = NuScenesDataset(nusc, version, split='test',
                                       height=opt.height,
                                       width=opt.width,
                                       scales=opt.scales,
                                       transforms=transforms_test)
        test_loader = DataLoader(test_dataset,
                                 batch_size=opt.batch_size,
                                 shuffle=False,
                                 num_workers=opt.num_workers,
                                 collate_fn=flatten_collate_fn)
        print(f"Test samples: {len(test_loader)}")
    else:
        raise ValueError("Unsupported nuScenes dataset version: {}".format(version))

    output_dir = os.path.join(opt.load_weights_folder, "outputs_test")
    os.makedirs(output_dir, exist_ok=True)

    print("\n-> Computing predictions with size {}x{}".format(encoder_dict['height'], encoder_dict['width']))
    pred_disps = []
    gt_depths = []
    with torch.no_grad():
        for data in test_loader:
            input_color = data[("color", 0, 0)].cuda()

            # if opt.post_process:
            #     # Post-processed results require each image to have two forward passes
            #     input_color = torch.cat((input_color, torch.flip(input_color, [3])), 0)

            output = depth_decoder(encoder(input_color))
            pred_disp, pred_deth = disp_to_depth(output[("disp", 0)], opt.min_depth, opt.max_depth)
            # pred_deth = pred_deth[0].detach().cpu().permute(1, 2, 0).numpy()

            pred_disp = pred_disp.cpu()[:, 0].numpy()
            # if opt.post_process:
            #     N = pred_deth.shape[0] // 2
            #     pred_deth = batch_post_process_disparity(pred_deth[:N], pred_deth[N:, :, ::-1])
            pred_disps.append(pred_disp)

            gt_depth = data[("depth_gt", 0, 0)].cpu().numpy()
            gt_depths.append(gt_depth)

            # Save depth maps
            for i in range(pred_deth.shape[0]):
                # === Input RGB εικόνα ===
                input_img = input_color[i].detach().cpu().permute(1, 2, 0).numpy()
                input_img = np.clip(input_img, 0, 1)

                # === Predicted depth ===
                pred = 1 / pred_disp[i]

                # === Radar points ===
                lidar_points = data['depth_gt', 0, 0][i].cpu().numpy()  # shape: (N, 3) -> x, y, depth or intensity

                fig, axs = plt.subplots(1, 3, figsize=(18, 6))
                # === Αριστερά: RGB input ===
                axs[0].imshow(input_img)
                axs[0].axis('off')
                axs[0].set_title("Input Image")

                # === Μέση: Depth ===
                im = axs[1].imshow(pred, cmap='viridis')
                axs[1].axis('off')
                axs[1].set_title("Predicted Depth")
                cbar = fig.colorbar(im, ax=axs[1], fraction=0.046, pad=0.04)
                cbar.set_label("Depth (m)")

                # === Δεξιά: Lidar scatter ===
                axs[2].imshow(input_img)

                non_zero_indices = np.where(lidar_points > 0)
                non_zero_depths = lidar_points[non_zero_indices]

                scatter = axs[2].scatter(non_zero_indices[1], non_zero_indices[0],
                                         c=non_zero_depths, cmap='viridis', s=5)

                # Ορισμός χρωματικής κλίμακας
                if non_zero_depths.size > 0:
                    vmin, vmax = np.min(non_zero_depths), np.max(non_zero_depths)
                else:
                    vmin, vmax = 0, 100
                scatter.set_norm(Normalize(vmin=vmin, vmax=vmax))
                axs[2].axis('off')
                axs[2].set_title("Lidar Depth")
                fig.colorbar(scatter, ax=axs[2], fraction=0.046, pad=0.04, label="Lidar Depth (m)")

                # === Αποθήκευση ===
                output_path_img = os.path.join(output_dir, f"{data[('name', 0, 0)][i]}.png")
                plt.savefig(output_path_img, bbox_inches='tight', pad_inches=0.1)
                plt.close(fig)

        pred_disps = np.concatenate(pred_disps)
        gt_depths = np.concatenate(gt_depths)

    if opt.no_eval:
        print("-> Evaluation disabled. Done.")
        quit()

    print("-> Evaluating")
    print("   Mono evaluation - using median scaling")

    errors = []
    ratios = []
    for i in range(pred_disps.shape[0]):

        gt_depth = gt_depths[i]
        gt_height, gt_width = gt_depth.shape[:2]

        pred_disp = pred_disps[i]
        pred_disp = cv2.resize(pred_disp, (gt_width, gt_height))
        pred_depth = 1 / pred_disp

        # # Save depth maps
        # for i in range(pred_deth.shape[0]):
        #     # === Input RGB εικόνα ===
        #     input_img = input_color[i].detach().cpu().permute(1, 2, 0).numpy()
        #     input_img = np.clip(input_img, 0, 1)
        #
        #     # === Predicted depth ===
        #     pred = pred_deth[i].detach().cpu().permute(1, 2, 0).numpy()
        #     pred = np.squeeze(pred)
        #
        #     # === Radar points ===
        #     lidar_points = data['depth_gt', 0, 0][i].cpu().numpy()  # shape: (N, 3) -> x, y, depth or intensity
        #
        #     fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        #     # === Αριστερά: RGB input ===
        #     axs[0].imshow(input_img)
        #     axs[0].axis('off')
        #     axs[0].set_title("Input Image")
        #
        #     # === Μέση: Depth ===
        #     im = axs[1].imshow(pred, cmap='viridis')
        #     axs[1].axis('off')
        #     axs[1].set_title("Predicted Depth")
        #     cbar = fig.colorbar(im, ax=axs[1], fraction=0.046, pad=0.04)
        #     cbar.set_label("Depth (m)")
        #
        #     # === Δεξιά: Lidar scatter ===
        #     axs[2].imshow(input_img)
        #
        #     non_zero_indices = np.where(lidar_points > 0)
        #     non_zero_depths = lidar_points[non_zero_indices]
        #
        #     scatter = axs[2].scatter(non_zero_indices[1], non_zero_indices[0],
        #                              c=non_zero_depths, cmap='viridis', s=5)
        #
        #     # Ορισμός χρωματικής κλίμακας
        #     if non_zero_depths.size > 0:
        #         vmin, vmax = np.min(non_zero_depths), np.max(non_zero_depths)
        #     else:
        #         vmin, vmax = 0, 100
        #     scatter.set_norm(Normalize(vmin=vmin, vmax=vmax))
        #     axs[2].axis('off')
        #     axs[2].set_title("Lidar Depth")
        #     fig.colorbar(scatter, ax=axs[2], fraction=0.046, pad=0.04, label="Lidar Depth (m)")
        #
        #     # === Αποθήκευση ===
        #     output_path_img = os.path.join(output_dir, f"{data[('name', 0, 0)][i]}.png")
        #     plt.savefig(output_path_img, bbox_inches='tight', pad_inches=0.1)
        #     plt.close(fig)

        mask = gt_depth > 0
        pred_depth = pred_depth[mask]
        gt_depth = gt_depth[mask]

        pred_depth *= opt.pred_depth_scale_factor
        if not opt.disable_median_scaling:
            ratio = np.median(gt_depth) / np.median(pred_depth)
            ratios.append(ratio)
            pred_depth *= ratio

        pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
        pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH

        errors.append(compute_errors(gt_depth, pred_depth))

    if not opt.disable_median_scaling:
        ratios = np.array(ratios)
        med = np.median(ratios)
        print(" Scaling ratios | med: {:0.3f} | std: {:0.3f}".format(med, np.std(ratios / med)))

    mean_errors = np.array(errors).mean(0)

    print("\n  " + ("{:>8} | " * 7).format("abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"))
    print(("&{: 8.3f}  " * 7).format(*mean_errors.tolist()) + "\\\\")
    print("\n-> Done!")

import sys
if __name__ == "__main__":
    sys.argv = [
        "evaluate_depth_ioanna.py",
        "--load_weights_folder", "/media/ilias/4b3f6643-e758-40b9-9b58-9e98f88e5c79/dimitris/monodepth_thesis_nuscenes/tmp/nuscenes_monodepth_right_metrics/models/weights_49",
        "--dataset_version", "test"
    ]
    options = MonodepthOptions()
    evaluate(options.parse())