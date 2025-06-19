from __future__ import absolute_import, division, print_function

import os

import torch

import networks
import matplotlib.pyplot as plt

from options_ioanna import MonodepthOptions
from datasets.dataloader import create_dataloaders

def evaluate(opt):
    """Evaluates a pretrained model using a specified test set
    """
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


    result = create_dataloaders(opt, transform=None)
    version = "v1.0-" + opt.dataset_version
    if version == "v1.0-trainval":
        train_loader, val_loader = result
        print(f"Train samples: {len(train_loader)}")
        print(f"Validation samples: {len(val_loader)}")

    else:
        test_loader = result
        print(f"Test size: {len(test_loader)}")





def show_radar_map_batch(batch, idx=0):
    """Visualize radar points from a batch on the camera image"""
    sample_data = batch[idx]  # Get the first sample from the batch

    image = sample_data['image']

    if isinstance(image, torch.Tensor):
        image = image.numpy().transpose(1, 2, 0)

    # Create figure
    plt.figure(figsize=(12, 6))

    # Plot image
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Camera Image')

    # Plot radar points
    plt.subplot(1, 2, 2)
    plt.imshow(image)

    radar_pts_2d = sample_data['radar_points_2d']
    radar_pts_3d = sample_data['radar_points_3d']

    # Color points by depth (z-coordinate)
    if len(radar_pts_3d) > 0:  # Check if there are any radar points
        depths = radar_pts_3d[:, 2]
        scatter = plt.scatter(radar_pts_2d[:, 0], radar_pts_2d[:, 1], c=depths,
                              cmap='viridis', s=20, alpha=0.7)
        plt.colorbar(scatter, label='Depth (m)')

    plt.title(f'Radar Points (Count: {len(radar_pts_2d)})')
    plt.tight_layout()
    plt.show()

import sys
if __name__ == "__main__":
    sys.argv = [
        "evaluate_depth_ioanna.py",
        "--load_weights_folder", "/media/ilias/4b3f6643-e758-40b9-9b58-9e98f88e5c79/dimitris/monodepth2/tmp/mono_640x192"
    ]
    options = MonodepthOptions()
    evaluate(options.parse())

    # opt = options.parse()
    # dataroot = opt.data_path
    # version = "v1.0-" + opt.dataset_version
    #
    # # Get dataloaders
    # if version in ["v1.0-trainval", "v1.0-mini"]:
    #     train_loader, val_loader = create_dataloaders(dataroot, version, batch_size=4)
    #
    #     # Get a batch from train loader
    #     for i, batch in enumerate(train_loader):
    #         print(f"Showing batch {i}")
    #         show_radar_map_batch(batch)
    #
    #         # Show only first 3 batches for demonstration
    #         if i >= 2:
    #             break
    #
    # else:  # test version
    #     test_loader = create_dataloaders(dataroot, version, batch_size=4)
    #     batch = next(iter(test_loader))
    #     show_radar_map_batch(batch)