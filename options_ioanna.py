from __future__ import absolute_import, division, print_function

import os
import argparse

file_dir = os.path.dirname(__file__)  # the directory that options.py resides in

class MonodepthOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Monodepthv2 options for NuScenes")

        # PATHS
        self.parser.add_argument("--data_path",
                                type=str,
                                help="path to the NuScenes dataset",
                                default=os.path.join(file_dir, 'nuscenes_data'))

        self.parser.add_argument("--log_dir",
                                type=str,
                                help="log directory",
                                default=os.path.join(file_dir, 'tmp'))

        # TRAINING options
        self.parser.add_argument("--model_name",
                                type=str,
                                help="the name of the folder to save the model in",
                                default="nuscenes")

        self.parser.add_argument("--dataset_version",
                                type=str,
                                help="which training split to use",
                                choices=["mini", "trainval", "test"],
                                default="trainval")

        self.parser.add_argument("--num_layers",
                                type=int,
                                help="number of resnet layers",
                                default=18,
                                choices=[18, 34, 50, 101, 152])


        self.parser.add_argument("--height",
                                type=int,
                                help="input image height",
                                default=320)  # Adjusted for NuScenes aspect ratio

        self.parser.add_argument("--width",
                                type=int,
                                help="input image width",
                                default=576)

        self.parser.add_argument("--disparity_smoothness",
                                type=float,
                                help="disparity smoothness weight",
                                default=1e-3)

        self.parser.add_argument("--scales",
                                nargs="+",
                                type=int,
                                help="scales used in the loss",
                                default=[0, 1, 2, 3])

        self.parser.add_argument("--min_depth",
                                type=float,
                                help="minimum depth",
                                default=0.1)

        self.parser.add_argument("--max_depth",
                                type=float,
                                help="maximum depth",
                                default=80.0)

        self.parser.add_argument("--frame_ids",
                                nargs="+",
                                type=int,
                                help="frames to load",
                                default=[0, -1, 1])

        self.parser.add_argument("--camera_name",
                                type=str,
                                help="which camera to use",
                                default="CAM_FRONT",
                                choices=["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
                                         "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"])

        # OPTIMIZATION options
        self.parser.add_argument("--batch_size",
                                type=int,
                                help="batch size",
                                default=8)  # Reduced for NuScenes higher resolution
        self.parser.add_argument("--learning_rate",
                                type=float,
                                help="learning rate",
                                default=1e-4)
        self.parser.add_argument("--num_epochs",
                                type=int,
                                help="number of epochs",
                                default=20)
        self.parser.add_argument("--scheduler_step_size",
                                type=int,
                                help="step size of the scheduler",
                                default=15)

        # ABLATION options
        self.parser.add_argument("--v1_multiscale",
                                help="if set, uses monodepth v1 multiscale",
                                action="store_true")
        self.parser.add_argument("--avg_reprojection",
                                help="if set, uses average reprojection loss",
                                action="store_true")
        self.parser.add_argument("--disable_automasking",
                                help="if set, doesn't do auto-masking",
                                action="store_true")
        self.parser.add_argument("--predictive_mask",
                                help="if set, uses a predictive masking scheme as in Zhou et al",
                                action="store_true")
        self.parser.add_argument("--no_ssim",
                                help="if set, disables ssim in the loss",
                                action="store_true")
        self.parser.add_argument("--weights_init",
                                type=str,
                                help="pretrained or scratch",
                                default="pretrained",
                                choices=["pretrained", "scratch"])
        self.parser.add_argument("--pose_model_input",
                                type=str,
                                help="how many images the pose network gets",
                                default="pairs",
                                choices=["pairs", "all"])
        self.parser.add_argument("--pose_model_type",
                                type=str,
                                help="normal or shared",
                                default="separate_resnet",
                                choices=["posecnn", "separate_resnet", "shared"])

        # SYSTEM options
        self.parser.add_argument("--no_cuda",
                                help="if set disables CUDA",
                                action="store_true")
        self.parser.add_argument("--num_workers",
                                type=int,
                                help="number of dataloader workers",
                                default=8)  # Reduced for memory constraints

        # LOADING options
        self.parser.add_argument("--load_weights_folder",
                                type=str,
                                help="name of model to load")
        self.parser.add_argument("--models_to_load",
                                nargs="+",
                                type=str,
                                help="models to load",
                                default=["encoder", "depth", "pose_encoder", "pose"])

        # LOGGING options
        self.parser.add_argument("--log_frequency",
                                type=int,
                                help="number of batches between each tensorboard log",
                                default=500)
        self.parser.add_argument("--save_frequency",
                                type=int,
                                help="number of epochs between each save",
                                default=1)

        # EVALUATION options
        self.parser.add_argument("--eval_stereo",
                                help="if set evaluates in stereo mode",
                                default=False)
        self.parser.add_argument("--eval_mono",
                                help="if set evaluates in mono mode",
                                default=True)
        self.parser.add_argument("--disable_median_scaling",
                                help="if set disables median scaling in evaluation",
                                action="store_true")
        self.parser.add_argument("--pred_depth_scale_factor",
                                help="if set multiplies predictions by this number",
                                type=float,
                                default=1)
        self.parser.add_argument("--ext_disp_to_eval",
                                type=str,
                                default='/media/ilias/4b3f6643-e758-40b9-9b58-9e98f88e5c79/dimitris/monodepth2/tmp/nuscenes_monodepth/something',
                                help="optional path to a .npy disparities file to evaluate")
        self.parser.add_argument("--eval_split",
                                type=str,
                                default="test",
                                choices=["val", "test"],
                                help="which split to run eval on")
        self.parser.add_argument("--save_pred_disps",
                                help="if set saves predicted disparities",
                                default="store_true")
        self.parser.add_argument("--no_eval",
                                help="if set disables evaluation",
                                action="store_true")
        self.parser.add_argument("--eval_out_dir",
                                help="if set will output the disparities to this folder",
                                type=str)
        self.parser.add_argument("--post_process",
                                help="if set will perform the flipping post processing "
                                     "from the original monodepth paper",
                                action="store_true")

        self.parser.add_argument("--gpu",
                                 type=int,
                                 default=1,
                                 help="GPU id to use (default: 1)")

    def parse(self):
        self.options = self.parser.parse_args()
        return self.options