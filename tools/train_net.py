# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
r"""
Basic training script for PyTorch
"""

# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from mega_core.utils.env import setup_environment  # noqa F401 isort:skip
from mega_core.utils.dist_env import init_dist
from mega_core.utils.distributed import ompi_rank, ompi_local_rank

import argparse
import time
import os

import torch
from torch.utils.tensorboard import SummaryWriter
from mega_core.config import cfg
from mega_core.data import make_data_loader
from mega_core.solver import make_lr_scheduler
from mega_core.solver import make_optimizer
from mega_core.engine.inference import inference
from mega_core.engine.trainer import do_train
from mega_core.modeling.detector import build_detection_model
from mega_core.utils.checkpoint import DetectronCheckpointer
from mega_core.utils.collect_env import collect_env_info
from mega_core.utils.comm import synchronize, get_rank
from mega_core.utils.imports import import_file
from mega_core.utils.logger import setup_logger
from mega_core.utils.miscellaneous import mkdir, save_config
from mega_core.modeling.detector.diffusion_det import add_diffusiondet_config

# See if we can use apex.DistributedDataParallel instead of the torch default,
# and enable mixed-precision via apex.amp
try:
    from apex import amp
except ImportError:
    raise ImportError('Use APEX for multi-precision via apex.amp')


def train(cfg, local_rank, distributed):
    model = build_detection_model(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    optimizer = make_optimizer(cfg, model)
    scheduler = make_lr_scheduler(cfg, optimizer)

    # Initialize mixed-precision training
    use_mixed_precision = cfg.DTYPE == "float16"
    amp_opt_level = 'O1' if use_mixed_precision else 'O0'
    model, optimizer = amp.initialize(model, optimizer, opt_level=amp_opt_level)

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,
            find_unused_parameters=False
        )

    arguments = {}
    arguments["iteration"] = 0

    output_dir = cfg.OUTPUT_DIR

    save_to_disk = get_rank() == 0
    checkpointer = DetectronCheckpointer(
        cfg, model, optimizer, scheduler, output_dir, save_to_disk
    )
    extra_checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT, ignore=cfg.MODEL.VID.IGNORE, skip_modules=['class_logits'])
    if cfg.MODEL.VID.METHOD in ("fgfa", "dff") and cfg.MODEL.VID.FLOWNET_WEIGHT != "":
        checkpointer.load_flownet(cfg.MODEL.VID.FLOWNET_WEIGHT)

    if not cfg.MODEL.VID.IGNORE:
        arguments.update(extra_checkpoint_data)

    print(f"cfg.MODEL.WEIGHT: {cfg.MODEL.WEIGHT}")
    print(f"cfg.MODEL.WEIGHT: {cfg.MODEL.VID.FLOWNET_WEIGHT}")
    data_loader = make_data_loader(
        cfg,
        is_train=True,
        is_distributed=distributed,
        start_iter=arguments["iteration"],
    )

    test_period = cfg.SOLVER.TEST_PERIOD * cfg.SOLVER.ACCUMULATION_STEPS
    if test_period > 0:
        data_loader_val = make_data_loader(cfg, is_train=False, is_distributed=distributed, is_for_period=False)
    else:
        data_loader_val = None

    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD * cfg.SOLVER.ACCUMULATION_STEPS

    if get_rank() == 0 and cfg.TENSORBOARD:
        if arguments["iteration"] == 0:
            tensorboard_writer = SummaryWriter(output_dir)
        else:
            tensorboard_writer = SummaryWriter(output_dir, purge_step=arguments["iteration"] + 1)
    else:
        tensorboard_writer = None

    do_train(
        cfg,
        model,
        data_loader,
        data_loader_val,
        optimizer,
        scheduler,
        checkpointer,
        device,
        checkpoint_period,
        test_period,
        arguments,
        tensorboard_writer
    )

    return model


def run_test(cfg, model, distributed, motion_specific=False):
    if distributed:
        model = model.module
    torch.cuda.empty_cache()  # TODO check if it helps
    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder
    data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)
    for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
        inference(
            cfg,
            model,
            data_loader_val,
            dataset_name=dataset_name,
            iou_types=iou_types,
            motion_specific=motion_specific,
            box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
            bbox_aug=cfg.TEST.BBOX_AUG.ENABLED,
            device=cfg.MODEL.DEVICE,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=output_folder,
        )
        synchronize()

import yacs.config as config
import yaml
import ast
def coerce_cfg_merge(cfg_old, cfg_new):
    """
    递归遍历两个配置文件，并将不匹配的值从 cfg_old 中转换为 cfg_new 中相应项的类型。
    
    Args:
        cfg_old: 旧的YACS配置对象
        cfg_new: 新的YACS配置对象
    """
    if isinstance(cfg_old, config.CfgNode) and isinstance(cfg_new, config.CfgNode):
        for key in cfg_new.keys():
            if key in cfg_old:
                if isinstance(cfg_new[key], config.CfgNode):
                    coerce_cfg_merge(cfg_old[key], cfg_new[key])
                elif type(cfg_old[key]) != type(cfg_new[key]):
                    try:
                        if isinstance(cfg_old[key], tuple):
                            cfg_old[key] = ast.literal_eval(cfg_new[key])
                        else:
                            cfg_old[key] = cfg_new[key]
                    except:
                        pass  # 忽略无法转换的值

def main():
    #os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # for debugging
    parser = argparse.ArgumentParser(description="PyTorch Video Object Detection Training")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        '--launcher',
        choices=['pytorch', 'mpi'],
        default='pytorch',
        help='job launcher')
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument("--master_port", "-mp", type=str, default='29999')
    parser.add_argument("--save_name", default="", help="Where to store the log", type=str)
    parser.add_argument(
        "--motion-specific",
        "-ms",
        action="store_true",
        help="if True, evaluate motion-specific iou"
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        print("Before distributed initialization - Environment Variables:")
        args.local_rank = int(os.environ['LOCAL_RANK'])
        print("Before init_dist")
        print(args)
        init_dist(args.launcher, args=args)
        print("After init_dist")
    
    # this is similar to the behavior of detectron2, which I think is a nice option.
    BASE_CONFIG = "configs/BASE_RCNN_{}gpu.yaml".format(num_gpus)
    
    cfg.merge_from_file(BASE_CONFIG)
    if 'Diffusion' in args.config_file:
        add_diffusiondet_config(cfg)
    
    with open(args.config_file, "r") as f:
        cfg_new = config.load_cfg(f)
    coerce_cfg_merge(cfg, cfg_new)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    output_dir = cfg.OUTPUT_DIR

    if output_dir:
        mkdir(output_dir)
    cfg.freeze()
    print(cfg.OUTPUT_DIR)
    
    logger = setup_logger("mega_core", output_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    if get_rank() == 0:
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, "r") as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
        logger.info("Running with config:\n{}".format(cfg))

    output_config_path = os.path.join(cfg.OUTPUT_DIR, 'config.yml')
    logger.info("Saving config into: {}".format(output_config_path))
    # save overloaded model config in the output directory
    save_config(cfg, output_config_path)

    if args.launcher == "mpi":
        args.local_rank = ompi_local_rank()
    model = train(cfg, args.local_rank, args.distributed)

    if not args.skip_test:
        run_test(cfg, model, args.distributed, args.motion_specific)


if __name__ == "__main__":
    main()
