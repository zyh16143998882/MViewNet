# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import argparse
import time


def get_args_from_command_line():
    """
    config the parameter
    """
    parser = argparse.ArgumentParser(description="The argument parser of R2Net runner")

    # choose model
    parser.add_argument("--model", type=str, default="pcf2dnet", help="sparenet, pcf2dnet")

    # choose train mode
    parser.add_argument("--gan", dest="gan", help="use gan", action="store_true", default=True)

    # choose pretrain model
    parser.add_argument("--pretrain", dest="pretrain", help="Initialize network from the weights file",
                        default="/data/zhayaohua/project/pccomplection/PCF2dNet/checkpoint/pretrain/ckpt-best.pth")

    # choose load model
    parser.add_argument("--weights", dest="weights", help="Initialize network from the weights file", default=None)

    # setup gpu
    parser.add_argument("--gpu", dest="gpu_id", help="GPU device to use", default="0", type=str)

    # setup gpu
    parser.add_argument("--batch_size", dest="batch_size", help="batch_size", default=2, type=int)

    # setup workdir
    parser.add_argument("--workdir", dest="workdir", help="where to save files", default=None)
    return parser.parse_args()


def main():
    args = get_args_from_command_line()

    # Set GPU to use
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    # update config
    from configs.base_config import cfg, cfg_from_file, cfg_update


    if args.gan:
        cfg_from_file("configs/" + args.model + "_gan.yaml")
    else:
        cfg_from_file("configs/" + args.model + ".yaml")        # 从配置文件加载配置
    output_dir = cfg_update(args)                               # 在cfg中写超参
    cfg.TRAIN.batch_size = args.batch_size                      # 设置batch size
    cfg.DATASETS.shapenet.train_img_path = "/data/zhayaohua/Datasets/ShapeNetImg/train/partial/%s_%s_%s_%s.jpg"
    cfg.DATASETS.shapenet.test_img_path = "/data/zhayaohua/Datasets/ShapeNetImg/test/partial/%s_%s_%s_%s.jpg"

    # Set up folders for logs and checkpoints
    if not os.path.exists(cfg.DIR.logs):
        os.makedirs(cfg.DIR.logs)
    from utils.misc import set_logger                           # 载入log工具

    logger = set_logger(os.path.join(cfg.DIR.logs, "log.txt"))  # 生成log对象
    logger.info("save into dir: %s" % cfg.DIR.logs)

    # Start train/inference process
    if args.gan:
        runners = __import__("runners." + args.model + "_gan_runner")   # 从runner文件里面载入
        module = getattr(runners, args.model + "_gan_runner")
        model = getattr(module, args.model + "GANRunner")(cfg, logger)  # 载入模型

    else:
        runners = __import__("runners." + args.model + "_runner")
        module = getattr(runners, args.model + "_runner")
        model = getattr(module, args.model + "Runner")(cfg, logger)

    model.runner()                                                      # 模型启动


if __name__ == "__main__":
    start = time.time()
    print("开始时间 == ", start)
    main()
    end = time.time()
    print("结束时间 == ", end)
    print("总时间 == ", end-start)
