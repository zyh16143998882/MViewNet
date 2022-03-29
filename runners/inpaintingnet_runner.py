# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import torch
import random
import logging
from time import time

from torchvision.transforms import ToPILImage

import utils.misc as um
from utils.p2i_utils import N_VIEWS_PREDEFINED, N_VIEWS_PREDEFINED_GEN
from utils.model_init import discriminator_init, renderer_init, renderer_init2
import cuda.emd.emd_module as emd
from cuda.chamfer_distance import ChamferDistance, ChamferDistanceMean
from runners.misc import AverageMeter
from runners.base_runner import BaseRunner
from utils.visualizer import VISUALIZER_PRE, get_ptcloud_img, VIS_PATH_GT, VIS_PATH_PC, VIS_PATH_PC_ALL, \
    VIS_PATH_PARTIAL, VIS_INPUT_PATH_POINT, VIS_REAL_PATH_POINT
from PIL import Image
from  torchvision import utils as vutils


class inpaintingnetRunner(BaseRunner):
    """Define the SpareNet GAN runner class"""

    def __init__(self, config, logger):
        super().__init__(config, logger)        # 先调baserunner的初始化方法
        self.losses = AverageMeter(
            ["CoarseLoss", "MiddleLoss", "RefineLoss"]
        )
        self.test_losses = AverageMeter(
            ["CoarseLoss", "MiddleLoss", "RefineLoss"]
        )
        self.test_metrics = AverageMeter(um.Metrics.names())
        self.chamfer_dist = None
        self.chamfer_dist_mean = None
        self.emd_dist = None

    def build_models(self):
        super().build_models()  # 这里是对netG的初始化


    def build_train_loss(self):
        # Set up loss functions
        self.chamfer_dist = torch.nn.DataParallel(
            ChamferDistance().to(self.gpu_ids[0]), device_ids=self.gpu_ids
        )
        self.chamfer_dist_mean = torch.nn.DataParallel(
            ChamferDistanceMean().to(self.gpu_ids[0]), device_ids=self.gpu_ids
        )
        self.emd_dist = torch.nn.DataParallel(
            emd.emdModule().to(self.gpu_ids[0]), device_ids=self.gpu_ids
        )
        self.feat_l1 = torch.nn.DataParallel(
            torch.nn.L1Loss().to(self.gpu_ids[0]), device_ids=self.gpu_ids
        )

    def build_val_loss(self):
        # Set up loss functions
        self.chamfer_dist = ChamferDistance().cuda()
        self.chamfer_dist_mean = ChamferDistanceMean().cuda()
        self.emd_dist = emd.emdModule().cuda()

    def train_step(self, items):
        # prepare the data and label
        _, (_, labels, code, data) = items
        for k, v in data.items():
            data[k] = v.float().to(self.gpu_ids[0])
        # run the completion network
        _loss, refine, middle, coarse, refine_loss, middle_loss, coarse_loss, rec_l1 = self.completion_wo_recurefine(data, code)   # 这里completion和sparenet_runner的completion函数一模一样，只不过middle_ptcloud必须获取
        self.models.zero_grad()
        _loss.backward()
        self.optimizers.step()

        self.loss["coarse_loss"] = coarse_loss * 1000
        self.loss["middle_loss"] = middle_loss * 1000
        self.loss["refine_loss"] = refine_loss * 1000
        self.loss["rec_loss"] = _loss
        self.losses.update([coarse_loss.item() * 1000, middle_loss * 1000, refine_loss.item() * 1000])


    def val_step(self, items):
        _, (_, _, code, data) = items
        for k, v in data.items():
            data[k] = v.float().to(self.gpu_ids[0])

        _loss, refine, middle, coarse, refine_loss, middle_loss, coarse_loss, rec_l1  = self.completion_wo_recurefine(data, code)
        self.test_losses.update([coarse_loss.item() * 1000, middle_loss * 1000, refine_loss.item() * 1000])
        self.metrics = um.Metrics.get(refine, data["gtcloud"])      # 在都用4卡的情况下，测试集的batch需设置为4，这两个的形状为torch.Size([4, 16384, 3])
        self.ptcloud = refine

    def completion_wo_recurefine(self, data, code):
        (
            coarse_ptcloud,
            middle_ptcloud,
            refine_ptcloud,
            expansion_penalty,
            res_fake,
        ) = self.models(data, code)        # image是torch.Size([2, 32, 256, 256])
        _loss=0.0
        # rec_l1 = self.feat_l1(res_fake, data["mview_gt"]).mean()
        rec_l1 = 0.0
        _loss += rec_l1
        if self.config.NETWORK.metric == "chamfer":
            coarse_loss = self.chamfer_dist_mean(coarse_ptcloud, data["gtcloud"]).mean()
            middle_loss = self.chamfer_dist_mean(middle_ptcloud, data["gtcloud"]).mean()
            if self.config.NETWORK.use_recurefine == True:
                refine_loss = self.chamfer_dist_mean(refine_ptcloud, data["gtcloud"]).mean()
                _loss += refine_loss + expansion_penalty.mean() * 0.1
            else:
                refine_loss = middle_loss

        elif self.config.NETWORK.metric == "emd":
            emd_coarse, _ = self.emd_dist(coarse_ptcloud, data["gtcloud"], eps=0.005, iters=50)
            emd_middle, _ = self.emd_dist(middle_ptcloud, data["gtcloud"], eps=0.005, iters=50)
            coarse_loss = torch.sqrt(emd_coarse).mean(1).mean()
            middle_loss = torch.sqrt(emd_middle).mean(1).mean()
            if self.config.NETWORK.use_recurefine == True:
                emd_refine, _ = self.emd_dist(refine_ptcloud, data["gtcloud"], eps=0.005, iters=50)
                refine_loss = torch.sqrt(emd_refine).mean(1).mean()
                _loss += refine_loss + expansion_penalty.mean() * 0.1
            else:
                refine_loss = middle_loss

        else:
            raise Exception("unknown training metric")

        _loss += coarse_loss + middle_loss


        if self.config.NETWORK.use_consist_loss:
            if self.config.NETWORK.use_recurefine == True:
                ptcloud = refine_ptcloud
            else:
                ptcloud = middle_ptcloud
            dist1, _ = self.chamfer_dist(ptcloud, data["gtcloud"])
            cd_input2fine = torch.mean(dist1).mean()
            _loss += cd_input2fine * 0.5

        return (
            _loss,
            refine_ptcloud,
            middle_ptcloud,
            coarse_ptcloud,
            refine_loss,
            middle_loss,
            coarse_loss,
            rec_l1,
        )