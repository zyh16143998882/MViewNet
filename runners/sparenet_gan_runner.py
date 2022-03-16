# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import torch
import random
import logging
from time import time
import utils.misc as um
from utils.p2i_utils import N_VIEWS_PREDEFINED, N_VIEWS_PREDEFINED_GEN
from utils.model_init import discriminator_init, renderer_init
import cuda.emd.emd_module as emd
from cuda.chamfer_distance import ChamferDistance, ChamferDistanceMean
from runners.misc import AverageMeter
from runners.base_runner import BaseRunner


class sparenetGANRunner(BaseRunner):
    """Define the SpareNet GAN runner class"""

    def __init__(self, config, logger):
        super().__init__(config, logger)        # 先调baserunner的初始化方法
        self.losses = AverageMeter(
            ["CoarseLoss", "RefineLoss", "errG", "errG_D", "DisRealLoss", "DisFakeLoss"]
        )
        self.test_losses = AverageMeter(
            ["CoarseLoss", "RefineLoss", "errG", "errG_D", "DisRealLoss", "DisFakeLoss"]
        )
        self.test_metrics = AverageMeter(um.Metrics.names())
        self.chamfer_dist = None
        self.chamfer_dist_mean = None
        self.emd_dist = None
        self.criterionD = torch.nn.MSELoss()

    def build_models(self):
        super().build_models()      # 这里是对netG的初始化
        self.renderer_gen, self.renderer_dis = renderer_init(self.config)      # 初始化渲染器
        self.models_D, self.optimizers_D, self.lr_schedulers_D = discriminator_init(self.config)        # 初始化netD

    def data_parallel(self):
        super().data_parallel()
        self.models_D = torch.nn.DataParallel(
            self.models_D.to(self.gpu_ids[0]), device_ids=self.gpu_ids
        )
        self.renderer_gen = torch.nn.DataParallel(
            self.renderer_gen.to(self.gpu_ids[0]), device_ids=self.gpu_ids
        )
        self.renderer_dis = torch.nn.DataParallel(
            self.renderer_dis.to(self.gpu_ids[0]), device_ids=self.gpu_ids
        )

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
        labels = torch.tensor(labels, dtype=torch.long).to(self.gpu_ids[0])

        # 获取点云的gt和partial的深度图
        self.get_depth_image(data)
        # self.input_imgs = data["mview"]

        # create GAN positive & negative labels
        _batch_size = data["partial_cloud"].size()[0]
        self.real_label = (                         # 全部填1，大小batch_size
            torch.FloatTensor(_batch_size)
                .resize_([_batch_size, 1])
                .data.fill_(1)
                .to(self.gpu_ids[0])
        )
        self.fake_label = (                         # 全部填0，大小batch_size
            torch.FloatTensor(_batch_size)
                .resize_([_batch_size, 1])
                .data.fill_(0)
                .to(self.gpu_ids[0])
        )

        # run the completion network
        if self.models.module.use_RecuRefine == True:
            _loss, _, middle_ptcloud, _, refine_loss, coarse_loss = self.completion(data)       # 这里completion和sparenet_runner的completion函数一模一样，只不过middle_ptcloud必须获取
        else:
            _loss, _, middle_ptcloud, _, refine_loss, coarse_loss = self.completion_wo_recurefine(data, code)   # 这里completion和sparenet_runner的completion函数一模一样，只不过middle_ptcloud必须获取
        rec_loss = _loss
        rendered_ptcloud = middle_ptcloud

        errD_real, errD_fake = self.discriminator_backward(  # labels是items[1][1]
            data, labels, rendered_ptcloud
        )
        errG, errG_D = self.generator_backward(data, labels, rec_loss)

        self.loss["coarse_loss"] = coarse_loss * 1000
        self.loss["refine_loss"] = refine_loss * 1000
        self.loss["rec_loss"] = _loss
        self.loss["errG"] = errG
        self.loss["errG_D"] = errG_D
        self.loss["errD_real"] = errD_real
        self.loss["errD_fake"] = errD_fake

        self.losses.update(
            [
                coarse_loss.item() * 1000,
                refine_loss.item() * 1000,
                errG.item(),
                errG_D.item(),
                errD_real.item(),
                errD_fake.item(),
                ]
        )


    def val_step(self, items):
        _, (_, _, code, data) = items
        for k, v in data.items():
            data[k] = um.var_or_cuda(v)

        # 获取点云的gt和partial的深度图
        self.get_depth_image(data)
        # self.input_imgs = data["mview"]

        if self.models.module.use_RecuRefine == True:
            _loss, refine_ptcloud, _, _, refine_loss, coarse_loss = self.completion(data)
        else:
            _loss, _, refine_ptcloud, _, refine_loss, coarse_loss = self.completion_wo_recurefine(data, code)
        self.test_losses.update([coarse_loss.item() * 1000, refine_loss.item() * 1000])
        self.metrics = um.Metrics.get(refine_ptcloud, data["gtcloud"])      # 在都用4卡的情况下，测试集的batch需设置为4，这两个的形状为torch.Size([4, 16384, 3])
        self.ptcloud = refine_ptcloud

    def completion_wo_recurefine(self, data, code):

        (
            coarse_ptcloud,
            middle_ptcloud,
            refine_ptcloud,
            expansion_penalty,
        ) = self.models(data, self.input_imgs, code)        # image是torch.Size([2, 32, 256, 256])

        if self.config.NETWORK.metric == "chamfer":
            coarse_loss = self.chamfer_dist_mean(coarse_ptcloud, data["gtcloud"]).mean()
            middle_loss = self.chamfer_dist_mean(middle_ptcloud, data["gtcloud"]).mean()
            # refine_loss = self.chamfer_dist_mean(refine_ptcloud, data["gtcloud"]).mean()

        elif self.config.NETWORK.metric == "emd":
            emd_coarse, _ = self.emd_dist(
                coarse_ptcloud, data["gtcloud"], eps=0.005, iters=50
            )
            emd_middle, _ = self.emd_dist(
                middle_ptcloud, data["gtcloud"], eps=0.005, iters=50
            )
            # emd_refine, _ = self.emd_dist(
            #     refine_ptcloud, data["gtcloud"], eps=0.005, iters=50
            # )
            coarse_loss = torch.sqrt(emd_coarse).mean(1).mean()
            # refine_loss = torch.sqrt(emd_refine).mean(1).mean()
            middle_loss = torch.sqrt(emd_middle).mean(1).mean()

        else:
            raise Exception("unknown training metric")

        # _loss = coarse_loss + middle_loss + refine_loss + expansion_penalty.mean() * 0.1
        _loss = coarse_loss + middle_loss + expansion_penalty.mean() * 0.1

        if self.config.NETWORK.use_consist_loss:
            dist1, _ = self.chamfer_dist(middle_ptcloud, data["gtcloud"])
            cd_input2fine = torch.mean(dist1).mean()
            _loss += cd_input2fine * 0.5

        return (
            _loss,
            refine_ptcloud,
            middle_ptcloud,
            coarse_ptcloud,
            middle_loss,
            coarse_loss,
        )
    def completion(self, data):
        """
        inputs:
            cfg: EasyDict
            data: tensor
                -partical_cloud: b x npoints1 x num_dims
                -gtcloud: b x npoints2 x num_dims

        outputs:
            _loss: float32
            refine_ptcloud: b x npoints2 x num_dims
            middle_ptcloud: b x npoints2 x num_dims
            coarse_ptcloud: b x npoints2 x num_dims
            refine_loss: float32
            coarse_loss: float32
        """
        (
            coarse_ptcloud,
            middle_ptcloud,
            refine_ptcloud,
            expansion_penalty,
        ) = self.models(data, self.input_imgs)

        if self.config.NETWORK.metric == "chamfer":
            coarse_loss = self.chamfer_dist_mean(coarse_ptcloud, data["gtcloud"]).mean()
            middle_loss = self.chamfer_dist_mean(middle_ptcloud, data["gtcloud"]).mean()
            refine_loss = self.chamfer_dist_mean(refine_ptcloud, data["gtcloud"]).mean()

        elif self.config.NETWORK.metric == "emd":
            emd_coarse, _ = self.emd_dist(
                coarse_ptcloud, data["gtcloud"], eps=0.005, iters=50
            )
            emd_middle, _ = self.emd_dist(
                middle_ptcloud, data["gtcloud"], eps=0.005, iters=50
            )
            emd_refine, _ = self.emd_dist(
                refine_ptcloud, data["gtcloud"], eps=0.005, iters=50
            )
            coarse_loss = torch.sqrt(emd_coarse).mean(1).mean()
            refine_loss = torch.sqrt(emd_refine).mean(1).mean()
            middle_loss = torch.sqrt(emd_middle).mean(1).mean()

        else:
            raise Exception("unknown training metric")

        _loss = coarse_loss + middle_loss + refine_loss + expansion_penalty.mean() * 0.1

        if self.config.NETWORK.use_consist_loss:
            dist1, _ = self.chamfer_dist(refine_ptcloud, data["gtcloud"])
            cd_input2fine = torch.mean(dist1).mean()
            _loss += cd_input2fine * 0.5

        return (
            _loss,
            refine_ptcloud,
            middle_ptcloud,
            coarse_ptcloud,
            refine_loss,
            coarse_loss,
        )

    def get_depth_image(self, data):
        real_render_imgs_dict = {}  # 渲染的一个点云gt的所有img       shape：8*2*1*256*256
        input_render_imgs_dict = {}  # 渲染的一个点云partial的所有img
        random_radius = random.sample(self.config.RENDER.radius_list, 1)[0]  # 随机半径
        random_view_ids = list(range(0, N_VIEWS_PREDEFINED_GEN, 1))  # 随机视角ID  从0到7

        for _view_id in random_view_ids:
            # get real_imgs, gen_imgs and input_render_imgs
            real_render_imgs_dict[_view_id],_ = self.renderer_gen(
                data["gtcloud"], view_id=_view_id, radius_list=[random_radius]
            )
            input_render_imgs_dict[_view_id],_ = self.renderer_gen(
                data["partial_cloud"], view_id=_view_id, radius_list=[random_radius]
            )

        _view_id = random_view_ids[0]
        self.real_imgs = real_render_imgs_dict[_view_id]  # 第0个2*1*256*256
        self.input_imgs = input_render_imgs_dict[_view_id]  # partial的视图
        for _index in range(1, len(random_view_ids)):  # 对每个点云将8个视图concat起来，最终real_imgs等变为2*8*256*256
            _view_id = random_view_ids[_index]
            self.real_imgs = torch.cat(
                (self.real_imgs, real_render_imgs_dict[_view_id]), dim=1
            ).to(self.gpu_ids[0])
            self.input_imgs = torch.cat(
                (self.input_imgs, input_render_imgs_dict[_view_id]), dim=1
            ).to(self.gpu_ids[0])

    # 核心代码
    def discriminator_backward(self, data, labels, rendered_ptcloud):
        """
        inputs:
            data: tensor
                -partical_cloud: b x npoints1 x num_dims
                -gtcloud: b x npoints2 x num_dims
            labels: tensor
            rendered_ptcloud: b x npoints2 x num_dims

        outputs:
            input_imgs: b x views x [img_size, img_size]
            fake_imgs: b x views x [img_size, img_size]
            real_imgs: b x views x [img_size, img_size]
            errD_real: float32
            errD_fake: float32
        """
        self.optimizers_D.zero_grad()
        real_render_imgs_dict = {}          # 渲染的一个点云gt的所有img       shape：8*2*1*256*256
        gen_render_imgs_dict = {}           # 渲染的一个生成点云的所有img
        input_render_imgs_dict = {}         # 渲染的一个点云partial的所有img
        random_radius = random.sample(self.config.RENDER.radius_list, 1)[0]         # 随机半径
        random_view_ids = list(range(0, N_VIEWS_PREDEFINED, 1))                     # 随机视角ID  从0到7

        for _view_id in random_view_ids:
            # get real_imgs, gen_imgs and input_render_imgs
            real_render_imgs_dict[_view_id],_ = self.renderer_dis(
                data["gtcloud"], view_id=_view_id, radius_list=[random_radius]
            )
            gen_render_imgs_dict[_view_id], _ = self.renderer_dis(
                rendered_ptcloud, view_id=_view_id, radius_list=[random_radius]
            )
            input_render_imgs_dict[_view_id],_ = self.renderer_dis(
                data["partial_cloud"], view_id=_view_id, radius_list=[random_radius]
            )
            # real_render_imgs_dict[_view_id] = self.renderer_dis(
            #     data["gtcloud"], view_id=_view_id, radius_list=[random_radius]
            # )
            # gen_render_imgs_dict[_view_id] = self.renderer_dis(
            #     rendered_ptcloud, view_id=_view_id, radius_list=[random_radius]
            # )
            # input_render_imgs_dict[_view_id] = self.renderer_dis(
            #     data["partial_cloud"], view_id=_view_id, radius_list=[random_radius]
            # )

        _view_id = random_view_ids[0]
        self.real_imgs = real_render_imgs_dict[_view_id]    # 第0个2*1*256*256
        self.fake_imgs = gen_render_imgs_dict[_view_id]
        self.input_imgs = input_render_imgs_dict[_view_id]  # partial的视图
        for _index in range(1, len(random_view_ids)):       # 对每个点云将8个视图concat起来，最终real_imgs等变为2*8*256*256
            _view_id = random_view_ids[_index]
            self.real_imgs = torch.cat(
                (self.real_imgs, real_render_imgs_dict[_view_id]), dim=1
            )
            self.fake_imgs = torch.cat(
                (self.fake_imgs, gen_render_imgs_dict[_view_id]), dim=1
            )
            self.input_imgs = torch.cat(
                (self.input_imgs, input_render_imgs_dict[_view_id]), dim=1
            )

        errD_real = 0.0
        errD_fake = 0.0

        if self.config.GAN.use_cgan:        # models_D前向传播，其实不重要
            D_real_pred = self.models_D(    # partial和gt的视图作输入,D_real_pred和D_fake_pred分别是分类的预测
                torch.cat((self.input_imgs, self.real_imgs), dim=1).detach(), y=labels
            )
            D_fake_pred = self.models_D(    # partial和gen的视图作输入
                torch.cat((self.input_imgs, self.fake_imgs), dim=1).detach(), y=labels
            )
        else:
            D_real_pred = self.models_D(
                torch.cat((self.input_imgs, self.real_imgs), dim=1).detach()
            )
            D_fake_pred = self.models_D(
                torch.cat((self.input_imgs, self.fake_imgs), dim=1).detach()
            )

        errD_real += self.criterionD(D_real_pred, self.real_label)
        errD_fake += self.criterionD(D_fake_pred, self.fake_label)
        errD = errD_real + errD_fake                # models_D的损失
        errD.backward()                             # 反向传播，更新models_D
        self.optimizers_D.step()
        return errD_real, errD_fake

    def generator_backward(self, data, labels, rec_loss):
        """
        inputs:
            data: tensor
                -partical_cloud: b x npoints1 x num_dims
                -gtcloud: b x npoints2 x num_dims
            labels: tensor
            input_imgs: b x views x [img_size, img_size]
            fake_imgs: b x views x [img_size, img_size]
            real_imgs: b x views x [img_size, img_size]
            rec_loss: float

        outputs:
            errG: float32
            errG_D: float32
        """
        self.optimizers.zero_grad()

        errG_D = 0.0
        loss_fm = 0.0
        loss_im = 0.0

        if self.config.GAN.use_fm:  # get feature matching
            if self.config.GAN.use_cgan:
                D_fake_pred, D_fake_features = self.models_D(       # 这里返回预测结果和每层的特征
                    torch.cat((self.input_imgs, self.fake_imgs), dim=1),
                    feat=True,
                    y=labels,
                )
                _, D_real_features = self.models_D(
                    torch.cat((self.input_imgs, self.real_imgs), dim=1),
                    feat=True,
                    y=labels,
                )

            else:
                # Calculate output of image discriminator (PatchGAN)
                D_fake_pred, D_fake_features = self.models_D(
                    torch.cat((self.input_imgs, self.fake_imgs), dim=1), feat=True
                )
                _, D_real_features = self.models_D(
                    torch.cat((self.input_imgs, self.real_imgs), dim=1), feat=True
                )

            # Feature match loss is weighted by number of feature maps
            map_nums = [feat.shape[1] for feat in D_fake_features]
            feat_weights = [float(i) / sum(map_nums) for i in map_nums]
            for j in range(
                    len(D_fake_features)
            ):  # the final loss is the sum of all features
                loss_fm += feat_weights[j] * torch.mean(
                    (D_fake_features[j] - D_real_features[j].detach()) ** 2
                )
        else:
            if self.config.GAN.use_cgan:
                D_fake_pred = self.models_D(
                    torch.cat((self.input_imgs, self.fake_imgs), dim=1), y=labels
                )
            else:
                # Calculate output of image discriminator (PatchGAN)
                D_fake_pred = self.models_D(
                    torch.cat((self.input_imgs, self.fake_imgs), dim=1)
                )
        errG_D += self.criterionD(D_fake_pred, self.real_label)

        if self.config.GAN.use_im:  # Get image matching (L1_loss)
            loss_im += torch.nn.L1Loss()(self.fake_imgs, self.real_imgs.detach())           # 图片上的损失

        errG = (
                self.config.GAN.weight_l2 * rec_loss + self.config.GAN.weight_gan * errG_D
        )
        # the sum of recloss and GAN_loss (and feature matching and image matching)
        if self.config.GAN.use_fm:
            errG += self.config.GAN.weight_fm * loss_fm
        if self.config.GAN.use_im:
            errG += self.config.GAN.weight_im * loss_im
        errG.backward()
        self.optimizers.step()

        return errG, errG_D