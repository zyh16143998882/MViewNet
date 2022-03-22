# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import functools
import os

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import cuda.MDS.MDS_module as MDS_module
import cuda.expansion_penalty.expansion_penalty_module as expansion
from  torchvision import utils as vutils
from PIL import Image


# 生成器
from cuda.pointnet2 import pointnet2_utils
from models.unet import UnetEncoder, UnetGanEncoder, MViewEncoder, EasyUnetGenerator
from utils.visualizer import get_ptcloud_img, VISUALIZER, VIS_PATH_PC, VISUALIZER_PC, plot_pcd_three_views, \
    VIS_PATH_PC_3


class SpareNetGenerator(nn.Module):
    """
    inputs:
    - data:
        -partical_cloud: b x npoints1 x num_dims
        -gtcloud: b x npoints2 x num_dims

    outputs:
    - coarse pcd: b x npoints2 x num_dims
    - middle pcd: b x npoints2 x num_dims
    - refine pcd: b x npoints2 x num_dims
    - loss_mst:
    """

    def __init__(
        self,
        n_primitives: int = 32,
        hide_size: int = 4096,
        bottleneck_size: int = 4096,
        num_points: int = 16382,
        use_SElayer: bool = False,
        use_RecuRefine: bool = False,
        use_AdaIn: str = "no_use",
        encode: str = "Pointfeat",
        decode: str = "Sparenet",
    ):
        super(SpareNetGenerator, self).__init__()
        self.num_points = num_points
        self.bottleneck_size = bottleneck_size
        self.n_primitives = n_primitives
        self.use_AdaIn = use_AdaIn
        self.hide_size = hide_size
        self.use_RecuRefine = use_RecuRefine
        self.conv1 = nn.Conv1d(3, 64, 1)
        # 定义编码器
        self.encoder = SpareNetEncode(
            hide_size=self.hide_size,
            bottleneck_size=self.bottleneck_size,
            use_SElayer=use_SElayer,
            encode=encode,
        )
        # 定义解码器
        self.decoder = SpareNetDecode(
            num_points=self.num_points,
            n_primitives=self.n_primitives,
            bottleneck_size=self.bottleneck_size,
            use_AdaIn=self.use_AdaIn,
            use_SElayer=use_SElayer,
            decode=decode,
        )
        # 定义微调器
        self.refine = SpareNetRefine(
            num_points=self.num_points,
            n_primitives=self.n_primitives,
            use_SElayer=use_SElayer,
        )

    def forward(self, data, point_imgs, code="default"):
        partial_x = data["partial_cloud"]  # 拿不全的点云
        partial_x = partial_x.transpose(1, 2).contiguous()  # [bs, 3, in_points]    开始是[bs, in_points, 3]之后只转置2 3维得到前面的
        partial = partial_x  # [batch_size, 3, in_points]

        # encode
        style = self.encoder(partial_x)  # [batch_size, 1024]       # 输入到encoder得到1024维特征

        # decode
        outs = self.decoder(style, point_imgs, code)                       # 用1024维特征使用风格解码器申城粗略点云 torch.Size([2, 3, 16384])
        if VISUALIZER == True:
            img_te = get_ptcloud_img(outs.cpu())
            img_te = Image.fromarray(img_te)
            img_te.save('./output/cartest/pc_all/{}_0.jpg'.format(str(code[0])))
        coarse = outs.transpose(1, 2).contiguous()  # [batch_size, out_points, 3]   # 再转置回来，contiguous()操作往往和转置相搭配 torch.Size([2, 16384, 3])


        # refine first time
        middle, loss_mst = self.refine(outs, partial, coarse)       # 部分点云和粗略点云做第一次微调，得到微调出的点云Yr1和其微调loss
        if VISUALIZER == True:
            img_te = get_ptcloud_img(middle.transpose(2, 1).contiguous().cpu())
            img_te = Image.fromarray(img_te)
            img_te.save('./output/cartest/pc_all/{}_1.jpg'.format(str(code[0])))

        if self.use_RecuRefine == True:
            # refine second time
            outs_2 = middle.transpose(1, 2).contiguous()
            refine, _ = self.refine(outs_2, partial, middle)            # Yr1和部分点云做微调得到最终的生成结果Yr2即Y

            return coarse, middle, refine, loss_mst                     # 返回粗略点云，第一次微调点云，最终点云以及第一次微调loss
        else:
            return coarse, middle, None, loss_mst                     # 返回粗略点云，第一次微调点云，最终点云以及第一次微调loss

class MViewNetGenerator(nn.Module):
    """
    inputs:
    - data:
        -partical_cloud: b x npoints1 x num_dims
        -gtcloud: b x npoints2 x num_dims
    outputs:
    - coarse pcd: b x npoints2 x num_dims
    - middle pcd: b x npoints2 x num_dims
    - refine pcd: b x npoints2 x num_dims
    - loss_mst:
    """

    def __init__(
        self,
        n_primitives: int = 32,
        hide_size: int = 4096,
        bottleneck_size: int = 4096,
        num_points: int = 16382,
        use_SElayer: bool = False,
        use_RecuRefine: bool = False,
        use_AdaIn: str = "no_use",
        encode: str = "Pointfeat",
        decode: str = "Sparenet",
    ):
        super(MViewNetGenerator, self).__init__()
        self.num_points = num_points
        self.bottleneck_size = bottleneck_size
        self.n_primitives = n_primitives
        self.use_AdaIn = use_AdaIn
        self.hide_size = hide_size
        self.use_RecuRefine = use_RecuRefine

        self.conv1 = nn.Conv1d(3, 64, 1)
        # 定义编码器
        self.encoder = SpareNetEncode(
            hide_size=self.hide_size,
            bottleneck_size=self.bottleneck_size,
            use_SElayer=use_SElayer,
            encode=encode,
        )
        # 定义解码器
        self.decoder = MViewNetDecode(
            num_points=self.num_points,
            n_primitives=self.n_primitives,
            bottleneck_size=self.bottleneck_size,
            use_AdaIn=self.use_AdaIn,
            use_SElayer=use_SElayer,
            decode=decode,
        )
        # 定义微调器
        self.refine = SpareNetRefine(
            num_points=self.num_points,
            n_primitives=self.n_primitives,
            use_SElayer=use_SElayer,
        )

    def forward(self, data, point_imgs, code="default"):
        partial_x = data["partial_cloud"]  # 拿不全的点云
        partial_x = partial_x.transpose(1, 2).contiguous()  # [bs, 3, in_points]    开始是[bs, in_points, 3]之后只转置2 3维得到前面的
        partial = partial_x  # [batch_size, 3, in_points]

        # encode
        style = self.encoder(partial_x)  # [batch_size, 1024]       # 输入到encoder得到1024维特征

        # decode
        outs, fake_maps, dec_inputs = self.decoder(style, point_imgs, code)  # 用1024维特征使用风格解码器申城粗略点云 torch.Size([2, 3, 16384])

        if VISUALIZER == True:
            img_te = get_ptcloud_img(outs.cpu())
            img_te = Image.fromarray(img_te)
            img_te.save('./output/cartest/pc_all/{}_0.jpg'.format(str(code[0])))
        coarse = outs.transpose(1, 2).contiguous()  # [batch_size, out_points, 3]   # 再转置回来，contiguous()操作往往和转置相搭配 torch.Size([2, 16384, 3])


        # refine first time
        middle, loss_mst = self.refine(outs, partial, coarse)       # 部分点云和粗略点云做第一次微调，得到微调出的点云Yr1和其微调loss
        if VISUALIZER == True:
            img_te = get_ptcloud_img(middle.transpose(2, 1).contiguous().cpu())
            img_te = Image.fromarray(img_te)
            img_te.save('./output/cartest/pc_all/{}_1.jpg'.format(str(code[0])))

        if self.use_RecuRefine == True:
            # refine second time
            outs_2 = middle.transpose(1, 2).contiguous()
            refine, _ = self.refine(outs_2, partial, middle)            # Yr1和部分点云做微调得到最终的生成结果Yr2即Y

            return coarse, middle, refine, loss_mst, fake_maps, dec_inputs                     # 返回粗略点云，第一次微调点云，最终点云以及第一次微调loss
        else:
            return coarse, middle, None, loss_mst, fake_maps, dec_inputs                     # 返回粗略点云，第一次微调点云，最终点云以及第一次微调loss

class MViewPointNetGenerator(nn.Module):
    """
    inputs:
    - data:
        -partical_cloud: b x npoints1 x num_dims
        -gtcloud: b x npoints2 x num_dims

    outputs:
    - coarse pcd: b x npoints2 x num_dims
    - middle pcd: b x npoints2 x num_dims
    - refine pcd: b x npoints2 x num_dims
    - loss_mst:
    """

    def __init__(
        self,
        n_primitives: int = 32,
        hide_size: int = 4096,
        bottleneck_size: int = 4096,
        num_points: int = 16382,
        use_SElayer: bool = False,
        use_RecuRefine: bool = False,
        use_AdaIn: str = "no_use",
        encode: str = "Pointfeat",
        decode: str = "Sparenet",
    ):
        super(MViewPointNetGenerator, self).__init__()
        self.num_points = num_points
        self.bottleneck_size = bottleneck_size
        self.n_primitives = n_primitives
        self.use_AdaIn = use_AdaIn
        self.hide_size = hide_size
        self.use_RecuRefine = use_RecuRefine
        self.conv1 = nn.Conv1d(3, 64, 1)

        # 定义解码器
        self.decoder = MViewPointNetDecode(
            num_points=self.num_points,
            n_primitives=self.n_primitives,
            bottleneck_size=self.bottleneck_size,
            use_AdaIn=self.use_AdaIn,
            use_SElayer=use_SElayer,
            decode=decode,
        )
        # 定义微调器
        self.refine = SpareNetRefine(
            num_points=self.num_points,
            n_primitives=self.n_primitives,
            use_SElayer=use_SElayer,
        )

    def forward(self, data, point_imgs, code="default"):
        partial_x = data["partial_cloud"]  # 拿不全的点云
        partial_x = partial_x.transpose(1, 2).contiguous()  # [bs, 3, in_points]    开始是[bs, in_points, 3]之后只转置2 3维得到前面的
        partial = partial_x  # [batch_size, 3, in_points]

        # decode
        outs,res_fake= self.decoder(point_imgs, code)                         # 这里的outs torch.Size([2, 3, 131072])

        outs = pointnet2_utils.gather_operation(outs.transpose(1, 2).contiguous(), pointnet2_utils.furthest_point_sample(outs, 16384))      # 这里的outs torch.Size([2, 3, 16384])
        coarse = outs.transpose(1,2).contiguous()  # [batch_size, out_points, 3]   # 再转置回来，contiguous()操作往往和转置相搭配 torch.Size([2, 16384, 3])

        # refine first time
        middle, loss_mst = self.refine(outs, partial, coarse)  # 部分点云和粗略点云做第一次微调，得到微调出的点云Yr1和其微调loss

        if self.use_RecuRefine == True:
            # refine second time
            outs_2 = middle.transpose(1, 2).contiguous()
            refine, _ = self.refine(outs_2, partial, middle)  # Yr1和部分点云做微调得到最终的生成结果Yr2即Y

            return coarse, middle, refine, loss_mst, res_fake  # 返回粗略点云，第一次微调点云，最终点云以及第一次微调loss
        else:
            return coarse, middle, None, loss_mst, res_fake

class InpaintingNetGenerator(nn.Module):
    """
    inputs:
    - data:
        -partical_cloud: b x npoints1 x num_dims
        -gtcloud: b x npoints2 x num_dims

    outputs:
    - coarse pcd: b x npoints2 x num_dims
    - middle pcd: b x npoints2 x num_dims
    - refine pcd: b x npoints2 x num_dims
    - loss_mst:
    """

    def __init__(
        self,
        n_primitives: int = 32,
        hide_size: int = 4096,
        bottleneck_size: int = 4096,
        num_points: int = 16382,
        use_SElayer: bool = False,
        use_RecuRefine: bool = False,
        use_AdaIn: str = "no_use",
        encode: str = "Pointfeat",
        decode: str = "Sparenet",
    ):
        super(InpaintingNetGenerator, self).__init__()
        self.num_points = num_points
        self.bottleneck_size = bottleneck_size
        self.n_primitives = n_primitives
        self.use_AdaIn = use_AdaIn
        self.hide_size = hide_size
        self.use_RecuRefine = use_RecuRefine
        self.conv1 = nn.Conv1d(3, 64, 1)

        # 定义解码器
        self.decoder = InpaintingNetDecode(
            num_points=self.num_points,
            n_primitives=self.n_primitives,
            bottleneck_size=self.bottleneck_size,
            use_AdaIn=self.use_AdaIn,
            use_SElayer=use_SElayer,
            decode=decode,
        )
        # 定义微调器
        self.refine = SpareNetRefine(
            num_points=self.num_points,
            n_primitives=self.n_primitives,
            use_SElayer=use_SElayer,
        )

    def forward(self, data, point_imgs, code="default"):
        partial_x = data["partial_cloud"]  # 拿不全的点云
        partial_x = partial_x.transpose(1, 2).contiguous()  # [bs, 3, in_points]    开始是[bs, in_points, 3]之后只转置2 3维得到前面的
        partial = partial_x  # [batch_size, 3, in_points]

        # decode
        outs,res_fake= self.decoder(point_imgs, code)                         # 这里的outs torch.Size([2, 3, 131072])
        if VISUALIZER_PC == True:
            plot_pcd_three_views(VIS_PATH_PC_3+'{}_all.jpg'.format(str(code[0])),outs.cpu(),"1","2")
            # img_te = get_ptcloud_img(outs.permute(0,2,1).cpu())
            # img_te = Image.fromarray(img_te)
            # img_te.save(VIS_PATH_PC+'{}_all.jpg'.format(str(code[0])))
        outs = pointnet2_utils.gather_operation(outs.transpose(1, 2).contiguous(), pointnet2_utils.furthest_point_sample(outs, 16384))      # 这里的outs torch.Size([2, 3, 16384])
        if VISUALIZER_PC == True:
            plot_pcd_three_views(VIS_PATH_PC_3 + '{}_coarse.jpg'.format(str(code[0])), outs.permute(0,2,1).cpu(), "1", "2")
            # img_te = get_ptcloud_img(outs.cpu())
            # img_te = Image.fromarray(img_te)
            # img_te.save(VIS_PATH_PC+'{}_coarse.jpg'.format(str(code[0])))


        coarse = outs.transpose(1,2).contiguous()  # [batch_size, out_points, 3]   # 再转置回来，contiguous()操作往往和转置相搭配 torch.Size([2, 16384, 3])

        # refine first time
        middle, loss_mst = self.refine(outs, partial, coarse)  # 部分点云和粗略点云做第一次微调，得到微调出的点云Yr1和其微调loss

        if VISUALIZER_PC == True:
            temp = middle.cpu().detach()
            plot_pcd_three_views(VIS_PATH_PC_3 + '{}_middle.jpg'.format(str(code[0])), temp, "1", "2")
            # img_te = get_ptcloud_img(temp)
            # img_te = Image.fromarray(img_te)
            # img_te.save(VIS_PATH_PC+'{}_middle.jpg'.format(str(code[0])))

            plot_pcd_three_views(VIS_PATH_PC_3 + '{}_partial.jpg'.format(str(code[0])), partial.permute(0,2,1).cpu(), "1", "2")
            # img_te = get_ptcloud_img(partial.cpu())
            # img_te = Image.fromarray(img_te)
            # img_te.save(VIS_PATH_PC + '{}_partial.jpg'.format(str(code[0])))

            plot_pcd_three_views(VIS_PATH_PC_3 + '{}_gt.jpg'.format(str(code[0])), data["gtcloud"].cpu(), "1", "2")
            # img_te = get_ptcloud_img(data["gtcloud"].permute(0,2,1).cpu())
            # img_te = Image.fromarray(img_te)
            # img_te.save(VIS_PATH_PC + '{}_gt.jpg'.format(str(code[0])))

        if self.use_RecuRefine == True:
            # refine second time
            outs_2 = middle.transpose(1, 2).contiguous()
            refine, _ = self.refine(outs_2, partial, middle)  # Yr1和部分点云做微调得到最终的生成结果Yr2即Y

            return coarse, middle, refine, loss_mst, res_fake  # 返回粗略点云，第一次微调点云，最终点云以及第一次微调loss
        else:
            return coarse, coarse, None, loss_mst, res_fake

class PCF2dNetGenerator(nn.Module):
    """
    inputs:
    - data:
        -partical_cloud: b x npoints1 x num_dims
        -gtcloud: b x npoints2 x num_dims

    outputs:
    - coarse pcd: b x npoints2 x num_dims
    - middle pcd: b x npoints2 x num_dims
    - refine pcd: b x npoints2 x num_dims
    - loss_mst:
    """

    def __init__(
        self,
        n_primitives: int = 32,
        hide_size: int = 4096,
        bottleneck_size: int = 4096,
        num_points: int = 16382,
        use_SElayer: bool = False,
        use_RecuRefine: bool = False,
        use_AdaIn: str = "no_use",
        encode: str = "Pointfeat",
        decode: str = "Sparenet",
    ):
        super(PCF2dNetGenerator, self).__init__()
        self.num_points = num_points
        self.bottleneck_size = bottleneck_size
        self.n_primitives = n_primitives
        self.use_AdaIn = use_AdaIn
        self.hide_size = hide_size
        self.use_RecuRefine = use_RecuRefine

        self.conv1 = nn.Conv1d(3, 64, 1)
        # 定义编码器
        self.encoder = SpareNetEncode(
            hide_size=self.hide_size,
            bottleneck_size=self.bottleneck_size,
            use_SElayer=use_SElayer,
            encode=encode,
        )
        # 定义解码器
        self.decoder = PCF2dNetDecode(
            num_points=self.num_points,
            n_primitives=self.n_primitives,
            bottleneck_size=self.bottleneck_size,
            use_AdaIn=self.use_AdaIn,
            use_SElayer=use_SElayer,
            decode=decode,
        )
        # 定义微调器
        self.refine = SpareNetRefine(
            num_points=self.num_points,
            n_primitives=self.n_primitives,
            use_SElayer=use_SElayer,
        )

    def forward(self, data, partial_imgs, gt_imgs,code="default"):
        partial_x = data["partial_cloud"]       # 拿不全的点云
        partial_x = partial_x.transpose(1, 2).contiguous()  # [bs, 3, in_points]    开始是[bs, in_points, 3]之后只转置2 3维得到前面的
        partial = partial_x  # [batch_size, 3, in_points]

        # encode
        style = self.encoder(partial_x)  # [batch_size, 1024]       # 输入到encoder得到1024维特征

        # decode
        outs, feature_loss = self.decoder(style, partial_x, partial_imgs, gt_imgs, code)                       # 用1024维特征使用风格解码器申城粗略点云 torch.Size([2, 3, 16384])
        if VISUALIZER == True:
            img_te = get_ptcloud_img(outs.cpu())
            img_te = Image.fromarray(img_te)
            img_te.save('./output/cartest/pc_all/{}_0.jpg'.format(str(code[0])))
        coarse = outs.transpose(1, 2).contiguous()  # [batch_size, out_points, 3]   # 再转置回来，contiguous()操作往往和转置相搭配 torch.Size([2, 16384, 3])


        # refine first time
        middle, loss_mst = self.refine(outs, partial, coarse)       # 部分点云和粗略点云做第一次微调，得到微调出的点云Yr1和其微调loss
        if VISUALIZER == True:
            img_te = get_ptcloud_img(middle.transpose(2, 1).contiguous().cpu())
            img_te = Image.fromarray(img_te)
            img_te.save('./output/cartest/pc_all/{}_1.jpg'.format(str(code[0])))

        if self.use_RecuRefine == True:
            # refine second time
            outs_2 = middle.transpose(1, 2).contiguous()
            refine, _ = self.refine(outs_2, partial, middle)            # Yr1和部分点云做微调得到最终的生成结果Yr2即Y

            return coarse, middle, refine, loss_mst, feature_loss                     # 返回粗略点云，第一次微调点云，最终点云以及第一次微调loss
        else:
            return coarse, middle, None, loss_mst, feature_loss                     # 返回粗略点云，第一次微调点云，最终点云以及第一次微调loss





class SpareNetEncode(nn.Module):
    """
    input
    - point_cloud: b x num_dims x npoints1

    output
    - feture:  one x feature_size
    """

    def __init__(
        self,
        bottleneck_size=4096,
        use_SElayer=False,
        encode="Pointfeat",
        hide_size=4096,
    ):
        super(SpareNetEncode, self).__init__()
        print(encode)
        if encode == "Residualnet":
            self.feat_extractor = EdgeConvResFeat(
                use_SElayer=use_SElayer, k=8, output_size=hide_size, hide_size=4096
            )
        else:
            self.feat_extractor = PointNetfeat(
                global_feat=True, use_SElayer=use_SElayer, hide_size=hide_size
            )
        self.linear = nn.Linear(hide_size, bottleneck_size)
        self.bn = nn.BatchNorm1d(bottleneck_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        # 提取3d特征
        x = self.feat_extractor(x)
        x = self.linear(x)
        x = self.bn(x)
        x = self.relu(x)

        return x




class EdgeConvResFeat(nn.Module):
    """
    input
    - point_cloud: b x num_dims x npoints1

    output
    - feture:  b x feature_size
    """

    def __init__(
        self,
        num_point: int = 16382,
        use_SElayer: bool = False,
        k: int = 8,
        hide_size: int = 2048,
        output_size: int = 4096,
    ):
        super(EdgeConvResFeat, self).__init__()
        self.use_SElayer = use_SElayer
        self.k = k
        self.hide_size = hide_size
        self.output_size = output_size

        self.conv1 = nn.Conv2d(6, self.hide_size // 16, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(
            self.hide_size // 8, self.hide_size // 16, kernel_size=1, bias=False
        )
        self.conv3 = nn.Conv2d(
            self.hide_size // 8, self.hide_size // 8, kernel_size=1, bias=False
        )
        self.conv4 = nn.Conv2d(
            self.hide_size // 4, self.hide_size // 4, kernel_size=1, bias=False
        )
        self.conv5 = nn.Conv1d(
            self.hide_size // 2, self.output_size // 2, kernel_size=1, bias=False
        )

        self.relu1 = nn.LeakyReLU(negative_slope=0.2)
        self.relu2 = nn.LeakyReLU(negative_slope=0.2)
        self.relu3 = nn.LeakyReLU(negative_slope=0.2)
        self.relu4 = nn.LeakyReLU(negative_slope=0.2)
        self.relu5 = nn.LeakyReLU(negative_slope=0.2)

        if use_SElayer:
            self.se1 = SELayer(channel=self.hide_size // 16)
            self.se2 = SELayer(channel=self.hide_size // 16)
            self.se3 = SELayer(channel=self.hide_size // 8)
            self.se4 = SELayer(channel=self.hide_size // 4)

        self.bn1 = nn.BatchNorm2d(self.hide_size // 16)
        self.bn2 = nn.BatchNorm2d(self.hide_size // 16)
        self.bn3 = nn.BatchNorm2d(self.hide_size // 8)
        self.bn4 = nn.BatchNorm2d(self.hide_size // 4)
        self.bn5 = nn.BatchNorm1d(self.output_size // 2)

        # 再第2，3，4个CAE中的输入直接到输出的Linear层，也就是F3
        self.resconv1 = nn.Conv1d(
            self.hide_size // 16, self.hide_size // 16, kernel_size=1, bias=False
        )
        self.resconv2 = nn.Conv1d(
            self.hide_size // 16, self.hide_size // 8, kernel_size=1, bias=False
        )
        self.resconv3 = nn.Conv1d(
            self.hide_size // 8, self.hide_size // 4, kernel_size=1, bias=False
        )

    def forward(self, x):
        # x : [bs, 3, num_points]   torch.Size([2, 3, 3000])
        batch_size = x.size(0)      # batch_size = 2
        if self.use_SElayer:
            x = get_graph_feature(x, k=self.k)      # 计算边的特征 结果为：torch.Size([2, 6, 3000, 8])   [2, 3, 3000, 8]和[2, 3, 3000, 8]相cat，前面是qij-pi后面是pi
            x = self.relu1(self.se1(self.bn1(self.conv1(x))))   # conv1是CAE模块的第一个MLP，即F1，然后进入自注意力，然后外层relu
            x1 = x.max(dim=-1, keepdim=False)[0]    # 最后的最大池化

            x2_res = self.resconv1(x1)              # F3
            x = get_graph_feature(x1, k=self.k)     # KNN模块
            x = self.relu2(self.se2(self.bn2(self.conv2(x))))
            x2 = x.max(dim=-1, keepdim=False)[0]
            x2 = x2 + x2_res

            x3_res = self.resconv2(x2)
            x = get_graph_feature(x2, k=self.k)
            x = self.relu3(self.se3(self.bn3(self.conv3(x))))
            x3 = x.max(dim=-1, keepdim=False)[0]
            x3 = x3 + x3_res

            x4_res = self.resconv3(x3)              # 最后一个CAE的残差
            x = get_graph_feature(x3, k=self.k)     # knn
            x = self.relu4(self.se4(self.bn4(self.conv4(x))))
        else:
            x = get_graph_feature(x, k=self.k)
            x = self.relu1(self.bn1(self.conv1(x)))
            x1 = x.max(dim=-1, keepdim=False)[0]

            x2_res = self.resconv1(x1)
            x = get_graph_feature(x1, k=self.k)
            x = self.relu2(self.bn2(self.conv2(x)))
            x2 = x.max(dim=-1, keepdim=False)[0]
            x2 = x2 + x2_res

            x3_res = self.resconv2(x2)
            x = get_graph_feature(x2, k=self.k)
            x = self.relu3(self.bn3(self.conv3(x)))
            x3 = x.max(dim=-1, keepdim=False)[0]
            x3 = x3 + x3_res

            x4_res = self.resconv3(x3)
            x = get_graph_feature(x3, k=self.k)
            x = self.relu4(self.bn4(self.conv4(x)))

        x4 = x.max(dim=-1, keepdim=False)[0]    # 最后一层的最大池化
        x4 = x4 + x4_res                        # 残差和特征加起来

        x = torch.cat((x1, x2, x3, x4), dim=1)  # torch.Size([2, 256, 3000]) torch.Size([2, 256, 3000]) torch.Size([2, 512, 3000]) torch.Size([2, 1024, 3000]) 合起来为torch.Size([2, 2048, 3000])
        x = self.relu5(self.bn5(self.conv5(x))) # conv5 bn5 relu5合起来构成encoder最终的MLP  torch.Size([2, 2048, 3000])用MLP转成torch.Size([2, 2048, 3000])

        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)  # [bs, 2048]    最大池化，3000个点的特征合成一个点的特征
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)  # [bs, 2048]    平均池化
        x = torch.cat((x1, x2), 1)  # 两个[bs, 2048]转成[bs, 4096]

        x = x.view(-1, self.output_size)
        return x


class PointNetfeat(nn.Module):
    """
    input
    - point_cloud： b x num_dims x npoints_1

    output
    - feture:  b x feature_size
    """

    def __init__(
        self, num_points=16382, global_feat=True, use_SElayer=False, hide_size=4096
    ):
        super(PointNetfeat, self).__init__()
        self.use_SElayer = use_SElayer
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, hide_size, 1)
        self.hide_size = hide_size
        if use_SElayer:
            self.se1 = SELayer1D(channel=64)
            self.se2 = SELayer1D(channel=128)

        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(hide_size)

        self.num_points = num_points
        self.global_feat = global_feat

    def forward(self, x):
        batchsize = x.size()[0]  # x: [batch_size, 3, num_points]
        if self.use_SElayer:
            x = F.relu(self.se1(self.bn1(self.conv1(x))))
            x = F.relu(self.se2(self.bn2(self.conv2(x))))
            x = self.bn3(self.conv3(x))
        else:
            x = F.relu(self.bn1(self.conv1(x)))  # x: [batch_size, 64, num_points]
            x = F.relu(self.bn2(self.conv2(x)))  # x: [batch_size, 128, num_points]
            x = self.bn3(self.conv3(x))  # x: [batch_size, 1024, num_points]
        x, _ = torch.max(x, 2)  # x: [batch_size, num_points]
        x = x.view(-1, self.hide_size)
        return x


class SpareNetDecode(nn.Module):
    """
    inputs:
    - style(feature): b x feature_size

    outputs:
    - out: b x num_dims x num_points
    """

    def __init__(
        self,
        num_points: int = 16382,
        n_primitives: int = 32,
        bottleneck_size: int = 4096,
        use_AdaIn: str = "no_use",
        decode="Sparenet",
        use_SElayer: bool = False,
    ):
        super(SpareNetDecode, self).__init__()
        self.use_AdaIn = use_AdaIn
        self.num_points = num_points
        self.n_primitives = n_primitives
        self.bottleneck_size = bottleneck_size
        self.decode = decode
        # 这里share的意思是那32个生成粗略点云的网络是用的同一个网络
        # 我这里修改一下，原本是32个生成有512个粗略点的点云生成网络。改成8个生成2048个粗略点的点云生成网络

        # 思路：使用一个点云的8张深度图来生成粗略点云，使用一个共享的网络，类似于上面的share模式，这个网络以一张深度图(batch*1*256*256)作为输入,输出为batch*2048*3
        # 网络结构为类似unet，
        # 定义一个二维特征提取网络，使用与UNet相同的编码器.将输入batch*1*256*256变为batch*2048*1*1。这里不能把unet定义到StyleBasedMViewNet，这样做会占大量内存
        # 定义编码器
        # self.mviewnet_encoder = MViewEncoder(3, 32, norm_layer=functools.partial(nn.InstanceNorm2d, affine=True,track_running_stats=False),use_spectral_norm=False)
        self.unet_encoder = UnetEncoder(3, 64, norm_layer=functools.partial(nn.InstanceNorm2d, affine=True,
                                                                            track_running_stats=False),
                                        use_spectral_norm=False)
        self.decoder = nn.ModuleList(
            [
                StyleBasedAdaIn(
                    input_dim=4,  # 这里从2改为1是说原本输入是随机采样的n*2现在改为图片经过多层卷积后得到的二维特征n*1
                    style_dim=self.bottleneck_size,
                    # 这里之前只是将self.bottleneck_size赋值给了style_dim，并没有赋给bottleneck_size,不赋的话用默认的1026
                    use_SElayer=use_SElayer,
                )
                for i in range(self.n_primitives)
            ]
        )

        # MLP to generate AdaIN parameters
        self.mlp = nn.Sequential(
            nn.Linear(self.bottleneck_size, self.bottleneck_size),
            nn.ReLU(),
            nn.Linear(self.bottleneck_size, get_num_adain_params(self.decoder[0])),
        )

    def forward(self, style, point_imgs, code="default"):
        outs = []
        adain_params = self.mlp(style)
        for i in range(self.n_primitives):
            point_img = point_imgs[:, i, :, :, :]  # torch.Size([8, 3, 256, 256])
            # 这里的regular_grid应该改为对应的深度图 size为batch*1*256*256
            # 将输入batch*1*256*256变为batch*512*1*1,再将batch*512*1*1变为batch*1*512作为dec的输入

            x = self.unet_encoder(point_img)
            x = x.view(x.size(0), x.size(1), x.size(2) * x.size(3))
            # x每个元素的取值由于输出时使用sigmoid进行激活，变为[0,1]之间，所以这里要将[0,1]变为[-1,1]
            x = ((x - 0.5) * 2).contiguous()
            # 之前x是torch.Size([2, 2048, 1, 1])需要变为torch.Size([2, 1, 512])
            temp_pc = self.decoder[i](x.view(x.size(0), x.size(2), x.size(1)), adain_params)
            # 在这里进行每个点云的可视化
            if VISUALIZER == True:
                img_te = get_ptcloud_img(temp_pc.cpu())
                img_te = Image.fromarray(img_te)
                img_te.save('./output/cartest/pc/{}_{}.jpg'.format(str(code[0]), str(i)))
                # vutils.save_image(img[0, :, :, :], './output/cartest/gt/{}_{}.jpg'.format(str(code[0]), str(i)), normalize=True)
            outs.append(temp_pc)  # 将整个decoder的输入torch.Size([2, 2, 512])输入到第i个decoder中。主要在self.decoder中进行处理

        return torch.cat(outs, 2).contiguous()

class MViewPointNetDecode(nn.Module):
    """
    inputs:
    - style(feature): b x feature_size

    outputs:
    - out: b x num_dims x num_points
    """

    def __init__(
        self,
        num_points: int = 16382,
        n_primitives: int = 32,
        bottleneck_size: int = 4096,
        use_AdaIn: str = "no_use",
        decode="Sparenet",
        use_SElayer: bool = False,
    ):
        super(MViewPointNetDecode, self).__init__()
        self.use_AdaIn = use_AdaIn
        self.num_points = num_points
        self.n_primitives = n_primitives
        self.bottleneck_size = bottleneck_size
        self.decode = decode
        self.criterionL1_loss = torch.nn.L1Loss()
        # 这里share的意思是那32个生成粗略点云的网络是用的同一个网络
        # 我这里修改一下，原本是32个生成有512个粗略点的点云生成网络。改成8个生成2048个粗略点的点云生成网络

        # 思路：使用一个点云的8张深度图来生成粗略点云，使用一个共享的网络，类似于上面的share模式，这个网络以一张深度图(batch*1*256*256)作为输入,输出为batch*2048*3
        # 网络结构为类似unet，
        # 定义一个二维特征提取网络，使用与UNet相同的编码器.将输入batch*1*256*256变为batch*2048*1*1。这里不能把unet定义到StyleBasedMViewNet，这样做会占大量内存
        # 定义编码器
        # self.mviewnet_encoder = MViewEncoder(3, 32, norm_layer=functools.partial(nn.InstanceNorm2d, affine=True,track_running_stats=False),use_spectral_norm=False)
        self.unet = EasyUnetGenerator(3,3,64, norm_layer=functools.partial(nn.InstanceNorm2d, affine=True,track_running_stats=False),use_spectral_norm=False)
        self.decoder = nn.ModuleList(
            [
                StyleBasedAdaIn(
                    input_dim=4,  # 这里从2改为1是说原本输入是随机采样的n*2现在改为图片经过多层卷积后得到的二维特征n*1
                    style_dim=self.bottleneck_size,
                    # 这里之前只是将self.bottleneck_size赋值给了style_dim，并没有赋给bottleneck_size,不赋的话用默认的1026
                    use_SElayer=use_SElayer,
                )
                for i in range(self.n_primitives)
            ]
        )

        # MLP to generate AdaIN parameters
        self.mlp = nn.Sequential(
            nn.Linear(self.bottleneck_size, self.bottleneck_size),
            nn.ReLU(),
            nn.Linear(self.bottleneck_size, get_num_adain_params(self.decoder[0])),
        )

    def forward(self, point_imgs, code="default"):
        point_img = point_imgs[:, 0, :, :, :]
        fake_pointmap, _ = self.unet(point_img)
        res_fake = fake_pointmap
        outs = torch.squeeze(fake_pointmap,1)
        outs = outs.permute(0,2,3,1)
        outs = torch.flatten(outs, start_dim=1,end_dim=2)

        for i in range(1, self.n_primitives):
            point_img = point_imgs[:, i, :, :, :]
            fake_pointmap, _ = self.unet(point_img)
            res_fake = torch.cat((res_fake, fake_pointmap), dim=1)
            outs_temp = torch.squeeze(fake_pointmap, 1)
            outs_temp = outs_temp.permute(0, 2, 3, 1)
            outs_temp = torch.flatten(outs_temp, start_dim=1, end_dim=2)
            outs = torch.cat((outs, outs_temp), dim=1)
        return outs, res_fake

class InpaintingNetDecode(nn.Module):
    """
    inputs:
    - style(feature): b x feature_size

    outputs:
    - out: b x num_dims x num_points
    """

    def __init__(
        self,
        num_points: int = 16382,
        n_primitives: int = 32,
        bottleneck_size: int = 4096,
        use_AdaIn: str = "no_use",
        decode="Sparenet",
        use_SElayer: bool = False,
    ):
        super(InpaintingNetDecode, self).__init__()
        self.use_AdaIn = use_AdaIn
        self.num_points = num_points
        self.n_primitives = n_primitives
        self.bottleneck_size = bottleneck_size
        self.decode = decode
        self.criterionL1_loss = torch.nn.L1Loss()
        # 这里share的意思是那32个生成粗略点云的网络是用的同一个网络
        # 我这里修改一下，原本是32个生成有512个粗略点的点云生成网络。改成8个生成2048个粗略点的点云生成网络

        # 思路：使用一个点云的8张深度图来生成粗略点云，使用一个共享的网络，类似于上面的share模式，这个网络以一张深度图(batch*1*256*256)作为输入,输出为batch*2048*3
        # 网络结构为类似unet，
        # 定义一个二维特征提取网络，使用与UNet相同的编码器.将输入batch*1*256*256变为batch*2048*1*1。这里不能把unet定义到StyleBasedMViewNet，这样做会占大量内存
        # 定义编码器
        # self.mviewnet_encoder = MViewEncoder(3, 32, norm_layer=functools.partial(nn.InstanceNorm2d, affine=True,track_running_stats=False),use_spectral_norm=False)
        # self.unet = EasyUnetGenerator(3,3,64, norm_layer=functools.partial(nn.InstanceNorm2d, affine=True,track_running_stats=False),use_spectral_norm=False)
        # self.decoder = nn.ModuleList(
        #     [
        #         StyleBasedAdaIn(
        #             input_dim=4,  # 这里从2改为1是说原本输入是随机采样的n*2现在改为图片经过多层卷积后得到的二维特征n*1
        #             style_dim=self.bottleneck_size,
        #             # 这里之前只是将self.bottleneck_size赋值给了style_dim，并没有赋给bottleneck_size,不赋的话用默认的1026
        #             use_SElayer=use_SElayer,
        #         )
        #         for i in range(self.n_primitives)
        #     ]
        # )
        #
        # # MLP to generate AdaIN parameters
        # self.mlp = nn.Sequential(
        #     nn.Linear(self.bottleneck_size, self.bottleneck_size),
        #     nn.ReLU(),
        #     nn.Linear(self.bottleneck_size, get_num_adain_params(self.decoder[0])),
        # )

    def forward(self, point_imgs, code="default"):
        point_img = point_imgs[:, 0, :, :, :]
        # fake_pointmap, _ = self.unet(point_img)
        fake_pointmap = point_img
        res_fake = fake_pointmap
        outs = torch.squeeze(fake_pointmap,1)
        outs = outs.permute(0,2,3,1)
        outs = torch.flatten(outs, start_dim=1,end_dim=2)

        for i in range(1, self.n_primitives):
            point_img = point_imgs[:, i, :, :, :]
            # fake_pointmap, _ = self.unet(point_img)
            fake_pointmap= point_img
            res_fake = torch.cat((res_fake, fake_pointmap), dim=1)
            outs_temp = torch.squeeze(fake_pointmap, 1)
            outs_temp = outs_temp.permute(0, 2, 3, 1)
            outs_temp = torch.flatten(outs_temp, start_dim=1, end_dim=2)
            outs = torch.cat((outs, outs_temp), dim=1)
        return outs, res_fake


class MViewNetDecode(nn.Module):
    """
    inputs:
    - style(feature): b x feature_size
    outputs:
    - out: b x num_dims x num_points
    """

    def __init__(
        self,
        num_points: int = 16382,
        n_primitives: int = 32,
        bottleneck_size: int = 4096,
        use_AdaIn: str = "no_use",
        decode="Sparenet",
        use_SElayer: bool = False,
    ):
        super(MViewNetDecode, self).__init__()
        self.use_AdaIn = use_AdaIn
        self.num_points = num_points
        self.n_primitives = n_primitives
        self.bottleneck_size = bottleneck_size
        self.decode = decode
        # 这里share的意思是那32个生成粗略点云的网络是用的同一个网络
        # 我这里修改一下，原本是32个生成有512个粗略点的点云生成网络。改成8个生成2048个粗略点的点云生成网络
        if decode == "Sparenet":
            # 只有原始的Sparenet才需要self.grid
            self.grid = grid_generation(self.num_points, self.n_primitives)     # 生成n_primitives个2d网格，二维列表，有32个元素，每个元素又是有512个[x,y]的列表
            if use_AdaIn == "share":
                self.decoder = nn.ModuleList(
                    [
                        StyleBasedAdaIn(
                            input_dim=2,
                            style_dim=self.bottleneck_size,
                            use_SElayer=use_SElayer,
                        )
                        for i in range(self.n_primitives)
                    ]
                )           # 生成32个StyleBasedAdaIn

                # MLP to generate AdaIN parameters
                self.mlp = nn.Sequential(
                    nn.Linear(self.bottleneck_size, self.bottleneck_size),
                    nn.ReLU(),
                    nn.Linear(self.bottleneck_size, get_num_adain_params(self.decoder[0])),
                )
            elif use_AdaIn == "no_share":
                self.decoder = nn.ModuleList(
                    [
                        AdaInPointGenCon(
                            input_dim=2,
                            style_dim=self.bottleneck_size,
                            use_SElayer=use_SElayer,
                        )
                        for i in range(self.n_primitives)
                    ]
                )

            elif use_AdaIn == "no_use":
                self.decoder = nn.ModuleList(
                    [
                        PointGenCon(
                            input_dim=2 + self.bottleneck_size, use_SElayer=use_SElayer
                        )
                        for i in range(self.n_primitives)
                    ]
                )
        elif decode == "Mviewnet":
            self.unet = EasyUnetGenerator(3,3,64, norm_layer=functools.partial(nn.InstanceNorm2d, affine=True,track_running_stats=False),use_spectral_norm=False)
            self.decoder = nn.ModuleList(
                [
                    StyleBasedAdaIn(
                        input_dim=3,        # 这里从2改为1是说原本输入是随机采样的n*2现在改为图片经过多层卷积后得到的二维特征n*1
                        style_dim=self.bottleneck_size,     # 这里之前只是将self.bottleneck_size赋值给了style_dim，并没有赋给bottleneck_size,不赋的话用默认的1026
                        use_SElayer=use_SElayer,
                    )
                    for i in range(self.n_primitives)
                ]
            )

            # MLP to generate AdaIN parameters
            self.mlp = nn.Sequential(
                nn.Linear(self.bottleneck_size, self.bottleneck_size),
                nn.ReLU(),
                nn.Linear(self.bottleneck_size, get_num_adain_params(self.decoder[0])),
            )
        else:
            print("error, decode not exit")
            exit()


    def forward(self, style, point_imgs, code="default"):
        outs = []
        fake_maps = []
        dec_inputs = []
        adain_params = self.mlp(style)
        for i in range(self.n_primitives):
            point_img = point_imgs[:, i, :, :, :]  # torch.Size([8, 3, 256, 256])
            # 这里的regular_grid应该改为对应的深度图 size为batch*1*256*256
            # 将输入batch*1*256*256变为batch*512*1*1,再将batch*512*1*1变为batch*1*512作为dec的输入

            fake_img = self.unet(point_img)
            # 将fake_img转为点云 最远点采样2048个点作为输入，对8个输入拼起来和gt用cd距离来约束，对fake_img用l1距离来约束
            dec_input = torch.flatten(torch.squeeze(fake_img,1).permute(0,2,3,1),start_dim=1,end_dim=2).contiguous()  # b n c
            dec_input = pointnet2_utils.gather_operation(dec_input.transpose(1, 2).contiguous(), pointnet2_utils.furthest_point_sample(dec_input, 16384//self.n_primitives))

            # 之前x是torch.Size([2, 2048, 1, 1])需要变为torch.Size([2, 1, 512])
            temp_pc = self.decoder[i](dec_input, adain_params)
            # 在这里进行每个点云的可视化
            if VISUALIZER == True:
                img_te = get_ptcloud_img(temp_pc.cpu())
                img_te = Image.fromarray(img_te)
                img_te.save('./output/cartest/pc/{}_{}.jpg'.format(str(code[0]), str(i)))
                # vutils.save_image(img[0, :, :, :], './output/cartest/gt/{}_{}.jpg'.format(str(code[0]), str(i)), normalize=True)
            outs.append(temp_pc)  # 将整个decoder的输入torch.Size([2, 2, 512])输入到第i个decoder中。主要在self.decoder中进行处理
            fake_maps.append(fake_img)
            dec_inputs.append(dec_input)

        return torch.cat(outs, 2).contiguous(), torch.cat(fake_maps, 1).contiguous(), torch.cat(dec_inputs,2).contiguous().permute(0, 2, 1)

class PCF2dNetDecode(nn.Module):
    """
    inputs:
    - style(feature): b x feature_size

    outputs:
    - out: b x num_dims x num_points
    """

    def __init__(
        self,
        num_points: int = 16382,
        n_primitives: int = 32,
        bottleneck_size: int = 4096,
        use_AdaIn: str = "no_use",
        decode="Sparenet",
        use_SElayer: bool = False,
    ):
        super(PCF2dNetDecode, self).__init__()
        self.use_AdaIn = use_AdaIn
        self.num_points = num_points
        self.n_primitives = n_primitives
        self.bottleneck_size = bottleneck_size
        self.decode = decode
        # 这里share的意思是那32个生成粗略点云的网络是用的同一个网络
        # 我这里修改一下，原本是32个生成有512个粗略点的点云生成网络。改成8个生成2048个粗略点的点云生成网络
        if decode == "Sparenet":
            # 只有原始的Sparenet才需要self.grid
            self.grid = grid_generation(self.num_points, self.n_primitives)     # 生成n_primitives个2d网格，二维列表，有32个元素，每个元素又是有512个[x,y]的列表
            if use_AdaIn == "share":
                self.decoder = nn.ModuleList(
                    [
                        StyleBasedAdaIn(
                            input_dim=2,
                            style_dim=self.bottleneck_size,
                            use_SElayer=use_SElayer,
                        )
                        for i in range(self.n_primitives)
                    ]
                )           # 生成32个StyleBasedAdaIn

                # MLP to generate AdaIN parameters
                self.mlp = nn.Sequential(
                    nn.Linear(self.bottleneck_size, self.bottleneck_size),
                    nn.ReLU(),
                    nn.Linear(self.bottleneck_size, get_num_adain_params(self.decoder[0])),
                )
            elif use_AdaIn == "no_share":
                self.decoder = nn.ModuleList(
                    [
                        AdaInPointGenCon(
                            input_dim=2,
                            style_dim=self.bottleneck_size,
                            use_SElayer=use_SElayer,
                        )
                        for i in range(self.n_primitives)
                    ]
                )

            elif use_AdaIn == "no_use":
                self.decoder = nn.ModuleList(
                    [
                        PointGenCon(
                            input_dim=2 + self.bottleneck_size, use_SElayer=use_SElayer
                        )
                        for i in range(self.n_primitives)
                    ]
                )
        elif decode == "Mviewnet":
            # 思路：使用一个点云的8张深度图来生成粗略点云，使用一个共享的网络，类似于上面的share模式，这个网络以一张深度图(batch*1*256*256)作为输入,输出为batch*2048*3
            # 网络结构为类似unet，
            # 定义一个二维特征提取网络，使用与UNet相同的编码器.将输入batch*1*256*256变为batch*2048*1*1。这里不能把unet定义到StyleBasedMViewNet，这样做会占大量内存
            self.unet_gan_encoder = UnetGanEncoder(1,64,norm_layer=functools.partial(nn.InstanceNorm2d, affine=True,track_running_stats=False),use_spectral_norm=False)

            self.decoder = nn.ModuleList(
                [
                    StyleBasedAdaIn(
                        input_dim=4,        # 这里从2改为1是说原本输入是随机采样的n*2现在改为图片经过多层卷积后得到的二维特征n*1
                        style_dim=self.bottleneck_size,     # 这里之前只是将self.bottleneck_size赋值给了style_dim，并没有赋给bottleneck_size,不赋的话用默认的1026
                        use_SElayer=use_SElayer,
                    )
                    for i in range(self.n_primitives)
                ]
            )

            # MLP to generate AdaIN parameters
            self.mlp = nn.Sequential(
                nn.Linear(self.bottleneck_size, self.bottleneck_size),
                nn.ReLU(),
                nn.Linear(self.bottleneck_size, get_num_adain_params(self.decoder[0])),
            )
        else:
            print("error, decode not exit")
            exit()


    def forward(self, style, partial_x, partial_imgs, gt_imgs, code="default"):
        outs = []
        feature_loss = 0
        if self.decode == "Sparenet":
            if self.use_AdaIn == "share":
                adain_params = self.mlp(style)
                for i in range(self.n_primitives):
                    regular_grid = torch.cuda.FloatTensor(self.grid[i])                 # 获取第i个网格并且将其挂到cuda上 list[512]变为torch.Size([512, 2])
                    regular_grid = regular_grid.transpose(0, 1).contiguous().unsqueeze(0)   # 变为了torch.Size([1, 2, 512])
                    regular_grid = regular_grid.expand(                                     # partial_x.size(0)是batch
                        partial_x.size(0), regular_grid.size(1), regular_grid.size(2)
                    ).contiguous()                                                          # 变为了torch.Size([2, 2, 512])
                    regular_grid = ((regular_grid - 0.5) * 2).contiguous()                  # 把值域[0,1]变为了[-1,1]

                    outs.append(self.decoder[i](regular_grid, adain_params))     # 将整个decoder的输入torch.Size([2, 2, 512])输入到第i个decoder中。主要在self.decoder中进行处理
            elif self.use_AdaIn == "no_share":
                for i in range(self.n_primitives):
                    regular_grid = torch.cuda.FloatTensor(self.grid[i])
                    regular_grid = regular_grid.transpose(0, 1).contiguous().unsqueeze(0)
                    regular_grid = regular_grid.expand(
                        partial_x.size(0), regular_grid.size(1), regular_grid.size(2)
                    ).contiguous()
                    regular_grid = ((regular_grid - 0.5) * 2).contiguous()

                    outs.append(self.decoder[i](regular_grid, style))
            elif self.use_AdaIn == "no_use":
                for i in range(self.n_primitives):
                    regular_grid = torch.cuda.FloatTensor(self.grid[i])
                    regular_grid = regular_grid.transpose(0, 1).contiguous().unsqueeze(0)
                    regular_grid = regular_grid.expand(
                        partial_x.size(0), regular_grid.size(1), regular_grid.size(2)
                    ).contiguous()
                    regular_grid = ((regular_grid - 0.5) * 2).contiguous()

                    y = (
                        style.unsqueeze(2)
                        .expand(style.size(0), style.size(1), regular_grid.size(2))
                        .contiguous()
                    )
                    y = torch.cat((regular_grid, y), 1).contiguous()
                    outs.append(self.decoder[i](y))
        elif self.decode == "Mviewnet":
            adain_params = self.mlp(style)
            for i in range(self.n_primitives):
                partial_img = partial_imgs[:,i,:,:].view(partial_imgs.size(0),1,partial_imgs.size(2),partial_imgs.size(3))
                gt_img = gt_imgs[:,i,:,:].view(gt_imgs.size(0),1,gt_imgs.size(2),gt_imgs.size(3))
                # 这里的regular_grid应该改为对应的深度图 size为batch*1*256*256
                # 将输入batch*1*256*256变为batch*512*1*1,再将batch*512*1*1变为batch*1*512作为dec的输入
                x, _loss = self.unet_gan_encoder(partial_img,gt_img)
                feature_loss += _loss
                x = x.view(x.size(0), x.size(1), x.size(2) * x.size(3))
                # x每个元素的取值由于输出时使用sigmoid进行激活，变为[0,1]之间，所以这里要将[0,1]变为[-1,1]
                x = ((x - 0.5) * 2).contiguous()
                # 之前x是torch.Size([2, 2048, 1, 1])需要变为torch.Size([2, 1, 512])
                temp_pc = self.decoder[i](x.view(x.size(0),x.size(2),x.size(1)), adain_params)
                # 在这里进行每个点云的可视化
                if VISUALIZER == True:
                    img_te = get_ptcloud_img(temp_pc.cpu())
                    img_te = Image.fromarray(img_te)
                    img_te.save('./output/cartest/pc/{}_{}.jpg'.format(str(code[0]), str(i)))
                    vutils.save_image(partial_img[0, :, :, :], './output/cartest/gt/{}_{}.jpg'.format(str(code[0]), str(i)), normalize=True)
                outs.append(temp_pc)     # 将整个decoder的输入torch.Size([2, 2, 512])输入到第i个decoder中。主要在self.decoder中进行处理

        return torch.cat(outs, 2).contiguous(), feature_loss

class StyleBasedMViewNet(nn.Module):
    """
    inputs:
    - content: b x (x,y) x (num_points / nb_primitives)
    - style(feature): b x feature_size
    - adain_params: b x parameter_size

    outputs:
    - out: b x num_dims x (num_points / nb_primitives)
    """

    def __init__(
            self,
            input_dim: int = 1026,
            style_dim: int = 1024,
            bottleneck_size: int = 1026,
            use_SElayer: bool = False,
    ):
        super(StyleBasedMViewNet, self).__init__()
        self.bottleneck_size = bottleneck_size
        self.input_dim = input_dim
        self.style_dim = style_dim
        self.dec = MViewDecoder(
            self.input_dim, self.bottleneck_size, use_SElayer=use_SElayer
        )

    def forward(self, content, adain_params):
        assign_adain_params(adain_params, self.dec)
        return self.dec(content)

class StyleBasedAdaIn(nn.Module):
    """
    inputs:
    - content: b x (x,y) x (num_points / nb_primitives)
    - style(feature): b x feature_size
    - adain_params: b x parameter_size

    outputs:
    - out: b x num_dims x (num_points / nb_primitives)
    """

    def __init__(
        self,
        input_dim: int = 1026,
        style_dim: int = 1024,
        bottleneck_size: int = 1026,
        use_SElayer: bool = False,
    ):
        super(StyleBasedAdaIn, self).__init__()
        self.bottleneck_size = bottleneck_size
        self.input_dim = input_dim
        self.style_dim = style_dim
        self.dec = MViewDecoder(
            self.input_dim, self.bottleneck_size, use_SElayer=use_SElayer
        )

    def forward(self, content, adain_params):
        assign_adain_params(adain_params, self.dec)
        return self.dec(content)


class AdaInPointGenCon(nn.Module):
    """
    inputs:
    - content: b x (x,y) x (num_points / nb_primitives)
    - style(feature): b x feature_size

    outputs:
    - out: b x num_dims x (num_points / nb_primitives)
    """

    def __init__(
        self,
        input_dim: int = 1026,
        style_dim: int = 1024,
        bottleneck_size: int = 1026,
        use_SElayer: bool = False,
    ):
        super(AdaInPointGenCon, self).__init__()
        self.bottleneck_size = bottleneck_size
        self.input_dim = input_dim
        self.style_dim = style_dim
        self.dec = GridDecoder(
            self.input_dim, self.bottleneck_size, use_SElayer=use_SElayer
        )

        # MLP to generate AdaIN parameters
        self.mlp = nn.Sequential(
            nn.Linear(self.style_dim, self.style_dim),
            nn.ReLU(),
            nn.Linear(self.style_dim, get_num_adain_params(self.dec)),
        )

    def forward(self, content, style):
        adain_params = self.mlp(style)
        assign_adain_params(adain_params, self.dec)
        return self.dec(content)


class PointGenCon(nn.Module):
    """
    inputs:
    - content: b x (x,y) x (num_points / nb_primitives)

    outputs:
    - out: b x num_dims x (num_points / nb_primitives)
    """

    def __init__(
        self,
        input_dim: int = 4098,
        bottleneck_size: int = 1026,
        use_SElayer: bool = False,
        dropout: bool = False,
    ):
        self.input_dim = input_dim
        self.bottleneck_size = bottleneck_size
        super(PointGenCon, self).__init__()
        self.conv1 = torch.nn.Conv1d(self.input_dim, self.bottleneck_size, 1)
        self.conv2 = torch.nn.Conv1d(self.bottleneck_size, self.bottleneck_size // 2, 1)
        self.conv3 = torch.nn.Conv1d(
            self.bottleneck_size // 2, self.bottleneck_size // 4, 1
        )
        self.conv4 = torch.nn.Conv1d(self.bottleneck_size // 4, 3, 1)

        self.use_SElayer = use_SElayer
        if self.use_SElayer:
            self.se1 = SELayer1D(channel=self.bottleneck_size)
            self.se2 = SELayer1D(channel=self.bottleneck_size // 2)
            self.se3 = SELayer1D(channel=self.bottleneck_size // 4)

        self.th = nn.Tanh()
        self.bn1 = torch.nn.BatchNorm1d(self.bottleneck_size)
        self.bn2 = torch.nn.BatchNorm1d(self.bottleneck_size // 2)
        self.bn3 = torch.nn.BatchNorm1d(self.bottleneck_size // 4)
        self.dropout = dropout
        if self.dropout:
            self.drop1 = nn.Dropout(0.4)
            self.drop2 = nn.Dropout(0.4)
            self.drop3 = nn.Dropout(0.4)

    def forward(self, x):
        if self.use_SElayer:
            x = F.relu(self.se1(self.bn1(self.conv1(x))))
        else:
            x = F.relu(self.bn1(self.conv1(x)))
        if self.dropout:
            x = self.drop1(x)

        if self.use_SElayer:
            x = F.relu(self.se2(self.bn2(self.conv2(x))))
        else:
            x = F.relu(self.bn2(self.conv2(x)))
        if self.dropout:
            x = self.drop2(x)

        if self.use_SElayer:
            x = F.relu(self.se3(self.bn3(self.conv3(x))))
        else:
            x = F.relu(self.bn3(self.conv3(x)))
        if self.dropout:
            x = self.drop3(x)
        x = self.conv4(x)  # [batch_size, 3, 512] 3 features(position) for 512 points
        return x


class SpareNetRefine(nn.Module):
    """
    inputs:
    - inps: b x npoints2 x num_dims
    - partial: b x npoints1 x num_dims
    - coarse: b x num_dims x npoints2

    outputs:
    - refine_result: b x num_dims x npoints2
    - loss_mst: float32
    """

    def __init__(
        self,
        n_primitives: int = 32,
        num_points: int = 16382,
        use_SElayer: bool = False,
    ):
        super(SpareNetRefine, self).__init__()
        self.num_points = num_points
        self.n_primitives = n_primitives
        self.expansion = expansion.expansionPenaltyModule()
        self.edgeres = False             # 不用边缘卷积
        if self.edgeres:
            self.residual = EdgeRes(use_SElayer=use_SElayer)
        else:
            self.residual = PointNetRes(use_SElayer=use_SElayer)

    def forward(self, inps, partial, coarse):
        dist, _, mean_mst_dis = self.expansion(                         # 先算一次在cuda的expansion损失
            coarse, 512, 1.5
        )
        loss_mst = torch.mean(dist)
        id0 = torch.zeros(inps.shape[0], 1, inps.shape[2]).cuda().contiguous()

        inps = torch.cat((inps, id0), 1)  # [batch_size, 4, out_points]
        id1 = torch.ones(partial.shape[0], 1, partial.shape[2]).cuda().contiguous()
        partial = torch.cat((partial, id1), 1)  # [batch_size, 4, in_points]
        base = torch.cat((inps, partial), 2)  # [batch_size, 4, out_points+ in_points]

        resampled_idx = MDS_module.minimum_density_sample(          # 最小密度采样
            base[:, 0:3, :].transpose(1, 2).contiguous(), coarse.shape[1], mean_mst_dis
        )
        base = MDS_module.gather_operation(base, resampled_idx)

        delta = self.residual(base)  # [batch_size, 3, out_points]
        base = base[:, 0:3, :]  # [batch_size, 3, out_points]
        outs = base + delta
        refine_result = outs.transpose(2, 1).contiguous()  # [batch_size, out_points, 3]
        return refine_result, loss_mst


class PointNetRes(nn.Module):
    """
    input:
    - inp: b x (num_dims+id) x num_points

    outputs:
    - out: b x num_dims x num_points
    """

    def __init__(self, use_SElayer: bool = False):
        super(PointNetRes, self).__init__()
        self.conv1 = torch.nn.Conv1d(4, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.conv4 = torch.nn.Conv1d(1088, 512, 1)
        self.conv5 = torch.nn.Conv1d(512, 256, 1)
        self.conv6 = torch.nn.Conv1d(256, 128, 1)
        self.conv7 = torch.nn.Conv1d(128, 3, 1)

        self.use_SElayer = use_SElayer
        if use_SElayer:
            self.se1 = SELayer1D(channel=64)
            self.se2 = SELayer1D(channel=128)
            self.se4 = SELayer1D(channel=512)
            self.se5 = SELayer1D(channel=256)
            self.se6 = SELayer1D(channel=128)

        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(1024)
        self.bn4 = torch.nn.BatchNorm1d(512)
        self.bn5 = torch.nn.BatchNorm1d(256)
        self.bn6 = torch.nn.BatchNorm1d(128)
        self.bn7 = torch.nn.BatchNorm1d(3)
        self.th = nn.Tanh()

    def forward(self, x):
        npoints = x.size()[2]
        # x: [batch_size, 4, num_points]
        if self.use_SElayer:
            x = F.relu(self.se1(self.bn1(self.conv1(x))))
        else:
            x = F.relu(self.bn1(self.conv1(x)))
        pointfeat = x  # [batch_size, 64, num_points]

        if self.use_SElayer:
            x = F.relu(self.se2(self.bn2(self.conv2(x))))
        else:
            x = F.relu(self.bn2(self.conv2(x)))

        x = self.bn3(self.conv3(x))  # [batch_size, 1024, num_points]
        x, _ = torch.max(x, 2)  # [batch_size, 1024]
        x = x.view(-1, 1024)  # [batch_size, 1024]
        x = x.view(-1, 1024, 1).repeat(1, 1, npoints)  # [batch_size, 1024, num_points]
        x = torch.cat([x, pointfeat], 1)  # [batch_size, 1088, num_points]
        if self.use_SElayer:
            x = F.relu(self.se4(self.bn4(self.conv4(x))))
            x = F.relu(self.se5(self.bn5(self.conv5(x))))
            x = F.relu(self.se6(self.bn6(self.conv6(x))))
        else:
            x = F.relu(self.bn4(self.conv4(x)))
            x = F.relu(self.bn5(self.conv5(x)))
            x = F.relu(self.bn6(self.conv6(x)))
        x = self.th(self.conv7(x))  # [batch_size, 3, num_points]
        return x


class EdgeRes(nn.Module):
    """
    input:
    - inp: b x (num_dims+id) x num_points

    outputs:
    - out: b x num_dims x num_points
    """

    def __init__(self, use_SElayer: bool = False):
        super(EdgeRes, self).__init__()
        self.k = 8
        self.conv1 = torch.nn.Conv2d(8, 64, kernel_size=1, bias=False)
        self.conv2 = torch.nn.Conv2d(128, 128, kernel_size=1, bias=False)
        self.conv3 = torch.nn.Conv2d(256, 1024, kernel_size=1, bias=False)
        self.conv4 = torch.nn.Conv2d(2176, 512, kernel_size=1, bias=False)
        self.conv5 = torch.nn.Conv2d(1024, 256, kernel_size=1, bias=False)
        self.conv6 = torch.nn.Conv2d(512, 128, kernel_size=1, bias=False)
        self.conv7 = torch.nn.Conv2d(256, 3, kernel_size=1, bias=False)

        self.use_SElayer = use_SElayer
        if use_SElayer:
            self.se1 = SELayer(channel=64)
            self.se2 = SELayer(channel=128)
            self.se4 = SELayer(channel=512)
            self.se5 = SELayer(channel=256)
            self.se6 = SELayer(channel=128)

        self.bn1 = torch.nn.BatchNorm2d(64)
        self.bn2 = torch.nn.BatchNorm2d(128)
        self.bn3 = torch.nn.BatchNorm2d(1024)
        self.bn4 = torch.nn.BatchNorm2d(512)
        self.bn5 = torch.nn.BatchNorm2d(256)
        self.bn6 = torch.nn.BatchNorm2d(128)
        self.bn7 = torch.nn.BatchNorm2d(3)
        self.th = nn.Tanh()

    def forward(self, x):
        npoints = x.size()[2]
        # x: [batch_size, 4, num_points]
        if self.use_SElayer:
            x = get_graph_feature(x, k=self.k)  # [bs, 8, num_points, k]
            x = F.relu(self.se1(self.bn1(self.conv1(x))))  # [bs, 64, num_points, k]
            x = x.max(dim=-1, keepdim=False)[0]  # [bs, 64, num_points]
            pointfeat = x  # [batch_size, 64, num_points]
            x = get_graph_feature(x, k=self.k)  # [bs, 128, num_points, k]
            x = F.relu(self.se2(self.bn2(self.conv2(x))))
            x = x.max(dim=-1, keepdim=False)[0]  # [bs, 128, num_points]
        else:
            x = get_graph_feature(x, k=self.k)  # [bs, 8, num_points, k]
            x = F.relu(self.bn1(self.conv1(x)))  # [bs, 64, num_points, k]
            x = x.max(dim=-1, keepdim=False)[0]  # [bs, 64, num_points]
            pointfeat = x  # [batch_size, 64, num_points]
            x = get_graph_feature(x, k=self.k)  # [bs, 128, num_points, k]
            x = F.relu(self.bn2(self.conv2(x)))
            x = x.max(dim=-1, keepdim=False)[0]  # [bs, 128, num_points]

        x = get_graph_feature(x, k=self.k)  # [bs, 256, num_points, k]
        x = self.bn3(self.conv3(x))  # [batch_size, 1024, num_points, k]
        x = x.max(dim=-1, keepdim=False)[0]  # [bs, 1024, num_points]

        x, _ = torch.max(x, 2)  # [batch_size, 1024]
        x = x.view(-1, 1024)  # [batch_size, 1024]
        x = x.view(-1, 1024, 1).repeat(1, 1, npoints)  # [batch_size, 1024, num_points]
        x = torch.cat([x, pointfeat], 1)  # [batch_size, 1088, num_points]

        if self.use_SElayer:
            x = get_graph_feature(x, k=self.k)  # [bs, 2176, num_points, k]
            x = F.relu(self.se4(self.bn4(self.conv4(x))))
            x = x.max(dim=-1, keepdim=False)[0]  # [bs, 512, num_points]
            x = get_graph_feature(x, k=self.k)  # [bs, 1024, num_points, k]
            x = F.relu(self.se5(self.bn5(self.conv5(x))))
            x = x.max(dim=-1, keepdim=False)[0]  # [bs, 256, num_points]
            x = get_graph_feature(x, k=self.k)  # [bs, 512, num_points, k]
            x = F.relu(self.se6(self.bn6(self.conv6(x))))
            x = x.max(dim=-1, keepdim=False)[0]  # [bs, 128, num_points]
        else:
            x = get_graph_feature(x, k=self.k)  # [bs, 2176, num_points, k]
            x = F.relu(self.bn4(self.conv4(x)))
            x = x.max(dim=-1, keepdim=False)[0]  # [bs, 512, num_points]
            x = get_graph_feature(x, k=self.k)  # [bs, 1024, num_points, k]
            x = F.relu(self.bn5(self.conv5(x)))
            x = x.max(dim=-1, keepdim=False)[0]  # [bs, 256, num_points]
            x = get_graph_feature(x, k=self.k)  # [bs, 512, num_points, k]
            x = F.relu(self.bn6(self.conv6(x)))
            x = x.max(dim=-1, keepdim=False)[0]  # [bs, 128, num_points]
        x = get_graph_feature(x, k=self.k)  # [bs, 256, num_points, k]
        x = self.th(self.conv7(x))
        x = x.max(dim=-1, keepdim=False)[0]  # [bs, 3, num_points]
        return x

# 自注意力层
class SELayer(nn.Module):
    """
    input:
        x:(b, c, m, n)

    output:
        out:(b, c, m', n')
    """

    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)     # F1之后的一个avg_pool
        y = self.fc(y).view(b, c, 1, 1)     # F2的MLP
        return x * y.expand_as(x)           # 最后全局乘以边特征


class SELayer1D(nn.Module):
    """
    input:
        x:(b, c, m)

    output:
        out:(b, c, m')
    """

    def __init__(self, channel, reduction=16):
        super(SELayer1D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        (b, c, _) = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


def grid_generation(num_points, nb_primitives):
    """
    inputs:
    - num_points: int
    - nb_primitives: int

    outputs:
    - 2D grid: nb_primitives * (num_points / nb_primitives) * 2
    """
    num_points = num_points / nb_primitives
    grain_x = 2 ** np.floor(np.log2(num_points) / 2) - 1
    grain_y = 2 ** np.ceil(np.log2(num_points) / 2) - 1

    vertices = []
    for i in range(int(grain_x + 1)):
        for j in range(int(grain_y + 1)):
            vertices.append([i / grain_x, j / grain_y])

    print("generating 2D grid")
    return [vertices for i in range(nb_primitives)]


def get_num_adain_params(model):
    """
    input:
    - model: nn.module

    output:
    - num_adain_params: int
    """
    # return the number of AdaIN parameters needed by the model
    num_adain_params = 0
    for m in model.modules():
        if m.__class__.__name__ == "AdaptiveInstanceNorm1d":
            num_adain_params += 2 * m.num_features
    return num_adain_params


def assign_adain_params(adain_params, model):

    """
    inputs:
    - adain_params: b x parameter_size
    - model: nn.module

    function:
    assign_adain_params
    """
    # assign the adain_params to the AdaIN layers in model
    for m in model.modules():
        if m.__class__.__name__ == "AdaptiveInstanceNorm1d":
            mean = adain_params[:, : m.num_features]
            std = adain_params[:, m.num_features : 2 * m.num_features]
            m.bias = mean.contiguous().view(-1)
            m.weight = std.contiguous().view(-1)
            if adain_params.size(1) > 2 * m.num_features:
                adain_params = adain_params[:, 2 * m.num_features :]


def knn(x, k: int):
    """
    inputs:
    - x: b x npoints1 x num_dims (partical_cloud)
    - k: int (the number of neighbor)

    outputs:
    - idx: int (neighbor_idx)
    """
    # x : (batch_size, feature_dim, num_points)
    # Retrieve nearest neighbor indices

    if torch.cuda.is_available():
        from knn_cuda import KNN

        ref = x.transpose(2, 1).contiguous()  # (batch_size, num_points, feature_dim)
        query = ref
        _, idx = KNN(k=k, transpose_mode=True)(ref, query)

    else:
        inner = -2 * torch.matmul(x.transpose(2, 1), x)
        xx = torch.sum(x ** 2, dim=1, keepdim=True)
        pairwise_distance = -xx - inner - xx.transpose(2, 1)
        idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)

    return idx


def get_graph_feature(x, k: int = 20, idx=None):
    """
    inputs:
    - x: b x npoints1 x num_dims (partical_cloud)
    - k: int (the number of neighbor)
    - idx: neighbor_idx

    outputs:
    - feature: b x npoints1 x (num_dims*2)
    """

    batch_size = x.size(0)
    num_points = x.size(2)  # 3000个点
    x = x.view(batch_size, -1, num_points)      # 将torch.Size([2, 3, 3000])转成torch.Size([2, 3, 3000])，没转
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)        # torch.Size([2, 3000, 8]) batch是2，3000个点，每个点8个近邻
    device = idx.device
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)      # 全部转成1维
    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()          # torch.Size([2, 3, 3000])转置成为torch.Size([2, 3000, 3])
    feature = x.view(batch_size * num_points, -1)[idx, :]       # torch.Size([2, 3000, 3])转成torch.Size([48000, 3])
    feature = feature.view(batch_size, num_points, k, num_dims) # 转成torch.Size([2, 3000, 8, 3]) 2个batch，每个有3000个点，每个点8个近邻，每个近邻有x,y,z这3个特征
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()       # cat的是近邻点和每个点与base点的差以及base点
    return feature


class AdaptiveInstanceNorm1d(nn.Module):
    """
    input:
    - inp: (b, c, m)

    output:
    - out: (b, c, m')
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
    ):
        super(AdaptiveInstanceNorm1d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x):
        assert (
            self.weight is not None and self.bias is not None
        ), "Please assign weight and bias before calling AdaIN!"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])

        out = F.batch_norm(
            x_reshaped,
            running_mean,
            running_var,
            self.weight,
            self.bias,
            True,
            self.momentum,
            self.eps,
        )

        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + "(" + str(self.num_features) + ")"

class MViewDecoder(nn.Module):
    """
    input:
    - x: b x (x,y) x (num_points / nb_primitives)

    output:
    - out: b x num_dims x (num_points / nb_primitives)
    """

    def __init__(
            self,
            input_dim: int = 2,
            bottleneck_size: int = 1026,
            use_SElayer: bool = False,
            use_sine: bool = False,             # 外面没有传这个参数，所以这里一定是默认值False
    ):
        super(MViewDecoder, self).__init__()
        self.bottleneck_size = bottleneck_size
        self.input_dim = input_dim

        self.use_sine = use_sine

        self.conv1 = torch.nn.Conv1d(self.input_dim, self.bottleneck_size // 4, 1)
        self.conv4 = torch.nn.Conv1d(self.bottleneck_size // 4, 3, 1)
        self.th = nn.Tanh()


        self.adain1 = AdaptiveInstanceNorm1d(self.bottleneck_size // 4)

        self.bn1 = torch.nn.BatchNorm1d(
            self.bottleneck_size // 4
        )

        self.use_SElayer = use_SElayer
        if self.use_SElayer:
            self.se1 = SELayer1D(channel=self.bottleneck_size // 4)

    def forward(self, x):
        if self.use_SElayer:
            x = F.relu(self.se1(self.bn1(self.adain1(self.conv1(x)))))
        else:
            x = F.relu(self.bn1(self.adain1(self.conv1(x))))
        x = self.th(self.conv4(x))
        return x

class GridDecoder(nn.Module):
    """
    input:
    - x: b x (x,y) x (num_points / nb_primitives)

    output:
    - out: b x num_dims x (num_points / nb_primitives)
    """

    def __init__(
        self,
        input_dim: int = 2,
        bottleneck_size: int = 1026,
        use_SElayer: bool = False,
        use_sine: bool = False,             # 外面没有传这个参数，所以这里一定是默认值False
    ):
        super(GridDecoder, self).__init__()
        self.bottleneck_size = bottleneck_size
        self.input_dim = input_dim

        self.use_sine = use_sine
        if not self.use_sine:
            self.conv1 = torch.nn.Conv1d(self.input_dim, self.bottleneck_size, 1)
            self.conv2 = torch.nn.Conv1d(
                self.bottleneck_size, self.bottleneck_size // 2, 1
            )
            self.conv3 = torch.nn.Conv1d(
                self.bottleneck_size // 2, self.bottleneck_size // 4, 1
            )
            self.conv4 = torch.nn.Conv1d(self.bottleneck_size // 4, 3, 1)
            self.th = nn.Tanh()
        else:
            first_omega_0 = 30.0
            hidden_omega_0 = 30.0
            self.linear1 = SineLayer(
                self.input_dim,
                self.bottleneck_size,
                is_first=True,
                omega_0=first_omega_0,
            )
            self.linear2 = SineLayer(
                self.bottleneck_size,
                self.bottleneck_size // 2,
                is_first=False,
                omega_0=hidden_omega_0,
            )
            self.linear3 = SineLayer(
                self.bottleneck_size // 2,
                self.bottleneck_size // 4,
                is_first=False,
                omega_0=hidden_omega_0,
            )
            self.linear4 = SineLayer(
                self.bottleneck_size // 4,
                self.bottleneck_size // 4,
                is_first=False,
                omega_0=hidden_omega_0,
            )
            self.linear5 = nn.Conv1d(self.bottleneck_size // 4, 3, 1)

            with torch.no_grad():
                self.linear5.weight.uniform_(
                    -np.sqrt(6 / self.bottleneck_size) / hidden_omega_0,
                    np.sqrt(6 / self.bottleneck_size) / hidden_omega_0,
                )

        self.adain1 = AdaptiveInstanceNorm1d(self.bottleneck_size)
        self.adain2 = AdaptiveInstanceNorm1d(self.bottleneck_size // 2)
        self.adain3 = AdaptiveInstanceNorm1d(self.bottleneck_size // 4)

        self.bn1 = torch.nn.BatchNorm1d(
            self.bottleneck_size
        )  # default with Learnable Parameters
        self.bn2 = torch.nn.BatchNorm1d(self.bottleneck_size // 2)
        self.bn3 = torch.nn.BatchNorm1d(self.bottleneck_size // 4)

        self.use_SElayer = use_SElayer
        if self.use_SElayer:
            self.se1 = SELayer1D(channel=self.bottleneck_size)
            self.se2 = SELayer1D(channel=self.bottleneck_size // 2)
            self.se3 = SELayer1D(channel=self.bottleneck_size // 4)

    def forward(self, x):
        if self.use_sine:
            x = x.clone().detach().requires_grad_(True)
            x = self.linear1(x)
            x = self.linear2(x)
            x = self.linear3(x)
            x = self.linear4(x)
            x = self.linear5(x)
        else:
            if self.use_SElayer:
                x = F.relu(self.se1(self.bn1(self.adain1(self.conv1(x)))))          # torch.Size([2, 2, 512]) --> torch.Size([2, 1026, 512])
                x = F.relu(self.se2(self.bn2(self.adain2(self.conv2(x)))))          # torch.Size([2, 1026, 512]) --> torch.Size([2, 513, 512])
                x = F.relu(self.se3(self.bn3(self.adain3(self.conv3(x)))))          # torch.Size([2, 513, 512]) --> torch.Size([2, 256, 512])
            else:
                x = F.relu(self.bn1(self.adain1(self.conv1(x))))
                x = F.relu(self.bn2(self.adain2(self.conv2(x))))
                x = F.relu(self.bn3(self.adain3(self.conv3(x))))
            x = self.th(self.conv4(x))
        return x


class SineLayer(nn.Module):
    """
    input:
    - x: b x (x,y) x (num_points / nb_primitives)

    output:
    - out: b x num_dims x (num_points / nb_primitives)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        is_first: bool = False,
        omega_0: int = 30,
    ):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Conv1d(in_features, out_features, 1, bias=bias)
        self.adain = AdaptiveInstanceNorm1d(out_features)
        self.bn = torch.nn.BatchNorm1d(out_features)
        self.init_weights()

    def init_weights(self):
        """
        input:
        - x: b x (x,y) x (num_points / nb_primitives)

        output:
        - out: b x num_dims x (num_points / nb_primitives)
        """
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(
                    -np.sqrt(6 / self.in_features) / self.omega_0,
                    np.sqrt(6 / self.in_features) / self.omega_0,
                )

    def forward(self, input):
        return torch.sin(self.adain(self.omega_0 * self.linear(input)))


