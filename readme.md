# 0311：添加Gan网络在2d特征域上学习从partial到gt的映射
这是添加2d特征生成网络的版本

train.py：是使用gt的img来预训练编码器解码器网络的入口

## 添加：

train_partial.py：是固定编解码器权重训练2d特征生成网络的入口

pcf2dnet_gan_runner.py：相关流程代码

pcf2dnet_gan.yaml：生成网络的配置文件

sparenet_generator.py：中定义PCF2dNetGenerator、PCF2dNetDecode类

unet.py：中添加UNetGanEncoder、EasyUnetGenerator类

dataloader.py：中添加Mview2ShapeNetDataLoader类

修改损失函数为3d重建损失和2d特征域l2损失，载入预训练模型后冻结相关层权重

# 0315：修改PointNet编码器为多视图编码器MViewEncoder

此版本添加新的编码器MViewEncoder，尝试使用2d卷积提取点云特征

## 添加

mviewnet_gan.yaml

mviewnet_gan_runner.py

MViewNetGenerator、MViewNetDecode、MViewEncoder类

index2point()函数

修改get_depth_image()函数


# 0318 Update

将SpareNet的输入完全改为坐标图

MViewNet的dec改为8个并行 render init那里改为只有一个render

添加MViewPontNet


# 0320 添加UNet生成网络及Inpainting网络

# 0321 使用UNet生成粗略pc

