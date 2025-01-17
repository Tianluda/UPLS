from .resnet import *
import torch.nn as nn
import torch
import torch.nn.functional as F
from functools import reduce

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)
def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8,ReduceDimensions='True'):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        if ReduceDimensions=='True':
            # 利用1x1卷积代替全连接
            self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
            self.relu1 = nn.ReLU()
            self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        elif ReduceDimensions=='UP':
            # 升维
            self.fc1   = nn.Conv2d(in_planes, in_planes*2, 1, bias=False)
            self.relu1 = nn.ReLU()
            self.fc2   = nn.Conv2d(in_planes*2, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)
    
class ChannelFcaLayerAttention(nn.Module):
    def __init__(self, in_planes, ratio=8,ReduceDimensions='True'):
        super(ChannelFcaLayerAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        if ReduceDimensions=='True':
            # 利用1x1卷积代替全连接
            self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
            self.relu1 = nn.ReLU()
            self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        elif ReduceDimensions=='UP':
            # 升维
            self.fc1   = nn.Conv2d(in_planes, in_planes*2, 1, bias=False)
            self.relu1 = nn.ReLU()
            self.fc2   = nn.Conv2d(in_planes*2, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

        self.width = 12
        self.height = 12
        self.channel = in_planes
        self.reduction = ratio
        # 注册一个缓冲区用于存储DCT权重，这些权重会在前向传播时预计算并重复使用。
        self.register_buffer('pre_computed_dct_weights', get_dct_weights(self.width, self.height, in_planes))
    def forward(self, x):
        avg_x = self.avg_pool(x)
        avg_out = self.fc2(self.relu1(self.fc1(avg_x)))
        max_x = self.max_pool(x)
        max_out = self.fc2(self.relu1(self.fc1(max_x)))
        b, c, _, _ = x.size()  # 从输入张量形状中提取批次大小和通道数。
        y = F.adaptive_avg_pool2d(x, (self.height, self.width)) # 对特征图应用自适应平均池化，减小空间维度。
        y = torch.sum(y * self.pre_computed_dct_weights, dim=(2, 3)).view(b, c, 1, 1) # 使用预计算的DCT权重计算池化特征的加权和。
        Fca_out = self.fc2(self.relu1(self.fc1(y)))  # 重塑序列网络的输出为通道级标量，对应于每个特征图。
        out = (Fca_out - avg_out) + (max_out - avg_out)
        return self.sigmoid(out)
    
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
    
# CBAM通道空间混合注意力模块
class CBAM(nn.Module):
    def __init__(self, channel, ratio=8, kernel_size=3,ReduceDimensions='True',Res='False'):
        super(CBAM, self).__init__()
        # self.channelattention = ChannelAttention(channel, ratio=ratio,ReduceDimensions=ReduceDimensions)
        self.channelattention = ChannelFcaLayerAttention(channel, ratio=ratio,ReduceDimensions=ReduceDimensions)
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)
        self.Res=Res
        if Res=='True':
            self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        if self.Res=='True':
            residual = x
        x = x * self.channelattention(x)
        x = x * self.spatialattention(x)
        if self.Res=='True':
            x += residual
            x = self.relu(x)
        return x
def get_1d_dct(i, freq, L):
    """
    计算一维离散余弦变换(DCT)的第i个系数。
    参数:
    - i: 指示第i个系数。
    - freq: 频率系数。
    - L: DCT的长度。
    返回值:
    - 计算得到的一维DCT系数。
    """
    # 计算DCT的基本函数值
    result = math.cos(math.pi * freq * (i + 0.5) / L)
    # 对于频率为0的特殊情况，直接返回结果，不乘以sqrt(2)
    if freq == 0:
        return result
    else:
        # 对于非零频率，结果乘以sqrt(2)以满足DCT的规范）
        return result * math.sqrt(2)

def get_dct_weights(width, height, channel, fidx_u=[0, 0, 6, 0, 0, 1, 1, 4, 5, 1, 3, 0, 0, 0, 2, 3],
                    fidx_v=[0, 1, 0, 5, 2, 0, 2, 0, 0, 6, 0, 4, 6, 3, 2, 5]):
    """
    计算并返回一个给定输入尺寸、通道数的DCT权重矩阵。
    参数:
    - width : 输入的宽度。
    - height : 输入的高度。
    - channel : 输入的通道数。
    - fidx_u : 指定的水平频率索引。
    - fidx_v : 指定的垂直频率索引。
    返回值:
    - dct_weights : 一个形如1 x channel x width x height的Tensor，包含了DCT权重。

    """
    # 根据输入尺寸调整频率索引的尺度
    scale_ratio = width // 7
    fidx_u = [u * scale_ratio for u in fidx_u]
    fidx_v = [v * scale_ratio for v in fidx_v]
    # 初始化DCT权重矩阵
    dct_weights = torch.zeros(1, channel, width, height)
    c_part = channel // len(fidx_u)  # 将通道数按频率索引分割
    
    # 对每个频率索引，计算并应用1D DCT权重
    for i, (u_x, v_y) in enumerate(zip(fidx_u, fidx_v)):
        for t_x in range(width):
            for t_y in range(height):
                # 应用水平和垂直的DCT权重
                dct_weights[:, i * c_part: (i + 1) * c_part, t_x, t_y] \
                    = get_1d_dct(t_x, u_x, width) * get_1d_dct(t_y, v_y, height)
    return dct_weights

class FcaLayer(nn.Module):
    """
    FcaLayer 类实现了一个特征通道注意力层，用于调整特征图中不同通道的重要性。它利用一个轻量级神经网络来学习通道级的注意力权重。
    - channel (int): 输入特征图的通道数。
    - reduction (int, 可选): 通道维度的压缩比例，默认为16。
    """
    def __init__(self,channel,reduction=16):
        super(FcaLayer, self).__init__()
        self.width = 12
        self.height = 12
        self.channel = channel
        self.reduction = reduction
        # 注册一个缓冲区用于存储DCT权重，这些权重会在前向传播时预计算并重复使用。
        self.register_buffer('pre_computed_dct_weights', get_dct_weights(self.width, self.height, channel))
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()  # 应用Sigmoid激活函数以产生通道级的注意力权重。
        )
    def forward(self, x):
        # 计算通道级的注意力权重，并将这些权重应用于输入特征图。
        b, c, _, _ = x.size()  # 从输入张量形状中提取批次大小和通道数。
        y = F.adaptive_avg_pool2d(x, (self.height, self.width)) # 对特征图应用自适应平均池化，减小空间维度。
        y = torch.sum(y * self.pre_computed_dct_weights, dim=(2, 3)) # 使用预计算的DCT权重计算池化特征的加权和。
        y = self.fc(y).view(b, c, 1, 1)  # 重塑序列网络的输出为通道级标量，对应于每个特征图。
        return x * y.expand_as(x)  # 扩展注意力权重以匹配输入特征图的形状，然后逐元素相乘。

def get_freq_indices(method):
    """
    根据指定的方法获取频率索引。
    参数: - method: 字符串，指定频率选择方法，可以是'top1'到'top32'，'bot1'到'bot32'，或'low1'到'low32'之一。
    返回:
    - mapper_x: 列表，包含在x方向上的频率索引。
    - mapper_y: 列表，包含在y方向上的频率索引。
    fidx_u=[0, 0, 6, 0, 0, 1, 1, 4, 5, 1, 3, 0, 0, 0, 2, 3],
    fidx_v=[0, 1, 0, 5, 2, 0, 2, 0, 0, 6, 0, 4, 6, 3, 2, 5]):
    """
    assert method in ['top1','top2','top4','top8','top16','top32',
                      'bot1','bot2','bot4','bot8','bot16','bot32',
                      'low1','low2','low4','low8','low16','low32']
    num_freq = int(method[3:])
    if 'top' in method:
        # 提取顶部频率索引
        # all_top_indices_x = [0,0,6,0,0,1,1,4,5,1,3,0,0,0,3,2,4,6,3,5,5,2,6,5,5,3,3,4,2,2,6,1]
        # all_top_indices_y = [0,1,0,5,2,0,2,0,0,6,0,4,6,3,5,2,6,3,3,3,5,1,1,2,4,2,1,1,3,0,5,3]
        all_top_indices_x = [0,0,6,0,0,1,1,4,5,1,3,0,0,0,2,3,4,6,3,5,5,2,6,5,5,3,3,4,2,2,6,1]
        all_top_indices_y = [0,1,0,5,2,0,2,0,0,6,0,4,6,3,2,5,6,3,3,3,5,1,1,2,4,2,1,1,3,0,5,3]
        mapper_x = all_top_indices_x[:num_freq]
        mapper_y = all_top_indices_y[:num_freq]
    elif 'low' in method:
        # 提取底部频率索引
        all_low_indices_x = [0,0,1,1,0,2,2,1,2,0,3,4,0,1,3,0,1,2,3,4,5,0,1,2,3,4,5,6,1,2,3,4]
        all_low_indices_y = [0,1,0,1,2,0,1,2,2,3,0,0,4,3,1,5,4,3,2,1,0,6,5,4,3,2,1,0,6,5,4,3]
        mapper_x = all_low_indices_x[:num_freq]
        mapper_y = all_low_indices_y[:num_freq]
    elif 'bot' in method:
        # 提取底部频率索引
        all_bot_indices_x = [6,1,3,3,2,4,1,2,4,4,5,1,4,6,2,5,6,1,6,2,2,4,3,3,5,5,6,2,5,5,3,6]
        all_bot_indices_y = [6,4,4,6,6,3,1,4,4,5,6,5,2,2,5,1,4,3,5,0,3,1,1,2,4,2,1,1,5,3,3,3]
        mapper_x = all_bot_indices_x[:num_freq]
        mapper_y = all_bot_indices_y[:num_freq]
    else:
        raise NotImplementedError
    return mapper_x, mapper_y

class MultiSpectralDCTLayer(nn.Module):
    """
    生成多光谱离散余弦变换（DCT）层。
    """
    def __init__(self, height, width, mapper_x, mapper_y, channel):
        super(MultiSpectralDCTLayer, self).__init__()
        assert len(mapper_x) == len(mapper_y)
        assert channel % len(mapper_x) == 0
        self.num_freq = len(mapper_x)
        # 注册缓冲区作为DCT滤波器的权重
        self.register_buffer('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))
        
    def forward(self, x):
        """
        前向传播。
        参数:- x: 输入特征。
        返回:- 经过DCT层处理后的结果。
        """
        assert len(x.shape) == 4, 'x must been 4 dimensions, but got ' + str(len(x.shape))
        x = x * self.weight
        result = torch.sum(x, dim=[2,3])
        return result

    def build_filter(self, pos, freq, POS):
        """
        构建DCT滤波器。
        参数:
        pos: 位置。freq: 频率。POS: 空间位置总数。
        返回:- 构建的滤波器值。
        """
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS) 
        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)
    
    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, channel):
        """
        生成DCT滤波器。
        参数:
        - tile_size_x: x方向上的瓦片大小。        - tile_size_y: y方向上的瓦片大小。
        - mapper_x: x方向上的频率索引。        - mapper_y: y方向上的频率索引。
        - channel: 通道数。
        返回:- dct_filter: DCT滤波器的张量。
        """
        dct_filter = torch.zeros(channel, tile_size_x, tile_size_y)
        c_part = channel // len(mapper_x)
        for i, (u_x, v_y) in enumerate(zip(mapper_x, mapper_y)):
            for t_x in range(tile_size_x):
                for t_y in range(tile_size_y):
                    dct_filter[i * c_part: (i+1)*c_part, t_x, t_y] = self.build_filter(t_x, u_x, tile_size_x) * self.build_filter(t_y, v_y, tile_size_y)
        return dct_filter
    
class MultiSpectralAttentionLayer(torch.nn.Module):
    def __init__(self, channel, dct_h=12, dct_w=12, reduction = 16, freq_sel_method = 'top16'):
        # top16 top32 bot16 bot32 low16 low32
        super(MultiSpectralAttentionLayer, self).__init__()
        self.reduction = reduction
        self.dct_h = dct_h
        self.dct_w = dct_w
        mapper_x, mapper_y = get_freq_indices(freq_sel_method)
        self.num_split = len(mapper_x)
        mapper_x = [temp_x * (dct_h // 7) for temp_x in mapper_x] 
        mapper_y = [temp_y * (dct_w // 7) for temp_y in mapper_y]
        # 将不同大小的频率映射到一个7x7的频率空间中，以保持兼容性
        self.dct_layer = MultiSpectralDCTLayer(dct_h, dct_w, mapper_x, mapper_y, channel)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        # self.fc = nn.Sequential(
        #     nn.Linear(channel, channel * 2, bias=False),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(channel *2 , channel, bias=False),
        #     nn.Sigmoid()
        # )
    def forward(self, x):
        """
        前向传播。
        参数:- x: 输入特征。
        返回:- 加权后的输入特征。
        """
        n,c,h,w = x.shape
        x_pooled = x
        if h != self.dct_h or w != self.dct_w:
            # 对输入进行适应性平均池化
            x_pooled = torch.nn.functional.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))
        y = self.dct_layer(x_pooled)
        y = self.fc(y).view(n, c, 1, 1)
        return x * y.expand_as(x)
    
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None
    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class BasicRFB(nn.Module):
    def __init__(self, in_planes, out_planes, stride=2, scale = 1, visual = 2):
    # def __init__(self, in_planes, out_planes, stride=1, scale = 0.1, visual = 1):
        super(BasicRFB, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // 8
        self.branch0 = nn.Sequential(
                BasicConv(in_planes, 2*inter_planes, kernel_size=1, stride=stride),
                BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=visual, dilation=visual, relu=False)
                )
        self.branch1 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, 2*inter_planes, kernel_size=(3,3), stride=stride, padding=(1,1)),
                BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=visual+1, dilation=visual+1, relu=False)
                )
        self.branch2 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, (inter_planes//2)*3, kernel_size=3, stride=1, padding=1),
                BasicConv((inter_planes//2)*3, 2*inter_planes, kernel_size=3, stride=stride, padding=1),
                BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=2*visual+1, dilation=2*visual+1, relu=False)
                )
        self.ConvLinear = BasicConv(6*inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)
    def forward(self,x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        out = torch.cat((x0,x1,x2),1)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out*self.scale + short
        out = self.relu(out)
        return out

class BasicRFB_s(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, scale = 1):
    # def __init__(self, in_planes, out_planes, stride=1, scale = 0.1):
        super(BasicRFB_s, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes //4

        self.branch0 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=1,relu=False)
                )
        self.branch1 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, inter_planes, kernel_size=(3,1), stride=1, padding=(1,0)),
                BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=3, dilation=3, relu=False)
                )
        self.branch2 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, inter_planes, kernel_size=(1,3), stride=stride, padding=(0,1)),
                BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=3, dilation=3, relu=False)
                )
        self.branch3 = nn.Sequential(
                BasicConv(in_planes, inter_planes//2, kernel_size=1, stride=1),
                BasicConv(inter_planes//2, (inter_planes//4)*3, kernel_size=(1,3), stride=1, padding=(0,1)),
                BasicConv((inter_planes//4)*3, inter_planes, kernel_size=(3,1), stride=stride, padding=(1,0)),
                BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=5, dilation=5, relu=False)
                )

        self.ConvLinear = BasicConv(4*inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)
    def forward(self,x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        out = torch.cat((x0,x1,x2,x3),1)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out*self.scale + short
        out = self.relu(out)
        return out

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x
    
class RFB(nn.Module):
    # RFB-like multi-scale module
    def __init__(self, in_channel, out_channel):
        super(RFB, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)
    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))
        x = self.relu(x_cat + self.conv_res(x))
        return x
    
class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__(
            DoubleConv(in_channels, out_channels),
            nn.MaxPool2d(2, stride=2)
        )

class ResNetSeries(nn.Module):
    def __init__(self, pretrained, parameters='False'):
        # parameters='False': 默认
        # parameters='True': CBAM
        # parameters='UP': CBAM_UP
        # parameters='Res': CBAM_Res
        # parameters='BS': BackgroundSuppression 背景抑制
        # parameters='Down': 第三层Down函数（先DoubleConv再MaxPool2d）
        # parameters='3': 返回后三层 [x4, x3, x2]
        # parameters='2': 返回后2层 [x4, x3] 均使用RFB
        super(ResNetSeries, self).__init__()
        self.parameters=parameters.split(',')
        if pretrained == 'supervised':  #'https://download.pytorch.org/models/resnet50-19c8e357.pth'
            print(f'Loading supervised pretrained parameters!')
            model = resnet50(pretrained=True)
        elif pretrained == 'mocov2':
            print(f'Loading unsupervised {pretrained} pretrained parameters!')
            model = resnet50(pretrained=False)
            checkpoint = torch.load('moco_r50_v2-e3b0c442.pth', map_location="cpu")
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        elif pretrained == 'detco':
            print(f'Loading unsupervised {pretrained} pretrained parameters!')
            model = resnet50(pretrained=False)
            checkpoint = torch.load('detco_200ep.pth', map_location="cpu")
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        elif pretrained == 'plant':
            print(f'Loading unsupervised {pretrained} pretrained parameters!')
            model = resnet50(pretrained=False)
            checkpoint = torch.load('ResNet50-Plant-model-80.pth', map_location="cpu")
            # 删除加载的权重文件中全连接层相关的参数
            del checkpoint['fc.weight']
            del checkpoint['fc.bias']
            model.load_state_dict(checkpoint, strict=False)
        else:
            raise NotImplementedError
        self.conv1 = model.conv1

        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        if 'True' in self.parameters:
            if 'Res' in self.parameters:
                self.cbam1 = CBAM(256,ReduceDimensions='True',Res='True')
                self.cbam2 = CBAM(512,ReduceDimensions='True',Res='True')
                self.cbam3 = CBAM(1024,ReduceDimensions='True',Res='True')
                self.cbam4 = CBAM(2048,ReduceDimensions='True',Res='True')
            else:
                self.cbam1 = CBAM(256,ReduceDimensions='True')
                self.cbam2 = CBAM(512,ReduceDimensions='True')
                self.cbam3 = CBAM(1024,ReduceDimensions='True')
                self.cbam4 = CBAM(2048,ReduceDimensions='True')
        if 'UP' in self.parameters:
            if 'Res' in self.parameters:
                self.cbam1 = CBAM(256,ReduceDimensions='UP',Res='True')
                self.cbam2 = CBAM(512,ReduceDimensions='UP',Res='True')
                self.cbam3 = CBAM(1024,ReduceDimensions='UP',Res='True')
                self.cbam4 = CBAM(2048,ReduceDimensions='UP',Res='True')
            else:
                self.cbam1 = CBAM(256,ReduceDimensions='UP')
                self.cbam2 = CBAM(512,ReduceDimensions='UP')
                self.cbam3 = CBAM(1024,ReduceDimensions='UP')
                self.cbam4 = CBAM(2048,ReduceDimensions='UP')
        if 'BS' in self.parameters:
            self.bg_suppression = BackgroundSuppression()
        if 'RFB' in self.parameters:
            self.rfb2 = RFB(512, 512)
        elif 'BasicRFB_s' in self.parameters:
            self.rfb2 = BasicRFB_s(512, 512)
        elif 'BasicRFB' in self.parameters:
            self.rfb2 = BasicRFB(512, 512)
        elif '3' in self.parameters:
            self.rfb2 = RFB(512, 512)
        if 'Down' in self.parameters:
            self.down2= Down(512, 512)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        if '3' in self.parameters:
            x1 = self.layer1(x)
            x2 = self.layer2(x1)
            x2 = self.rfb2(x2)
            if 'BS' in self.parameters:
                x2 = self.bg_suppression(x2)
            if ('True' in self.parameters) or ('UP' in self.parameters):
                x3 = self.layer3(x2)
                x3 = self.cbam3(x3)
                x4 = self.layer4(x3)
                x4 = self.cbam4(x4)
            else:
                x3 = self.layer3(x2)
                x4 = self.layer4(x3)
            if 'Down' in self.parameters:
                h2, w2 = x2.size(2), x2.size(3)
                pad_h = (2 - h2 % 2) % 2
                pad_w = (2 - w2 % 2) % 2
                if pad_h > 0 or pad_w > 0:
                    x2 = F.pad(x2, (0, pad_w, 0, pad_h), mode='constant', value=0)
                x2 = self.down2(x2)
            else:
                x2 = F.adaptive_avg_pool2d(x2, output_size=x3.size()[2:])
            return torch.cat([x4, x3, x2], dim=1)
        elif '2' in self.parameters:
            x1 = self.layer1(x)
            x2 = self.layer2(x1)
            if 'RFB' in self.parameters:
                x2 = self.rfb2(x2)
            if ('True' in self.parameters) or ('UP' in self.parameters):
                x3 = self.layer3(x2)
                x3 = self.cbam3(x3)
                x4 = self.layer4(x3)
                x4 = self.cbam4(x4)
            else:
                x3 = self.layer3(x2)
                x4 = self.layer4(x3)
            return torch.cat([x4, x3], dim=1)
        else:
            x1 = self.layer1(x)
            x2 = self.layer2(x1)
            if ('True' in self.parameters) or ('UP' in self.parameters):
                x3 = self.layer3(x2)
                x3 = self.cbam3(x3)
                x4 = self.layer4(x3)
                x4 = self.cbam4(x4)
            else:
                x3 = self.layer3(x2)
                x4 = self.layer4(x3)
            return torch.cat([x4, x3], dim=1)

class Disentangler(nn.Module):
    def __init__(self, cin):
        super(Disentangler, self).__init__()
        self.activation_head = nn.Conv2d(cin, 1, kernel_size=3, padding=1, bias=False)
        self.bn_head = nn.BatchNorm2d(1)
    def forward(self, x, inference=False):
        N, C, H, W = x.size() # 获取输入特征图的尺寸

        if inference:
            ccam = self.bn_head(self.activation_head(x))  # 不使用Sigmoid激活，用于推理
        else:
            ccam = torch.sigmoid(self.bn_head(self.activation_head(x)))  # 使用Sigmoid激活，用于训练 [N, 1, H, W][256, 1, 12, 12]

        ccam_ = ccam.reshape(N, 1, H * W)                          # [N, 1, H*W]

        x = x.reshape(N, C, H * W).permute(0, 2, 1).contiguous()   # [N, H*W, C]
        # 计算前景和背景特征
        fg_feats = torch.matmul(ccam_, x) / (H * W)                # [N, 1, C]
        bg_feats = torch.matmul(1 - ccam_, x) / (H * W)            # [N, 1, C]

        return fg_feats.reshape(x.size(0), -1), bg_feats.reshape(x.size(0), -1), ccam

class NewDisentangler(nn.Module):
    def __init__(self, cin, Disentangle_spatial='False',Disentangle_cbam='False',Disentangle_Fca='False'):
        super(NewDisentangler, self).__init__()
        self.activation_head = nn.Conv2d(cin, 1, kernel_size=3, padding=1, bias=False)
        self.bn_head = nn.BatchNorm2d(1)
        self.Disentangle_spatial=Disentangle_spatial
        self.Disentangle_cbam=Disentangle_cbam
        self.Disentangle_Fca=Disentangle_Fca
        if self.Disentangle_spatial == 'True':
            self.spatialattention = SpatialAttention(kernel_size=3)
        if self.Disentangle_cbam == 'True':
            self.cbam = CBAM(cin,ReduceDimensions='True')
        if self.Disentangle_Fca == 'True':
            self.FcaLayer=FcaLayer(cin)
    def forward(self, x, inference=False):
        N, C, H, W = x.size() # 获取输入特征图的尺寸
        if self.Disentangle_spatial == 'True':
            x = x * self.spatialattention(x)
        if self.Disentangle_cbam == 'True':
            x = self.cbam(x)
        if self.Disentangle_Fca == 'True':
            x = self.FcaLayer(x)

        if inference:
            ccam = self.bn_head(self.activation_head(x))  # 不使用Sigmoid激活，用于推理
        else:
            ccam = torch.sigmoid(self.bn_head(self.activation_head(x)))  # 使用Sigmoid激活，用于训练
        ccam_ = ccam.reshape(N, 1, H * W)                          # [N, 1, H*W]
        x = x.reshape(N, C, H * W).permute(0, 2, 1).contiguous()   # [N, H*W, C]
        # 计算前景和背景特征
        fg_feats = torch.matmul(ccam_, x) / (H * W)                # [N, 1, C]
        bg_feats = torch.matmul(1 - ccam_, x) / (H * W)            # [N, 1, C]

        # 计算全局平均池化后的特征
        x = x.permute(0, 2, 1).contiguous().view(N, C, H, W)
        # global_avg_pool = F.avg_pool2d(x.view(N, C, H, W), kernel_size=(H, W)).view(N, C, 1, 1) 
        global_avg_pool = F.adaptive_avg_pool2d(x, (1,1)).view(N, C, 1, 1)
        fg_feats = fg_feats.view(N, C, 1, 1)
        bg_feats = bg_feats.view(N, C, 1, 1)

        diff_fg = fg_feats - global_avg_pool
        diff_bg = bg_feats - global_avg_pool
        return fg_feats.reshape(x.size(0), -1), bg_feats.reshape(x.size(0), -1), ccam, diff_fg, diff_bg

class NewDisentangler_1(nn.Module):
    def __init__(self, cin, Disentangle_spatial='False',Disentangle_cbam='False',Disentangle_Fca='False'):
        super(NewDisentangler_1, self).__init__()
        self.activation_head = nn.Conv2d(cin, 1, kernel_size=3, padding=1, bias=False)
        self.bn_head = nn.BatchNorm2d(1)
        self.Disentangle_spatial=Disentangle_spatial
        self.Disentangle_cbam=Disentangle_cbam
        self.Disentangle_Fca=Disentangle_Fca
        if self.Disentangle_spatial == 'True':
            self.spatialattention = SpatialAttention(kernel_size=3)
        if self.Disentangle_cbam == 'True':
            self.cbam = CBAM(cin,ReduceDimensions='True')
        if self.Disentangle_Fca == 'True':
            self.FcaLayer=FcaLayer(cin)
    def forward(self, x, inference=False):
        N, C, H, W = x.size() # 获取输入特征图的尺寸
        if self.Disentangle_spatial == 'True':
            x = x * self.spatialattention(x)
        if self.Disentangle_cbam == 'True':
            x = self.cbam(x)
        if self.Disentangle_Fca == 'True':
            x = self.FcaLayer(x)

        if inference:
            ccam = self.bn_head(self.activation_head(x))  # 不使用Sigmoid激活，用于推理
        else:
            ccam = torch.sigmoid(self.bn_head(self.activation_head(x)))  # 使用Sigmoid激活，用于训练
        ccam_ = ccam.reshape(N, 1, H * W)                          # [N, 1, H*W]
        x = x.reshape(N, C, H * W).permute(0, 2, 1).contiguous()   # [N, H*W, C]
        # 计算前景和背景特征
        fg_feats = torch.matmul(ccam_, x) / (H * W)                # [N, 1, C]
        bg_feats = torch.matmul(1 - ccam_, x) / (H * W)            # [N, 1, C]
        # print(fg_feats.reshape(x.size(0), -1).size())
        # 计算全局平均池化后的特征
        x = x.permute(0, 2, 1).contiguous().view(N, C, H, W)
        global_feats = F.adaptive_avg_pool2d(x, (1,1)).view(N, C, 1, 1)
        return fg_feats.reshape(x.size(0), -1), bg_feats.reshape(x.size(0), -1), ccam, global_feats.reshape(x.size(0), -1)
    
class NewDisentangler_2(nn.Module):
    def __init__(self, cin, Disentangle_spatial='False',Disentangle_cbam='False',Disentangle_Fca='False'):
        super(NewDisentangler_2, self).__init__()
        self.activation_head = nn.Conv2d(cin, 1, kernel_size=3, padding=1, bias=False)
        self.bn_head = nn.BatchNorm2d(1)
        self.Disentangle_spatial=Disentangle_spatial
        self.Disentangle_cbam=Disentangle_cbam
        self.Disentangle_Fca=Disentangle_Fca
        if self.Disentangle_spatial == 'True':
            self.spatialattention = SpatialAttention(kernel_size=3)
        if self.Disentangle_cbam == 'True':
            self.cbam = CBAM(cin,ReduceDimensions='True')
        if self.Disentangle_Fca == 'True':
            self.FcaLayer=FcaLayer(cin)
    def forward(self, x, inference=False):
        N, C, H, W = x.size() # 获取输入特征图的尺寸
        if self.Disentangle_spatial == 'True':
            x = x * self.spatialattention(x)
        if self.Disentangle_cbam == 'True':
            x = self.cbam(x)
        if self.Disentangle_Fca == 'True':
            x = self.FcaLayer(x)
        if inference:
            ccam = self.bn_head(self.activation_head(x))  # 不使用Sigmoid激活，用于推理
        else:
            ccam = torch.sigmoid(self.bn_head(self.activation_head(x)))  # 使用Sigmoid激活，用于训练

        ccam_ = ccam.reshape(N, 1, H * W)                          # [N, 1, H*W]
        x = x.reshape(N, C, H * W).permute(0, 2, 1).contiguous()   # [N, H*W, C]
        # 计算前景和背景特征
        fg_feats = torch.matmul(ccam_, x) / (H * W)                # [N, 1, C]
        bg_feats = torch.matmul(1 - ccam_, x) / (H * W)            # [N, 1, C]

        # 先最大池化再最大池化后的特征
        x = x.permute(0, 2, 1).contiguous().view(N, C, H, W)
        global_feats = F.adaptive_max_pool2d(x,(1,1))
        return fg_feats.reshape(x.size(0), -1), bg_feats.reshape(x.size(0), -1), ccam, global_feats.reshape(x.size(0), -1)

class NewDisentangler_3(nn.Module):
    def __init__(self, cin,Disentangle_spatial='False',Disentangle_cbam='False',Disentangle_Fca='False'):
        super(NewDisentangler_3, self).__init__()
        self.activation_head = nn.Conv2d(cin, 1, kernel_size=3, padding=1, bias=False)
        self.bn_head = nn.BatchNorm2d(1)
        self.Disentangle_spatial=Disentangle_spatial
        self.Disentangle_cbam=Disentangle_cbam
        self.Disentangle_Fca=Disentangle_Fca
        if self.Disentangle_spatial == 'True':
            self.spatialattention = SpatialAttention(kernel_size=3)
        if self.Disentangle_cbam == 'True':
            self.cbam = CBAM(cin,ReduceDimensions='True')
        if self.Disentangle_Fca == 'True':
            self.FcaLayer=FcaLayer(cin)
    def forward(self, x, inference=False):
        N, C, H, W = x.size() # 获取输入特征图的尺寸#[256, 3072, 12, 12]
        if self.Disentangle_spatial == 'True':
            x = x * self.spatialattention(x)
        if self.Disentangle_cbam == 'True':
            x = self.cbam(x)
        if self.Disentangle_Fca == 'True':
            x = self.FcaLayer(x)

        if inference:
            ccam = self.bn_head(self.activation_head(x))  # 不使用Sigmoid激活，用于推理
        else:
            ccam = torch.sigmoid(self.bn_head(self.activation_head(x)))  # 使用Sigmoid激活，用于训练

        ccam_ = ccam.reshape(N, 1, H * W)                          # [N, 1, H*W]
        x = x.reshape(N, C, H * W).permute(0, 2, 1).contiguous()   # [N, H*W, C]
        # 计算前景和背景特征
        fg_feats = torch.matmul(ccam_, x) / (H * W)                # [N, 1, C]
        bg_feats = torch.matmul(1 - ccam_, x) / (H * W)            # [N, 1, C]

        x = x.reshape(N, H * W, C).permute(0, 2, 1).contiguous().view(N, C, H, W)
        # 先最大池化再全局平均池化后的特征
        global_max_pool = F.adaptive_max_pool2d(x, (2,2))
        global_feats = F.adaptive_avg_pool2d(global_max_pool,(1,1))

        return fg_feats.reshape(x.size(0), -1), bg_feats.reshape(x.size(0), -1), ccam, global_feats.reshape(x.size(0), -1)
class NewDisentangler_4(nn.Module):
    def __init__(self, cin, Disentangle_spatial='False',Disentangle_cbam='False',Disentangle_Fca='False'):
        super(NewDisentangler_4, self).__init__()
        self.activation_head = nn.Conv2d(cin, 1, kernel_size=3, padding=1, bias=False)
        self.bn_head = nn.BatchNorm2d(1)
        self.Disentangle_spatial=Disentangle_spatial
        self.Disentangle_cbam=Disentangle_cbam
        self.Disentangle_Fca=Disentangle_Fca
        if self.Disentangle_spatial == 'True':
            self.spatialattention = SpatialAttention(kernel_size=3)
        if self.Disentangle_cbam == 'True':
            self.cbam = CBAM(cin,ReduceDimensions='True')
        if self.Disentangle_Fca == 'True':
            self.FcaLayer=FcaLayer(cin)
    def forward(self, x, inference=False):
        N, C, H, W = x.size() # 获取输入特征图的尺寸
        if self.Disentangle_spatial == 'True':
            x = x * self.spatialattention(x)
        if self.Disentangle_cbam == 'True':
            x = self.cbam(x)
        if self.Disentangle_Fca == 'True':
            x = self.FcaLayer(x)
        if inference:
            ccam = self.bn_head(self.activation_head(x))  # 不使用Sigmoid激活，用于推理
        else:
            ccam = torch.sigmoid(self.bn_head(self.activation_head(x)))  # 使用Sigmoid激活，用于训练
        ccam_ = ccam.reshape(N, 1, H * W)                          # [N, 1, H*W]
        x = x.reshape(N, C, H * W).permute(0, 2, 1).contiguous()   # [N, H*W, C]
        # 计算前景和背景特征
        fg_feats = torch.matmul(ccam_, x) / (H * W)                # [N, 1, C]
        bg_feats = torch.matmul(1 - ccam_, x) / (H * W)            # [N, 1, C]

        x = x.permute(0, 2, 1).contiguous().view(N, C, H, W)
        # 先最大池化再全局平均池化后的特征
        global_feats = F.adaptive_max_pool2d(x,(1,1))
        fg_feats = fg_feats.view(N, C, 1, 1)
        bg_feats = bg_feats.view(N, C, 1, 1)
        cbam = CBAM(C,ReduceDimensions='UP').cuda()
        global_feats  = cbam(global_feats)
        fg_feats  = cbam(fg_feats)
        bg_feats  = cbam(bg_feats)
        return fg_feats.reshape(x.size(0), -1), bg_feats.reshape(x.size(0), -1), ccam, global_feats.reshape(x.size(0), -1)

class NewDisentangler_5(nn.Module):
    def __init__(self, cin, Disentangle_spatial='False',Disentangle_cbam='False',Disentangle_Fca='False'):
        super(NewDisentangler_5, self).__init__()
        self.activation_head = nn.Conv2d(cin, 1, kernel_size=3, padding=1, bias=False)
        self.bn_head = nn.BatchNorm2d(1)
    def forward(self, x, inference=False):
        N, C, H, W = x.size() # 获取输入特征图的尺寸
        if inference:
            ccam = self.bn_head(self.activation_head(x))  # 不使用Sigmoid激活，用于推理
        else:
            ccam = torch.sigmoid(self.bn_head(self.activation_head(x)))  # 使用Sigmoid激活，用于训练
        ccam_ = ccam.reshape(N, 1, H * W)                          # [N, 1, H*W]
        x = x.reshape(N, C, H * W).permute(0, 2, 1).contiguous()   # [N, H*W, C]
        # 计算前景和背景特征
        fg_feats = torch.matmul(ccam_, x) / (H * W)                # [N, 1, C]
        bg_feats = torch.matmul(1 - ccam_, x) / (H * W)            # [N, 1, C]
        x = x.permute(0, 2, 1).contiguous().view(N, C, H, W)
        # 先最大池化再全局平均池化后的特征
        global_max_pool = F.adaptive_max_pool2d(x, (2,2))
        global_feats = F.adaptive_avg_pool2d(global_max_pool,(1,1))

        fg_feats = fg_feats.view(N, C, 1, 1)
        bg_feats = bg_feats.view(N, C, 1, 1)
        cbam = CBAM(C,ReduceDimensions='UP').cuda()
        global_feats  = cbam(global_feats)
        fg_feats  = cbam(fg_feats)
        bg_feats  = cbam(bg_feats)

        return fg_feats.reshape(x.size(0), -1), bg_feats.reshape(x.size(0), -1), ccam, global_feats.reshape(x.size(0), -1)

class NewDisentangler_6(nn.Module):
    def __init__(self, cin, Disentangle_spatial='False',Disentangle_cbam='False',Disentangle_Fca='False'):
        super(NewDisentangler_6, self).__init__()
        self.activation_head = nn.Conv2d(cin, 1, kernel_size=3, padding=1, bias=False)
        self.bn_head = nn.BatchNorm2d(1)
        self.attention = SelfAttention(in_dim=cin)

    def forward(self, x, inference=False):
        N, C, H, W = x.size() # 获取输入特征图的尺寸
        if inference:
            ccam = self.bn_head(self.activation_head(x))  # 不使用Sigmoid激活，用于推理
        else:
            ccam = torch.sigmoid(self.bn_head(self.activation_head(x)))  # 使用Sigmoid激活，用于训练
        ccam_ = ccam.reshape(N, 1, H * W)                          # [N, 1, H*W]
        x = x.reshape(N, C, H * W).permute(0, 2, 1).contiguous()   # [N, H*W, C]
        # 计算前景和背景特征
        fg_feats = torch.matmul(ccam_, x) / (H * W)                # [N, 1, C]
        bg_feats = torch.matmul(1 - ccam_, x) / (H * W)            # [N, 1, C]

        x = x.permute(0, 2, 1).contiguous().view(N, C, H, W)
        x, attention = self.attention(x)
        global_feats = F.adaptive_max_pool2d(x,(1,1))
        return fg_feats.reshape(x.size(0), -1), bg_feats.reshape(x.size(0), -1), ccam, global_feats.reshape(x.size(0), -1)
    
class NewDisentangler_7(nn.Module):
    def __init__(self, cin, Disentangle_spatial='False',Disentangle_cbam='False',Disentangle_Fca='False'):
        super(NewDisentangler_7, self).__init__()
        self.activation_head = nn.Conv2d(cin, 1, kernel_size=3, padding=1, bias=False)
        self.bn_head = nn.BatchNorm2d(1)
        self.bg_suppression = BackgroundSuppression()
    def forward(self, x, inference=False):
        N, C, H, W = x.size() # 获取输入特征图的尺寸
        # x = self.bg_suppression(x)
        if inference:
            ccam = self.bn_head(self.activation_head(x))  # 不使用Sigmoid激活，用于推理
        else:
            ccam = torch.sigmoid(self.bn_head(self.activation_head(x)))  # 使用Sigmoid激活，用于训练
        ccam_ = ccam.reshape(N, 1, H * W)                          # [N, 1, H*W]
        x = x.reshape(N, C, H * W).permute(0, 2, 1).contiguous()   # [N, H*W, C]
        # 计算前景和背景特征
        fg_feats = torch.matmul(ccam_, x) / (H * W)                # [N, 1, C]
        bg_feats = torch.matmul(1 - ccam_, x) / (H * W)            # [N, 1, C]

        x = x.permute(0, 2, 1).contiguous().view(N, C, H, W)
        x = self.bg_suppression(x)
        global_feats = F.adaptive_max_pool2d(x,(1,1))
        return fg_feats.reshape(x.size(0), -1), bg_feats.reshape(x.size(0), -1), ccam, global_feats.reshape(x.size(0), -1)
    
class BackgroundSuppression(nn.Module):
    def __init__(self):
        super(BackgroundSuppression, self).__init__()
    def forward(self, x):
        # 输入特征图的全局平均值
        avg = torch.mean(x, dim=(2, 3), keepdim=True)
        # 计算输入特征图与全局平均值的差异
        diff = x - avg
        if torch.isnan(diff).any() or torch.isinf(diff).any():
            print("Input tensor contains NaN or Inf.")
        # 核函数 K(x) = - (1/pi) * arctan(|x|) + 1/2
        kernel = torch.exp(torch.abs(diff))  # 计算背景增强权重
        # 可以选择在这里记录或处理异常
        # 背景抑制计算公式
        out = x + diff * kernel
        # out = torch.clamp(out, -1, 1) #第二好
        out = torch.clamp(out, 0, 1) #第一好
        return out
    
class Network(nn.Module):
    def __init__(self, pretrained='mocov2', cin=None, New='False', CBAM='False',Disentangle_spatial='False',Disentangle_cbam='False',Disentangle_Fca='False'):
        super(Network, self).__init__()
        self.CBAM=CBAM
        self.NewDisentangler=New
        self.backbone = ResNetSeries(pretrained=pretrained,parameters=CBAM)
        if self.NewDisentangler=='False':
            self.ac_head = Disentangler(cin)
        elif self.NewDisentangler=='True':
            self.ac_head = NewDisentangler(cin,Disentangle_spatial=Disentangle_spatial,Disentangle_cbam=Disentangle_cbam,Disentangle_Fca=Disentangle_Fca)
        elif self.NewDisentangler=='1':
            self.ac_head = NewDisentangler_1(cin,Disentangle_spatial=Disentangle_spatial,Disentangle_cbam=Disentangle_cbam,Disentangle_Fca=Disentangle_Fca)
        elif self.NewDisentangler=='2':
            self.ac_head = NewDisentangler_2(cin,Disentangle_spatial=Disentangle_spatial,Disentangle_cbam=Disentangle_cbam,Disentangle_Fca=Disentangle_Fca)
        elif self.NewDisentangler=='3':
            self.ac_head = NewDisentangler_3(cin,Disentangle_spatial=Disentangle_spatial,Disentangle_cbam=Disentangle_cbam,Disentangle_Fca=Disentangle_Fca)
        elif self.NewDisentangler=='4':
            self.ac_head = NewDisentangler_4(cin,Disentangle_spatial=Disentangle_spatial,Disentangle_cbam=Disentangle_cbam,Disentangle_Fca=Disentangle_Fca)
        elif self.NewDisentangler=='5':
            self.ac_head = NewDisentangler_5(cin,Disentangle_spatial=Disentangle_spatial,Disentangle_cbam=Disentangle_cbam,Disentangle_Fca=Disentangle_Fca)
        elif self.NewDisentangler=='6':
            self.ac_head = NewDisentangler_6(cin,Disentangle_spatial=Disentangle_spatial,Disentangle_cbam=Disentangle_cbam,Disentangle_Fca=Disentangle_Fca)
        elif self.NewDisentangler=='7':
            self.ac_head = NewDisentangler_7(cin,Disentangle_spatial=Disentangle_spatial,Disentangle_cbam=Disentangle_cbam,Disentangle_Fca=Disentangle_Fca)
        self.from_scratch_layers = [self.ac_head]
    def forward(self, x, inference=False):
        feats = self.backbone(x)
        if self.NewDisentangler=='False':
            fg_feats, bg_feats, ccam = self.ac_head(feats, inference=inference)
            return fg_feats, bg_feats, ccam
        elif self.NewDisentangler=='True':
            fg_feats, bg_feats, ccam, diff_fg, diff_bg = self.ac_head(feats, inference=inference)
            return fg_feats, bg_feats, ccam, diff_fg, diff_bg
        else:
            fg_feats, bg_feats, ccam, global__feats = self.ac_head(feats, inference=inference)
            return fg_feats, bg_feats, ccam, global__feats
    def get_parameter_groups(self):
        groups = ([], [], [], [])
        print('======================================================')
        for m in self.modules():
            if (isinstance(m, nn.Conv2d) or isinstance(m, nn.modules.normalization.GroupNorm)):
                if m.weight.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[2].append(m.weight)
                    else:
                        groups[0].append(m.weight)
                if m.bias is not None and m.bias.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[3].append(m.bias)
                    else:
                        groups[1].append(m.bias)
        return groups

def get_model(pretrained, cin=None, NewDisentangler='False', CBAM='False',Disentangle_spatial='False',Disentangle_cbam='False',Disentangle_Fca='False'):
    if '3' in CBAM.split(','):
        cin=2048+1024+512
        # cin=2048+1024+2048
    else:
        cin=2048+1024
    return Network(pretrained=pretrained, cin=cin, New=NewDisentangler, CBAM=CBAM, Disentangle_spatial=Disentangle_spatial,Disentangle_cbam=Disentangle_cbam,Disentangle_Fca=Disentangle_Fca)


#####################使用自注意力机制提取全局特征###############################
class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        # self.gamma = nn.Parameter(torch.zeros(1))
        self.gamma = nn.Parameter(torch.ones(1))
    def forward(self, x):
        batch_size, C, width, height = x.size()
        # query, key, value
        query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)  # (N, H*W, C//8)
        key = self.key_conv(x).view(batch_size, -1, width * height)  # (N, C//8, H*W)
        value = self.value_conv(x).view(batch_size, -1, width * height)  # (N, C, H*W)
        # Attention map
        attention = torch.bmm(query, key)  # (N, H*W, H*W)
        attention = torch.softmax(attention, dim=-1)  # (N, H*W, H*W)
        # Weighted value
        out = torch.bmm(value, attention.permute(0, 2, 1))  # (N, C, H*W)
        out = out.view(batch_size, C, width, height)  # (N, C, H, W)
        out = self.gamma * out + x
        return out, attention
    
#####################使用全局注意力机制（GAM）提取全局特征###############################
class GlobalAttention(nn.Module):
    def __init__(self, in_dim):
        super(GlobalAttention, self).__init__()
        self.in_dim = in_dim
        self.theta = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.phi = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.g = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        # self.gamma = nn.Parameter(torch.zeros(1))
        self.gamma = nn.Parameter(torch.ones(1))
    def forward(self, x):
        batch_size, C, H, W = x.size()
        # Theta path
        theta = self.theta(x).view(batch_size, C, -1)  # (N, C, H*W)
        # Phi path
        phi = self.phi(x).view(batch_size, C, -1)  # (N, H*W, C)
        # Attention map
        attention = torch.bmm(theta.permute(0, 2, 1), phi)  # (N, H*W, H*W)
        attention = self.softmax(attention)  # Apply softmax to get attention map
        # g path
        g = self.g(x).view(batch_size, C, -1)  # (N, C, H*W)
        # Apply attention
        out = torch.bmm(g, attention)  # (N, C, H*W)
        out = out.view(batch_size, C, H, W)  # (N, C, H, W)
        # Weighted sum
        out = self.gamma * out + x
        return out, attention

#####################使用多头自注意力机制提取全局特征###############################
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, in_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        assert in_dim % num_heads == 0, "in_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = in_dim // num_heads
        
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.out_conv = nn.Conv2d(in_channels=in_dim // 8, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.ones(1))

    def forward(self, x):
        N, C, H, W = x.size()
        # Multi-head attention
        def split_heads(tensor, N):
            return tensor.view(N, self.num_heads, self.head_dim // 8, -1).permute(0, 1, 3, 2)  # (N, num_heads, H*W, head_dim // 8)

        query = split_heads(self.query_conv(x), N)  # (N, num_heads, H*W, head_dim // 8)
        key = split_heads(self.key_conv(x), N).permute(0, 1, 3, 2)  # (N, num_heads, head_dim // 8, H*W)
        value = split_heads(self.value_conv(x), N)  # (N, num_heads, H*W, head_dim // 8)

        # Scaled Dot-Product Attention
        attention = torch.matmul(query, key) / (self.head_dim ** 0.5)  # (N, num_heads, H*W, H*W)
        attention = torch.softmax(attention, dim=-1)  # (N, num_heads, H*W, H*W)

        out = torch.matmul(attention, value)  # (N, num_heads, H*W, head_dim // 8)
        out = out.permute(0, 1, 3, 2).contiguous()  # (N, num_heads, head_dim // 8, H*W)
        out = out.view(N, -1, H, W)  # (N, num_heads * head_dim // 8, H, W)
        out = self.out_conv(out)  # (N, in_dim, H, W)
        
        out = self.gamma * out + x
        return out, attention