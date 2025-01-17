import torch
import torch.nn as nn
import torch.nn.functional as F

# 计算两个输入向量之间的余弦相似度
def cos_simi(embedded_fg, embedded_bg):
    embedded_fg = F.normalize(embedded_fg, dim=1)
    embedded_bg = F.normalize(embedded_bg, dim=1)
    sim = torch.matmul(embedded_fg, embedded_bg.T)
    return torch.clamp(sim, min=0.0005, max=0.9995)
    # return torch.clamp(sim, min=-0.9995, max=0.9995)

def cos_distance(embedded_fg, embedded_bg):
    embedded_fg = F.normalize(embedded_fg, dim=1)
    embedded_bg = F.normalize(embedded_bg, dim=1)
    sim = torch.matmul(embedded_fg, embedded_bg.T)
    return 1 - sim

def l2_distance(embedded_fg, embedded_bg):
    N, C = embedded_fg.size()
    # embedded_fg = F.normalize(embedded_fg, dim=1)
    # embedded_bg = F.normalize(embedded_bg, dim=1)
    embedded_fg = embedded_fg.unsqueeze(1).expand(N, N, C)
    embedded_bg = embedded_bg.unsqueeze(0).expand(N, N, C)
    return torch.pow(embedded_fg - embedded_bg, 2).sum(2) / C

# Minimize Similarity, e.g., push representation of foreground and background apart.
# 用于最小化相似度的损失函数 SimMinLoss 类。它的作用是推开前景和背景的表示，以确保它们在特征空间中不相似。
class SimMinLoss(nn.Module):
    def __init__(self, metric='cos', reduction='mean'):
        super(SimMinLoss, self).__init__()
        self.metric = metric
        self.reduction = reduction
    def forward(self, embedded_bg, embedded_fg):
        """
        :param embedded_fg: [N, C]
        :param embedded_bg: [N, C]
        """
        if self.metric == 'l2':
            raise NotImplementedError
        elif self.metric == 'cos':
            sim = cos_simi(embedded_bg, embedded_fg)
            loss = -torch.log(1 - sim)
        else:
            raise NotImplementedError
        if self.reduction == 'mean':
            return torch.mean(loss),sim
        elif self.reduction == 'sum':
            return torch.sum(loss),sim

# Maximize Similarity, e.g., pull representation of background and background together.
# 用于最大化相似度的损失函数 SimMaxLoss 类。它的作用是拉近前景和前景、背景和背景之间的表示，以确保它们在特征空间中更相似。
class SimMaxLoss(nn.Module):
    def __init__(self, metric='cos', alpha=0.25, reduction='mean'):
        super(SimMaxLoss, self).__init__()
        self.metric = metric
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, embedded_bg):
        """
        :param embedded_fg: [N, C]
        :param embedded_bg: [N, C]
        :return:
        """
        if self.metric == 'l2':
            raise NotImplementedError

        elif self.metric == 'cos':
            sim = cos_simi(embedded_bg, embedded_bg)
            loss = -torch.log(sim)
            # loss = 1 / (sim + 1) - 0.5
            # loss = torch.exp(-sim) - torch.exp(-1)
            loss[loss < 0] = 0
            _, indices = sim.sort(descending=True, dim=1)
            _, rank = indices.sort(dim=1)
            rank = rank - 1
            rank_weights = torch.exp(-rank.float() * self.alpha)
            loss = loss * rank_weights
        else:
            raise NotImplementedError

        if self.reduction == 'mean':
            return torch.mean(loss),sim
        elif self.reduction == 'sum':
            return torch.sum(loss),sim
        
# 减小前景特征与全局特征的相似度的损失函数
class DecreaseSimLoss(nn.Module):
    def __init__(self, metric='cos', reduction='mean'):
        super(DecreaseSimLoss, self).__init__()
        self.metric = metric
        self.reduction = reduction
    def forward(self, embedded_gg, embedded_fg):
        """
        :param embedded_fg: [N, C]
        :param embedded_gg: [N, C]
        """
        if self.metric == 'l2':
            raise NotImplementedError
        elif self.metric == 'cos':
            sim = cos_simi(embedded_gg, embedded_fg)
            sim = torch.diag(sim)
            loss = -torch.log(1 - sim)
        else:
            raise NotImplementedError
        if self.reduction == 'mean':
            return torch.mean(loss),sim
        elif self.reduction == 'sum':
            return torch.sum(loss),sim
        
# 增大背景特征与全局特征的相似度的损失函数
class IncreaseSimLoss(nn.Module):
    def __init__(self, metric='cos', alpha=0.25, reduction='mean'):
        super(IncreaseSimLoss, self).__init__()
        self.metric = metric
        self.alpha = alpha
        self.reduction = reduction
    def forward(self, embedded_bg, embedded_gg):
        """
        :param embedded_gg: [N, C]
        :param embedded_bg: [N, C]
        :return:
        """
        if self.metric == 'l2':
            raise NotImplementedError
        elif self.metric == 'cos':
            sim = cos_simi(embedded_bg, embedded_gg)
            sim = torch.diag(sim)
            loss = -torch.log(sim)
        else:
            raise NotImplementedError
        if self.reduction == 'mean':
            return torch.mean(loss),sim
        elif self.reduction == 'sum':
            return torch.sum(loss),sim
        
# 增大前景特征与全局平均池化后的特征的差异的损失函数
class IncreaseFgDiffLoss(nn.Module):
    def __init__(self, reduction='mean',alpha=0.25):
        super(IncreaseFgDiffLoss, self).__init__()
        self.reduction = reduction
        self.alpha = alpha
    def forward(self, diff_fg):
        """
        :param fg_feats: 前景特征，维度为(N, C)。
        :param global_avg_pool: 全局平均池化后的特征，维度为(N, C)。
        :return: 增大前景特征与全局平均池化后的特征的差异的损失值
        """
        # loss = self.alpha * 1 / torch.norm(diff_fg, dim=1)  # 使用L2范数的倒数作为损失函数
        loss = 1 / torch.norm(diff_fg, dim=1)  # 使用L2范数的倒数作为损失函数
        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        
# 减小背景特征与全局平均池化后的特征的差异的损失函数
class DecreaseBgDiffLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(DecreaseBgDiffLoss, self).__init__()
        self.reduction = reduction
        # self.sigmoid = nn.Sigmoid()
    def forward(self, diff_bg):
        """
        :param bg_feats: 背景特征，维度为(N, C)。
        :param global_avg_pool: 全局平均池化后的特征，维度为(N, C)。
        :return: 减小背景特征与全局平均池化后的特征的差异的损失值
        """
        # min_val = diff_bg.min(dim=1, keepdim=True)[0]
        # max_val = diff_bg.max(dim=1, keepdim=True)[0]
        # diff_bg = (diff_bg - min_val) / (max_val - min_val + 1e-8)  # 加 1e-8 避免除以零

        loss = torch.norm(diff_bg, p=2, dim=1)  # 使用L2范数作为损失函数
        # loss = self.sigmoid(loss) 
        max_loss = loss.max().clone().detach().requires_grad_(True)
        loss = torch.log(loss + 1) / torch.log(max_loss + 1.0)
        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        
if __name__ == '__main__':
    fg_embedding = torch.randn((3, 4))
    bg_embedding = torch.randn((3, 4))
    print('fg_embedding:\n', fg_embedding)
    print('bg_embedding:\n', bg_embedding)
    print('cos_simi fg_embedding:\n',cos_simi(fg_embedding, fg_embedding))
    print('cos_simi bg_embedding,fg_embedding:\n',cos_simi(fg_embedding, bg_embedding))
    print('cos_simi bg_embedding:\n',cos_simi(bg_embedding, bg_embedding))
    print()
    loss1,sim1=SimMaxLoss()(fg_embedding)
    print('SimMaxLoss fg_embedding:\n',loss1)
    print("sim1:\n",sim1)
    loss2,sim2=SimMinLoss()(bg_embedding,fg_embedding)
    print('SimMinLoss bg_embedding,fg_embedding:\n',loss2)
    print("sim2:\n",sim2)
    loss3,sim3=SimMaxLoss()(bg_embedding)
    print('SimMaxLoss bg_embedding:\n',loss3)
    print("sim3:\n",sim3)
    # constraint_term = torch.sigmoid(sim3 - sim2) - 0.5
    constraint_term = torch.sqrt(2*(1-sim3))-torch.sqrt(2*(1-sim2))
    print('constraint_term:\n',constraint_term)
    constraint_term = constraint_term.mean()  # 将约束项的值取平均，确保它是一个标量
    print('constraint_term:\n',constraint_term)
    print()
    # neg_contrast = NegContrastiveLoss(metric='cos')
    # neg_loss = neg_contrast(fg_embedding, bg_embedding)
    # print(neg_loss)

    # pos_contrast = PosContrastiveLoss(metric='cos')
    # pos_loss = pos_contrast(fg_embedding)
    # print(pos_loss)

    examplar = torch.tensor([[1, 2, 3, 4], [2, 3, 1, 4], [4, 2, 1, 3]])

    _, indices = examplar.sort(descending=True, dim=1)
    print(indices)
    _, rank = indices.sort(dim=1)
    print(rank)
    rank_weights = torch.exp(-rank.float() * 0.25)
    print(rank_weights)
