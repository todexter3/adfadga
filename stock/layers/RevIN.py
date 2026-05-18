import torch
from torch import nn
import torch
from torch import nn

class RevIN(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.gamma = nn.Parameter(torch.ones(num_features))
            self.beta = nn.Parameter(torch.zeros(num_features))
        
        # 关键修改：注册为 buffer，而不是动态挂载属性
        # persistent=False 意味着它们不会被保存在 pth 模型权重文件中
        self.register_buffer('mean', torch.zeros(1), persistent=False)
        self.register_buffer('stdev', torch.zeros(1), persistent=False)

    def forward(self, x, mode):
        if mode == 'norm':
            # 直接赋值给 buffer
            self.mean = torch.mean(x, dim=1, keepdim=True).detach()
            self.stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + self.eps).detach()
            
            x = (x - self.mean) / self.stdev
            if self.affine:
                x = x * self.gamma + self.beta
                
        elif mode == 'denorm':
            if self.affine:
                x = (x - self.beta) / self.gamma
            # 现在 stdev 和 mean 是已知的 buffer，不会报 KeyError
            x = x * self.stdev + self.mean
        return x