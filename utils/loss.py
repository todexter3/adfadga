import torch 
import torch.nn as nn

class WeightedMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def get_x_V_P(self, batch_x, device):
        # 兼容性处理：判断是 [B, T, L, F] 还是 [N, L, F]
        dim = batch_x.dim()
        
        if dim == 4:
            # batch_x: [B, T, L, F]
            # 提取最后一个时间步的第1个特征，在 T 维度取均值
            phi = batch_x[:, :, -1, 1].mean(dim=1) 
        elif dim == 3:
            # batch_x: [N, L, F] (被有效掩码筛选后的数据)
            # 此时已经没有 T 维度了，直接取每个样本最后一个时间步的特征
            phi = batch_x[:, -1, 1] 
        else:
            raise ValueError(f"Unexpected batch_x dimension: {dim}")

        batch_size = phi.shape[0]
        V = torch.eye(batch_size, device=device)
        
        # 归一化 phi
        phi = (phi / (torch.linalg.norm(phi) + 1e-8)).unsqueeze(0) # [1, BatchSize]
        P = torch.mm(phi.T, phi) # [BatchSize, BatchSize]
        return V, P

    def forward(self, batch_x, outputs, targets, tau_hat=None, tau=None, c_norms=None):
        # 注意：这里的 outputs, targets 应该是已经根据 valid_mask 筛选过后的 [N] 维向量
        # 或者在 forward 内部进行掩码操作。
        
        # 1. 基础 MSE 计算
        error = outputs - targets
        mse_loss = torch.mean(error**2)
        
        if tau_hat is None:
            return {'total': mse_loss, 'mse': mse_loss}
        
        device = outputs.device
        V, P = self.get_x_V_P(batch_x, device)
        
        # 2. 截面误差向量 e_vec
        # 如果 outputs 是 [N]，那么 e_vec 就是 [N, 1]
        e_vec = error.unsqueeze(1)
        
        # 3. 计算加权损失
        # V 是单位阵，V*e_vec 其实就是 e_vec 本身，这里保留逻辑结构
        v_loss = torch.mean(torch.matmul(V, e_vec) ** 2)
        p_loss = torch.mean(torch.matmul(P, e_vec) ** 2)
        
        # 根据你的公式组合 total_loss
        total_loss = tau_hat * v_loss + tau * p_loss
        
        return {'total': total_loss, 'mse': mse_loss}