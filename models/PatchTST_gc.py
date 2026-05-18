import torch
from torch import nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding_group, FlattenHead
from layers.RevIN import RevIN

import torch.nn.functional as F

class IndustryAttentionPooling(nn.Module):
    """
    使用注意力机制从行业内的个股特征中聚合出行业代表向量
    """
    def __init__(self, d_model):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_model)) # 可学习的行业查询向量
        self.attn = nn.MultiheadAttention(d_model, n_heads=1, batch_first=True)
        
    def forward(self, x, mask=None):
        # x: [N_stocks_in_industry, P, D]
        # 这里的聚合是在个股维度进行的
        # 将 P (Patch) 维度平铺或者对每个 Patch 分别做聚合
        N, P, D = x.shape
        # 简单的做法：先对 Patch 维做平均，提取个股的核心特征
        stock_summary = x.mean(dim=1) # [N, D]
        
        # 使用 Cross-Attention 聚合
        # Query: 固定的行业模板, Key/Value: 行业内所有个股
        attn_out, _ = self.attn(self.query.expand(1, -1, -1), 
                               stock_summary.unsqueeze(0), 
                               stock_summary.unsqueeze(0))
        return attn_out.squeeze(0) # [1, D]

class HierarchicalInteraction(nn.Module):
    def __init__(self, d_model, n_heads, num_industries, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_industries = num_industries
        
        self.intra_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.inter_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        
        self.industry_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout)
        )

        self.res_scale = nn.Parameter(torch.full((1,), 0.1))
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, group_feat, industry_ids, mask_data):
        B, T, P, D = group_feat.shape
        device = group_feat.device

        # --- Step 1: 提取行业代表向量 ---
        ind_sum = torch.zeros(B, self.num_industries, P, D, device=device)
        # 记录每个行业有多少个有效股票
        ind_count = torch.zeros(B, self.num_industries, 1, 1, device=device)
        
        m = mask_data.view(B, T, 1, 1)
        expanded_ids = industry_ids.view(B, T, 1, 1).expand(-1, -1, P, D)
        
        ind_sum.scatter_add_(1, expanded_ids, group_feat * m)
        ind_count.scatter_add_(1, industry_ids.view(B, T, 1, 1), m)
        
        # 识别今天完全不存在的行业
        # ind_present: [B, I], True表示该行业有股票，False表示全空
        ind_present = (ind_count.view(B, self.num_industries) > 0)
        
        # 安全除法：防止分母为0
        protos = ind_sum / (ind_count + 1e-9)

        # --- Step 2: 行业间交互 (增加 Padding Mask) ---
        inter_in = protos.mean(dim=2) # [B, I, D]
        
        # 生成行业掩码：attn 需要的是 (B, I)，1/True 表示被屏蔽(没有股票的行业)
        # 注意：PyTorch MultiheadAttention 的 key_padding_mask 中 True 是 mask 掉
        industry_mask = ~ind_present 
        
        # 行业间自注意力，只在存在的行业之间进行
        inter_out, _ = self.inter_attn(
            inter_in, inter_in, inter_in, 
            key_padding_mask=industry_mask
        )
        
        # 只对存在的行业更新增强特征
        protos_enhanced = self.industry_mlp((inter_in + inter_out)) # [B, I, D]
        # 只有存在的行业才叠加增强信息，缺失行业保持为0
        protos = protos + (protos_enhanced * ind_present.unsqueeze(-1)).unsqueeze(2)

        # --- Step 3: 行业内反馈 ---
        target_protos = torch.gather(protos, 1, expanded_ids) # [B, T, P, D]

        q = group_feat.reshape(B * T, P, D)
        kv = target_protos.reshape(B * T, P, D)
        
        # 如果某股票所属行业今天不存在，其 target_proto 为 0
        refined_feat, _ = self.intra_attn(q, kv, kv)
        refined_feat = refined_feat.reshape(B, T, P, D)

        # --- Step 4: 强残差融合 ---
        combined = torch.cat([group_feat, refined_feat], dim=-1)
        g = self.gate(combined)
        
        # mask_data 再次确保无效股票（填充位置）不产生输出贡献
        out = group_feat + self.res_scale * (g * refined_feat)
        out = out * m # 显式屏蔽无效股票特征
        
        res = out
        out = self.norm(out.reshape(-1, D)) 
        out = self.ffn(out).reshape(B, T, P, D) 
        
        return (out + res) * m # 最终返回前再次确保 Mask


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False): 
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)


class MultiScaleConv(nn.Module):
    """轻量并行多尺度卷积（在 patch/time 维上提取短/中/长期特征）"""
    def __init__(self, d_model, kernel_sizes=(3,7,15)):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=k, padding=k//2, bias=True)
            for k in kernel_sizes
        ])
        self.proj = nn.Linear(len(kernel_sizes)*d_model, d_model)
        self.act = nn.GELU()

    def forward(self, x):
        # x: [B_total, seq_len, d_model]
        x = x.transpose(1, 2)     
        outs = [conv(x) for conv in self.convs]   
        out = torch.cat(outs, dim=1)              
        out = out.transpose(1, 2)                  
        out = self.proj(out)                      
        return self.act(out)                      


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.alpha = torch.nn.Parameter(torch.tensor(configs.tau_hat_init))
        padding = configs.stride
        patch_len = configs.patch_len
        stride = configs.stride
        self.configs = configs

        # feature groups
        self.feature_group = [[0,1],[2],[3,4,5,6,7],[8],[9]] 

        self.revin_layer = RevIN(configs.enc_in, affine=True)

        self.multi_scale = MultiScaleConv(configs.d_model, kernel_sizes=(3,7,15))

        self.patch_embedding = PatchEmbedding_group(configs.d_model, patch_len, stride, padding, configs.dropout, self.feature_group)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False), 
                        configs.d_model, configs.n_heads
                    ),
                    configs.d_model, configs.d_ff, dropout=configs.dropout, activation=configs.activation
                ) 
                for _ in range(configs.e_layers)
            ],
            norm_layer=nn.Sequential(Transpose(1,2), nn.BatchNorm1d(configs.d_model), Transpose(1,2))
        )
        
       
        # 行业交互模块
        self.industry_interaction = HierarchicalInteraction(configs.d_model, configs.n_heads,configs.num_industries)
        
        # 特征融合层：将原始时序特征与行业交互特征拼接降维
        self.fusion_mlp = nn.Sequential(
            nn.Linear(configs.d_model * 2, configs.d_model * 2),
            nn.GELU(),
            nn.Linear(configs.d_model * 2, configs.d_model)
        )
        # Prediction Head    
        
        self.flatten = nn.Flatten(start_dim=-2)
        self.dropout = nn.Dropout(configs.dropout)
        
        self.head_nf = (configs.seq_len + padding - patch_len) // stride + 1
        # 直接输出 1 维分数，不需要与输入维度对齐，因为我们移除了 Denorm
        self.patch_reduction = nn.Linear(self.head_nf, 8) # 将 Patch 维从 head_nf 降到 8
        #d_flatten = len(self.feature_group) * configs.d_model * 8
        #d_flatten = len(self.feature_group) * configs.d_model * self.head_nf
        d_flatten = len(self.feature_group) * configs.d_model 

        self.projection = nn.Sequential(
            nn.Linear(d_flatten, configs.d_model * 4), # 55M 参数量说明之前这里太大了，建议降维
            nn.BatchNorm1d(configs.d_model * 4),
            nn.GELU(),
            nn.Dropout(configs.dropout),
            nn.Linear(configs.d_model * 4, configs.d_model),
            nn.GELU(),
            nn.Linear(configs.d_model, 1)
        )
        
  
    def regression(self, x_total, industry_ids, mask_data):
        B, T, L, F = x_total.shape

     
        m = mask_data.unsqueeze(-1).unsqueeze(-1) # [B, T, 1, 1]
        valid_count = m.sum(dim=1, keepdim=True) + 1e-6
        
        x_mean = (x_total * m).sum(dim=1, keepdim=True) / valid_count
        x_var = (((x_total - x_mean) * m) ** 2).sum(dim=1, keepdim=True) / valid_count
        x_std = torch.sqrt(x_var + 1e-6)
        
        # 标准化，同时将停牌/无效票置为 0
        x_total = (x_total - x_mean) / x_std * m 

        x_enc = x_total.reshape(B * T, L, F)

        # 针对时序维度的 RevIN，用于对齐不同 feature 的尺度
        x_enc = self.revin_layer(x_enc, mode='norm')

        # Patch Embedding & Encoder 
        x_in = x_enc.permute(0, 2, 1) 
        enc_out_list, n_vars = self.patch_embedding(x_in)
        enc_inputs = torch.cat(enc_out_list, dim=0) # [Num_Groups * B * T, P, D]

        enc_inputs = self.multi_scale(enc_inputs)
        
        enc_outputs, _ = self.encoder(enc_inputs)


        num_groups = len(self.feature_group)
        enc_outputs = enc_outputs.reshape(num_groups, B, T, -1, self.configs.d_model) 

        final_outputs = []
        for g in range(num_groups):
            raw_feat = enc_outputs[g] # 纯个股的时序特征 [B, T, P, D]
            
            # 行业交互特征
            inter_feat = self.industry_interaction(raw_feat, industry_ids, mask_data)
            

            combined = torch.cat([raw_feat, inter_feat], dim=-1) # [B, T, P, 2D]
            fused_feat = self.fusion_mlp(combined.reshape(-1, self.configs.d_model * 2)).reshape(B, T, -1, self.configs.d_model) # [B, T, P, D]
            
            final_outputs.append(fused_feat)

    
        out = torch.stack(final_outputs, dim=0).permute(1, 2, 0, 3, 4) # [B, T, Groups, P, D]
        
        

        out = out.mean(dim=-2) # 对 Patch 维度取平均，[B, T, Groups, D]
        output = out.reshape(B * T, -1) # [B*T, Groups * D]
        pred = self.projection(output)
    


        return pred.reshape(B, T)* mask_data
    
    def forward(self, x_total, industry_ids, mask_data):
        return self.regression(x_total, industry_ids, mask_data)