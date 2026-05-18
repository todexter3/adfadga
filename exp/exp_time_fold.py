import torch
import torch.nn as nn
from torch import optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import Dataset
from torch.utils.data import Subset, DataLoader, ConcatDataset
import torch.nn.functional as F

import numpy as np
import pandas as pd
import ast
import os
import time
import warnings
from sklearn.preprocessing import RobustScaler
import joblib
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

from utils.loss import WeightedMSELoss
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric
import utils.plt_heiyi as plt_heiyi

import data_provider.data_loader_time_fold as data_loader_stock_daily
from data_provider.data_loader_time_fold import daily_collate_fn

from exp.exp_basic import Exp_Basic


warnings.filterwarnings('ignore')



def print_model_parameters(model, only_num=True):
    print('*************************Model Total Parameter************************')
    if not only_num:
        for name, param in model.named_parameters():
            print(name, param.shape, param.requires_grad)
    total_num = sum([param.nelement() for param in model.parameters()])
    print('Total params num: {}'.format(total_num))
    print('************************Finish Parameter************************')


class CCC(nn.Module):
    def __init__(self):
        super(CCC, self).__init__()
        self.cos = nn.CosineSimilarity(dim=0, eps=1e-6)

    def forward(self, x, y):
        x_mean = torch.mean(x)
        y_mean = torch.mean(y)
        numerator = torch.sum((x - x_mean) * (y - y_mean))
        denominator = torch.sqrt(torch.sum((x - x_mean) ** 2) * torch.sum((y - y_mean) ** 2))
        correlation_coefficient = numerator / denominator
        loss = 1 - correlation_coefficient
        return loss


def cosine_similarity(x, y, dim=2, eps=1e-6):
    """
    计算两个张量之间的余弦相似度。

    参数:
        x (Tensor): 形状为 [batch_size, seq_len, dim] 的张量。
        y (Tensor): 形状为 [batch_size, seq_len, dim] 的张量。
        dim (int): 沿着哪个维度计算余弦相似度（默认为最后一个维度）。
        eps (float): 为了防止除以零而添加的小常数。

    返回:
        Tensor: 形状为 [batch_size, seq_len] 的张量，包含每对向量的余弦相似度。
    """
    x_norm = torch.norm(x, dim=dim, keepdim=True)
    y_norm = torch.norm(y, dim=dim, keepdim=True)
    product = (x * y).sum(dim=dim, keepdim=True)
    cosine = product / (x_norm * y_norm + eps)
    return cosine.squeeze(dim=dim)


def interval_loss(y_true, y_pred):
    """
    区间损失函数，包括区间覆盖损失和区间宽度损失
    y_pred[:, 0] 是区间下界，y_pred[:, 1] 是区间上界
    """
    # 区间覆盖损失：如果真实值不在预测区间内，则惩罚
    coverage_loss = torch.where(
        (y_true >= y_pred[:, 0]) & (y_true <= y_pred[:, 1]),
        torch.tensor(0.0, device=y_true.device),
        torch.min((y_true - y_pred[:, 0]) ** 2, (y_true - y_pred[:, 1]) ** 2)
    ).mean()

    # 区间宽度损失：惩罚过宽的区间
    width_loss = ((y_pred[:, 1] - y_pred[:, 0]) ** 2).mean()

    # 总损失是覆盖损失和宽度损失的加权和
    total_loss = coverage_loss + 0.5 * width_loss
    return total_loss


def cosine_loss(x, y, dim=1, reduction='mean'):
    """
    计算余弦损失函数。

    参数:
        x (Tensor): 形状为 [batch_size, seq_len, dim] 的张量。
        y (Tensor): 形状为 [batch_size, seq_len, dim] 的张量。
        dim (int): 沿着哪个维度计算余弦相似度（默认为最后一个维度）。
        reduction (str): 指定如何减少损失：'none' | 'mean' | 'sum'。

    返回:
        Tensor: 如果 reduction 不是 'none'，则返回一个标量损失值；否则返回与输入形状相同的张量。
    """
    x = x[:, -1]
    y = y[:, -1]
    cosine_sim = cosine_similarity(x, y, dim)
    if reduction == 'none':
        return 1 - cosine_sim
    elif reduction == 'mean':
        return 1 - cosine_sim.mean()
    elif reduction == 'sum':
        return 1 - cosine_sim.sum()
    else:
        raise ValueError(f"Unsupported reduction: {reduction}")


def correlation_loss(x, y):
    # x = x[:,-1:]
    # y = y[:,-1:]
    x_mean = torch.mean(x)
    y_mean = torch.mean(y)
    numerator = torch.sum((x - x_mean) * (y - y_mean))
    denominator = torch.sqrt(torch.sum((x - x_mean) ** 2) * torch.sum((y - y_mean) ** 2))
    correlation_coefficient = numerator / denominator
    loss = 1 - correlation_coefficient
    return loss


class TDataset(Dataset):

    def __init__(self, X, y, time_gra):
        self.X = X
        self.time_gra = time_gra
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx], self.time_gra[idx]


class Exp_Multiple_Regression_Fold(Exp_Basic):
    def __init__(self, args, single_fold=None):
        super(Exp_Multiple_Regression_Fold, self).__init__(args)
        self.all_test_preds = np.array([])
        self.single_fold = single_fold  # 如果指定则只训练单个fold
        # self.all_test_trues = []
        # self.args = args
        # self.device = self._acquire_device()
        # self.model = model.to(self.device)
        # self.model = model.to(self.device)

    def _build_model(self):
        # model init
        self.model = self.model_dict[self.args.model].Model(self.args).float().to(self.device)
        if self.args.use_multi_gpu and self.args.use_gpu:
            # self.model = nn.DataParallel(self.model, device_ids=[3,0,1,2])
            self.model = nn.DataParallel(self.model, device_ids=self.args.device_ids)
        # for name, param in self.model.named_parameters():
        #     if param.requires_grad:
        #         print(name, param.data.shape)
        # sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        # print("Number of Parameters:", sum(p.numel() for p in self.model.parameters() if p.requires_grad))
        import torch
        if int(torch.__version__.split('.')[0]) >= 2:
            print("Optimizing model with torch.compile...")
            import torch._inductor.config
            torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = True
            #self.model = torch.compile(self.model,dynamic=True)
        return self.model

    def _acquire_device(self):
        if self.args.use_gpu:
            # os.environ["CUDA_VISIBLE_DEVICES"] = str(
            #     self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def __init_normal_(self, model, init_type='norm '):
        if init_type == 'norm':
            for m in model.parameters():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight.data)
                    nn.init.constant_(m.bias, 0)
                if isinstance(m, nn.LSTM):
                    nn.init.xavier_normal_(m.weight_ih_l0)
                    nn.init.orthogonal_(m.weight_hh_l0)
                    nn.init.constant_(m.bias_ih_l0, 0)
                    nn.init.constant_(m.bias_hh_l0, 0)
        elif init_type == 'unif':
            for p in model.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
                else:
                    nn.init.uniform_(p)

    
    def _get_data(self, flag):
        self.args.size = [self.args.seq_len]
        if flag == 'train':
            return data_loader_stock_daily.Dataset_regression_train_val(self.args)
        elif flag == 'test':
            return data_loader_stock_daily.Dataset_regression_test(self.args)    
    
    def _select_optimizer(self):
        optim_type = self.args.optim_type
        if optim_type == 'SGD':
            model_optim = optim.SGD(self.model.parameters(), lr=self.args.learning_rate, momentum=0.9,
                                    weight_decay=self.args.weight_decay)
        elif optim_type == 'Adam':
            model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate,
                                     weight_decay=self.args.weight_decay)
        else:
            raise ValueError("can't find your optimizer! please defined a optimizer!")
        scheduler = None
        if self.args.lradj == 'cos':
            scheduler = lr_scheduler.CosineAnnealingLR(model_optim, T_max=self.args.train_epochs // 2)
        elif self.args.lradj == 'steplr':
            scheduler = lr_scheduler.StepLR(model_optim, step_size=2, gamma=0.5)
        return model_optim, scheduler

    def _select_criterion(self):
        loss_func = self.args.loss
        if loss_func == 'MSE':
            criterion = nn.MSELoss(reduction='none')
        elif loss_func == 'MSE_with_weak':
            criterion = WeightedMSELoss()
        else:
            raise ValueError("can't find your loss function! please defined it!")
        return criterion

    def prepared_dataset(self, train_data):
        # train_data 现在每个 item 返回: (day_x, day_y, day_info)
        # day_info 是一个 List[Dict]
        dates = []
        tickers = []
        # 这里只是为了统计日期分布，batch_size 可以大一些
        temp_loader = DataLoader(dataset=train_data, batch_size=1, shuffle=False, num_workers=4)
        
        for i, (_, _, day_info) in enumerate(temp_loader):
            # 注意：DataLoader 会把 List[Dict] 变成 Dict[List]
            # day_info['CalcDate'] 现在是一个 tuple/list，长度为当天股票数
            d = day_info['CalcDate'][0] # 取当天第一个股票的日期即可
            dates.append(d)
            # tickers 这里如果不需要精确统计，可以跳过，或者：
            # tickers.extend(day_info['Code'])

        dates = np.array(dates)
        unique_dates = np.unique(dates)
        sorted_dates = np.sort(unique_dates)
        num_dates = len(sorted_dates)

        return num_dates, sorted_dates, dates
        

    def load_dataset(self, num_dates, fold, sorted_dates, dates, train_data):
        fold_size = num_dates // self.args.num_fold
        start_idx = fold * fold_size
        end_idx = (fold + 1) * fold_size if fold != self.args.num_fold - 1 else num_dates
        val_dates = sorted_dates[start_idx:end_idx]

        if (fold + 1) == 1:
            train_dates = np.concatenate([sorted_dates[:start_idx], sorted_dates[end_idx + self.args.pred_task:]])
        elif (fold + 1) > 1 and (fold + 1) < self.args.num_fold:
            train_dates = np.concatenate(
                [sorted_dates[:start_idx - self.args.pred_task], sorted_dates[end_idx + self.args.pred_task:]])
        elif (fold + 1) == self.args.num_fold:
            train_dates = np.concatenate([sorted_dates[:start_idx - self.args.pred_task], sorted_dates[end_idx:]])

        val_indices = np.where(np.isin(dates, val_dates))[0]
        train_indices = np.where(np.isin(dates, train_dates))[0]

        # 直接复用原始 Dataset，零内存拷贝！
        train_dataset = Subset(train_data, train_indices)
        val_dataset = Subset(train_data, val_indices)

        train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True, pin_memory=True,
                                  drop_last=False, num_workers=4, collate_fn=daily_collate_fn)  # 建议 num_workers 降到 4 缓解 CPU 内存压力
        vali_loader = DataLoader(val_dataset, batch_size=self.args.batch_size, shuffle=False, pin_memory=True,
                                 drop_last=False, num_workers=4, collate_fn=daily_collate_fn)

        return train_dataset, train_loader, val_dataset, vali_loader

    def load_dataset_time(self, num_dates, fold, sorted_dates, data_x, data_y, dates):
        val_size = int(0.2 * num_dates)
        each_train_size = (num_dates - val_size) // self.args.num_fold
        start_idx = fold * each_train_size  # train idx
        end_idx = num_dates - val_size
        # 去掉val前面的
        val_dates = sorted_dates[end_idx + self.args.pred_task:]
        train_dates = sorted_dates[start_idx:end_idx]

        # 创建mask
        val_mask = np.isin(dates, val_dates)
        train_mask = np.isin(dates, train_dates)
        # 分割数据
        train_set_x = data_x[train_mask]
        train_set_y = data_y[train_mask]
        val_set_x = data_x[val_mask]
        val_set_y = data_y[val_mask]
        dates_train_x = dates[train_mask]
        dates_val_x = dates[val_mask]

        # 创建数据集和数据加载器
        train_dataset = TDataset(train_set_x, train_set_y, dates_train_x)
        val_dataset = TDataset(val_set_x, val_set_y, dates_val_x)
        train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True, pin_memory=True,
                                  drop_last=False, num_workers=8, collate_fn=daily_collate_fn)
        vali_loader = DataLoader(val_dataset, batch_size=self.args.batch_size, shuffle=False, pin_memory=True,
                                 drop_last=False, num_workers=8, collate_fn=daily_collate_fn)
        return train_dataset, train_loader, val_dataset, vali_loader

    def load_dataset_last(self, num_dates, fold, sorted_dates, data_x, data_y, dates):
        # 计算当前fold的验证集日期范围 最后20%
        fold_size = int(0.2 * num_dates)
        start_idx = 0
        end_idx = num_dates - fold_size  # train idx
        train_dates = sorted_dates[start_idx:end_idx - self.args.pred_task]
        val_dates = np.concatenate([sorted_dates[:start_idx], sorted_dates[end_idx:]])
        # 创建mask
        train_mask = np.isin(dates, train_dates)
        val_mask = np.isin(dates, val_dates)
        # 分割数据
        train_set_x = data_x[train_mask]
        train_set_y = data_y[train_mask]
        val_set_x = data_x[val_mask]
        val_set_y = data_y[val_mask]
        dates_train_x = dates[train_mask]
        dates_val_x = dates[val_mask]

        # 创建数据集和数据加载器
        train_dataset = TDataset(train_set_x, train_set_y, dates_train_x)
        val_dataset = TDataset(val_set_x, val_set_y, dates_val_x)
        train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True, pin_memory=True,
                                  drop_last=False, num_workers=8)
        vali_loader = DataLoader(val_dataset, batch_size=self.args.batch_size, shuffle=False, pin_memory=True,
                                 drop_last=False, num_workers=8)
        return train_dataset, train_loader, val_dataset, vali_loader

    def train(self, setting):
        train_data, nowcast_dataset = self._get_data(flag='train')
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        val_ratio = 1.0 / self.args.num_fold
        print(f'val_ratio:{val_ratio}\n')
        print_model_parameters(self.model)

        if hasattr(train_data, 'valid_dates'):
            sorted_dates = np.array(train_data.valid_dates)
            num_dates = len(sorted_dates)
            dates = sorted_dates
        else:
            num_dates, sorted_dates, dates = self.prepared_dataset(train_data)


        print('__________ Start training !____________')
        start_training_time = time.time()

        best_train_corr_list = []
        best_val_losses = []
        best_val_corr_list = []
        best_val_metric_list = []
        best_val_sr_list = []
        nowcast_corr_list = []

        # 如果指定了single_fold，只训练该fold
        fold_range = [self.single_fold] if self.single_fold is not None else range(self.args.num_fold)

        for fold in fold_range:
            start_fold_time = time.time()
            print(f"Training fold {fold + 1}/{self.args.num_fold}")

            if self.args.num_fold == 1:
                train_dataset, train_loader, val_dataset, vali_loader = self.load_dataset_last(num_dates, fold,
                                                                                               sorted_dates, dates)
            else:
                train_dataset, train_loader, val_dataset, vali_loader = self.load_dataset(num_dates, fold, sorted_dates,
                                                                                           dates,train_data)
            train_steps = len(train_loader)
            self.model = self._build_model()  # 每个折叠重新初始化模型
            model_optim, scheduler = self._select_optimizer()  # 每次初始化模型后也要重新初始化优化器和调度器
            criterion = self._select_criterion()
            scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                                steps_per_epoch=train_steps,
                                                pct_start=self.args.pct_start,
                                                epochs=self.args.train_epochs,
                                                max_lr=self.args.learning_rate)

            if self.args.use_amp:
                scaler = torch.cuda.amp.GradScaler()

            early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

            best_epoch = 0
            best_train_corr = -1
            best_val_metric = -999
            best_val_corr = -1
            best_val_sr = -1
            best_train_mse = 999
            best_val_mse = 999
            # Logging setup for this fold
            log_file_path = f'{self.args.save_path}/training_logs_fold_{fold + 1}.txt'
            with open(log_file_path, 'w') as file:
                file.write('Item\tTrain Loss\tBatch Correlation\n')

            with open(f'{self.args.save_path}/_result_of_multiple_regression.txt', 'a') as file:
                file.write(f'training fold {fold + 1}\n')
            for epoch in range(self.args.train_epochs):

                iter_count = 0
                train_loss_list = []
                # batch_corr = torch.tensor([], device=self.device)
                preds_list = []
                trues_list = []
                # corr_loss = []
                # mse_loss_list = []
                self.model.train()
                epoch_time = time.time()

                
                
                model_optim.zero_grad()

                for i, (batch_x, batch_y,batch_industry, batch_mask,  batch_time) in enumerate(train_loader):
                    if i == 0: print(batch_x.shape, batch_y.shape)
                    if batch_x.shape[1] <= 1:
                        continue
                    iter_count += 1
                    model_optim.zero_grad()

                    B, T, L, F = batch_x.shape
                    
                   

                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)

                    batch_industry = batch_industry.to(self.device)
                    batch_mask = batch_mask.to(self.device)


                    outputs = self.model(batch_x, batch_industry, batch_mask)
                    outputs = outputs.squeeze(-1)# [B, T]
                    batch_y = batch_y.squeeze().reshape(outputs.shape)
                    valid_mask = (batch_mask > 0) & (~torch.isnan(batch_y))

                    if i == 0: print('output and batch_y', outputs.shape, batch_y.shape)
                
                    if self.args.loss == 'MSE_with_weak':
                        tau_hat = torch.sigmoid(self.model.alpha)
                        tau = 1 - tau_hat
                        loss_dict = criterion(batch_x[valid_mask], outputs[valid_mask], batch_y[valid_mask], tau_hat, tau,self.args.c_norms)
                        mse = loss_dict['total']
                    else:
                        mse = torch.mean(criterion(outputs[valid_mask], batch_y[valid_mask]))
                   
                    loss = mse 
                    loss.backward()

                    
                    if self.args.use_amp:
                        if self.args.grad_norm:
                            scaler.unscale_(model_optim)
                            nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_value)
                        scaler.step(model_optim)
                        scaler.update()
                    else:
                        if self.args.grad_norm:
                            nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_value)
                        model_optim.step()
                    
                    model_optim.zero_grad()
                    scheduler.step()


                    train_loss_list.append(torch.tensor([loss.item()], device=self.device))

                    
                    corr = torch.corrcoef(torch.stack([outputs[valid_mask].reshape(-1), batch_y[valid_mask].reshape(-1)]))[0, 1]

                    # mse_loss_list.append(torch.tensor([mse.item()], device=self.device))

                    with open(f'{self.args.save_path}/training_logs.txt', 'a') as file:
                        file.write(f'{i}\t{loss:.4f}\t{corr:.4f}\n')
                    if (i == 0) or ((i + 1) % 1000 == 0):
                        print("\titers: {0}, epoch: {1} | loss: {2:.7f} | corr: {3:.8f}".format(i + 1, epoch + 1, loss.item(), corr))

                    

                # Epoch end statistics
                print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
                train_loss, corr, sr = self.vali(train_loader, criterion,
                                                 fold)  # 保持和val一致，每个epoch模型固定后train的corr
                vali_loss, vali_corr, vali_sr = self.vali(vali_loader, criterion, fold)
                # test_loss, test_corr = self.vali(test_loader, criterion,epoch,fold)
                # train_loss = torch.mean(torch.cat(train_loss_list).to(self.device))
                mse_loss = train_loss  # torch.mean(torch.cat(mse_loss_list).to(self.device)) # amd模型和path模型还有moe loss，没有加在里面

                print(
                    f"Epoch {epoch + 1} | Train Loss: {train_loss:.7f} | mse:{mse_loss:.8f} | Train Corr: {corr:.8f} "
                    f"| Val Loss: {vali_loss:.8f} | Val Corr: {vali_corr:.8f} | Val Sr: {vali_sr:.8f}")

                with open(f'{self.args.save_path}/_result_of_multiple_regression.txt', 'a') as file:
                    file.write(f"Epoch {epoch + 1} | Train Loss: {train_loss:.7f} | Train Corr: {corr:.8f} "
                               f"| Val Loss: {vali_loss:.8f} | Val Corr: {vali_corr:.8f} | Val Sr: {vali_sr:.8f}\n")

                if self.args.task_name == 'Long_term_forecasting':
                    early_stopping(-vali_loss, vali_sr + vali_corr, self.model, path)
                else:
                    early_stopping(vali_corr, vali_sr,self.model,
                                   f'{path}/best_model_fold_{fold + 1}.pth')  # vali_corr vali_loss

                if early_stopping.counter == 0:
                    print(f"Update best record at epoch {epoch + 1}")  # 可选：调试用
                    best_epoch = epoch + 1

                    # 这里记录的直接就是“当前这一轮”的所有指标
                    # 因为这一轮被保存了，所以这些就是“最后保存模型”对应的指标
                    best_train_corr = corr  # 当前轮的训练相关性
                    best_train_mse = train_loss  # 当前轮的训练Loss

                    best_val_corr = vali_corr  # 当前轮的验证相关性
                    best_val_sr = vali_sr  # 当前轮的验证SR
                    best_val_mse = vali_loss  # 当前轮的验证Loss

                    best_val_metric = vali_corr + vali_sr

                if early_stopping.early_stop:
                    print("Early stopping triggered.")
                    break
                    # early_triggered = True
                if self.args.lradj != 'not':
                    adjust_learning_rate(model_optim, epoch + 1, self.args)

            best_val_losses.append(best_val_mse.cpu().numpy())
            best_train_corr_list.append(best_train_corr.cpu().numpy())
            best_val_corr_list.append(best_val_corr.cpu().numpy())
            best_val_metric_list.append(best_val_metric.cpu().numpy())
            best_val_sr_list.append(best_val_sr.cpu().numpy())
            fold_time = (time.time() - start_fold_time) / 60

            nowcast_loader = DataLoader(dataset=nowcast_dataset, batch_size=self.args.batch_size, shuffle=False,
                                        pin_memory=True, drop_last=False, num_workers=10, collate_fn=daily_collate_fn)

            _, nowcast_corr, vali_sr = self.vali(nowcast_loader, criterion, fold)
            nowcast_corr_list.append(nowcast_corr.cpu().numpy())
            print(
                f"best train corr: {best_train_corr:.6f} | best train mse: {best_train_mse} | nowcast corr: {nowcast_corr:.6f} | best val sr: {best_val_sr:.6f} | best val corr: {best_val_corr:.6f} | best val mse: {best_val_mse}")

            with open(f'{self.args.save_path}/_result_of_multiple_regression.txt', 'a') as file:
                file.write(
                    f'best train corr: {best_train_corr:.6f}\n Best validation metric for fold {fold + 1}: {best_val_sr} at epoch {best_epoch}\nfold{fold + 1} training time: {fold_time:.2f} minutes\n')

            # Final training summary
        total_time = (time.time() - start_training_time) / 60

        # 保存单折训练结果
        if self.single_fold is not None:
            result_path = f'{self.args.save_path}/fold_{self.single_fold + 1}_results.npy'
            np.save(result_path, {
                'best_train_corr': best_train_corr_list[0] if best_train_corr_list else None,
                'best_val_loss': best_val_losses[0] if best_val_losses else None,
                'best_val_corr': best_val_corr_list[0] if best_val_corr_list else None,
                'best_val_metric': best_val_metric_list[0] if best_val_metric_list else None,
                'best_val_sr': best_val_sr_list[0] if best_val_sr_list else None,
                'nowcast_corr': nowcast_corr_list[0] if nowcast_corr_list else None,
            })
            print(f"Fold {self.single_fold + 1} training completed in {total_time:.2f} minutes")
        else:
            # 多折训练汇总
            with open(f'{self.args.save_path}/_result_of_multiple_regression.txt', 'a') as f:
                f.write(
                    f'Total training time: {total_time:.2f} minutes\n best train corr: {np.mean(best_train_corr_list)}\n best val loss:{np.mean(best_val_losses)} best val corr:{np.mean(best_val_corr_list):.6f} best val metric: {np.mean(best_val_metric_list):.6f}\n')
            print(f"Total training time: {total_time:.2f} minutes")
            print(f"average best train corr: {np.mean(best_train_corr_list):.6f}")
            print(f"average best val loss: {np.mean(best_val_losses):.6f}")
            print(f"average best val corr: {np.mean(best_val_corr_list):.6f}")
            print(f"average best val sr: {np.mean(best_val_sr_list):.6f}")
            print(f"average best val metric: {np.mean(best_val_metric_list):.6f}")
            print(f"average nowcast corr: {np.mean(nowcast_corr_list):.6f}")
        # Load the best model after training
        # self.model.load_state_dict(torch.load(best_model_path))

        return self.model
    
    def vali(self, vali_loader, criterion, fold):
        total_loss_list = []
        self.model.eval()
        preds_list = []
        trues_list = []
        
        with torch.no_grad():
            for i, (batch_x, batch_y,batch_industry, batch_mask, batch_time) in enumerate(vali_loader):
                # 如果 bs=1，原来的 <= 1 会导致跳过所有数据
                if batch_x.shape[0] < 1: 
                    continue

                B, T, L, F = batch_x.shape
                
                # 提取行业 ID 和 Mask
                batch_industry = batch_industry.to(self.device)
                batch_mask = batch_mask.to(self.device)

                batch_x = batch_x.float().to(self.device)
                outputs = self.model(batch_x, batch_industry, batch_mask)
                
                # 处理 batch_y 维度
                batch_y = batch_y.float().to(self.device)
                batch_y = batch_y.squeeze().reshape(outputs.shape)
                
                # 有效性掩码
                valid_mask = (batch_mask > 0) & (~torch.isnan(batch_y))

                if not valid_mask.any(): 
                    continue
              
                # Loss 计算
                if self.args.loss == 'MSE_with_weak':
                    tau_hat = torch.sigmoid(self.model.alpha)
                    tau = 1 - tau_hat
                    # 这里的 batch_x[valid_mask] 变成 3 维，loss 里的逻辑要能兼容
                    loss_dict = criterion(batch_x[valid_mask], outputs[valid_mask], batch_y[valid_mask], tau_hat, tau)
                    loss_val = loss_dict['total']
                else:
                    loss_val = torch.mean(criterion(outputs[valid_mask], batch_y[valid_mask]))
                
                total_loss_list.append(loss_val.item())
                preds_list.append(outputs[valid_mask].detach())
                trues_list.append(batch_y[valid_mask].detach())

        # --- 关键防护 ---
        if not total_loss_list:
            self.model.train()
            return torch.tensor(0.0, device=self.device), 0.0, 0.0

        total_loss = torch.tensor(total_loss_list).mean().to(self.device)
        preds = torch.cat(preds_list)
        trues = torch.cat(trues_list)
        
        # 过滤 NaN 并计算指标
        valid_idx = ~torch.isnan(preds) & ~torch.isnan(trues)
        if valid_idx.sum() < 2: # 至少2个点算相关系数
            return total_loss, 0.0, 0.0
            
        preds_f = preds[valid_idx]
        trues_f = trues[valid_idx]
        
        vali_corr = torch.corrcoef(torch.stack([preds_f, trues_f]))[0, 1]
        # 这里的 SR 计算注意分母不能为 0
        std_val = (trues_f * preds_f).std()
        vali_sr = (preds_f * trues_f).mean() / (std_val + 1e-8)
        
        self.model.train()
        return total_loss, vali_corr, vali_sr

    '''
    def vali(self, vali_loader, criterion, fold):
        total_loss_list = []
        self.model.eval()
        preds_list = []
        trues_list = []
        y_times = []
        # batch_corr = []
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_time) in enumerate(vali_loader):
                if batch_x.shape[0] <= 1:
                    continue

                B, T, L, F = batch_x.shape
                    
                industry_ids = torch.tensor([item.get('industry_sw1', 0) for item in batch_time]).long()
                mask_data = torch.tensor([item.get('mask_data', 0.0) for item in batch_time]).float()
                # 关键：Reshape 成与 batch_y 一致的 [B, T]
                industry_ids = industry_ids.view(B, T).to(self.device)
                mask_data = mask_data.view(B, T).to(self.device)

                batch_x = batch_x.float().to(self.device)
                outputs = self.model(batch_x, industry_ids, mask_data)
                batch_y = batch_y.squeeze().reshape(outputs.shape).to(self.device)
                valid_mask = (mask_data > 0) & (~torch.isnan(batch_y))

                if not valid_mask.any(): continue

                if self.args.loss == 'MSE_with_weak':
                    tau_hat = torch.sigmoid(self.model.alpha)
                    tau = 1 - tau_hat
                    loss_dict = criterion(batch_x[valid_mask], outputs[valid_mask], batch_y[valid_mask], tau_hat, tau,self.args.c_norms)
                    mse = loss_dict['total']
                else:
                    mse = torch.mean(criterion(outputs[valid_mask], batch_y[valid_mask]))
                loss = mse
                
                total_loss_list.append(torch.tensor([loss.item()]).to(self.device))
                
                preds_list.append(outputs[valid_mask].detach())
                trues_list.append(batch_y[valid_mask].detach())

            # golbel_scaler = joblib.load(f'{self.args.save_path}/{fold}_robust_scaler.pkl')
            total_loss = torch.mean(torch.cat(total_loss_list))
            preds = torch.cat(preds_list).to(self.device)
            trues = torch.cat(trues_list).to(self.device)
            # cos_loss = torch.cat(cos_loss_list)
            # preds = preds * golbel_scaler.scale_[0] + golbel_scaler.center_[0]
            # trues = trues * golbel_scaler.scale_[0] + golbel_scaler.center_[0]
            valid_mask = ~torch.isnan(preds) & ~torch.isnan(trues)
            # 过滤掉 NaN 值
            preds_filtered = preds[valid_mask]
            trues_filtered = trues[valid_mask]
            # total_loss = criterion(preds_filtered, trues_filtered).item()
            # cos_loss = torch.mean(cos_loss)
            # print('vali shape:', preds.shape, trues.shape)
            vali_corr = torch.corrcoef(torch.stack([preds_filtered.reshape(-1), trues_filtered.reshape(-1)]))[0, 1]
            vali_sr = (preds_filtered * trues_filtered).mean() / (trues_filtered * preds_filtered).std()
            self.model.train()
            return total_loss, vali_corr, vali_sr

    '''
    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')

        print("\n开始测试所有5折模型...\n")
        for fold in range(self.args.num_fold):
            # print(f'cycle:{cycle}-epoch:{epoch}')
            print('loading model')
            self.model.load_state_dict(
                torch.load(os.path.join(self.args.checkpoints + '/' + setting, f'best_model_fold_{fold + 1}.pth'),map_location=torch.device('cpu')),
                strict=True)
            self.model.to(self.device)
            # scalers = joblib.load(f'{self.args.save_path}/robust_scaler.pkl')
            # self.model.load_state_dict(torch.load(f'/cpfs/dss/dev/lxjie/lxj_results/stock/patchtst_base/train_start_2010/model/best_model_fold_{fold+1}.pth'))
            criterion = self._select_criterion()
            folder_path = self.args.save_path + '/multi_reg_results/' + setting + '/'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            preds = []
            trues = []
            y_tickers = np.array([], dtype=str)
            y_times = np.array([], dtype=str)

            mse_loss = []
            self.model.eval()
            with torch.no_grad():
                for i, (batch_x, batch_y,batch_industry, batch_mask, batch_time) in enumerate(test_loader):

                    current_date_str = str(batch_time[0].get('CalcDate', ''))
                    if str(self.args.test_year) not in current_date_str:
                        continue

                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)
                    batch_industry = batch_industry.to(self.device) # 已经是 Tensor 了
                    batch_mask =batch_mask.to(self.device)

                    B, T, L, F = batch_x.shape
                    
                    
                    y_ticker = [str(item.get('Code', '')) for item in batch_time]
                    y_time = [str(item.get('CalcDate', '')) for item in batch_time]

                    # 维度对齐，如果模型要求 [B, T] 且 T=1
                    if len(batch_y.shape) > 1:
                        batch_industry = batch_industry.view(batch_y.shape)
                        batch_mask = batch_mask.view(batch_y.shape)
                


                    outputs = self.model(batch_x, batch_industry, batch_mask)
                    outputs = outputs

                    outputs = outputs.view(B, T)
                    batch_y = batch_y.view(B, T)

                    if i == 0: print('output and batch_y', outputs.shape, batch_y.shape)

                    outputs[batch_mask <= 0] = torch.nan

                    pred = outputs.detach().cpu().numpy().flatten()
                    true = batch_y.detach().cpu().numpy().flatten()
                    preds = np.append(preds, pred)
                    trues = np.append(trues, true)

                    
                    # 优化日期格式转换
                    # 优化日期格式转换
                    new_y_time = []
                    for t in y_time:
                        if isinstance(t, str):
                            try:
                                # 如果是带 T 的长格式：'2022-06-27T00:00:00.000000'
                                if 'T' in t:
                                    dt_obj = datetime.strptime(t[:26], '%Y-%m-%dT%H:%M:%S.%f')
                                    new_y_time.append(dt_obj.strftime('%Y-%m-%d'))
                                # 如果已经是短格式：'2022-06-27'，直接跳过转换
                                elif len(t) <= 10: 
                                    new_y_time.append(t)
                                else:
                                    # 其他可能的格式尝试
                                    new_y_time.append(t[:10]) 
                            except Exception:
                                # 万一解析失败，保留前10位(YYYY-MM-DD)
                                new_y_time.append(str(t)[:10])
                        else:
                            new_y_time.append(t)
                    y_time = new_y_time
                    

                    # 优化字符串处理
                    y_ticker = [t.strip("[]' ") for t in y_ticker]

                    # 扩展列表
                    y_tickers = np.concatenate([y_tickers, y_ticker])
                    y_times = np.concatenate([y_times, y_time])

                y_tickers = np.array(y_tickers)
                y_times = np.array(y_times)
                valid_mask = ~np.isnan(preds) & ~np.isnan(trues)
                # 过滤掉 NaN 值
                if np.any(valid_mask):  # 只有存在有效值才计算
                    preds_filtered = preds[valid_mask]
                    trues_filtered = trues[valid_mask]
                    total_loss = np.mean(np.square(preds_filtered - trues_filtered)).item()

                    # 至少需要两个点才能计算相关系数 Pearson
                    if len(preds_filtered) > 1:
                        corr, _ = pearsonr(preds_filtered, trues_filtered)
                    else:
                        corr = np.nan

                    print('test data mse: ', total_loss)

                else:
                    print("Warning: No valid data points found in this fold.")
                    total_loss, corr = np.nan, np.nan
                # corr = np.corrcoef(pred_list, true_list)[0, 1] # 所有test的corr（拼接完一起）1折的
                if self.all_test_preds.size == 0:
                    self.all_test_preds = preds.reshape(1, -1)
                else:
                    self.all_test_preds = np.concatenate((self.all_test_preds, preds.reshape(1, -1)))

                data = {'Code': y_tickers, 'CalcDate': y_times, 'True Values': trues, 'Predicted Values': preds}
                df = pd.DataFrame(data)
                

                # 2. 修改 daily_ic 函数，增加鲁棒性
                def daily_ic(sub):
                    # 过滤掉当前日期中的 NaN
                    sub_valid = sub.dropna(subset=['True Values', 'Predicted Values'])
                    if len(sub_valid) < 2:  # 相关性计算至少需要2个样本
                        return np.nan
                    return np.corrcoef(sub_valid['True Values'], sub_valid['Predicted Values'])[0, 1]

                ic_series = df.groupby('CalcDate').apply(daily_ic).dropna()  # dropna 排除掉无法计算IC的日期

                # 3. 计算 SR，需确保 ic_series 不为空且标准差不为 0
                if not ic_series.empty and ic_series.std() != 0:
                    sr = np.sqrt(len(ic_series)) * ic_series.mean() / ic_series.std()
                else:
                    sr = np.nan
                print('the  test corr result is {} ;sr is {}'.format(corr, sr))
                with open(f'{self.args.save_path}/_result_of_multiple_regression.txt', 'a') as f:
                    f.write(f'the  test corr result is {corr} ;sr is {sr}\n')
                # Define the path where you want to store the CSV file
                csv_file_path = self.args.save_path + '/' + self.args.model + self.args.task_name + self.args.test_year + f'predicted_true_values_{fold + 1}.csv'

                df.to_csv(csv_file_path, index=False)

                print("created with the true and predicted values.")

                print("True and predicted values have been saved to:", csv_file_path)
             
                plt_heiyi.plt_epoch_train_val_trend_fold(self.args,
                                                         f'{self.args.save_path}/_result_of_multiple_regression.txt')

        all_test_mean_preds = np.mean(self.all_test_preds, axis=0)
        mask = ~np.isnan(trues) & ~np.isnan(all_test_mean_preds)
        mean_df = pd.DataFrame({
            'Code': y_tickers,
            'CalcDate': y_times,
            'True Values': trues,
            'mean Predicted Values': all_test_mean_preds
        })
        test_mean_csv_file_path = self.args.save_path + '/' + self.args.model + self.args.task_name + self.args.test_year + f'predicted_true_values_mean.csv'
        mean_df.to_csv(test_mean_csv_file_path, index=False)
        # 4. 只有在 mask 有效时才计算最终指标
        if np.any(mask) and np.sum(mask) > 1:
            all_test_corr = np.corrcoef(all_test_mean_preds[mask], trues[mask])[0, 1]
            mse = np.mean(np.square(all_test_mean_preds[mask] - trues[mask]))
            # 重新应用鲁棒的 daily_ic
            def daily_ic_mean(sub):
                sub_valid = sub.dropna(subset=['True Values', 'mean Predicted Values'])
                if len(sub_valid) < 2:
                    return np.nan
                return np.corrcoef(sub_valid['True Values'], sub_valid['mean Predicted Values'])[0, 1]

            ic_series_mean = mean_df.groupby('CalcDate').apply(daily_ic_mean).dropna()

            if not ic_series_mean.empty and ic_series_mean.std() != 0:
                all_test_sr = np.sqrt(len(ic_series_mean)) * ic_series_mean.mean() / ic_series_mean.std()
            else:
                all_test_sr = np.nan
            print(f'the average mse value of {mse}')
            print(f'the average corr value of {all_test_corr}')
            print(f'the average sr value of {all_test_sr}')
            with open(f'{self.args.save_path}/_result_of_multiple_regression.txt', 'a') as f:
                f.write(f'the average mse value of {mse}\n'
                        f'the average corr value of {all_test_corr}\n'
                        f'the average sr value of {all_test_sr}\n'
                        )
        else:
            print("Error: No sufficient valid data for average metrics calculation.")

        return