import numpy as np
import torch
import matplotlib.pyplot as plt
import time
import torch.nn as nn


from datetime import datetime
import torch.optim.lr_scheduler as lr_scheduler
from collections import OrderedDict
import torch.nn.functional as F
from einops import rearrange

# plt.switch_backend('agg')


def adjust_learning_rate(optimizer, epoch, args, scheduler=None):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    if args.lradj == 'cos':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_epochs // 5)
    elif args.lradj == 'steplr':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.5)
    # elif args.lradj in ['cos', 'steplr']:
        # assert scheduler != None
        # scheduler.step()
        # lr_adjust = {epoch: scheduler.get_last_lr()[-1]}

    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))
    return scheduler


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_val_metric = -np.inf # 因为是 Corr+SR，越大越好，初始设为负无穷
        self.delta = delta

    def __call__(self, vali_corr, vali_sr, model, path):
        # 聚合指标：将相关性和夏普率相加作为综合得分
        score = vali_corr + vali_sr
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, model, path)
        elif score < self.best_score + self.delta:
            # 如果新得分没有超过历史最好得分（加上偏移量），计数增加
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # 表现变好了
            self.best_score = score
            self.save_checkpoint(score, model, path)
            self.counter = 0

    def save_checkpoint(self, score, model, path):
        if self.verbose:
            print(f'Validation metric increased ({self.best_val_metric:.6f} --> {score:.6f}). Saving model ...')
        
        # 实际保存模型
        torch.save(model.state_dict(), path)
        self.best_val_metric = score
        
'''
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        # torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss
    def reset(self):
        """
        重置早停状态。
        """
        if self.verbose:
            print("Resetting early stopping state.")
        self.counter = 0
        self.best_score = None
        self.early_stop = False

'''

