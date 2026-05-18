import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler
import joblib
# import feather
import torch
import bisect
import re
from typing import Dict, List, Optional

import os


class Dataset_regression():
    '''
    daily 原始特征
        - 数据处理：前值填充，robust
        - 不在此处划分train val，对除了test之外的数据做截断，删除异常值
    '''

    def __init__(self, args, data_path='/data/daily_label_research_v1_extend_factors.csv', flag='train',
                 size=None, train_start_year='2010', train_end_year='2018', test_year='2014', val_start_year='2019',
                 ticker_type=0):
        if size == None:
            self.seq_len = 0
        else:
            self.seq_len = size[0]

        self.flag = flag
        assert flag in ['train', 'test']
        type_map = {'train': 0, 'test': 1}
        self.set_type = type_map[flag]
        self.train_start_year = train_start_year
        self.train_end_year = train_end_year

        self.test_year =test_year
        # if test_year == '2026':
        #     self.test_year = train_end_year
        self.data_path = data_path
        self.ticker_type = ticker_type
        self.args = args
        self.pred_task = 'y' + str(self.args.pred_task)

        self.feature_cols = ['ret_slp', 'ret','tr_ret', 'close_adj', 'high_adj', 'low_adj', 'open_adj', 'vwap_adj', 'volume_adj', 'capvol0']
        self.columns=['CalcDate', 'Code',self.pred_task]+self.feature_cols+['industry_sw1', 'mask_data']

        self.__read_data__()

    def __get_data__(self):
        
        if self.data_path.endswith('.feather'):
            df = pd.read_feather(self.data_path)
        else:
            df = pd.read_hdf(self.data_path)

        df = df.replace([-np.inf, np.inf], np.nan)
        df = df.rename(columns={'date': 'CalcDate', 'ticker': 'Code'})
        df['CalcDate'] = pd.to_datetime(df['CalcDate'].astype(str), format='%Y%m%d')

        feature_cols=self.feature_cols
        
        # Remove anomalies
        if 'mask_data' in df.columns:
            df.loc[df['mask_data'] == 0, self.pred_task] = np.nan
        df.loc[df[self.pred_task].abs() > 50, self.pred_task] = np.nan
       
        columns = self.columns
        df = df[columns]

        grouped_df = df.groupby('Code')
        # 填充规则 (保持不变，用于指导 FFILL/0 填充)
        fill_rules = {
            **{c: 0 for c in feature_cols if c.startswith(('ret', 'volume'))},
            **{c: 'ffill' for c in feature_cols if not c.startswith(('ret', 'volume'))}
        }

        # 对所有特征进行序列填充
        for col, rule in fill_rules.items():
            if rule == 'ffill':
                df[col] = grouped_df[col].ffill()
            else:
                df[col] = df[col].fillna(rule)

        # Label 截面标准化 (Z-score)  
        df[self.pred_task] = df.groupby('CalcDate')[self.pred_task].transform(
            lambda x: (x - x.mean()) / x.std())

        train_set, nowcast_set, test_set = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        grouped = df.groupby('Code')
        for _, group in grouped:
            group.dropna(subset=feature_cols, inplace=True, how='any')

            data = group[(group['CalcDate'] >= str(self.train_start_year + '-01-01')) & (group['CalcDate'] <= str(self.train_end_year + '-12-31'))]
            test = group[(group['CalcDate'] >= str(self.test_year + '-01-01')) & (group['CalcDate'] <= str(self.test_year + '-12-31'))]

            if len(data) <= self.args.pred_task + self.seq_len - 1:  # 当训练数据不足以生成完整的序列，这里多加了predtask
                if self.seq_len > 1:
                    test = pd.concat([data.iloc[-(self.seq_len - 1):], test])
                    if len(test) <= self.seq_len - 1:  # 如果训练数据不足生成完整序列就打算把他放到test中生成新的test数据，但是需要防止新的test不能生成完整序列，这里如果加入了不完整的数据会导致加载数据错位，这个地方是不是不应该加predtask
                        continue
                    test_set = pd.concat([test_set, test])  # 短数据拼接到test，生成新的test数据
                else:
                    if len(test) <= self.seq_len - 1:
                        continue
                    test_set = pd.concat([test_set, test])
                continue
            # 选取当前 Code 的数据
            Code_data = data.copy()
            if len(Code_data.iloc[:-(self.args.pred_task)]) < self.seq_len:
                if len(test) <= self.seq_len - 1:
                    continue
                else:
                    if self.seq_len > 1:
                        test = pd.concat([data.iloc[-(self.seq_len - 1):], test])  # 为了预测test起始数据
                        test_set = pd.concat([test_set, test])
                    else:
                        test_set = pd.concat([test_set, test])
            else:
                train_data = Code_data.iloc[:-(self.args.pred_task)]
                train_set = pd.concat([train_set, train_data])
                if self.seq_len > 1:
                    test = pd.concat([data.iloc[-(self.seq_len - 1):], test])
                    if len(test) <= self.seq_len - 1:
                        continue
                    test_set = pd.concat([test_set, test])
                else:
                    if len(test) <= self.seq_len - 1:
                        continue
                    test_set = pd.concat([test_set, test])

        nowcast_grouped = train_set.groupby('Code')
        for _, group in nowcast_grouped:
            data = group[
                (group['CalcDate'] >= str(self.train_start_year + '-01-01')) & (
                        group['CalcDate'] <= str(str(int(self.train_end_year) - 1) + '-12-31'))]
            vali = group[
                (group['CalcDate'] >= str(self.train_end_year + '-01-01')) & (
                        group['CalcDate'] <= str(self.train_end_year + '-12-31'))]

            if len(data) <= self.args.pred_task + self.seq_len - 1:  # 当训练数据不足以生成完整的序列，这里多加了predtask
                if self.seq_len > 1:
                    vali = pd.concat([data.iloc[-(self.seq_len - 1):], vali])
                    if len(vali) <= self.seq_len - 1:  # 如果训练数据不足生成完整序列就打算把他放到test中生成新的test数据，但是需要防止新的test不能生成完整序列，这里如果加入了不完整的数据会导致加载数据错位，这个地方是不是不应该加predtask
                        continue
                    nowcast_set = pd.concat([nowcast_set, vali])  # 短数据拼接到test，生成新的test数据
                else:
                    if len(vali) <= self.seq_len - 1:
                        continue
                    nowcast_set = pd.concat([nowcast_set, vali])
                continue
            # 选取当前 Code 的数据
            Code_data = data.copy()
            if len(Code_data.iloc[:-(self.args.pred_task)]) < self.seq_len:
                if len(vali) <= self.seq_len - 1:
                    continue
                else:
                    if self.seq_len > 1:
                        vali = pd.concat([data.iloc[-(self.seq_len - 1):], vali])  # 为了预测test起始数据
                        nowcast_set = pd.concat([nowcast_set, vali])
                    else:
                        nowcast_set = pd.concat([nowcast_set, vali])
            else:
                if self.seq_len > 1:
                    vali = pd.concat([data.iloc[-(self.seq_len - 1):], vali])
                    if len(vali) <= self.seq_len - 1:
                        continue
                    nowcast_set = pd.concat([nowcast_set, vali])
                else:
                    if len(vali) <= self.seq_len - 1:
                        continue
                    nowcast_set = pd.concat([nowcast_set, vali])

        # 对异常值做截断，排除y
        for col in feature_cols + [self.pred_task]:
            q = train_set[col].quantile([0.01, 0.99])
            train_set[col] = np.clip(train_set[col], q[0.01], q[0.99])


        scaler = RobustScaler(quantile_range=(5, 95))
        #
        train_set[feature_cols] = scaler.fit_transform(train_set[feature_cols])
        nowcast_set[feature_cols] = scaler.transform(nowcast_set[feature_cols])
        test_set[feature_cols] = scaler.transform(test_set[feature_cols])

        joblib.dump(scaler, f'{self.args.save_path}/robust_scaler.pkl')
        # 计算每个通道的l2
        self.args.c_norms = [np.linalg.norm(train_set[c], ord=2) for c in feature_cols]
        
        return train_set, nowcast_set, test_set

    def __read_data__(self):
        train_df, nowcast_df, test_df = self.__get_data__()
        '''
        df_raw.columns: ['tk', 'y_5', 'y_20',...(features), ...(time_mark)
        '''

        if self.set_type == 0:
            self.train_Code = train_df[['Code']]
            self.train_stamp = train_df[['CalcDate']]
            self.train_industry = torch.tensor(train_df['industry_sw1'].values, dtype=torch.long)
            self.train_mask = torch.tensor(train_df['mask_data'].values, dtype=torch.float32)
            self.train_set = torch.tensor(train_df[self.feature_cols].values, dtype=torch.float32)
            self.train_label = torch.tensor(train_df[self.pred_task].values, dtype=torch.float32)
            
            # 3. 验证集处理 (nowcast)
            self.nowcast_Code = nowcast_df[['Code']]
            self.nowcast_stamp = nowcast_df[['CalcDate']]
            self.nowcast_industry = torch.tensor(nowcast_df['industry_sw1'].values, dtype=torch.long)
            self.nowcast_mask = torch.tensor(nowcast_df['mask_data'].values, dtype=torch.float32)
            self.nowcast_set = torch.tensor(nowcast_df[self.feature_cols].values, dtype=torch.float32)
            self.nowcast_label = torch.tensor(nowcast_df[self.pred_task].values, dtype=torch.float32)

        else:
            self.test_Code = test_df[['Code']]
            self.test_stamp = test_df[['CalcDate']]
            self.test_industry = torch.tensor(test_df['industry_sw1'].values, dtype=torch.long)
            self.test_mask = torch.tensor(test_df['mask_data'].values, dtype=torch.float32)
            self.test_set = torch.tensor(test_df[self.feature_cols].values, dtype=torch.float32)
            self.test_label = torch.tensor(test_df[self.pred_task].values, dtype=torch.float32)



class Dataset_regression_dataset(Dataset):
    def __init__(self, data_x, data_y, Codes, data_stamp, seq_len,stride=1, industry_ids=None, mask_data=None):
        self.data_x = data_x
        self.data_y = data_y
        self.Codes = Codes
        self.seq_len = seq_len
        self.data_stamp = data_stamp
        self.stride = stride
        self.industry_ids = industry_ids  # tensor of shape [N,]
        self.mask_data = mask_data        # tensor of shape [N,]

        #  获取所有唯一的日期并排序
        # 确保 CalcDate 是 datetime 格式或统一的字符串格式
        self.data_stamp['CalcDate'] = pd.to_datetime(self.data_stamp['CalcDate'])
        self.unique_dates = sorted(self.data_stamp['CalcDate'].unique())
        # 预先建立 日期 -> 样本行索引 的映射，提高读取速度
        # 只有当全局索引 i >= seq_len - 1 时，才能构造出完整的 seq_x
        date_series = self.data_stamp['CalcDate'].reset_index(drop=True)
        date_groups = date_series.groupby(date_series).indices

        self.valid_dates = []
        self.date_to_indices = {}

        for date in self.unique_dates:
            # 获取该日期对应的所有行索引
            indices = date_groups[date]
            # 过滤掉由于序列长度不足而无法构造 window 的索引
            # (在原始 __get_data__ 逻辑中，各股票起始已补全或筛选，此处做二次保险)
            valid_idx = [i for i in indices if i >= self.seq_len - 1]
            
            if len(valid_idx) > 0:
                self.date_to_indices[date] = valid_idx
                self.valid_dates.append(date)


    def __getitem__(self, index):
        target_date = self.valid_dates[index]
        sample_indices = self.date_to_indices[target_date]

        day_x = []
        day_y = []
        day_info = []

        for idx in sample_indices:
            # 提取 seq_x: 形状 (seq_len, feature_dim)
            s_begin = idx - self.seq_len + 1
            s_end = idx + 1
            seq_x = self.data_x[s_begin:s_end]
            
            # 提取 seq_y: 目标值
            seq_y = self.data_y[idx]
            
            # 提取辅助信息
            info = {
                'Code': str(self.Codes.iloc[idx]['Code']),
                'CalcDate': target_date.strftime('%Y-%m-%d'),
                'industry_sw1': self.industry_ids[idx].item() if self.industry_ids is not None else 0,
                'mask_data': self.mask_data[idx].item() if self.mask_data is not None else 1.0
            }
            
            day_x.append(seq_x)
            day_y.append(seq_y)
            day_info.append(info)

        # 返回 Tensor 列表，形状分别为：
        # [Num_Stocks, seq_len, feat_dim], [Num_Stocks], [List of Dict]
        return torch.stack(day_x), torch.tensor(day_y), day_info

       

    def __len__(self):
        return len(self.valid_dates)
    
def daily_collate_fn(batch):
    combined_x = []
    combined_y = []
    combined_info = []
    
    # 获取这 batch 32天中，股票最多的一天有多少只
    max_stocks = max([day[0].shape[0] for day in batch])
    feat_dim = batch[0][0].shape[-1]
    seq_len = batch[0][0].shape[-2]

    new_batch_x = []
    new_batch_y = []
    new_batch_industry = []
    new_batch_mask = []
    combined_info = []
    
    for day_x, day_y, day_info in batch:
        curr_t = day_x.shape[0]
        pad_size = max_stocks - curr_t
        # 填充 X: [T, L, F] -> [T_max, L, F]
        pad_x = torch.zeros((pad_size, seq_len, feat_dim))
        new_batch_x.append(torch.cat([day_x, pad_x], dim=0))
        
        # 填充 Y: [T] -> [T_max]
        pad_y = torch.zeros((pad_size))
        new_batch_y.append(torch.cat([day_y, pad_y], dim=0))

        day_ind = torch.tensor([info['industry_sw1'] for info in day_info], dtype=torch.long)
        pad_ind = torch.zeros(pad_size, dtype=torch.long) # 0 通常作为空行业 padding
        new_batch_industry.append(torch.cat([day_ind, pad_ind], dim=0))

        day_m = torch.tensor([info['mask_data'] for info in day_info], dtype=torch.float32)
        pad_m = torch.zeros(pad_size, dtype=torch.float32) # Padding 部分 mask 为 0
        new_batch_mask.append(torch.cat([day_m, pad_m], dim=0))

        full_day_info = list(day_info)
        for _ in range(pad_size):
            full_day_info.append({'industry_sw1': 0, 'mask_data': 0.0, 'Code': 'PAD', 'CalcDate': day_info[0]['CalcDate']})
        combined_info.extend(full_day_info)

    # 现在可以 stack 了，得到 [B, T_max, L, F]
    return (
        torch.stack(new_batch_x), 
        torch.stack(new_batch_y), 
        torch.stack(new_batch_industry), 
        torch.stack(new_batch_mask),
        combined_info
    )

def Dataset_regression_train_val(args):
    """
    获取训练和验证数据集
    """
    dataset = Dataset_regression(
        args, data_path=args.data_path, flag='train', size=args.size,
        train_start_year=args.train_start_year,
        train_end_year=args.train_end_year,
        test_year=args.test_year
    )
    
    seq_len = args.size[0]
    
    train_dataset = Dataset_regression_dataset(
        dataset.train_set, dataset.train_label, dataset.train_Code,
        dataset.train_stamp, seq_len,
        industry_ids=dataset.train_industry,  
        mask_data=dataset.train_mask          
    )
    
    nowcast_dataset = Dataset_regression_dataset(
        dataset.nowcast_set, dataset.nowcast_label, dataset.nowcast_Code,
        dataset.nowcast_stamp, seq_len,
        industry_ids=dataset.nowcast_industry,   
        mask_data=dataset.nowcast_mask          
    )
    
    # 注意：在外部创建 DataLoader 时，必须指定 collate_fn=daily_collate_fn
    return train_dataset, nowcast_dataset

def Dataset_regression_test(args):
    """
    获取测试数据集及 Loader
    """
    dataset = Dataset_regression(
        args, data_path=args.data_path, flag='test', size=args.size,
        train_start_year=args.train_start_year,
        train_end_year=args.train_end_year,
        test_year=args.test_year
    )
    
    seq_len = args.size[0]
    
    test_dataset = Dataset_regression_dataset(
        dataset.test_set, dataset.test_label, dataset.test_Code,
        dataset.test_stamp, seq_len,
        industry_ids=dataset.test_industry,   
        mask_data=dataset.test_mask          
    )

    # 这里的 batch_size 指的是“天数”
    test_loader = DataLoader(
        dataset=test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        pin_memory=True,
        drop_last=False, 
        num_workers=10,
        collate_fn=daily_collate_fn  # 必须使用自定义拼合函数
    )
    
    return test_dataset, test_loader