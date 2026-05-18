import warnings
warnings.filterwarnings(
    "ignore",
    "The pynvml package is deprecated. Please install nvidia-ml-py instead.",
    FutureWarning
)
import yaml
import copy
import torch.multiprocessing as mp
import argparse
import os
import torch
from exp.exp_time_fold import Exp_Multiple_Regression_Fold
import random
import numpy as np
# from src.model_phi_heiyi import phi  # 加载phi
import joblib
import os
# 使用多进程并行训练5折
from multiprocessing import Process, set_start_method
import torch.multiprocessing
import time
import subprocess
import sys


os.environ["KMP_AFFINITY"] = "noverbose"

parser = argparse.ArgumentParser(description='phi2')

# basic config
parser.add_argument('--config',default='configs.yaml', type=str, help='Path to yaml config file')
parser.add_argument('--task_name', type=str, default='multiple_regression',
                    help='task name, options:[Long_term_forecasting, anomaly_detection, predict_feature,multiple_regression, LGB]')
parser.add_argument('--is_training', type=int, default=0, help='status')
parser.add_argument('--model_id', type=str, default='test', help='model id')
parser.add_argument('--model', type=str, default='PatchTST_gc',
                    help='model name, options: [GPT2TS, ]') # PatchTST_multi_scale
parser.add_argument('--asset', type=str, default='stock_daily', help='')

parser.add_argument('--train_log_dir', type=str, default='results/result5/logs/', help='path')
parser.add_argument('--save_dir', type=str, default= 'results/result5/fold/', help='path')

# data loader
parser.add_argument('--dataset', type=str, default='heiyi',
                    help='[ETTh1, ETTh2, ETTm1, ETTm2, weather, psm, smap]')
parser.add_argument('--prompt',type=str, default='Etth1')
parser.add_argument('--root_path', type=str, default='/home/liangxijie1/phi-2/dataset/',
                    help='root path of the data file:feature_1419_5, d1')
parser.add_argument('--data_path', type=str, default='LongtermForecast/ETT-small/',
                    help='data file, options: [ETT-small, electricity, exchange_rate, illness, traffic, weather]')
parser.add_argument('--freq', type=str, default='d',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--checkpoints', type=str, default='./checkpoints_heiyi/', help='location of model checkpoints')

parser.add_argument('--drop_ratio', type=float, default=0.1, help='Set a dropping ratio for feature_selection')
parser.add_argument('--train_data_start_year', type=int, default=2010)
parser.add_argument('--test_data_start_year', type=int, default=2021)
parser.add_argument('--feature_selection',type = bool, default=False, help='whether to use feature selection')
parser.add_argument('--extra_input',type = bool, default=False, help='whether to add tikcter')

# Forecast task
parser.add_argument('--seq_len', type=int, default=120, help='input sequence length')
parser.add_argument('--pred_len', type=int, default=1, help='prediction sequence length')

# phi-2
parser.add_argument('--block_size', type=int, default=1024)
parser.add_argument('--n_layer', type=int, default=6)
parser.add_argument('--n_head', type=int, default=12)
parser.add_argument('--n_embd', type=int, default=768)
parser.add_argument('--embd_pdrop', type=float, default=0.1)
parser.add_argument('--resid_pdrop', type=float, default=0.1)
parser.add_argument('--attn_pdrop', type=float, default=0.1)
parser.add_argument('--patch_len', type=int, default=32)
parser.add_argument('--stride', type=int, default=4)
parser.add_argument('--individual', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--r', type=int, default=8)

# model define
parser.add_argument('--expand', type=int, default=2, help='expansion factor for Mamba')
parser.add_argument('--d_conv', type=int, default=4, help='conv kernel size for Mamba')
parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
parser.add_argument('--enc_in', type=int, default=10, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=1, help='output size')
parser.add_argument('--d_model', type=int, default=128, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=3, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=32, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--channel_independence', type=int, default=0,
                    help='0: channel dependence 1: channel independence for FreTS model')
parser.add_argument('--decomp_method', type=str, default='moving_avg',
                    help='method of series decompsition, only support moving_avg or dft_decomp')
parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
# parser.add_argument('--down_sampling_layers', type=int, default=0, help='num of down sampling layers')
# parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size')
# parser.add_argument('--down_sampling_method', type=str, default=None,
#                     help='down sampling method, only support avg, max, conv')
parser.add_argument('--seg_len', type=int, default=48,
                    help='the length of segmen-wise iteration of SegRNN')
# LGB
parser.add_argument('--feature_path', type=str, default='/home/dmz-ai/liruoling/heiy/results/fea/PatchTST', help='npy')

# MLP
parser.add_argument('--MLP_hidden', type=int, default=128,
                    help='The middle tier scale of fc MLPn in ecoder')
parser.add_argument('--MLP_layers', type=int, default=3, help='layers of MLP')
parser.add_argument('--kernel_size', type=int, default=7, help='kernel size of fc conv')
parser.add_argument('--max_depth', type=int, default=2, help='kernel size of fc conv')
parser.add_argument('--weight_std', type=float, default=0.01, help='weight initializes standard deviation')

# timeMixer
parser.add_argument('--down_sampling_layers', type=int, default=3, help='num of down sampling layers')
parser.add_argument('--down_sampling_window', type=int, default=2, help='down sampling window size')
parser.add_argument('--down_sampling_method', type=str, default='avg',
                    help='down sampling method, only support avg, max, conv')
# Client
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--w_lin', type=float, default=1.0, help='initial weight of the linear model')
# Fredformer
parser.add_argument('--cf_dim',         type=int, default=640)   #feature dimension
parser.add_argument('--cf_drop',        type=float, default=0.2)#dropout
parser.add_argument('--cf_depth',       type=int, default=3)    #Transformer layer
parser.add_argument('--cf_heads',       type=int, default=8)    #number of multi-heads
#parser.add_argument('--cf_patch_len',  type=int, default=16)   #patch length
parser.add_argument('--cf_mlp',         type=int, default=640)  #ff dimension
parser.add_argument('--cf_head_dim',    type=int, default=32)   #dimension for single head
parser.add_argument('--cf_weight_decay',type=float, default=0)  #weight_decay
parser.add_argument('--cf_p',           type=int, default=1)    #patch_type
parser.add_argument('--use_nys',           type=int, default=1)    #use nystrom
parser.add_argument('--mlp_drop',           type=float, default=0.3)    #output type
parser.add_argument('--ablation',       type=int, default=0)    #ablation study 012.
parser.add_argument('--fc_dropout', type=float, default=0.05, help='fully connected dropout')
parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
# parser.add_argument('--mlp_hidden', type=int, default=64, help='hidden layer dimension of model')
# CycleNet.
parser.add_argument('--cycle', type=int, default=24, help='cycle length')
parser.add_argument('--model_type', type=str, default='mlp', help='model type, options: [linear, mlp]')
# optimization
parser.add_argument('--num_workers', type=int, default=8, help='data loader num workers')
parser.add_argument('--train_epochs', type=int, default=60, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--early_open', type=bool, default=True)
parser.add_argument('--patience', type=int, default=8, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='optimizer learning rate')
parser.add_argument('--optim_type', type=str, default='Adam', help='select optimizer type, optional[SGD, Adam]')
parser.add_argument('--weight_decay', type=float, default=2e-5, help='weight decay value')
parser.add_argument('--loss', type=str, default='MSE_with_weak', help='loss function, optional[ MSE, MAE, CCC]')
parser.add_argument('--lradj', type=str, default='not',
                    help='adjust learning rate, optional:[type1, type2, not, cos, steplr]')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--clip_value', type=float, default=0.5, help='clip grad')
parser.add_argument('--pct_start', type=int, default=0.6)
# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
parser.add_argument('--dataset_num', type=str, default='0', help='AIOps have 29 dataset,number:0-28')

# FITS
parser.add_argument('--train_mode', type=int,default=0)
parser.add_argument('--cut_freq', type=int,default=0)
parser.add_argument('--base_T', type=int,default=24)
parser.add_argument('--H_order', type=int,default=2)

# tsAMD
parser.add_argument('--n_block', type=int,default=1)
parser.add_argument('--mix_layer_num', type=int,default=2)
parser.add_argument('--mix_layer_scale', type=int,default=2)
parser.add_argument('--alpha', type=float,default=0.0)

# pathformer
parser.add_argument('--num_nodes', type=int, default=7)
parser.add_argument('--layer_nums', type=int, default=3)
parser.add_argument('--k', type=int, default=2, help='choose the Top K patch size at the every layer ')
parser.add_argument('--num_experts_list', type=list, default=[4, 4, 4])
parser.add_argument('--patch_size_list', nargs='+', type=int, default=[16,12,8,32,12,8,6,4,8,6,4,2])
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')
# parser.add_argument('--revin', type=int, default=1, help='whether to apply RevIN')
parser.add_argument('--drop', type=float, default=0.1, help='dropout ratio')
# parser.add_argument('--embed', type=str, default='timeF',
#                     help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--residual_connection', type=int, default=1)
parser.add_argument('--batch_norm', type=int, default=0)

# heiyi
parser.add_argument('--save_path', type=str, default='/data/lrlresults/multiscale_patch', help='train start year')
# parser.add_argument('--is_training', type=int, default=1)
parser.add_argument('--train_start_year', type=str, default='2010', help='train start year')
parser.add_argument('--train_end_year', type=str, default='2017', help='train end year')
parser.add_argument('--val_start_year', type=str, default='2014', help='vali start year')
parser.add_argument('--use_original_feature', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--kfold', action='store_true', help='use kfold', default=False)
parser.add_argument('--per20', action='store_true', help='use foldper20', default=False)
parser.add_argument('--num_fold', type=int, default=5, help='')
parser.add_argument('--pred_task', type=int, default=10, help='y5,y10,y20')
parser.add_argument('--lgb', action='store_true', help='use lgb regressor', default=False)
parser.add_argument('--output_channels', type=int,default=1)
parser.add_argument('--label_type', type=str,default='res')

parser.add_argument('--seed', type=int, default=1024, help='seed')
parser.add_argument('--single_fold', type=int, default=None, help='train single fold for parallel execution')
parser.add_argument('--fold_start', type=int, default=0, help='fold_start')
parser.add_argument('--fold_end', type=int, default=1, help='fold_end')
parser.add_argument('--gpu_list', type=str, default='0', help='GPU list for 5-fold parallel training, separated by comma')
parser.add_argument('--test_only', action='store_true', help='only run testing', default=False)
parser.add_argument('--num_industries', type=int, help='num_industries', default=32)
parser.add_argument('--data_new', type=str, default='5', help='data_new')
parser.add_argument('--n_splits', type=int, help='n_splits', default=3)

parser.add_argument('--random_zero_prob', type=float, help='random_zero_prob', default=0.0)
parser.add_argument('--random_mask_prob', type=float, help='random_mask_prob', default=0.0)
parser.add_argument('--tau_hat_init', type=float, help='tau_hat_init', default=4.5)
parser.add_argument('--test_year', type=str, default=None, help='test_year')



# 并行训练函数
def train_single_fold(fold_id, args_dict, setting):
    """单个fold的训练函数，用于多进程"""
    import torch
    import random
    import numpy as np
    from exp.exp_time_fold import Exp_Multiple_Regression_Fold
    import os

    # --- 关键修改：设置 CUDA 隔离 ---
    # 获取分配给该进程的物理 GPU ID
    assigned_gpu = args_dict['gpus'][fold_id-args_dict['fold_start']]  # 注意：这里要从字典里取 gpus
    # 限制该进程只能看到这一块 GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(assigned_gpu)

    # 重建args对象
    class Args:
        pass

    log_file_path = os.path.join(args_dict['train_log_dir'], f'fold_{fold_id + 1}_training.log')
    log_file = open(log_file_path, 'a', buffering=1)  # buffering=1 表示行缓冲，实时写入
    original_stdout = sys.stdout

    # 将当前进程的所有 print() 输出指向文件
    sys.stdout = log_file
    # 如果希望错误信息也进文件，取消下面这行的注释；如果希望报错在屏幕显示，则保留注释
    # sys.stderr = log_file
    args = Args()
    for key, value in args_dict.items():
        setattr(args, key, value)

    # --- 关键修改：重置内部 GPU ID 为 0 ---
    # 因为设置了 CUDA_VISIBLE_DEVICES，现在这就变成了该进程的第 0 号设备
    args.gpu = 0
    args.device = torch.device("cuda:0")
    # 打印调试信息
    print(f'>>>>>>> Fold {fold_id + 1}: PID {os.getpid()} using Physical GPU {assigned_gpu} (Logical cuda:0) >>>>>>>')
    print(f"日志文件路径: {log_file_path}")
    # 设置随机种子
    fix_seed = args.seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    try:
        # 创建实验并训练
        exp = Exp_Multiple_Regression_Fold(args, single_fold=fold_id)
        exp.train(setting)
        print(f'>>>>>>> Fold {fold_id + 1} Finished Successfully <<<<<<<')
    except Exception as e:
        import traceback
        traceback.print_exc(file=log_file)
        sys.stderr.write(f"\n!!!! Fold {fold_id + 1} Error !!!! 查看日志: {log_file_path}\n")
        traceback.print_exc()
    finally:
        # 关闭文件，虽然进程结束会自动关闭，但显式关闭是好习惯
        log_file.close()

    return fold_id

def check_fold_complete(log_file, fold_id):
    """
    检查指定fold的日志是否包含完成标志（固定格式）
    :param log_file: 日志文件绝对路径
    :param fold_id: 要检查的fold索引（3/4）
    :return: True=完成，False=未完成/日志不存在
    """
    if not os.path.exists(log_file):
        return 0

    # 匹配的核心标志（必须和日志输出完全一致）
    fold_num = fold_id + 1  # fold3→Fold4，fold4→Fold5
    complete_flag = f">>>>>>> Fold {fold_num} Finished Successfully <<<<<<<"

    count = 0
    try:
        # 逐行读取统计，避免内存问题
        with open(log_file, 'r', encoding='utf-8') as f:
            for line in f:
                if complete_flag in line:
                    count += 1
    except Exception as e:
        print(f"⚠️ 读取日志文件失败 {log_file}: {str(e)}")
        return False
    return count


def wait_serverB_folds(serverB_log_dir, wait_interval=300):
    # 服务器负责的 fold 索引
    target_folds = list(range(0,5))
    completed_folds = set()

    # 记录每个 fold 在任务分发瞬间的初始完成次数
    initial_counts = {}
    for fold_id in target_folds:
        log_path = os.path.join(serverB_log_dir, f'fold_{fold_id + 1}_training.log')
        initial_counts[fold_id] = check_fold_complete(log_path, fold_id)

    print(f"\n========== 开始监控训练进度 ==========")
    print(f"等待 Fold: {[f + 1 for f in target_folds]}")

    while len(completed_folds) < len(target_folds):
        for fold_id in target_folds:
            if fold_id in completed_folds:
                continue

            log_file = os.path.join(serverB_log_dir, f'fold_{fold_id + 1}_training.log')
            current_count = check_fold_complete(log_file, fold_id)

            # 条件：当前的完成次数必须严格大于启动时的次数
            if current_count > initial_counts[fold_id]:
                completed_folds.add(fold_id)
                print(f"✅ Fold {fold_id + 1} 已完成 (当前计数: {current_count})")
        if len(completed_folds) == 0:
            time.sleep(wait_interval)
        elif len(completed_folds) < len(target_folds):
            remaining = [f + 1 for f in target_folds if f not in completed_folds]
            print(f"⏳ 正在运行中... 剩余 fold: {remaining}，{wait_interval / 60:.1f} 分钟后再次检查")
            time.sleep(wait_interval)

    print(f"🎉 所有 fold 训练已同步完成！")
    return True

def summarize_fold_results(args, setting):
    """
    汇总所有fold的训练结果
    """
    print(f"\n汇总训练结果: {args.save_path}")

    results = {}
    missing_folds = []
    missing_models = []

    # 读取各fold的结果
    for fold in range(args.num_fold):
        result_file = f'{args.save_path}/fold_{fold + 1}_results.npy'
        model_file = os.path.join(args.checkpoints + '/' + setting, f'best_model_fold_{fold + 1}.pth')

        # 检查结果文件
        if os.path.exists(result_file):
            try:
                fold_result = np.load(result_file, allow_pickle=True).item()
                results[fold] = fold_result
                print(f"\n✓ Fold {fold + 1} 结果:")
                print(f"  - Best Train Corr: {fold_result.get('best_train_corr', 'N/A'):.6f}")
                print(f"  - Best Val Loss:   {fold_result.get('best_val_loss', 'N/A'):.6f}")
                print(f"  - Best Val Corr:   {fold_result.get('best_val_corr', 'N/A'):.6f}")
                print(f"  - Best Val SR:     {fold_result.get('best_val_sr', 'N/A'):.6f}")
                print(f"  - Best Val Metric: {fold_result.get('best_val_metric', 'N/A'):.6f}")
                print(f"  - Nowcast Corr:    {fold_result.get('nowcast_corr', 'N/A'):.6f}")
            except Exception as e:
                print(f"\n× Fold {fold + 1} 结果文件读取失败: {e}")
                missing_folds.append(fold + 1)
        else:
            print(f"\n× Fold {fold + 1} 结果文件未找到")
            missing_folds.append(fold + 1)

        # 检查模型文件
        if not os.path.exists(model_file):
            print(f"  × 模型文件未找到: {model_file}")
            missing_models.append(fold + 1)
        else:
            file_size = os.path.getsize(model_file) / (1024 * 1024)  # MB
            print(f"  ✓ 模型文件: {file_size:.2f} MB")

    # 计算平均值
    if results:
        print("\n" + "=" * 60)
        print("平均结果汇总:")
        print("=" * 60)

        metrics = ['best_train_corr', 'best_val_loss', 'best_val_corr',
                   'best_val_sr', 'best_val_metric', 'nowcast_corr']

        avg_results = {}
        for metric in metrics:
            values = [r.get(metric) for r in results.values() if r.get(metric) is not None]
            if values:
                values = [v.item() if hasattr(v, 'item') else v for v in values]
                mean_val = np.mean(values)
                std_val = np.std(values)
                avg_results[metric] = {'mean': mean_val, 'std': std_val}
                print(f"{metric:20s}: {mean_val:.6f} ± {std_val:.6f}")

        # 保存汇总结果
        with open(f'{args.save_path}/_result_of_multiple_regression.txt', 'a') as f:
            f.write("\n" + "=" * 60 + "\n")
            f.write(f"5折交叉验证汇总结果\n")
            f.write("=" * 60 + "\n\n")

            for fold, result in results.items():
                f.write(f"Fold {fold + 1}:\n")
                for metric in metrics:
                    val = result.get(metric, 'N/A')
                    if val != 'N/A':
                        val = val.item() if hasattr(val, 'item') else val
                        f.write(f"  {metric:20s}: {val:.6f}\n")
                f.write("\n")

            f.write("=" * 60 + "\n")
            f.write("平均结果:\n")
            f.write("=" * 60 + "\n")
            for metric in metrics:
                if metric in avg_results:
                    f.write(f"{metric:20s}: {avg_results[metric]['mean']:.6f}\n")

    # 检查是否可以开始测试
    print("\n" + "=" * 60)
    if missing_folds or missing_models:
        if missing_folds:
            print(f"⚠ 警告: 以下fold缺少结果文件: {missing_folds}")
        if missing_models:
            print(f"⚠ 警告: 以下fold缺少模型文件: {missing_models}")
        print("建议等待所有fold训练完成后再进行测试")
        print("=" * 60)
        return False
    else:
        print("✓ 所有fold训练已完成，模型文件完整，可以开始测试")
        print("=" * 60)
        return True


def worker_process(fold_id, task_queue, result_queue, assigned_gpu):
    """
    负责特定 fold 的常驻进程。启动后即锁定内存逻辑。
    包含完整的日志记录、异常处理和任务完成信号发送功能。
    """
    # --- 1. 环境初始化 (只执行一次) ---
    import os
    import sys
    import torch
    import numpy as np
    import random
    import traceback

    # 绑定物理 GPU
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(assigned_gpu)

    # 强制在子进程启动瞬间导入，之后不再重新 import
    from exp.exp_time_fold import Exp_Multiple_Regression_Fold

    # 打印初始化信息
    print(f'>>> [Init] Fold Worker {fold_id + 1} started. PID: {os.getpid()}, GPU: {assigned_gpu} >>>')

    # --- 2. 循环等待任务 ---
    while True:
        # 从任务队列获取任务
        task = task_queue.get()

        # 退出信号
        if task is None:
            print(f'>>> [Exit] Fold Worker {fold_id + 1} received exit signal. >>>')
            break

        # 解包任务
        if isinstance(task, tuple) and len(task) == 2:
            # 第一种格式: (args对象, setting字符串)
            task_args, setting = task
            args_dict = None
        elif isinstance(task, tuple) and len(task) == 3:
            # 第二种格式: (args字典, fold_id, setting)
            args_dict, task_fold_id, setting = task
            task_args = None
        else:
            print(f"!!! [Error] Fold Worker {fold_id + 1} received malformed task: {type(task)} !!!")
            result_queue.put((fold_id, False, "Malformed task"))
            continue

        # 重建Args对象
        if args_dict is not None:
            class Args:
                pass

            args = Args()
            for key, value in args_dict.items():
                setattr(args, key, value)
        else:
            args = task_args

        # 强制设置内部 GPU 为 0 (因为 CUDA_VISIBLE_DEVICES 已隔离)
        torch.cuda.set_device(assigned_gpu)
        args.gpu = assigned_gpu
        args.device = torch.device(f"cuda:{assigned_gpu}")

        # --- 日志设置 (每个任务追加写入) ---
        # 使用主进程传入的log_dir_base，或者使用args.train_log_dir
        if hasattr(args, 'train_log_dir'):
            log_dir = args.train_log_dir
        else:
            # 默认路径
            log_dir = f'results/logs/'

        log_file_path = os.path.join(log_dir, f'fold_{fold_id + 1}_training.log')

        # 确保目录存在
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

        # 打开日志文件
        log_file = open(log_file_path, 'a', buffering=1)
        original_stdout = sys.stdout
        sys.stdout = log_file

        print(f"\n{'=' * 20} New Task Started {'=' * 20}")
        print(f"Setting: {setting}")

        # 设置随机种子 (每个任务重置，保证复现)
        fix_seed = args.seed
        random.seed(fix_seed)
        torch.manual_seed(fix_seed)
        np.random.seed(fix_seed)

        # 设置线程数
        torch.set_num_threads(12)

        success = False
        error_msg = None

        try:
            # --- 执行训练 ---
            exp = Exp_Multiple_Regression_Fold(args, single_fold=fold_id)
            exp.train(setting)
            print(f'>>>>>>> Fold {fold_id + 1} Finished Successfully <<<<<<<')
            success = True

        except Exception as e:
            # 记录详细错误信息
            error_msg = str(e)
            traceback.print_exc(file=log_file)
            print(f"\n!!!! Fold {fold_id + 1} Error !!!!")
            print(f"Error message: {error_msg}")

        finally:
            # 恢复标准输出并关闭文件
            sys.stdout = original_stdout
            log_file.close()

            # 打印到控制台（方便主进程查看）
            if success:
                print(f'[Worker Fold {fold_id + 1}] 训练完成: {setting}')
            else:
                print(f'[Worker Fold {fold_id + 1}] 训练失败: {error_msg if error_msg else "Unknown error"}')

            # --- 发送完成信号 ---
            # 无论成功失败，都必须发送信号，否则主进程会死锁
            result_queue.put((fold_id, success, error_msg))

            # 清理显存
            torch.cuda.empty_cache()

if __name__ == '__main__':
    # 检查是否在守护进程中运行，避免"daemonic processes are not allowed to have children"错误
    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    # 3. 读取 YAML 并注入
    if args.config:
        with open(args.config, 'r', encoding='utf-8', errors='ignore') as f:
            config_data = yaml.safe_load(f)
            exp_config = config_data.get('experiment', {})

            for key, value in exp_config.items():
                # 只有当用户没有在命令行显式输入该参数时，才用 YAML 覆盖
                    setattr(args, key, value)
    # args.use_multi_gpu=1

    if args.use_gpu and args.use_multi_gpu:
        args.dvices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[-1]


    pred_task = args.pred_task

    args.grad_norm = False
    args.dropout = args.drop_ratio

    
    if args.test_year is None:
        args.test_year = str(int(args.train_end_year)+1)

    args.train_log_dir=args.train_log_dir+f'{args.model}_{args.train_start_year}_{args.test_year}_bs{args.batch_size}_ln{args.e_layers}_dm{args.d_model}_sq{args.seq_len}'
    args.save_dir=args.save_dir+f'{args.fold_type}/phi_ret_mean/{args.data_type}_price/'
    print(args.train_log_dir)
    print(args.save_dir)


    args.val = True


    args.epsilon = 2
    epsilon = 2
  
    if args.cut_freq == 0:
        args.cut_freq = int(args.seq_len // args.base_T + 1) * args.H_order + 10

    '''
        每个ticker的val取20%做kfold, 因子集和原始特征
    '''
   
    # args.is_training = 1
    i=0
    args.patch_size_list = np.array(args.patch_size_list).reshape(args.layer_nums, -1).tolist()

    if args.is_training:
        try:
            mp.set_start_method('spawn', force=True)
            print("Set multiprocessing start method to 'spawn'")
        except RuntimeError:
            # 如果已经设置过，忽略错误
            pass
        # ==========================================
        # TRAINING LOGIC (Multi-process Parallel)
        # ==========================================
        args.gpus = [int(gpu.strip()) for gpu in args.gpu_list.split(',')]
        target_folds = range(args.fold_start, args.fold_end)
        if len(args.gpus) != args.fold_end - args.fold_start:
            raise ValueError(
                f"GPU list size ({len(args.gpus)}) must match num_fold ({args.num_fold}).")

        workers = []
        worker_queues = {}
        result_queue = mp.Queue()  # 用于接收完成信号

        for i, fold_id in enumerate(target_folds):
            q = mp.Queue()
            assigned_gpu = args.gpus[i]

            p = mp.Process(
                target=worker_process,
                args=(fold_id, q, result_queue, assigned_gpu)
            )
            p.start()
            workers.append(p)
            worker_queues[fold_id] = q

        try:
            for args.seq_len in [120, 150]:
                for args.d_model in [256,128]:
                    args.d_ff = args.d_model * 4
                    for args.patch_len in [16, 24]:
                        args.stride = args.patch_len // 2
                        print('Args in experiment:')
                        print(args)

                        if args.data_type == 'daily':
                            if args.task_name == 'Long_term_forecasting':
                                args.pred_task = pred_task
                                args.pred_len = args.pred_task
                            elif args.task_name == 'multiple_regression':
                                args.pred_task = pred_task
                                args.pred_len = 1
                            elif args.task_name == 'predict_feature':
                                args.pred_task = pred_task
                                args.pred_len = 1
                        elif args.data_type == 'min15':
                            if args.task_name == 'Long_term_forecasting':
                                args.pred_task = pred_task
                                args.pred_len = args.pred_task
                            elif args.task_name == 'multiple_regression' or args.task_name == 'classification':
                                args.pred_task = pred_task
                                args.pred_len = 1
                            elif args.task_name == 'predict_feature':
                                args.pred_task = pred_task
                                args.pred_len = 1

                        fix_seed = args.seed
                        random.seed(fix_seed)
                        torch.manual_seed(fix_seed)
                        np.random.seed(fix_seed)
                        args.size = [args.seq_len, args.pred_len]

                        if args.loss == 'MSE_with_weak':
                            train_des = f"{args.model}_test_year{args.test_year}_tau_x{args.tau_hat_init}_kfold{args.kfold}_seq{args.seq_len}_pred{args.pred_len}_ep{args.train_epochs}_bs{args.batch_size}_early{args.patience}_lr{args.learning_rate}_wd{args.weight_decay}_"
                        else:
                            train_des = f"{args.model}_test_year{args.test_year}_kfold{args.kfold}_seq{args.seq_len}_pred{args.pred_len}_ep{args.train_epochs}_bs{args.batch_size}_early{args.patience}_lr{args.learning_rate}_wd{args.weight_decay}_"

                        if args.model == 'FITS':
                            model_des = f"nl{args.n_layer}_nh{args.n_head}_ne_{args.n_embd}_era_dp{args.drop_ratio}_{args.features}_inv{args.individual}_dmo{args.d_model}_dff{args.d_ff}_horder{args.H_order}"
                        else:
                            model_des = f"eps{args.epsilon}_nl{args.e_layers}_nh{args.n_head}_ne_{args.n_embd}_era_dp{args.drop_ratio}_{args.features}_inv{args.individual}_dmo{args.d_model}_dff{args.d_ff}"

                        patching_des = f'_pl{args.patch_len}_sr{args.stride}_val{args.val}'
                        setting = train_des + model_des + patching_des

                        args.save_path = os.path.join(args.save_dir, f'y{args.pred_task}/{args.model}_{setting}')
                        args.checkpoints = args.save_path
                        args.logs_dir = args.save_path + f'/logs'

                        if not os.path.exists(args.train_log_dir):
                            os.makedirs(args.train_log_dir, exist_ok=True)
                        if not os.path.exists(args.save_path):
                            os.makedirs(args.save_path)
                        test_mean_csv_file_path  =args.save_path + '/' + args.model + args.task_name + args.test_year + f'predicted_true_values_mean.csv'   
                        if os.path.exists(test_mean_csv_file_path):
                            continue

                        with open(f'{args.save_path}/_result_of_multiple_regression.txt', 'a') as file:
                            file.write('Args in experiment:\n' + f'{args}\n\n')

                        if args.single_fold is not None:
                            # 单折模式：直接训练指定的fold
                            print(
                                f'>>>>>>>start training fold {args.single_fold + 1} : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>')
                            Exp = Exp_Multiple_Regression_Fold
                            exp = Exp(args, single_fold=args.single_fold)
                            exp.train(setting)
                            print(
                                f'>>>>>>>fold {args.single_fold + 1} training completed<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
                        else:
                            start_time = time.time()
                            print(f"\n[Master] Dispatching tasks for setting: {setting}")

                            for fold_id in target_folds:
                                task_packet = copy.deepcopy(args)
                                task_packet.fold_id = fold_id
                                worker_queues[fold_id].put((task_packet, setting))

                            print(f"[Master] Batch finished in {time.time() - start_time:.2f}s.")
                            print(f"[Master] 等待全局 5 个 Fold 完成 (包含其他服务器)...")

                            if args.fold_start == 0:
                                wait_serverB_folds(args.train_log_dir, wait_interval=300)

                                print('\n====== Summarizing Results ======')
                                all_ready = summarize_fold_results(args, setting)

                                if all_ready and not args.test_only:
                                    print(f'Auto-testing {setting}...')
                                    exp = Exp_Multiple_Regression_Fold(args)
                                    exp.test(setting)
                                    del exp
                                    import gc

                                    gc.collect()
                                    torch.cuda.empty_cache()

                            finished_count = 0
                            while finished_count < len(target_folds):
                                fold_id, success, msg = result_queue.get()
                                finished_count += 1
                                print(
                                    f"[Master] Local Fold {fold_id + 1} signal received ({finished_count}/{len(target_folds)})")
        finally:
            if args.single_fold is None:
                print("\n>>> Shutting down workers...")
                for fold_id in target_folds:
                    worker_queues[fold_id].put(None)  # 发送停止信号
                for p in workers:
                    p.join()
                print(">>> All workers stopped.")

    else:
        # ==========================================
        # TESTING LOGIC (Purely Single-process)
        # ==========================================
        for args.seq_len in [120, 150]:
            for args.d_model in [128, 256]:
                args.d_ff = args.d_model * 2
                for args.patch_len in [16, 24]:
                    args.stride = args.patch_len // 2
                    print('Args in experiment (TESTING ONLY):')
                    print(args)

                    if args.data_type == 'daily':
                        if args.task_name == 'Long_term_forecasting':
                            args.pred_task = pred_task
                            args.pred_len = args.pred_task
                        elif args.task_name == 'multiple_regression':
                            args.pred_task = pred_task
                            args.pred_len = 1
                        elif args.task_name == 'predict_feature':
                            args.pred_task = pred_task
                            args.pred_len = 1
                    elif args.data_type == 'min15':
                        if args.task_name == 'Long_term_forecasting':
                            args.pred_task = pred_task
                            args.pred_len = args.pred_task
                        elif args.task_name == 'multiple_regression' or args.task_name == 'classification':
                            args.pred_task = pred_task
                            args.pred_len = 1
                        elif args.task_name == 'predict_feature':
                            args.pred_task = pred_task
                            args.pred_len = 1

                    fix_seed = args.seed
                    random.seed(fix_seed)
                    torch.manual_seed(fix_seed)
                    np.random.seed(fix_seed)
                    args.size = [args.seq_len, args.pred_len]

                    if args.loss == 'MSE_with_weak':
                        train_des = f"{args.model}_test_year{args.test_year}_tau_x{args.tau_hat_init}_kfold{args.kfold}_seq{args.seq_len}_pred{args.pred_len}_ep{args.train_epochs}_bs{args.batch_size}_early{args.patience}_lr{args.learning_rate}_wd{args.weight_decay}_"
                    else:
                        train_des = f"{args.model}_test_year{args.test_year}_kfold{args.kfold}_seq{args.seq_len}_pred{args.pred_len}_ep{args.train_epochs}_bs{args.batch_size}_early{args.patience}_lr{args.learning_rate}_wd{args.weight_decay}_"

                    if args.model == 'FITS':
                        model_des = f"nl{args.n_layer}_nh{args.n_head}_ne_{args.n_embd}_era_dp{args.drop_ratio}_{args.features}_inv{args.individual}_dmo{args.d_model}_dff{args.d_ff}_horder{args.H_order}"
                    else:
                        model_des = f"eps{args.epsilon}_nl{args.e_layers}_nh{args.n_head}_ne_{args.n_embd}_era_dp{args.drop_ratio}_{args.features}_inv{args.individual}_dmo{args.d_model}_dff{args.d_ff}"

                    patching_des = f'_pl{args.patch_len}_sr{args.stride}_val{args.val}'
                    setting = train_des + model_des + patching_des

                    args.save_path = os.path.join(args.save_dir, f'y{args.pred_task}/{args.model}_{setting}')
                    args.checkpoints = args.save_path
                    args.logs_dir = args.save_path + f'/logs'

                    Exp = Exp_Multiple_Regression_Fold
                    exp = Exp(args)
                    print(f'>>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
                    exp.test(setting)
                    torch.cuda.empty_cache()