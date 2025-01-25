
import gc
import re
import os
import ast
import sys
import json
import time
import math
import random
import argparse
import requests
import numpy as np
import pandas as pd
import scipy.io as sio

from tqdm import tqdm
from collections import defaultdict
from collections import Counter
from logging import getLogger
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
import scipy.sparse as sp
from ast import Global

import daisy
from daisy.utils.loader import RawDataReader, Preprocessor
from daisy.utils.splitter import TestSplitter, ValidationSplitter
from daisy.utils.config import init_seed, init_config, init_logger
from daisy.utils.metrics import MAP, NDCG, Recall, Precision, HR, MRR
from daisy.utils.sampler import BasicNegtiveSampler, SkipGramNegativeSampler, UniqueNegativeSampler
from daisy.utils.dataset import AEDataset, BasicDataset, CandidatesDataset, get_dataloader
from daisy.utils.utils import get_history_matrix, get_ur, build_candidates_set, ensure_dir, get_inter_matrix
from daisy.model.MFRecommender import MF
from daisy.model.FMRecommender import FM
from daisy.model.NFMRecommender import NFM
from daisy.model.NGCFRecommender import NGCF
from daisy.model.EASERecommender import EASE
from daisy.model.SLiMRecommender import SLiM
from daisy.model.VAECFRecommender import VAECF
from daisy.model.NeuMFRecommender import NeuMF
from daisy.model.PopRecommender import MostPop
from daisy.model.KNNCFRecommender import ItemKNNCF
from daisy.model.PureSVDRecommender import PureSVD
from daisy.model.Item2VecRecommender import Item2Vec
from daisy.model.LightGCNRecommender import LightGCN
from daisy.utils.metrics import calc_ranking_results
from sklearn.metrics import mean_squared_error


class BPRData(data.Dataset):
    def __init__(self, train_data_length):
        super(BPRData, self).__init__()
        self.train_data_length = train_data_length
        self.features_fill = None        

    def get_data(self, saved_data_path):
        import pickle
        print('load data: {}'.format(saved_data_path))
        with open(saved_data_path, 'rb') as fp:
            b = pickle.load(fp)
            self.features_fill = b            

    def __len__(self):
        return self.train_data_length

    def __getitem__(self, idx):
        features = self.features_fill 
        if True:    
            user = features[idx][0]
            pos1 = features[idx][1]
            pos2 = features[idx][2]        
            neg1 = features[idx][3]                    
            neg2 = features[idx][4]                                
            return user, pos1, pos2, neg1, neg2


def get_config():
    model_config = {
        'mostpop': MostPop,
        'slim': SLiM,
        'itemknn': ItemKNNCF,
        'puresvd': PureSVD,
        'mf': MF,
        'fm': FM,
        'ngcf': NGCF,
        'neumf': NeuMF,
        'nfm': NFM,
        'multi-vae': VAECF,
        'item2vec': Item2Vec,
        'ease': EASE,
        'lightgcn': LightGCN,
    }

    metrics_config = {
        "recall": Recall,
        "mrr": MRR,
        "ndcg": NDCG,
        "hr": HR,
        "map": MAP,
        "precision": Precision,
    }

    tune_params_config = {
        'mostpop': [],
        'itemknn': ['maxk'],
        'puresvd': ['factors'],
        'slim': ['alpha', 'elastic'],
        'mf': ['num_ng', 'factors', 'lr', 'batch_size', 'reg_1', 'reg_2'],
        'fm': ['num_ng', 'factors', 'lr', 'batch_size', 'reg_1', 'reg_2'],
        'neumf': ['num_ng', 'factors', 'num_layers', 'dropout', 'lr', 'batch_size', 'reg_1', 'reg_2'],
        'nfm': ['num_ng', 'factors', 'num_layers', 'dropout', 'lr', 'batch_size', 'reg_1', 'reg_2'],
        'ngcf': ['num_ng', 'factors', 'node_dropout', 'mess_dropout', 'batch_size', 'lr', 'reg_1', 'reg_2'],
        'multi-vae': ['latent_dim', 'dropout','batch_size', 'lr', 'anneal_cap'],
        'ease': ['reg'],
        'item2vec': ['context_window', 'rho', 'lr', 'factors'],
        'lightgcn': ['num_ng', 'factors', 'batch_size', 'lr', 'reg_1', 'reg_2', 'num_layers'],
    }

    param_type_config = {
        'num_layers': 'int',
        'maxk': 'int',
        'factors': 'int',
        'alpha': 'float',
        'elastic': 'float',
        'num_ng': 'int',
        'lr': 'float',
        'batch_size': 'int',
        'reg_1': 'float',
        'reg_2': 'float',
        'dropout': 'float',
        'node_dropout': 'float',
        'mess_dropout': 'float',
        'latent_dim': 'int',
        'anneal_cap': 'float',
        'reg': 'float',
        'context_window': 'int',
        'rho': 'float'
    }
    
    return model_config, metrics_config, tune_params_config, param_type_config



def train_backbone_model(model, train_samples, train_set, config, fine_tune = 1):
    model_config = config['model_config'] 
    logger = config['logger']

    # Start training
    s_time = time.time()
    algo_name = config['algo_name'].lower()
    if algo_name in ['itemknn', 'puresvd', 'slim', 'mostpop', 'ease']:
        if fine_tune == 0:
            print('Init model')
            model = model_config[algo_name](config)
        model.fit(train_set)

    elif algo_name in ['mf', 'fm', 'neumf', 'nfm', 'ngcf', 'lightgcn']:
        if algo_name in ['lightgcn', 'ngcf']:
            config['inter_matrix'] = get_inter_matrix(train_set, config)
        if fine_tune == 0:
            print('Init model')
            model = model_config[algo_name](config)
        train_dataset = BasicDataset(train_samples)
        train_loader = get_dataloader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
        model.fit(train_loader)

    else:
        raise NotImplementedError(f"Algorithm {algo_name} is not supported.")

    # Log training time
    elapsed_time = time.time() - s_time
    logger.info(f"Finish training: {config['dataset']} {config['prepro']} {config['algo_name']} with {config['loss_type']} and {config['sample_method']} sampling, {elapsed_time:.4f}")

    return model


def get_ur_l(df):
    """
    从 DataFrame 中构建用户-物品交互字典。

    参数:
        df: 包含 'user' 和 'item' 列的 DataFrame。

    返回:
        ur: 用户-物品交互字典，键是用户 ID，值是该用户交互过的物品列表。
    """
    # 确保 'user' 和 'item' 列是整数类型
    df['user'] = df['user'].astype(int)
    df['item'] = df['item'].astype(int)

    # 使用 groupby 和 apply(list) 构建用户-物品字典
    ur = df.groupby('user')['item'].apply(list).to_dict()

    return ur

def get_model_pred(user_set, pred):
    # data = np.array(df)
    # data = data.tolist()
    model_pred = {}
    for i in range(len(user_set)):
        u = user_set[i]
        model_pred[u] = pred[i]
    return model_pred


def cal_single_item_exp(rank_list, user_num):
    sum_exp = 0
    for x in rank_list:
        exp_i = 1 / math.log(x + 1,2)
        sum_exp += exp_i
    return sum_exp / user_num

def cal_all_item_exp(rec_results, user_num, item_num):
    exp_list = []
    for item_id in range(item_num):
        item_i_df = rec_results.loc[rec_results['item'] == item_id]
        if len(item_i_df) == 0:
            exp_i = 0
        else:
            rank_list = item_i_df['rank'].to_list()
            exp_i = cal_single_item_exp(rank_list, user_num)
        exp_list.append(exp_i)
    # exp_list = exp_list / np.sum(exp_list)
    return exp_list


def cal_tgf(exp_list, warm_item_list, cold_item_list):
    """
    计算分类的权重差异（tgf）

    参数:
    exp_list: 每个项目的经验值列表，形状为 [item_num]
    warm_item_list: 暖项目的列表，包含整数索引
    cold_item_list: 冷项目的列表，包含整数索引

    返回:
    cate_tgf: 分类的权重差异
    """
    # 如果 exp_list 为空，返回 0
    if len(exp_list) == 0:
        return 0

    # 计算总经验值并归一化
    total_exp = np.sum(exp_list)
    if total_exp == 0:
        return 0
    exp_list = exp_list / total_exp

    # 如果 warm_item_list 或 cold_item_list 为空，返回 0
    if len(warm_item_list) == 0 or len(cold_item_list) == 0:
        return 0

    # 将列表转换为 numpy 数组
    warm_item_list = np.array(warm_item_list, dtype=int)
    cold_item_list = np.array(cold_item_list, dtype=int)

    # 检查索引是否在有效范围内
    if np.any(warm_item_list >= len(exp_list)) or np.any(cold_item_list >= len(exp_list)):
        raise IndexError("索引超出 exp_list 的范围")

    # 提取暖项目和冷项目的经验值
    warm_exp_list = exp_list[warm_item_list]
    cold_exp_list = exp_list[cold_item_list]

    # 计算暖项目的权重
    warm_num = len(warm_exp_list)
    warm_weight = np.arange(warm_num, 0, -1)  # 从 warm_num 到 1

    # 计算冷项目的权重
    cold_num = len(cold_exp_list)
    if cold_num > 1:
        cold_weight = 1 + np.arange(cold_num) * (warm_num - 1) / (cold_num - 1)
    else:
        cold_weight = np.array([1])

    # 计算暖项目的部分
    warm_part = np.sum(warm_exp_list * warm_weight) / warm_num

    # 计算冷项目的部分
    cold_part = np.sum(cold_exp_list * cold_weight) / cold_num

    # 计算分类的权重差异
    cate_tgf = warm_part - cold_part
    if cate_tgf < 0:
        cate_tgf = cate_tgf / (warm_num / cold_num)
    return cate_tgf

def remove_duplicates(preds):
    preds = preds.astype(int)
    preds_new = []
    for lst in preds:
        seen = set()
        result = [x for x in lst if not (x in seen or seen.add(x))]
        preds_new.append(result)
    return preds_new


def evaluate_backbone(model, test_u, test_ur, test_ucands, config): 
    test_dataset = CandidatesDataset(test_ucands)
    test_loader = get_dataloader(test_dataset, batch_size=64, shuffle=False, num_workers=0)
    preds = model.rank(test_loader) # np.array (u, topk)
    preds_new = remove_duplicates(preds) 
    preds_topk = np.array([x[:config['topk']] for x in preds_new])
    
    rec_results = create_recommendation_df(test_u, preds_topk)
    return preds_topk, rec_results


def transform_to_single_row(df):
    # 初始化一个空字典，用于存储结果
    result = {}
    
    # 遍历每一行
    for _, row in df.iterrows():
        metric = row["KPI@K"]  # 获取指标名称（如 NDCG, Hit Ratio 等）
        for k in [10, 20, 50]:  # 遍历 K 值
            col_name = f"{metric}@{k}"  # 构造列名（如 NDCG@10）
            result[col_name] = row[k]  # 将值存入字典
    
    # 将字典转换为 DataFrame
    result_df = pd.DataFrame([result])
    return result_df

def get_rec_metric(test_u, test_ur, preds_topk, config):
    topk_list = config['topk_list']
    rec_metric = calc_ranking_results(test_ur, preds_topk, test_u, config)
    return rec_metric.loc[2:3, ['KPI@K'] + topk_list]


def get_evaluation_metric(test_u, test_ur, preds_topk, rec_results, config):
    rec_metric = get_rec_metric(test_u, test_ur, preds_topk, config)
    fairness_metric = get_fairness_metric(rec_results, test_ur, config)
    metric_result = pd.concat([rec_metric, fairness_metric])
    result_df = transform_to_single_row(metric_result)
    return result_df

def update_new_item(last_sate_set, config):
    warm_item_list = last_sate_set[config['IID_NAME']].unique()
    cold_item_list = list(set(range(config['item_num'])) - set(warm_item_list))
    return warm_item_list, cold_item_list


def add_user_inter(df, user_item_dict):
    """
    为输入的 DataFrame 增加一列 'user_inter'，根据规则进行采样。

    参数:
    df (pd.DataFrame): 包含 'user', 'item', 'rank' 列的 DataFrame。
    user_item_dict (dict): 字典，形式为 {'user': {item1, item2, ...}}。

    返回:
    pd.DataFrame: 增加 'user_inter' 列后的 DataFrame。
    """
    def sample_user_inter(row):
        user = row['user']
        item = row['item']
        rank = row['rank']

        # 检查 item 是否在字典中对应 user 的集合中
        if user in user_item_dict and item in user_item_dict[user]:
            # 计算采样概率
            prob = 1 / (np.log2(rank + 1))
            # 按概率采样
            return np.random.binomial(1, prob)
        else:
            # 如果不在字典中，设置为 0
            return 0

    # 使用 apply 对每一行应用函数
    df['user_inter'] = df.apply(sample_user_inter, axis=1)
    return df


def get_weight(warm_item_list, cold_item_list):
    """
    计算暖项目和冷项目的权重
    """
    warm_num = len(warm_item_list)
    warm_weight = np.arange(warm_num, 0, -1)  # 从 warm_num 到 1

    cold_num = len(cold_item_list)
    if cold_num > 1:
        cold_weight = 1 + np.arange(cold_num) * (warm_num - 1) / (cold_num - 1)
    else:
        cold_weight = np.array([1])
    return warm_weight, cold_weight


def set_values_by_id(interaction_history, item_num):
    """
    将用户的交互历史（列表）转换为一个固定长度的列表，交互过的物品位置为 1，未交互过的为 0。
    """
    # 将 interaction_history 转换为集合以提高成员检查效率
    interaction_set = set(interaction_history)
    return [1 if i in interaction_set else 0 for i in range(item_num)]


def calculate_mse_divergence(df1, df2, user_col='user', tgf_col='user_tgf'):
    """
    计算两个 DataFrame 中 user_tgf 列的 JS 散度。
    
    参数:
    df1 (pd.DataFrame): 第一个 DataFrame。
    df2 (pd.DataFrame): 第二个 DataFrame。
    user_col (str): user 列的名称，默认为 'user'。
    tgf_col (str): user_tgf 列的名称，默认为 'user_tgf'。
    
    返回:
    float: mse 值。
    """
    # 按 user 列升序排列
    df1 = df1.sort_values(by=user_col).reset_index(drop=True)
    df2 = df2.sort_values(by=user_col).reset_index(drop=True)
    
    # 取出两个 DataFrame 中 user 列重合的部分
    common_users = set(df1[user_col]).intersection(set(df2[user_col]))
    df1_common = df1[df1[user_col].isin(common_users)]
    df2_common = df2[df2[user_col].isin(common_users)]
    
    # 确保两个 DataFrame 的 user 列顺序一致
    df1_common = df1_common.sort_values(by=user_col).reset_index(drop=True)
    df2_common = df2_common.sort_values(by=user_col).reset_index(drop=True)
    
    # 提取 user_tgf 列
    p = df1_common[tgf_col].values
    q = df2_common[tgf_col].values
    
    # # 归一化，确保是概率分布
    # p = p / np.sum(p)
    # q = q / np.sum(q)
    
    # 计算 JS 散度
    mse_distance = mean_squared_error(p, q)
    return mse_distance

def get_fairness_metric(rec_results, test_ur, config):
    user_his = config['user_his']
    topk_list = config['topk_list']
    warm_item_list = config['warm_item_list']
    cold_item_list = config['cold_item_list']
    tgf_results = ['TGF']
    unf_results = ['UNF']
    nc_results = ['NC']
    for k in topk_list:
        temp_rec_results = rec_results.loc[rec_results['rank'] <= k]
        exp_list = cal_all_item_exp(temp_rec_results, config['user_num'], config['item_num'])
        tgf_value = cal_tgf(exp_list, warm_item_list, cold_item_list)
        tgf_results.append(tgf_value)
        cold_item = temp_rec_results.loc[temp_rec_results['item'].isin(cold_item_list)]
        nc_value = len(cold_item) / len(temp_rec_results)
        nc_results.append(nc_value)
        # temp_rec_results = add_user_inter(temp_rec_results, test_ur)
        # user_rec_records = generate_user_records(temp_rec_results)
        # user_rec_records = get_stage_user_tgf(user_rec_records, 'rec_list', config)
        # mse_distance = calculate_mse_divergence(user_his, user_rec_records, user_col='user', tgf_col='hist_tgf')
        # unf_results.append(mse_distance)
        
    # data = [tgf_results, nc_results, unf_results]
    data = [tgf_results, nc_results]
    columns = ['KPI@K'] + topk_list
    fairness_metric = pd.DataFrame(data, columns=columns)
    return fairness_metric




def negative_sampling(df):
    result = []  # 存储最终结果
    
    # 遍历每一行数据
    for index, row in df.iterrows():
        user = row['user']
        rec_list = row['rec_list']
        positive_inter = row['positive_inter']
        
        # 如果 positive_inter 为空，则跳过当前行
        if not positive_inter:
            continue
        
        # 计算候选负样本集合
        negative_candidates = list(rec_list - positive_inter)  # 转换为列表方便采样
        
        # 对每个正样本进行负采样
        for pos in positive_inter:
            # 从候选负样本中随机采样 4 个
            neg_samples = random.sample(negative_candidates, min(4, len(negative_candidates)))
            # 生成 [user, pos, neg] 并添加到结果中
            result.extend([[user, pos, neg] for neg in neg_samples])
    
    return result

def create_recommendation_df(user_ids, recommendations):
    """
    创建一个包含用户ID、item ID和排序位置的Pandas DataFrame。

    参数:
    user_ids (list): 用户ID列表。
    recommendations (list of lists): 每个用户的推荐列表，每个列表包含50个item id。

    返回:
    pandas.DataFrame: 包含三列（user, item, rank）的DataFrame。
    """
    # 初始化一个空的列表来存储数据
    data = []

    # 遍历每个用户的推荐列表
    for user, user_list in zip(user_ids, recommendations):
        for rank, item in enumerate(user_list, start=1):
            data.append([user, item, rank])

    # 创建DataFrame
    df = pd.DataFrame(data, columns=['user', 'item', 'rank'])

    return df


def pcc_train(model_here, train_data, item_pop_dict, config):
    data2 = train_data.copy()
    
    data2['user'] = data2['user'].apply(lambda x : int(x))
    data2['item'] = data2['item'].apply(lambda x : int(x))    
        
    filter_users = data2.user.value_counts()[data2.user.value_counts() > 1].index
    data2 = data2[data2.user.isin(filter_users)]
    data2 = data2.reset_index()[['user', 'item']]
    
    user_num = len(data2.user.unique())
    data_len = data2.shape[0]
    frac = 50
    frac_user_num = int(data_len/frac)
    
    predictions_list = torch.tensor([]).cuda()
    
    model_here.cuda()
    
    for itr in range(frac):
        tmp = data2.iloc[ (frac_user_num*itr) : (frac_user_num*(itr+1)) ].values    
        user = tmp[:, 0]
        user = user.astype(np.int64)        
        user = torch.from_numpy(user).cuda()
        item = tmp[:, 1]
        item = item.astype(np.int64)        
        item = torch.from_numpy(item).cuda()
        if config['algo_name'] == 'mf':
            predictions = model_here(user, item)
        elif config['algo_name'] == 'lightgcn':
            user_emb_all, item_emb_all = model_here.forward()
            user_emb = user_emb_all[user]
            item_emb = item_emb_all[item]
            predictions = (user_emb * item_emb).sum(dim=-1)
        
        predictions_list = torch.hstack((predictions_list, predictions))

        
        if itr+1 == frac:
            tmp = data2.iloc[ (frac_user_num*(itr+1)):].values    
            user = tmp[:, 0]
            user = user.astype(np.int64)        
            user = torch.from_numpy(user).cuda()
            item = tmp[:, 1]
            item = item.astype(np.int64)        
            item = torch.from_numpy(item).cuda()

            if config['algo_name'] == 'mf':
                predictions = model_here(user, item)
                # predictions = F.softmax(predictions, dim=0)
            elif config['algo_name'] == 'lightgcn':
                user_emb_all, item_emb_all = model_here.forward()
                user_emb = user_emb_all[user]
                item_emb = item_emb_all[item]
                predictions = (user_emb * item_emb).sum(dim=-1)
                # predictions = F.softmax(predictions, dim=0)
            predictions_list = torch.hstack((predictions_list, predictions))

    # item_pop_dict = dict(item_pop.values)
    data2['item_pop_count'] = data2['item'].map(item_pop_dict)            
        
    values = predictions_list.reshape(-1, 1)
    item_pop_count = data2.item_pop_count.values
    item_pop_count = item_pop_count.astype(np.int32)
    item_pop_count = torch.from_numpy(item_pop_count).float().cuda()
    
    
    X = values
    Y = item_pop_count # item pop
    
    X = X.view([1, -1])
    Y = Y.view([1, -1])
    pcc = ((X - X.mean())*(Y - Y.mean())).sum() / ((X - X.mean())*(X- X.mean())).sum().sqrt() / ((Y - Y.mean())*(Y- Y.mean())).sum().sqrt()    
    
    return pcc

def calculate_loss(sample, pos_scores, neg_scores, pos, neg, config):
    item_pop_train_dict = config['item_pop_train_dict']
    warm_item_train = config['warm_item_list']
    cold_item_train = config['cold_item_list']
    acc_w = config['weight']
    pop_w = 1.0 - acc_w
    """计算损失"""
    if sample == 'bpr':
        acc_loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()
        return acc_loss, (acc_loss.item(), 0)
    elif sample in ['posneg']:
        acc_loss = -torch.log(torch.sigmoid(pos_scores - neg_scores)).mean() / 2
        pop_loss =  -(1 -(pos_scores + neg_scores).abs().tanh() + 1e-8).log().mean() / 2
        # pop_loss = -torch.log(1 - (pos_scores + neg_scores).abs().tanh() + 1e-8).mean() / 2
        return acc_loss * acc_w + pop_loss * pop_w, (acc_loss.item(), pop_loss.item())
    elif sample in ['pd']:
        pos_weight = torch.tensor([item_pop_train_dict[item.item()] for item in pos], device=pos.device)
        neg_weight = torch.tensor([item_pop_train_dict[item.item()] for item in neg], device=neg.device)
        m = nn.ELU()
        return -torch.log(torch.sigmoid((m(pos_scores) + 1.0) * pos_weight ** acc_w - (m(neg_scores) + 1.0) * neg_weight ** acc_w)).mean(), None
    # elif sample == 'ips':
    #     pos_weight = torch.tensor([1.0 / item_pop_train_dict[item.item()] for item in pos], device=pos.device)
    #     neg_weight = torch.tensor([1.0 / item_pop_train_dict[item.item()] for item in neg], device=neg.device)
    #     return -torch.log(torch.sigmoid(pos_scores) * pos_weight + torch.sigmoid(1 - neg_scores) * neg_weight).mean(), None
    elif sample == 'pearson':
        return -torch.log(torch.sigmoid(pos_scores - neg_scores)).mean(), None
    elif sample == 'pearson_new':
        acc_loss = -torch.log(torch.sigmoid(pos_scores - neg_scores)).mean()
        pearson_loss = pcc_loss(pos, pos_scores, item_pop_train_dict)
        return acc_loss + pop_w * pearson_loss, (acc_loss.item(), pearson_loss.item())
    else:
        raise ValueError(f"Unsupported sample method: {sample}")

def train_baseline_model(model, train_loader, train_set, config, device='cuda'):
    """训练函数"""
    item_pop_train_dict = config['item_pop_train_dict']
    warm_item_train = config['warm_item_list']
    cold_item_train = config['cold_item_list']
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    weight = config['weight']
    
    # 训练循环
    for epoch in range(config['epochs']):
        print(f'Epoch {epoch + 1}/{config["epochs"]}')
        model.train()
        model.to(device)
        
        if config['debias_method'] in ['cnif', 'posneg']:
            config['burnin'] = 'yes'
        # 动态选择采样方法
        sample = 'bpr' if (epoch < (config['epochs']/2) and config['burnin'] == 'yes') else config['debias_method']

        # 遍历数据加载器
        # if sample == 'cnif':
        #     for i in range(10):
        #         model.zero_grad()
        #         users = np.array(range(config['user_num']))
        #         items = np.array(range(config['item_num']))
        #         cgf_loss = cgf_topk_loss(model, users, items, config)   
        #         loss = weight * ((cgf_loss - 0.3430)**2)
        #         print(f'cgf_loss of epoch : {cgf_loss.item()}') 
        #         print(f'loss of epoch : {loss.item()}')  
        #         loss.backward()
        #         optimizer.step()
        if sample == 'cnif':
            for epoch in range(10):  # 假设训练10个epoch
                # 获取所有用户和物品
                users = np.array(range(config['user_num']))
                items = np.array(range(config['item_num']))

                batch_size = config['batch_size']
                num_batches = int(np.ceil(len(users) / batch_size))
                count_cnif = 0
                for batch_idx in range(num_batches):
                    model.zero_grad()  
                    start_idx = batch_idx * batch_size
                    end_idx = min((batch_idx + 1) * batch_size, len(users))
                    batch_users = users[start_idx:end_idx]

                    cgf_loss = cgf_topk_loss(model, batch_users, items, config)
                    loss = weight * ((cgf_loss - 0.3430)**2)
                    if count_cnif % 5 == 0:
                        print(f'cgf_loss of batch {batch_idx + 1}/{num_batches} in epoch {epoch + 1}: {cgf_loss.item()}')
                        print(f'loss of batch {batch_idx + 1}/{num_batches} in epoch {epoch + 1}: {loss.item()}')
                    count_cnif += 1
                    # 反向传播和优化
                    loss.backward()
                    optimizer.step()
        else:
            count = 0
            for user, pos1, neg1 in tqdm(train_loader, desc='Training'):
                # 将数据移动到设备
                user = user.to(device)
                pos = pos1.to(device)
                neg = neg1.to(device)

                # 清零梯度
                optimizer.zero_grad()

                # 计算 pos_score 和 neg_score
                if config['algo_name'] == 'mf':
                    user_emb = model.embed_user(user)
                    pos_emb = model.embed_item(pos)
                    neg_emb = model.embed_item(neg)
                    pos_scores = model(user, pos)
                    neg_scores = model(user, neg)
                    # print(f"pos_scores shape (MF): {pos_scores.shape}")  # Debugging
                    # print(f"neg_scores shape (MF): {neg_scores.shape}")  # Debugging
                    # pos_scores = F.softmax(pos_scores, dim=0)  # 对每一行的正样本得分计算 softmax
                    # neg_scores = F.softmax(neg_scores, dim=0)
                    
                elif config['algo_name'] == 'lightgcn':
                    user_emb_all, item_emb_all = model.forward()
                    user, pos, neg = to_long_tensor(user, pos, neg)
                    user_emb = user_emb_all[user]
                    pos_emb = item_emb_all[pos]
                    neg_emb = item_emb_all[neg]
                    pos_scores = (user_emb * pos_emb).sum(dim=-1)
                    neg_scores = (user_emb * neg_emb).sum(dim=-1)
                    # pos_scores = F.softmax(pos_scores, dim=0)  # 对每一行的正样本得分计算 softmax
                    # neg_scores = F.softmax(neg_scores, dim=0)

                # 计算损失
                loss, extra_loss_info = calculate_loss(sample, pos_scores, neg_scores, pos, neg, config)

                # 添加正则化项
                if config['add_reg'] == 'yes':
                    # loss += config['reg_1'] * (user_emb.norm(p=1) + pos_emb.norm(p=1) + neg_emb.norm(p=1))
                    # loss += config['reg_2'] * (user_emb.norm() + pos_emb.norm() + neg_emb.norm())
                    reg = (torch.norm(user_emb) ** 2 + torch.norm(pos_emb)** 2  + torch.norm(neg_emb) ** 2)/3 / config['batch_size']
                    loss += config['reg_2'] * reg
                    # loss += 1e-5 * reg
                    # loss += calculate_regularization(model, user, pos, neg, config['reg_1'], config['reg_2'], config['batch_size'])

                # 反向传播和优化
                loss.backward()
                optimizer.step()

                # 打印损失信息
                if count % 1000 == 0:
                    if extra_loss_info:
                        print(f'Loss: {loss.item()}, Extra acc Loss: {extra_loss_info[0]}, Extra pop Loss: {extra_loss_info[1]}')
                count = count + 1
            if sample == 'pearson':
                pcc = pcc_train(model, train_set, item_pop_train_dict, config) 
                loss = weight * (pcc**2)
                print(f'pcc loss of epoch : {pcc.item()}')
                print(f'loss of epoch : {loss.item()}')
                loss.backward()
                optimizer.step()

    return model

def cgf_topk_loss(model_cgf, users, items, config):
    warm_items = config['warm_item_list']
    cold_items = config['cold_item_list']
    item_num = len(items)
    warm_nums = len(warm_items)
    cold_nums = len(cold_items)
    top_score = config['topk']
    # print(f'all item num:{item_num}, warm_num:{warm_nums}, cold_num:{cold_nums}')
    
    # 将数据移动到 GPU
    model_cgf.cuda()
    users = torch.tensor(users).cuda()
    items = torch.tensor(items).cuda()
    
    # 获取用户和物品的嵌入
    if config['algo_name'] == 'mf':
        user_embed = model_cgf.embed_user(users)
        item_embed = model_cgf.embed_item(items)
    elif config['algo_name'] == 'lightgcn':
        user_emb_all, item_emb_all = model_cgf.forward()
        user_embed = user_emb_all[users]
        item_embed = item_emb_all[items]
    
    # 计算分数矩阵
    score_mat = torch.mm(item_embed, user_embed.t())
    # score_mat = F.softmax(score_mat, dim=0)
    
    # 获取 top-k 分数和索引
    _, item_index = score_mat.topk(top_score, dim=0)
    
    # 使用掩码操作替代 scatter_ 和 gather
    mask = torch.zeros_like(score_mat).cuda()
    mask.scatter_(0, item_index, 1)  # 创建一个掩码，标记 top-k 的位置
    score_top = score_mat * mask  # 通过掩码提取 top-k 分数
    
    # 计算期望分数（使用对数归一化）
    exp_item_list = score_top.mean(dim=1)
    exp_item_list = exp_item_list / exp_item_list.sum()
    
    # 批量计算权重
    weight_list = torch.zeros(item_num).cuda()
    weight_list[:warm_nums] = (warm_nums - torch.arange(warm_nums).cuda()) / warm_nums
    weight_list[warm_nums:] = -(1 + (torch.arange(cold_nums).cuda() * (warm_nums - 1) / (cold_nums - 1))) / cold_nums
    
    # 计算损失
    cgf_loss = (exp_item_list * weight_list).sum()
    
    return cgf_loss


def to_long_tensor(*args, device='cuda'):
    """
    将输入的变量（numpy 数组或 PyTorch 张量）转换为 torch.long 类型，并移动到指定设备。

    参数:
        *args: 输入的变量（numpy 数组或 PyTorch 张量）。
        device: 目标设备，默认为 'cuda'。

    返回:
        转换后的 torch.long 张量（或多个张量）。
    """
    results = []
    for arg in args:
        if isinstance(arg, np.ndarray):  # 如果是 numpy 数组
            arg = arg.astype(np.int64)  # 转换为 64 位整数
            arg = torch.from_numpy(arg).to(device)  # 转换为 PyTorch 张量并移动到设备
        elif isinstance(arg, torch.Tensor):  # 如果是 PyTorch 张量
            arg = arg.long().to(device)  # 转换为 long 类型并移动到设备
        else:
            raise TypeError(f"Unsupported type: {type(arg)}. Expected numpy array or PyTorch tensor.")
        results.append(arg)
    return results[0] if len(results) == 1 else tuple(results)



def calculate_nc(input_list, target_list):
    """
    统计给定集合中有多少比例的值出现在目标集合中

    参数:
    input_set: 输入的集合
    target_set: 目标集合

    返回:
    proportion: 出现在目标集合中的比例
    """
    if not input_list:
        return 0.0
    count = len(set(input_list) & set(target_list))  # 计算交集的大小
    return count / len(input_list)

def get_user_hist_tgf(exp_list, warm_item_list, cold_item_list, warm_weight, cold_weight):
    """
    计算用户的分类权重差异（tgf）
    """
    if np.sum(exp_list) == 0:
        return 0

    # 归一化经验值
    exp_list = exp_list / np.sum(exp_list)

    # 提取暖项目和冷项目的经验值
    warm_exp_list = exp_list[warm_item_list]
    cold_exp_list = exp_list[cold_item_list]

    # 计算暖项目和冷项目的部分
    warm_part = np.sum(warm_exp_list * warm_weight) / len(warm_item_list)
    cold_part = np.sum(cold_exp_list * cold_weight) / len(cold_item_list)

    # 计算分类的权重差异
    user_tgf = warm_part - cold_part
    if user_tgf < 0:
        user_tgf = user_tgf / (len(warm_item_list) / len(cold_item_list))
    return user_tgf

def update_hist_df(rec_results, rl_user_his_pref, test_ur, config):
    """
    更新用户历史偏好 DataFrame。

    参数:
    - rec_results: 推荐结果数据
    - rl_user_his_pref: 用户历史偏好 DataFrame
    - test_ur: 测试集的用户-物品交互数据
    - config: 配置字典

    返回:
    - rl_user_his_pref: 更新后的用户历史偏好 DataFrame
    """
    # 复制推荐结果并添加用户交互数据
    temp_rec_result = rec_results.copy(deep = True)
    temp_rec_result = add_user_inter(temp_rec_result, test_ur)
    
    # 生成用户推荐记录
    user_rec_records = generate_user_records(temp_rec_result)
    
    # 更新用户历史偏好
    rl_user_his_pref = update_user_history(rl_user_his_pref, user_rec_records)
   
    return rl_user_his_pref
    # 计算用户的 TGF + NC
    
def update_user_history(df1, df2):
    """
    更新用户历史记录。

    参数:
    - df1: 包含用户历史记录的 DataFrame
    - df2: 包含用户新增交互记录的 DataFrame

    返回:
    - df1: 更新后的 DataFrame
    """
    # 确保 user_history 是列表类型
    df1['user_history'] = df1['user_history'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    
    # 遍历 df2 更新 user_history
    for user, positive_inter in df2[['user', 'positive_inter']].itertuples(index=False):
        if positive_inter:  # 如果 positive_inter 不为空
            # 找到 df1 中对应的 user，并更新 user_history
            df1.loc[df1['user'] == user, 'user_history'] = df1.loc[df1['user'] == user, 'user_history'].apply(
                lambda x: x + [item for item in positive_inter if item not in x]
            )
    
    return df1  

def generate_user_records(df):
    """
    生成新的 DataFrame，包含 user, rec_list, positive_inter 列。

    参数:
    df (pd.DataFrame): 包含 'user', 'item', 'user_inter' 列的 DataFrame。

    返回:
    pd.DataFrame: 包含 user, rec_list, positive_inter 列的新 DataFrame。
    """
    # 按 user 分组，搜集 item 和 user_inter=1 的 item
    grouped = df.groupby('user').agg(
        rec_list=('item', set),  # 搜集所有 item
        positive_inter=('item', lambda x: set(x[df.loc[x.index, 'user_inter'] == 1]))  # 搜集 user_inter=1 的 item
    ).reset_index()

    return grouped 

def get_stage_user_tgf(df, col, config, nc_weight = 0.9):
    
    # 将 warm_item_list 和 cold_item_list 转换为 NumPy 数组
    warm_item_list = np.array(config['warm_item_list'])
    cold_item_list = np.array(config['cold_item_list'])

    # 预先计算权重
    warm_weight, cold_weight = get_weight(warm_item_list, cold_item_list)

    # 应用函数生成 exp_list
    df['exp_list'] = df[col].apply(lambda x: set_values_by_id(x, config['item_num']))

    # 计算用户的 TGF 值
    df['hist_tgf'] = df['exp_list'].apply(
        lambda x: get_user_hist_tgf(x, warm_item_list, cold_item_list, warm_weight, cold_weight)
    )

    # 将 cold_item_list 转换为集合
    cold_item_set = set(cold_item_list)
    

    # 计算用户的 NC（Novelty Coverage）值
     
    
    new_stage_nc = df[col].apply(lambda x: calculate_nc(x, cold_item_set))

    # 更新 hist_nc 列
    df['hist_nc'] = nc_weight * df['hist_nc'] + (1 - nc_weight) * new_stage_nc

    return df


import numpy as np

def generate_user_action_space(user_items_dict, user_prob_dict, new_item_list, config):
    """
    对每个用户，根据概率从 user_items_dict 和 new_item_list 中挑选商品，生成新的 user_action_space。

    参数:
    - user_items_dict: 字典，格式为 {'user': [item1, item2, ...]}
    - user_prob_dict: 字典，格式为 {'user': value}，value 是概率值
    - new_item_list: 列表，包含新的项目
    - config: 配置字典，包含随机种子和 topk 参数

    返回:
    - user_action_space: 新的字典，格式为 {'user': [item1, item2, ...]}
    """
    seed = config.get('seed', None)
    len_action_space = config.get('topk', 10) + 50  # 默认 topk 为 10

    # 设置随机种子
    if seed is not None:
        np.random.seed(seed)

    # 将 new_item_list 转换为集合以提高查找效率
    new_item_set = set(new_item_list)

    # 初始化新的字典
    user_action_space = {}

    # 遍历用户
    for user, items in user_items_dict.items():
        # 获取用户的概率值
        prob = user_prob_dict.get(user, 0)

        # 拆分 items 为 pred_old_item_list 和 pred_new_item_list
        pred_old_item_list = [item for item in items if item not in new_item_set]
        pred_new_item_list = [item for item in items if item in new_item_set]

        # 计算需要从 pred_old_item_list 中挑选的商品数量 N_old
        N_old = int(len_action_space * (1 - prob))
        N_old = min(N_old, len(pred_old_item_list))  # 不能超过 pred_old_item_list 的长度

        # 计算需要从 pred_new_item_list 中挑选的商品数量 N_new
        N_new = len_action_space - N_old

        # 从 pred_old_item_list 中取前 N_old 个旧品
        selected_old_items = pred_old_item_list[:N_old]

        # 从 pred_new_item_list 中取前 N_new 个新品
        if len(pred_new_item_list) >= N_new:
            selected_new_items = pred_new_item_list[:N_new]
        else:
            # 如果 pred_new_item_list 中的数量不够，则从 new_item_list 随机采样补齐
            remaining_new_items = list(set(new_item_list) - set(pred_new_item_list))
            selected_new_items = pred_new_item_list + np.random.choice(remaining_new_items, size=N_new - len(pred_new_item_list), replace=False).tolist()

        # 将 selected_new_items 随机插入到 selected_old_items 中，同时保持 selected_new_items 的顺序
        updated_items = selected_old_items.copy()
        insert_indices = np.random.choice(len(updated_items) + 1, size=N_new, replace=True)
        insert_indices.sort()  # 确保插入顺序正确
        for idx, item in zip(insert_indices, selected_new_items):
            updated_items.insert(idx, item)

        # 将结果添加到新的字典中
        user_action_space[user] = updated_items

    return user_action_space

def df_get_neighbors(input_df, obj, max_num):
    """
    Get users' neighboring items.
    return:
        nei_array - [max_num, neighbor array], use 0 to pad users which have no neighbors.
    """
    group = tuple(input_df.groupby(obj))
    keys, values = zip(*group)  # key: obj scalar, values: neighbor array

    keys = np.array(keys, dtype=np.int64)
    opp_obj = 'item' if obj == 'user' else 'user'
    values = list(map(lambda x: x[opp_obj].values, values))
    values.append(0)
    values = np.array(values, dtype=object)

    nei_array = np.zeros((max_num,), dtype=object)
    nei_array[keys] = values[:-1]
    return nei_array


from torch.cuda.amp import GradScaler, autocast

def train_model(model, optimizer, train_loader, config, device='cuda:0'):
    """
    训练模型并保存最佳模型。
    """
    # 将模型移动到指定设备
    model = model.to(device)
    for epoch in range(1, config['max_epoch'] + 1):
        model.train()
        epoch_loss = 0.0
        with tqdm(train_loader, unit="batch", desc=f"Epoch {epoch}/{config['max_epoch']}") as pbar:
            for batch_idx, batch in enumerate(pbar):
                pos_item_content, pos_item_emb, neg_item_content, neg_item_emb, user_emb_batch, pos_item_freq, neg_item_freq = batch
                pos_item_content = pos_item_content.to(device)
                pos_item_emb = pos_item_emb.to(device)
                neg_item_content = neg_item_content.to(device)
                neg_item_emb = neg_item_emb.to(device)
                user_emb_batch = user_emb_batch.to(device)
                pos_item_freq = pos_item_freq.to(device)
                neg_item_freq = neg_item_freq.to(device)

                optimizer.zero_grad()
                loss = model.compute_loss(pos_item_content, pos_item_emb, neg_item_content, neg_item_emb, user_emb_batch, pos_item_freq, neg_item_freq)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                pbar.set_postfix(loss=loss.item())
        epoch_loss /= len(train_loader)
        print(f'Epoch [{epoch}/{config["max_epoch"]}], Average Loss: {epoch_loss:.4f}')

    print("Finish training model at epoch {}.".format(epoch))
    return model

def get_test_predict(test_data, gen_user_emb, gen_item_emb, warm_items, cold_items):
    test_data = test_data[['user', 'item', 'label']]
    test_data['user'] = test_data['user'].apply(lambda x : int(x))
    test_data['item'] = test_data['item'].apply(lambda x : int(x))
    test_users_num = test_data['user'].nunique()   
    
    warm_data = test_data[test_data['item'].isin(warm_items)]
    cold_data = test_data[test_data['item'].isin(cold_items)]
    warm_predict_list = predict_score_embed(warm_data, gen_user_emb[0], gen_item_emb)
    warm_data['pred'] = warm_predict_list
    cold_predict_list = predict_score_embed(cold_data, gen_user_emb[1], gen_item_emb)
    cold_data['pred'] = cold_predict_list
    result_pd = pd.concat([warm_data, cold_data])
    return result_pd

def predict_score_embed(target_data, user_emb, item_emb , test_batch = 10000):
    data_len = target_data.shape[0]
    frac_user_num = int(data_len/test_batch)
    # print(frac_user_num)
    user_emb = np.array(user_emb)
    item_emb = np.array(item_emb)
    
    predictions_list = [] 
    i = 0
    while(i < frac_user_num):
        tmp = target_data.iloc[(test_batch * i) : (test_batch * (i + 1))].values        
        user = tmp[:, 0]
        user = user.astype(np.int32)
        item = tmp[:, 1]
        item = item.astype(np.int32)          
        select_user_emb = user_emb[user]
        select_item_emb = item_emb[item]
        predictions_tmp = select_user_emb * select_item_emb
        predictions_tmp = list(np.sum(predictions_tmp, axis = 1))
        predictions_list += predictions_tmp
        i = i + 1
    if frac_user_num * test_batch < data_len:
        tmp = target_data.iloc[(test_batch * (i)) : ].values
        user = tmp[:, 0]
        user = user.astype(np.int32)
        item = tmp[:, 1]
        item = item.astype(np.int32)          
        select_user_emb = user_emb[user]
        select_item_emb = item_emb[item]
        predictions_tmp = select_user_emb * select_item_emb
        predictions_tmp = list(np.sum(predictions_tmp, axis = 1))
        predictions_list += predictions_tmp
    return predictions_list



# def get_rec_result_aldi(df, max_item, topk=50):
#     # 按 user 分组并对 pred 排名
#     df = df[df['item'] <= max_item]
    
#     df['rank'] = df.groupby('user')['pred'].rank(ascending=False, method='first')

#     # 按 user 和 rank 排序
#     df = df.sort_values(by=['user', 'rank'])

#     # 取出每个 user 的前 topk 个 item
#     result_df = df.groupby('user').head(topk)

#     # 提取 user 列表
#     test_u = result_df['user'].unique()

#     # 提取每个 user 的前 topk 个 item 列表
#     preds_topk = result_df.groupby('user')['item'].apply(lambda x: x.head(topk).tolist()).tolist()

#     return test_u.tolist(), np.array(preds_topk), result_df


def get_rec_result_aldi(df, max_item, topk=50):
    # 按 user 分组并对 pred 排名
    df = df[df['item'] <= max_item]
    df['rank'] = df.groupby('user')['pred'].rank(ascending=False, method='first')

    # 按 user 和 rank 排序
    df = df.sort_values(by=['user', 'rank'])

    # 取出每个 user 的前 topk 个 item
    result_df = df.groupby('user').head(topk)

    # 提取 user 列表
    test_u = result_df['user'].unique()

    # 提取每个 user 的前 topk 个 item 列表
    preds_topk = result_df.groupby('user')['item'].apply(lambda x: x.head(topk).tolist()).tolist()

    return test_u.tolist(), preds_topk, result_df  # 返回列表的列表


# def get_rec_result_aldi(df, topk=50):
#     # 分组后对于每个 user 选出 pred 值最大的 topk 个行的索引
#     # 返回的是一个以 group label 为 level0，行索引为 level1 的 Series
#     top_indices = df.groupby('user')['pred'].nlargest(topk).index.droplevel(1)
    
#     # 从 df 中根据选取的索引得到所需行
#     result_df = df.loc[top_indices].copy()

#     # 此时行数据已经按照 pred 值降序排列
#     # 因为 nlargest 会按照分组后 pred 的值进行排序
#     # 如果需要对相同 pred 的情况进一步按照 rank 或其他方式排序可再自行指定
    
#     # 按出现顺序获取 user 列表
#     test_u = result_df['user'].unique().tolist()
    
#     # 依次获取每个 user 的 item 列表
#     # 因为 result_df 是分组后 nlargest 的结果，顺序已经根据 pred 降序排列
#     preds_topk = [result_df.loc[result_df['user'] == u, 'item'].tolist() for u in test_u]
    
#     return test_u, np.array(preds_topk), result_df
