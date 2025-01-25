import gc
import re
import os
import sys
import json
import time
import random
import optuna
import requests
import numpy as np
import pandas as pd
import scipy.io as sio
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from collections import defaultdict
from logging import getLogger
from ast import Global
from pathlib import Path
import daisy
from utils import *

from daisy.utils.config import init_seed, init_config, init_logger
from daisy.utils.metrics import MAP, NDCG, Recall, Precision, HR, MRR
from daisy.utils.sampler import BasicNegtiveSampler, SkipGramNegativeSampler, UniqueNegativeSampler
from daisy.utils.dataset import AEDataset, BasicDataset, CandidatesDataset, get_dataloader
from daisy.utils.utils import get_history_matrix, get_ur, build_candidates_set, ensure_dir, get_inter_matrix

from daisy.model.MFRecommender import MF
from daisy.model.LightGCNRecommender import LightGCN
from daisy.utils.metrics import calc_ranking_results



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training for Backbone and Baseline')
    # common settings
    parser.add_argument('--dataset', 
                        type=str, 
                        default='kuairec', 
                        help='select dataset')
    parser.add_argument('--algo_name', 
                        type=str, 
                        default='mf', 
                        help='mf, lightgcn')
    parser.add_argument('--debias_method', 
                        type=str, 
                        default='backbone_total', 
                        help='backbone, posneg, pd, cnif')
    parser.add_argument('--save', 
                        type=int, 
                        default=0, 
                        help='whether save results and model')
    parser.add_argument('--weight',
                        type=float, 
                        default=1.0,
                        help='weight of acc_loss')
    parser.add_argument('--batch_size',
                        type=int, 
                        default=8192,
                        help='batch size')
    parser.add_argument('--gpu', 
                        type=str, 
                        default='0', 
                        help='gpu card ID')
    args = parser.parse_args()
    
    # init config
    config = init_config(args)
    init_seed(config['seed'], config['reproducibility'])
    init_logger(config)
    logger = getLogger()
    logger.info(config)
    config['logger'] = logger
    model_config, metrics_config, tune_params_config, param_type_config = get_config()
    config['model_config'] = model_config
    
    # set path
    save_path = config['save_path'] + config['version']
    ensure_dir(save_path)

    file_path = save_path + f'{config["dataset"]}/'
    ensure_dir(file_path)

    saved_data_path = file_path + 'data/'
    ensure_dir(saved_data_path)

    saved_result_path = file_path + f'{config["algo_name"]}/'
    ensure_dir(saved_result_path)

    saved_model_path = saved_result_path + 'model/'
    ensure_dir(saved_model_path)

    saved_rec_path = saved_result_path + 'rec_list/'
    ensure_dir(saved_rec_path)
    
    saved_metric_path = saved_result_path + 'metric/'
    ensure_dir(saved_metric_path)
    logger.info('-' * 50)
    logger.info(f"choose dataset: {config['dataset']}")
    logger.info(f"choose algo: {config['algo_name']}")
    logger.info(f"choose debias method: {config['debias_method']}")
    logger.info(f"choose weight: {config['weight']}")
    
    # load data
    ui_num = np.load(saved_data_path + 'ui_cate.npy')
    config['user_num'] = ui_num[0]
    config['item_num'] = ui_num[1]
    # logger.info(f"user number: {config['user_num']}  item number: {config['item_num']}")
    train_total_set = pd.read_csv(saved_data_path + 'train_total_set.csv', index_col=0)
    train_total_samples = np.load(saved_data_path + 'train_total_samples.npy')
    test_u_all = np.load(saved_data_path + 'test_u_all.npy', allow_pickle=True)
    test_ucands_all = np.load(saved_data_path + 'test_ucands_all.npy',allow_pickle=True)
    test_ur_all = np.load(saved_data_path + 'test_ur_all.npy', allow_pickle=True)
    rl_user_his_pref = pd.read_csv(saved_data_path + 'rl_user_his_pref.csv', index_col = 0)
    config['user_his'] = rl_user_his_pref
    with open(saved_data_path + 'train_total_ur.json', 'r') as file:
        train_total_ur = json.load(file)
    config['train_ur'] = train_total_ur
    
    item_pop_train = pd.read_csv(saved_data_path + 'item_pop_train.csv',index_col=0)
    item_pop_train_dict = dict(list(zip(item_pop_train.item, item_pop_train.train_counts)))
    warm_item_list = np.load(saved_data_path + 'warm_item_list.npy')
    cold_item_list = np.load(saved_data_path + 'cold_item_list.npy')
    config['warm_item_list'] = warm_item_list
    config['cold_item_list'] = cold_item_list
    config['item_pop_train_dict'] = item_pop_train_dict
    
    # get backbone model
    if config['debias_method'] == 'backbone':
        config['weight'] = 'total'
    train_set = train_total_set
    model_pth = saved_model_path + f"{config['debias_method']}_{config['weight']}_{config['epochs']}.pth"
    file_path = Path(model_pth)  
    if file_path.exists() & (config['save'] != 2):
        logger.info(f"load pre-trained model {model_pth}")
        if config['algo_name'].lower() in ['lightgcn', 'ngcf']:
            config['inter_matrix'] = get_inter_matrix(train_set, config)
        rec_model = torch.load(model_pth)
    else:
        logger.info(f"train model {model_pth}")
        ## initialize model
        if config['algo_name'].lower() in ['lightgcn', 'ngcf']:
            config['inter_matrix'] = get_inter_matrix(train_set, config)
        rec_model = model_config[config['algo_name']](config)
        
        #train backbone/baseline model
        if config['debias_method'] == 'backbone':
            rec_model = train_backbone_model(rec_model, train_total_samples, train_set, config)
        elif config['debias_method'] in ['bpr', 'pd', 'pearson','posneg','cnif']:
            train_dataset = BasicDataset(train_total_samples)
            train_loader = get_dataloader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
            rec_model = train_baseline_model(rec_model, train_loader, train_set, config, device='cuda')
        else:
            raise ValueError(f'Invalid debias method...')
        if config['save'] > 0:
            torch.save(rec_model, model_pth)
    
    data_last_stage = train_total_set
    result_df_all = pd.DataFrame()
    for stage in range(config['test_stage']):
        logger.info(f'begin test stage {stage}')
        # load test data
        test_u = test_u_all[stage]
        test_ur = test_ur_all[stage]
        test_ucands = test_ucands_all[stage]
        test_set = pd.read_csv(saved_data_path + f'test_set_{stage}.csv')

        # update warm and cold items
        start_time = time.time()
        warm_item_list, cold_item_list = update_new_item(data_last_stage, config)
        config['warm_item_list'] = warm_item_list
        config['cold_item_list'] = cold_item_list
        config['topk_list'] = [10, 20, 50]
        data_last_stage = pd.concat([data_last_stage, test_set])
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f":finished update warm/cold_list: {elapsed_time:.6f} s")

        # evaluate rec+fairness metric
        logger.info(f'begin calculate metrics at test stage {stage}')
        start_time = time.time()
        preds_topk, rec_results = evaluate_backbone(rec_model, test_u, test_ur, test_ucands, config)
        result_df = get_evaluation_metric(test_u, test_ur, preds_topk, rec_results, config)
        # logger.info(result_df)
        rec_results.to_csv(saved_metric_path + f"{config['debias_method']}_{stage}.csv")
        result_df_all = pd.concat([result_df_all, result_df])
        #save 
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f":finished calculate metric: {elapsed_time:.6f} s")

        start_time = time.time()
        rl_user_his_pref = update_hist_df(rec_results, rl_user_his_pref, test_ur, config)
        # rl_user_his_pref = get_stage_user_tgf(rl_user_his_pref, 'user_history', config)
        config['user_his'] = rl_user_his_pref
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f":finished update user_hist: {elapsed_time:.6f} s")

        # get fine tune data
        logger.info(f'begin get fine-tune data test stage {stage}')
        start_time = time.time()
        temp_rec_result = add_user_inter(rec_results, test_ur)
        temp_rec_result.to_csv(saved_rec_path + f'rec_test_stage{stage}.csv') 
        user_rec_records = generate_user_records(temp_rec_result)
        fine_tune_data = negative_sampling(user_rec_records)
        if config['save'] > 0:
            np.save(saved_data_path + f'fine_tune_data_test_{stage}', fine_tune_data)
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f":finished get fine tune data: {elapsed_time:.6f} s")
        
        logger.info(f'begin fine-tune model at test stage {stage}')
        start_time = time.time()
        model_pth = saved_model_path + f"{config['debias_method']}_{config['weight']}_{config['epochs']}_test_{stage}.pth"
        file_path = Path(model_pth)  
        if file_path.exists() & (config['save'] != 2):
            logger.info(f"load pre-trained model {model_pth}")
            if config['algo_name'].lower() in ['lightgcn', 'ngcf']:
                config['inter_matrix'] = get_inter_matrix(train_set, config)
            rec_model = torch.load(model_pth)
        else:
            logger.info(f"fine tune model {model_pth}")
            train_samples = fine_tune_data
            
            ## train backbone / baseline:
            if config['debias_method'] == 'backbone':
                rec_model = train_backbone_model(rec_model, train_samples, train_set, config, fine_tune = 1)
            elif config['debias_method'] in ['bpr','pd', 'pearson','posneg','cnif']:
                train_dataset = BasicDataset(train_samples)
                train_loader = get_dataloader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)
                rec_model = train_baseline_model(rec_model, train_loader, train_set, config, device='cuda')
            else:
                raise ValueError(f'Invalid debias method...')
            
            if config['save'] > 0:
                logger.info(f'begin save model to {model_pth}')
                torch.save(rec_model, model_pth)
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f":finished train model stage {stage}: {elapsed_time:.6f} s")
    result_name = f"{config['dataset']}_{config['algo_name']}_{config['debias_method']}.csv"
    result_df_all.to_csv(f'./metric_results/{result_name}')
