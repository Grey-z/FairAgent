import gc
import re
import sys
import json
import requests
import pickle
import scipy.io as sio
import random
import argparse
import numpy as np
import pandas as pd
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from pprint import pprint
from tqdm import tqdm
from collections import Counter
from collections import defaultdict
from logging import getLogger
from ast import Global
from pathlib import Path

sys.path.append('/root/linghui/drs/TRD-main/daisyRec/')
import daisy

from ALDI import *
from utils import *
sys.path.append('/root/linghui/drs/TRD-main/daisyRec/')
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='Random Seed.')
    parser.add_argument('--gpu_id', type=int, default=0)
    # parser.add_argument('--n_jobs', type=int, default=4, help='Multiprocessing number.')

    parser.add_argument('--dataset', type=str, default="kuairec", help='Dataset to use.')
    parser.add_argument('--algo_name', type=str, default='mf', help='mf, lightgcn')
    # warm model
    parser.add_argument('--debias_method', type=str, default='backbone', help='debias method')  #backbone
    # training
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate")
    parser.add_argument('--batch_size', type=int, default=8192, help='Normal batch size.')
    parser.add_argument('--max_epoch', type=int, default=20)
    parser.add_argument('--save', type=int, default=0, help='whether save results and model')

    # cold-start method parameter
    parser.add_argument('--model', type=str, default='ALDI')
    parser.add_argument('--reg_1', type=float, default=1e-4, )
    parser.add_argument('--alpha', type=float, default=0.9)
    parser.add_argument('--beta', type=float, default=0.05)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--tws', type=int, default=0, choices=[0, 1])
    parser.add_argument('--freq_coef_M', type=float, default=4)
    args, _ = parser.parse_known_args()
    
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

    saved_trd_path = saved_result_path + f"{config['trd_version']}/"
    ensure_dir(saved_trd_path)

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
    
    
    test_u_all = np.load(saved_data_path + 'test_u_all.npy', allow_pickle=True)
    test_ucands_all = np.load(saved_data_path + 'test_ucands_all.npy',allow_pickle=True)
    test_ur_all = np.load(saved_data_path + 'test_ur_all.npy', allow_pickle=True)
    content_data = np.load(saved_data_path + f"{config['dataset']}_item_embedding.npy", allow_pickle=True)
    train_total_set = pd.read_csv(saved_data_path + 'train_total_set.csv', index_col=0)
    train_total_samples = np.load(saved_data_path + 'train_total_samples.npy')
    ui_num = np.load(saved_data_path + 'ui_cate.npy')
    config['user_num'] = ui_num[0]
    config['item_num'] = ui_num[1]
    
    warm_item_list = np.load(saved_data_path + 'warm_item_list.npy')
    cold_item_list = np.load(saved_data_path + 'cold_item_list.npy')
    rl_user_his_pref = pd.read_csv(saved_data_path + 'rl_user_his_pref.csv', index_col = 0)
    rl_user_his_pref['user_history'] = rl_user_his_pref['user_history'].apply(ast.literal_eval)
    rl_user_his_pref = rl_user_his_pref.sort_values(by='user', ascending=True)
    user_pref_dict = rl_user_his_pref.set_index('user')['hist_tgf'].to_dict()
    config['user_pref_dict'] = user_pref_dict
    config['user_his'] = rl_user_his_pref
    
    warm_item_train = train_total_set[config['IID_NAME']].unique()
    cold_item_train = list(set(range(config['item_num'])) - set(warm_item_train))
    config['warm_item_list'] = warm_item_list
    config['cold_item_list'] = cold_item_list
    config['emb_item_nb'] = df_get_neighbors(train_total_set, 'item', config['item_num'])
    config['emb_user_nb'] = df_get_neighbors(train_total_set, 'user', config['user_num'])
    item_freq = np.ones(shape=(config['item_num'],), dtype=np.float32)
    item_to_user_neighbors = config['emb_item_nb'][config['warm_item_list']]
    for item_index, user_neighbor_list in zip(config['warm_item_list'], item_to_user_neighbors):
        item_to_item_neighborhoods = config['emb_user_nb'][user_neighbor_list]
        item_freq[item_index] = sum([1.0 / len(neighborhood) for neighborhood in item_to_item_neighborhoods])
    x_expect = (len(train_total_set) / config['item_num']) * (1 / (len(train_total_set) / config['user_num']))
    config['freq_coef_a'] = config['freq_coef_M'] / x_expect
    logger.info('Finished computing item frequencies.')
    
    #load pre-trained model
    if config['algo_name'].lower() in ['lightgcn', 'ngcf']:
        config['inter_matrix'] = get_inter_matrix(train_total_set, config)
    rec_model = model_config[config['algo_name']](config)
    if config['debias_method'] == 'backbone':
        model_pth = saved_model_path + f"backbone_total_{config['epochs']}.pth"
    elif config['debias_method'] in ['pd', 'pearson', 'posneg', 'cnif']:
        if config['debias_method'] == 'cnif':
            weight = 10.0
        if config['debias_method'] == 'pd':
            weight = 0.01
        if config['debias_method'] == 'pearson':
            weight = 0.5
        if config['debias_method'] == 'posneg':
            weight = 0.9
        model_pth = saved_model_path + f"{config['debias_method']}_{weight}_{config['epochs']}.pth"
    rec_model = torch.load(model_pth, map_location=torch.device('cuda:0'))
    logger.info(f"load model {config['algo_name']}")
    if config['algo_name'] == 'mf':
        user_embed = rec_model.embed_user.weight
        item_embed = rec_model.embed_item.weight
    elif config['algo_name'] == 'lightgcn':
        user_embed, item_embed = rec_model.forward()
        
    
    # train base aldi
    saved_name =  f"ALDI_{config['debias_method']}_{config['max_epoch']}_base.pt"
    aldi_pt = saved_model_path + saved_name
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_aldi = ALDI(config, user_embed.shape[-1], content_data.shape[-1]).to(device)
    optimizer = optim.Adam(model_aldi.parameters(), lr=config['lr'])
    logger.info("=======model initial completed========")
    file_path = Path(aldi_pt)  
    # if file_path.exists() & (config['save'] != 2):
    if file_path.exists():
        logger.info(f"load pre-trained model {aldi_pt}")
        model_aldi.load_state_dict(torch.load(aldi_pt, map_location=torch.device('cuda')))
        model_aldi.to("cuda:0")
        
    else:
        start_time = time.time()
        logger.info(f"train model {aldi_pt}")
        train_dataset = TrainDataset(train_total_samples, content_data, item_embed, user_embed, item_freq)
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        
        model_aldi = train_model(model_aldi, optimizer, train_loader,config)
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f":finished train agent: {elapsed_time:.6f} s")
        torch.save(model_aldi.state_dict(), aldi_pt)
    
    data_last_stage = train_total_set
    result_df_all = pd.DataFrame()
    for stage in range(config['test_stage']):
        logger.info(f'begin test stage {stage}')
        # load test data
        test_u = test_u_all[stage]
        test_ur = test_ur_all[stage]
        test_ucands = test_ucands_all[stage]
        test_set = pd.read_csv(saved_data_path + f'test_set_{stage}.csv')
        test_pd = pd.read_csv(saved_data_path + f'test_df_{stage}', index_col = 0)

        # update warm ，cold set and new items in this stage
        start_time = time.time()
        warm_item_list, cold_item_list = update_new_item(data_last_stage, config)
        config['warm_item_list'] = warm_item_list
        config['cold_item_list'] = cold_item_list
        new_item_list = list(set(test_set[config['IID_NAME']].unique()) - set(warm_item_list))
        config['topk_list'] = [10, 20, 50]
        data_last_stage = pd.concat([data_last_stage, test_set])
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f":finished update warm/cold_list: {elapsed_time:.6f} s")

        # evaluate rec+fairness metric
        logger.info(f'begin calculate metrics at test stage {stage}')
        start_time = time.time()
        
        gen_user_emb = model_aldi.get_user_emb(torch.tensor(user_embed, dtype=torch.float32).to(device))
        gen_item_emb = model_aldi.get_item_emb(torch.tensor(content_data, dtype=torch.float32).to(device),
                                         torch.tensor(item_embed, dtype=torch.float32).to(device),
                                         warm_item_train, cold_item_train)

        predict_pd = get_test_predict(test_pd, gen_user_emb.cpu().numpy(), gen_item_emb.cpu().numpy(), config['warm_item_list'], config['cold_item_list'])
        max_item = np.max(new_item_list)
        test_u_aldi, preds_topk, rec_results = get_rec_result_aldi(predict_pd, max_item)
        result_df = get_evaluation_metric(test_u_aldi, test_ur, preds_topk, rec_results, config)
        ## get rec results
        result_df_all = pd.concat([result_df_all, result_df])
        rec_results.to_csv(saved_metric_path + f"rec_results_aldi_{config['debias_method']}_stage{stage}.csv")
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f":finished calculate metric: {elapsed_time:.6f} s")
        
        
        # update user interaction
        start_time = time.time()
        rl_user_his_pref = update_hist_df(rec_results, rl_user_his_pref, test_ur, config)
        len_hist = config['topk']
        rl_user_his_pref['user_history'] = rl_user_his_pref['user_history'].apply(lambda x: x[-len_hist:] if len(x) >= len_hist else x)
        rl_user_his_pref = get_stage_user_tgf(rl_user_his_pref, 'user_history', config)
        config['user_his'] = rl_user_his_pref
        user_pref_dict = rl_user_his_pref.set_index('user')['hist_tgf'].to_dict()
        config['user_pref_dict'] = user_pref_dict
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
        
        if stage < 4:
            # fine-tune backbone model
            logger.info(f'begin fine-tune model at test stage {stage}')
            start_time = time.time()
            
            aldi_pth = saved_model_path + f"ALDI_{config['debias_method']}_{config['max_epoch']}_test_{stage}.pt"
            file_path = Path(aldi_pth)  
            if file_path.exists() & (config['save'] != 2):
                logger.info(f"load pre-trained model {aldi_pth}")
                model_aldi.load_state_dict(torch.load(aldi_pth, map_location=torch.device('cuda')))
            else:
                train_samples = fine_tune_data
                train_dataset = TrainDataset(train_samples, content_data, item_embed, user_embed, item_freq)
                train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
                model_aldi = train_model(model_aldi, optimizer, train_loader,config)
                if config['save'] > 0:
                    logger.info(f'begin save ALDI model to {aldi_pth}')
                    torch.save(model_aldi.state_dict(), aldi_pth)
            end_time = time.time()
            elapsed_time = end_time - start_time
            logger.info(f":finished train aldi model at stage {stage}: {elapsed_time:.6f} s")
        
        
    result_name = f"{config['dataset']}_{config['algo_name']}_{config['debias_method']}.csv"
    mean_values = result_df_all.mean()
    result_df_all.loc['mean'] = mean_values
    result_df_all.to_csv(f'./metric_results/aldi/{result_name}')
        