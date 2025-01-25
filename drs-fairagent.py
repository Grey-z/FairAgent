import os
import gc
import re
import sys
import json
import time
import random
import requests
import argparse
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import scipy.io as sio
from tqdm import tqdm
from pathlib import Path
from collections import Counter
from collections import defaultdict
from logging import getLogger
from ast import Global

from trd import *
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
                        default='fair', 
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
    parser.add_argument('--alpha',
                        type=float, 
                        default=2,
                        help='alpha')
    parser.add_argument('--beta',
                        type=float, 
                        default=0.5,
                        help='beta')
    parser.add_argument('--gama',
                        type=float, 
                        default=0.1,
                        help='gama')
    parser.add_argument('--seed', 
                        type=int, 
                        default=2024, 
                        help='seed')
    
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
    
    # load data
    ui_num = np.load(saved_data_path + 'ui_cate.npy')
    config['user_num'] = ui_num[0]
    config['item_num'] = ui_num[1]
    # logger.info(f"user number: {config['user_num']}  item number: {config['item_num']}")
    train_base_set = pd.read_csv(saved_data_path + 'train_base_set.csv', index_col=0)
    train_rl_set = pd.read_csv(saved_data_path + 'train_rl_set.csv', index_col=0)
    train_total_set = pd.read_csv(saved_data_path + 'train_total_set.csv', index_col=0)
    train_total_samples = np.load(saved_data_path + 'train_total_samples.npy')
    train_rl_u = np.load(saved_data_path + 'train_rl_u.npy', allow_pickle=True)
    train_rl_ucands = np.load(saved_data_path + 'train_rl_ucands.npy',allow_pickle=True)
    test_u_all = np.load(saved_data_path + 'test_u_all.npy', allow_pickle=True)
    test_ucands_all = np.load(saved_data_path + 'test_ucands_all.npy',allow_pickle=True)
    test_ur_all = np.load(saved_data_path + 'test_ur_all.npy', allow_pickle=True)
    rl_user_his_pref = pd.read_csv(saved_data_path + 'rl_user_his_pref.csv', index_col = 0)
    rl_user_his_pref['user_history'] = rl_user_his_pref['user_history'].apply(ast.literal_eval)
    rl_user_his_pref = rl_user_his_pref.sort_values(by='user', ascending=True)
    user_pref_dict = rl_user_his_pref.set_index('user')['hist_tgf'].to_dict()
    config['user_pref_dict'] = user_pref_dict
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
    if config['algo_name'] in ['mf', 'lightgcn']:
        content_flag = 0
    elif config['algo_name'] in ['ALDI']:
        content_flag = 1
    train_set = train_total_set
    # load pre-trained backbone model and extract embeddings
    if config['algo_name'].lower() in ['lightgcn', 'ngcf']:
        config['inter_matrix'] = get_inter_matrix(train_total_set, config)
    rec_model = model_config[config['algo_name']](config)
    model_pth = saved_model_path + f"backbone_total_{config['epochs']}.pth"
    rec_model = torch.load(model_pth, map_location=torch.device('cuda:0'))
    logger.info(f"load model {config['algo_name']}")
    if config['algo_name'] == 'mf':
        user_embed = rec_model.embed_user.weight
        item_embed = rec_model.embed_item.weight
    elif config['algo_name'] == 'lightgcn':
        user_embed, item_embed = rec_model.forward()
        # user_embed = user_embed.weight
        # item_embed = item_embed.weight
    rec_model.cuda()
    
    # train fairagent or load existing model - initial
    param_name = f"fair_alpha{config['alpha']}_beta{config['beta']}_gama{config['gama']}"
    dqn_pt = saved_trd_path + f"rl_{param_name}_step{config['train_step']}_seed{config['seed']}.pt"
    n_actions = config['n_actions']
    model_dqn = DQN(user_embed, item_embed, config, n_actions)
    logger.info("=======model initial completed========")
    file_path = Path(dqn_pt)  
    if file_path.exists() & (config['save'] != 2):
    # if file_path.exists():
        logger.info(f"load pre-trained model {dqn_pt}")
        model_dqn.load_state_dict(torch.load(dqn_pt, map_location=torch.device('cuda')))
        model_dqn.to("cuda:0")
        
    else:
        logger.info(f"train model {dqn_pt}")
        ## initialize model
        
        # history_state -> train_rl_set
        history_state = get_ur_l(train_base_set)
        # ground_truth -> train_rl_set
        ground_truth = get_ur(train_rl_set)
        # train_user -> train_rl_set
        train_users = train_rl_u
        # construct action space
        test_dataset = CandidatesDataset(train_rl_ucands)
        test_loader = get_dataloader(test_dataset, batch_size=128, shuffle=False, num_workers=0)
        preds = rec_model.rank(test_loader) # np.array (u, topk)
        preds_new = remove_duplicates(preds) 
        preds_action_space = np.array([x[:config['cand_num']] for x in preds_new])
        print(np.shape(preds_action_space))
        action_space =  get_model_pred(train_rl_u, preds_action_space)
        start_time = time.time()
        RLTrainer = RLTraining(train_users, history_state, ground_truth, action_space, config)
        model_dqn = RLTrainer.train(model_dqn)
        end_time = time.time()
        elapsed_time = end_time - start_time
        saved_name = saved_trd_path + f"rl_{param_name}_step{config['train_step']}_seed{config['seed']}.pt"
        logger.info(f":finished train agent: {elapsed_time:.6f} s")
        torch.save(model_dqn.state_dict(), saved_name)
        
    data_last_stage = train_total_set
    result_df_all = pd.DataFrame()
    for stage in range(config['test_stage']):
        logger.info(f'begin test stage {stage}')
        # load test data
        test_u = test_u_all[stage]
        test_ur = test_ur_all[stage]
        test_ucands = test_ucands_all[stage]
        test_set = pd.read_csv(saved_data_path + f'test_set_{stage}.csv')

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
        model_dqn.eval()
        ## construct test action space for each user based his/her historical preference
        user_nc = rl_user_his_pref.set_index('user')['hist_nc'].to_dict()
        test_dataset = CandidatesDataset(test_ucands)
        test_loader = get_dataloader(test_dataset, batch_size=128, shuffle=False, num_workers=0)
        preds = rec_model.rank(test_loader) # np.array (u, topk)
        preds_new = remove_duplicates(preds) 
        preds_test_stage = np.array([x[:config['cand_num']] for x in preds_new])
        pred_test_dict =  get_model_pred(test_u, preds_test_stage)
        action_space_stage = generate_user_action_space(pred_test_dict, user_nc, new_item_list, config)
        ## get ground_truth and history state
        ground_truth = test_ur
        history_records = rl_user_his_pref.set_index('user')['user_history'].to_dict()
        
        preds_topk, rec_results = evaluate_agent(model_dqn, test_ur, history_records, action_space_stage, config)
        result_df = get_evaluation_metric(test_u, test_ur, preds_topk, rec_results, config)
        ## get rec results
        result_df_all = pd.concat([result_df_all, result_df])
        rec_results.to_csv(saved_metric_path + f'fairagent_rec_results_{param_name}_stage{stage}.csv')
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f":finished calculate metric: {elapsed_time:.6f} s")
        
        
        #update user interaction
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
            model_pth = saved_model_path + f"{config['debias_method']}_backbone_{config['epochs']}_test_{stage}.pth"
            file_path = Path(model_pth)  
            if file_path.exists() & (config['save'] != 2):
                logger.info(f"load pre-trained model {model_pth}")
                if config['algo_name'].lower() in ['lightgcn', 'ngcf']:
                    config['inter_matrix'] = get_inter_matrix(train_set, config)
                rec_model = torch.load(model_pth, map_location=torch.device('cuda:0'))
            else:
                logger.info(f"fine tune model {model_pth}")
                train_samples = fine_tune_data
                rec_model = train_backbone_model(rec_model, train_samples, train_set, config, fine_tune = 1)
                if config['save'] > 0:
                    logger.info(f'begin save backbone model to {model_pth}')
                    torch.save(rec_model, model_pth)
            end_time = time.time()
            elapsed_time = end_time - start_time
            logger.info(f":finished train backbone model at stage {stage}: {elapsed_time:.6f} s")

            #fine-tune fairagent
            dqn_test_pt = saved_trd_path + f"rl_{param_name}_step{config['train_step']}_seed{config['seed']}_stage{stage}.pt.pt"
            file_path = Path(dqn_test_pt)  
            if file_path.exists() & (config['save'] != 2):
                logger.info(f"load pre-trained model {dqn_pt}")
                model_dqn = DQN(user_embed, item_embed, config, n_actions)
                logger.info("=======model initial completed========")
                
                model_dqn.load_state_dict(torch.load(dqn_test_pt, map_location=torch.device('cuda')))
                model_dqn.cuda()
            else:
                logger.info(f"fine tune fairagent {model_pth}")
                tune_history_state = history_records
                ## construct action space for fine tuning
                pred_test_dict =  get_model_pred(test_u, preds_topk)
                user_nc = rl_user_his_pref.set_index('user')['hist_nc'].to_dict()
                action_space_stage = generate_user_action_space(pred_test_dict, user_nc, new_item_list, config)

                config['train_step'] = config['topk'] - 20
                tune_ground_truth = test_ur
                tune_users = test_u

                start_time = time.time()
                RLTrainer = RLTraining(tune_users, tune_history_state, tune_ground_truth, action_space_stage, config)
                model_dqn.train()
                model_dqn = RLTrainer.train(model_dqn)
                end_time = time.time()
                elapsed_time = end_time - start_time
                logger.info(f":finished train agent: {elapsed_time:.6f} s")
                if config['save'] > 0:
                    logger.info(f'begin save agent model to {dqn_test_pt}')
                    torch.save(model_dqn.state_dict(), dqn_test_pt)
        
        
    result_name = f"{config['dataset']}_{config['algo_name']}_{config['debias_method']}_{param_name}.csv"
    mean_values = result_df_all.mean()
    result_df_all.loc['mean'] = mean_values
    result_df_all.to_csv(f'./metric_results/fairagent/{result_name}')