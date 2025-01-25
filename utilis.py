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
import numpy as np
import pandas as pd
import scipy.io as sio
from tqdm import tqdm
from collections import defaultdict, Counter
from logging import getLogger
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
import scipy.sparse as sp
from torch.cuda.amp import GradScaler, autocast
from utils import *
import daisy
from daisy.utils.config import init_seed, init_config, init_logger
from daisy.utils.metrics import MAP, NDCG, Recall, Precision, HR, MRR
from daisy.utils.dataset import BasicDataset, CandidatesDataset, get_dataloader
from daisy.utils.utils import get_history_matrix, get_ur, build_candidates_set, ensure_dir, get_inter_matrix
from daisy.model.MFRecommender import MF
from daisy.model.LightGCNRecommender import LightGCN
from daisy.utils.metrics import calc_ranking_results
from sklearn.metrics import mean_squared_error


# ====================================================
# Data Loading and Processing
# ====================================================

def get_ur_l(df):
    """
    Build a user-item interaction dictionary from a DataFrame.

    Args:
        df: DataFrame containing 'user' and 'item' columns.

    Returns:
        ur: Dictionary where keys are user IDs and values are lists of interacted items.
    """
    df['user'] = df['user'].astype(int)
    df['item'] = df['item'].astype(int)
    ur = df.groupby('user')['item'].apply(list).to_dict()
    return ur

# ====================================================
# Fairness Evaluation
# ====================================================

def cal_single_item_exp(rank_list, user_num):
    """
    Calculate the exposure value for a single item based on its rank.

    Args:
        rank_list (list): List of ranks for the item.
        user_num (int): Total number of users.

    Returns:
        float: Exposure value for the item.
    """
    sum_exp = 0
    for x in rank_list:
        exp_i = 1 / math.log(x + 1, 2)  # Exposure decreases logarithmically with rank
        sum_exp += exp_i
    return sum_exp / user_num  # Normalize by the number of users


def cal_all_item_exp(rec_results, user_num, item_num):
    """
    Calculate the exposure values for all items.

    Args:
        rec_results (pd.DataFrame): DataFrame containing recommendation results with columns 'item' and 'rank'.
        user_num (int): Total number of users.
        item_num (int): Total number of items.

    Returns:
        exp_list (list): List of exposure values for all items.
    """
    exp_list = []
    for item_id in range(item_num):
        item_i_df = rec_results.loc[rec_results['item'] == item_id]
        if len(item_i_df) == 0:
            exp_i = 0  # If the item is not recommended, exposure is 0
        else:
            rank_list = item_i_df['rank'].to_list()  # Get ranks for the item
            exp_i = cal_single_item_exp(rank_list, user_num)  # Calculate exposure
        exp_list.append(exp_i)
    return exp_list


def cal_tgf(exp_list, warm_item_list, cold_item_list):
    """
    Calculate the Top-K Group Fairness (TGF) metric.

    Args:
        exp_list (list): List of exposure values for all items.
        warm_item_list (list): List of warm item indices.
        cold_item_list (list): List of cold item indices.

    Returns:
        cate_tgf (float): TGF value, representing the fairness gap between warm and cold items.
    """
    # If exp_list is empty, return 0
    if len(exp_list) == 0:
        return 0

    # Normalize exposure values
    total_exp = np.sum(exp_list)
    if total_exp == 0:
        return 0
    exp_list = exp_list / total_exp

    # If warm or cold item lists are empty, return 0
    if len(warm_item_list) == 0 or len(cold_item_list) == 0:
        return 0

    # Convert lists to numpy arrays for efficient indexing
    warm_item_list = np.array(warm_item_list, dtype=int)
    cold_item_list = np.array(cold_item_list, dtype=int)

    # Check if indices are within bounds
    if np.any(warm_item_list >= len(exp_list)) or np.any(cold_item_list >= len(exp_list)):
        raise IndexError("Indices out of bounds for exp_list.")

    # Extract exposure values for warm and cold items
    warm_exp_list = exp_list[warm_item_list]
    cold_exp_list = exp_list[cold_item_list]

    # Compute weights for warm items (decreasing from warm_num to 1)
    warm_num = len(warm_exp_list)
    warm_weight = np.arange(warm_num, 0, -1)

    # Compute weights for cold items (increasing from 1 to warm_num-like values)
    cold_num = len(cold_exp_list)
    if cold_num > 1:
        cold_weight = 1 + np.arange(cold_num) * (warm_num - 1) / (cold_num - 1)
    else:
        cold_weight = np.array([1])

    # Compute weighted exposure for warm and cold items
    warm_part = np.sum(warm_exp_list * warm_weight) / warm_num
    cold_part = np.sum(cold_exp_list * cold_weight) / cold_num

    # Compute TGF as the difference between warm and cold exposure
    cate_tgf = warm_part - cold_part
    if cate_tgf < 0:
        cate_tgf = cate_tgf / (warm_num / cold_num)  # Normalize if TGF is negative
    return cate_tgf

def calculate_nc(input_list, target_list):
    """
    Calculate the proportion of values in the input list that appear in the target list.

    Args:
        input_list (list): Input list of values.
        target_list (list): Target list of values.

    Returns:
        float: Proportion of values in the input list that appear in the target list.
    """
    if not input_list:
        return 0.0  # Return 0 if the input list is empty
    # Calculate the intersection of the input list and target list
    count = len(set(input_list) & set(target_list))
    return count / len(input_list)  # Return the proportion


# ====================================================
# Model Training and Evaluation
# ====================================================

def train_backbone_model(model, train_samples, train_set, config, fine_tune=1):
    """
    Train a backbone model.

    Args:
        model: Model object.
        train_samples: Training samples.
        train_set: Training set.
        config: Configuration dictionary.
        fine_tune: Whether to fine-tune the model, default is 1.

    Returns:
        model: Trained model.
    """
    model_config = config['model_config'] 
    logger = config['logger']
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

    elapsed_time = time.time() - s_time
    logger.info(f"Finish training: {config['dataset']} {config['prepro']} {config['algo_name']} with {config['loss_type']} and {config['sample_method']} sampling, {elapsed_time:.4f}")
    return model


# ====================================================
# Recommendation Results Processing
# ====================================================

def create_recommendation_df(user_ids, recommendations):
    """
    Create a DataFrame containing user IDs, item IDs, and their ranks.

    Args:
        user_ids (list): List of user IDs.
        recommendations (list of lists): List of recommendations for each user.

    Returns:
        pd.DataFrame: DataFrame with columns 'user', 'item', and 'rank'.
    """
    data = []
    for user, user_list in zip(user_ids, recommendations):
        for rank, item in enumerate(user_list, start=1):
            data.append([user, item, rank])
    df = pd.DataFrame(data, columns=['user', 'item', 'rank'])
    return df

def get_model_pred(user_set, pred):

def remove_duplicates(preds):
    """
    Remove duplicates from recommendation results.

    Args:
        preds: List of recommendations.

    Returns:
        preds_new: List of recommendations with duplicates removed.
    """
    preds = preds.astype(int)
    preds_new = []
    for lst in preds:
        seen = set()
        result = [x for x in lst if not (x in seen or seen.add(x))]
        preds_new.append(result)
    return preds_new


def transform_to_single_row(df):
    """
    Transform a multi-row DataFrame into a single-row DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame with columns like 'KPI@K', 10, 20, 50.

    Returns:
        result_df (pd.DataFrame): Single-row DataFrame with combined metrics.
    """
    result = {}  # Initialize an empty dictionary to store results
    
    # Iterate over each row in the DataFrame
    for _, row in df.iterrows():
        metric = row["KPI@K"]  # Get the metric name (e.g., NDCG, Hit Ratio)
        for k in [10, 20, 50]:  # Iterate over K values
            col_name = f"{metric}@{k}"  # Construct column name (e.g., NDCG@10)
            result[col_name] = row[k]  # Store the value in the dictionary
    
    # Convert the dictionary to a single-row DataFrame
    result_df = pd.DataFrame([result])
    return result_df

# ====================================================
# Model  Evaluation
# ====================================================

def evaluate_backbone(model, test_u, test_ur, test_ucands, config): 
    """
    Evaluate a backbone model.

    Args:
        model: Model object.
        test_u: List of test users.
        test_ur: Dictionary of user-item interactions for test users.
        test_ucands: Candidate items for test users.
        config: Configuration dictionary.

    Returns:
        preds_topk: Top-k predictions.
        rec_results: DataFrame containing recommendation results.
    """
    test_dataset = CandidatesDataset(test_ucands)
    test_loader = get_dataloader(test_dataset, batch_size=64, shuffle=False, num_workers=0)
    preds = model.rank(test_loader) # np.array (u, topk)
    preds_new = remove_duplicates(preds) 
    preds_topk = np.array([x[:config['topk']] for x in preds_new])
    
    rec_results = create_recommendation_df(test_u, preds_topk)
    return preds_topk, rec_results


    """
    Map user IDs to their corresponding predictions.

    Args:
        user_set (list): List of user IDs.
        pred (list): List of predictions corresponding to each user.

    Returns:
        model_pred (dict): Dictionary mapping user IDs to their predictions.
    """
    model_pred = {}
    for i in range(len(user_set)):
        u = user_set[i]
        model_pred[u] = pred[i]
    return model_pred


def get_rec_metric(test_u, test_ur, preds_topk, config):
    """
    Calculate recommendation metrics (e.g., NDCG, Recall) for the test set.

    Args:
        test_u (list): List of test user IDs.
        test_ur (dict): Dictionary of user-item interactions for test users.
        preds_topk (list): Top-K predictions for each user.
        config (dict): Configuration dictionary containing 'topk_list'.

    Returns:
        rec_metric (pd.DataFrame): DataFrame containing recommendation metrics.
    """
    topk_list = config['topk_list']  # Get the list of K values (e.g., [10, 20, 50])
    rec_metric = calc_ranking_results(test_ur, preds_topk, test_u, config)  # Calculate metrics
    return rec_metric.loc[2:3, ['KPI@K'] + topk_list]  # Filter relevant rows and columns


def get_evaluation_metric(test_u, test_ur, preds_topk, rec_results, config):
    """
    Combine recommendation metrics and fairness metrics into a single evaluation result.

    Args:
        test_u (list): List of test user IDs.
        test_ur (dict): Dictionary of user-item interactions for test users.
        preds_topk (list): Top-K predictions for each user.
        rec_results (pd.DataFrame): DataFrame containing recommendation results.
        config (dict): Configuration dictionary.

    Returns:
        result_df (pd.DataFrame): Single-row DataFrame with combined evaluation metrics.
    """
    # Get recommendation metrics
    rec_metric = get_rec_metric(test_u, test_ur, preds_topk, config)
    
    # Get fairness metrics
    fairness_metric = get_fairness_metric(rec_results, test_ur, config)
    
    # Combine recommendation and fairness metrics
    metric_result = pd.concat([rec_metric, fairness_metric])
    
    # Transform the combined metrics into a single-row DataFrame
    result_df = transform_to_single_row(metric_result)
    return result_df

def get_fairness_metric(rec_results, test_ur, config):
    """
    Calculate fairness metrics (TGF, NC) for the recommendation results.

    Args:
        rec_results (pd.DataFrame): DataFrame containing recommendation results with columns 'item' and 'rank'.
        test_ur (dict): Dictionary of user-item interactions for test users.
        config (dict): Configuration dictionary containing 'user_his', 'topk_list', 'warm_item_list', 'cold_item_list',
                       'user_num', and 'item_num'.

    Returns:
        fairness_metric (pd.DataFrame): DataFrame containing fairness metrics (TGF and NC) for each K value.
    """
    user_his = config['user_his']
    topk_list = config['topk_list']
    warm_item_list = config['warm_item_list']
    cold_item_list = config['cold_item_list']
    tgf_results = ['TGF']
    unf_results = ['UNF']
    nc_results = ['NC']

    for k in topk_list:
        # Filter recommendations within the top-K
        temp_rec_results = rec_results.loc[rec_results['rank'] <= k]
        
        # Calculate exposure values for all items
        exp_list = cal_all_item_exp(temp_rec_results, config['user_num'], config['item_num'])
        
        # Calculate TGF (Top-K Group Fairness)
        tgf_value = cal_tgf(exp_list, warm_item_list, cold_item_list)
        tgf_results.append(tgf_value)
        
        # Calculate NC (Novelty Coverage)
        cold_item = temp_rec_results.loc[temp_rec_results['item'].isin(cold_item_list)]
        nc_value = len(cold_item) / len(temp_rec_results)
        nc_results.append(nc_value)
        
        # (Optional) Calculate UNF (User-Normalized Fairness)
        # temp_rec_results = add_user_inter(temp_rec_results, test_ur)
        # user_rec_records = generate_user_records(temp_rec_results)
        # user_rec_records = get_stage_user_tgf(user_rec_records, 'rec_list', config)
        # mse_distance = calculate_mse_divergence(user_his, user_rec_records, user_col='user', tgf_col='hist_tgf')
        # unf_results.append(mse_distance)
        
    # Combine results into a DataFrame
    data = [tgf_results, nc_results]  # Add unf_results if UNF is calculated
    columns = ['KPI@K'] + topk_list
    fairness_metric = pd.DataFrame(data, columns=columns)
    return fairness_metric


# ====================================================
# DRS Setting
# ====================================================

def update_new_item(last_sate_set, config):
    """
    Update the list of warm and cold items based on the last state.

    Args:
        last_sate_set (pd.DataFrame): DataFrame containing the last state of items.
        config (dict): Configuration dictionary containing 'IID_NAME' and 'item_num'.

    Returns:
        warm_item_list (list): List of warm item IDs.
        cold_item_list (list): List of cold item IDs.
    """
    # Get warm items from the last state
    warm_item_list = last_sate_set[config['IID_NAME']].unique()
    
    # Get cold items as the complement of warm items
    cold_item_list = list(set(range(config['item_num'])) - set(warm_item_list))
    
    return warm_item_list, cold_item_list

def add_user_inter(df, user_item_dict):
    """
    Add a 'user_inter' column to the input DataFrame based on sampling rules.

    Args:
        df (pd.DataFrame): DataFrame containing 'user', 'item', and 'rank' columns.
        user_item_dict (dict): Dictionary of user-item interactions, where keys are user IDs
                              and values are sets of item IDs.

    Returns:
        pd.DataFrame: DataFrame with an additional 'user_inter' column.
    """
    def sample_user_inter(row):
        """
        Sample whether a user interacts with an item based on its rank.

        Args:
            row (pd.Series): A row of the DataFrame.

        Returns:
            int: 1 if the user interacts with the item, 0 otherwise.
        """
        user = row['user']
        item = row['item']
        rank = row['rank']

        # Check if the item is in the user's interaction set
        if user in user_item_dict and item in user_item_dict[user]:
            # Calculate sampling probability (inversely proportional to log2(rank + 1))
            prob = 1 / (np.log2(rank + 1))
            # Sample based on the probability
            return np.random.binomial(1, prob)
        else:
            # If the item is not in the user's interaction set, return 0
            return 0

    # Apply the sampling function to each row
    df['user_inter'] = df.apply(sample_user_inter, axis=1)
    return df

def negative_sampling(df):
    """
    Perform negative sampling for a given DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing columns 'user', 'rec_list', and 'positive_inter'.
                          - 'user': User ID.
                          - 'rec_list': Set of recommended items for the user.
                          - 'positive_inter': Set of items the user has interacted with.

    Returns:
        result (list): List of negative samples, where each sample is of the form [user, positive_item, negative_item].
    """
    result = []  # Initialize an empty list to store the results
    
    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        user = row['user']
        rec_list = row['rec_list']
        positive_inter = row['positive_inter']
        
        # Skip the row if there are no positive interactions
        if not positive_inter:
            continue
        
        # Compute the set of candidate negative items
        negative_candidates = list(rec_list - positive_inter)  # Convert to a list for sampling
        
        # Perform negative sampling for each positive item
        for pos in positive_inter:
            # Randomly sample 4 negative items (or fewer if not enough candidates)
            neg_samples = random.sample(negative_candidates, min(4, len(negative_candidates)))
            # Add [user, positive_item, negative_item] tuples to the result
            result.extend([[user, pos, neg] for neg in neg_samples])
    
    return result

def update_hist_df(rec_results, rl_user_his_pref, test_ur, config):
    """
    Update the user history preference DataFrame with new recommendation results.

    Args:
        rec_results (pd.DataFrame): DataFrame containing recommendation results.
        rl_user_his_pref (pd.DataFrame): DataFrame containing user history preferences.
        test_ur (dict): Dictionary of user-item interactions for test users.
        config (dict): Configuration dictionary.

    Returns:
        pd.DataFrame: Updated user history preference DataFrame.
    """
    # Copy the recommendation results and add user interaction data
    temp_rec_result = rec_results.copy(deep=True)
    temp_rec_result = add_user_inter(temp_rec_result, test_ur)
    
    # Generate user recommendation records
    user_rec_records = generate_user_records(temp_rec_result)
    
    # Update user history preferences
    rl_user_his_pref = update_user_history(rl_user_his_pref, user_rec_records)
   
    return rl_user_his_pref


def update_user_history(df1, df2):
    """
    Update the user history DataFrame with new interaction records.

    Args:
        df1 (pd.DataFrame): DataFrame containing user history records.
        df2 (pd.DataFrame): DataFrame containing new user interaction records.

    Returns:
        pd.DataFrame: Updated user history DataFrame.
    """
    # Ensure 'user_history' is of list type
    df1['user_history'] = df1['user_history'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    
    # Iterate over df2 to update user_history
    for user, positive_inter in df2[['user', 'positive_inter']].itertuples(index=False):
        if positive_inter:  # If positive_inter is not empty
            # Update user_history for the corresponding user
            df1.loc[df1['user'] == user, 'user_history'] = df1.loc[df1['user'] == user, 'user_history'].apply(
                lambda x: x + [item for item in positive_inter if item not in x]
            )
    
    return df1


def generate_user_records(df):
    """
    Generate a new DataFrame containing user, recommended items, and positive interactions.

    Args:
        df (pd.DataFrame): DataFrame containing 'user', 'item', and 'user_inter' columns.

    Returns:
        pd.DataFrame: New DataFrame with columns 'user', 'rec_list', and 'positive_inter'.
    """
    # Group by user and aggregate recommended items and positive interactions
    grouped = df.groupby('user').agg(
        rec_list=('item', set),  # Collect all recommended items
        positive_inter=('item', lambda x: set(x[df.loc[x.index, 'user_inter'] == 1]))  # Collect items with user_inter=1
    ).reset_index()

    return grouped


def generate_user_action_space(user_items_dict, user_prob_dict, new_item_list, config):
    """
    Generate a user action space by selecting items from user_items_dict and new_item_list based on probabilities.

    Args:
        user_items_dict (dict): Dictionary where keys are user IDs and values are lists of items.
        user_prob_dict (dict): Dictionary where keys are user IDs and values are probabilities.
        new_item_list (list): List of new items to consider.
        config (dict): Configuration dictionary containing 'seed' and 'topk' parameters.

    Returns:
        dict: A dictionary where keys are user IDs and values are lists of selected items.
    """
    seed = config.get('seed', None)  # Get the random seed from the config
    len_action_space = config.get('topk', 10) + 50  # Default topk is 10, add 50 for buffer

    # Set the random seed for reproducibility
    if seed is not None:
        np.random.seed(seed)

    # Convert new_item_list to a set for efficient membership checking
    new_item_set = set(new_item_list)

    # Initialize the new dictionary to store user action spaces
    user_action_space = {}

    # Iterate over each user and their items
    for user, items in user_items_dict.items():
        # Get the user's probability value (default to 0 if not found)
        prob = user_prob_dict.get(user, 0)

        # Split items into old items (not in new_item_list) and new items (in new_item_list)
        pred_old_item_list = [item for item in items if item not in new_item_set]
        pred_new_item_list = [item for item in items if item in new_item_set]

        # Calculate the number of old items to select (N_old)
        N_old = int(len_action_space * (1 - prob))
        N_old = min(N_old, len(pred_old_item_list))  # Ensure N_old does not exceed the available old items

        # Calculate the number of new items to select (N_new)
        N_new = len_action_space - N_old

        # Select the first N_old old items
        selected_old_items = pred_old_item_list[:N_old]

        # Select the first N_new new items
        if len(pred_new_item_list) >= N_new:
            selected_new_items = pred_new_item_list[:N_new]
        else:
            # If there are not enough new items, randomly sample from the remaining new items
            remaining_new_items = list(set(new_item_list) - set(pred_new_item_list))
            selected_new_items = pred_new_item_list + np.random.choice(
                remaining_new_items, size=N_new - len(pred_new_item_list), replace=False
            ).tolist()

        # Randomly insert selected_new_items into selected_old_items while maintaining order
        updated_items = selected_old_items.copy()
        insert_indices = np.random.choice(len(updated_items) + 1, size=N_new, replace=True)
        insert_indices.sort()  # Ensure insertion indices are in ascending order
        for idx, item in zip(insert_indices, selected_new_items):
            updated_items.insert(idx, item)

        # Add the updated items to the user_action_space dictionary
        user_action_space[user] = updated_items

    return user_action_space

# ====================================================
# UNF Calculation
# ====================================================

def set_values_by_id(interaction_history, item_num):
    """
    Convert a user's interaction history into a fixed-length binary list.

    Args:
        interaction_history (list): List of item IDs that the user has interacted with.
        item_num (int): Total number of items.

    Returns:
        list: A binary list where 1 indicates interaction and 0 indicates no interaction.
    """
    # Convert interaction_history to a set for efficient membership checking
    interaction_set = set(interaction_history)
    return [1 if i in interaction_set else 0 for i in range(item_num)]


def get_weight(warm_item_list, cold_item_list):
    """
    Compute weights for warm and cold items.

    Args:
        warm_item_list (list): List of warm item IDs.
        cold_item_list (list): List of cold item IDs.

    Returns:
        warm_weight (np.array): Weights for warm items (decreasing from warm_num to 1).
        cold_weight (np.array): Weights for cold items (increasing from 1 to warm_num-like values).
    """
    warm_num = len(warm_item_list)
    warm_weight = np.arange(warm_num, 0, -1)  # Weights decrease from warm_num to 1

    cold_num = len(cold_item_list)
    if cold_num > 1:
        # Weights increase from 1 to warm_num-like values
        cold_weight = 1 + np.arange(cold_num) * (warm_num - 1) / (cold_num - 1)
    else:
        cold_weight = np.array([1])  # If only one cold item, weight is 1

    return warm_weight, cold_weight

def calculate_mse_divergence(df1, df2, user_col='user', tgf_col='user_tgf'):
    """
    Calculate the Mean Squared Error (MSE) divergence between two DataFrames based on a specific column.

    Args:
        df1 (pd.DataFrame): First DataFrame.
        df2 (pd.DataFrame): Second DataFrame.
        user_col (str): Name of the user column, default is 'user'.
        tgf_col (str): Name of the TGF (Top-K Group Fairness) column, default is 'user_tgf'.

    Returns:
        float: MSE divergence value.
    """
    # Sort both DataFrames by the user column
    df1 = df1.sort_values(by=user_col).reset_index(drop=True)
    df2 = df2.sort_values(by=user_col).reset_index(drop=True)
    
    # Find common users between the two DataFrames
    common_users = set(df1[user_col]).intersection(set(df2[user_col]))
    df1_common = df1[df1[user_col].isin(common_users)]
    df2_common = df2[df2[user_col].isin(common_users)]
    
    # Ensure both DataFrames are sorted by the user column
    df1_common = df1_common.sort_values(by=user_col).reset_index(drop=True)
    df2_common = df2_common.sort_values(by=user_col).reset_index(drop=True)
    
    # Extract the TGF column values
    p = df1_common[tgf_col].values
    q = df2_common[tgf_col].values
    
    # Calculate the Mean Squared Error (MSE)
    mse_distance = mean_squared_error(p, q)
    return mse_distance


def get_user_hist_tgf(exp_list, warm_item_list, cold_item_list, warm_weight, cold_weight):
    """
    Calculate the Top-K Group Fairness (TGF) for a user based on their exposure distribution.

    Args:
        exp_list (list): List of exposure values for all items.
        warm_item_list (list): List of warm item indices.
        cold_item_list (list): List of cold item indices.
        warm_weight (np.array): Weights for warm items.
        cold_weight (np.array): Weights for cold items.

    Returns:
        float: TGF value, representing the fairness gap between warm and cold items for the user.
    """
    if np.sum(exp_list) == 0:
        return 0  # Return 0 if there is no exposure

    # Normalize exposure values
    exp_list = exp_list / np.sum(exp_list)

    # Extract exposure values for warm and cold items
    warm_exp_list = exp_list[warm_item_list]
    cold_exp_list = exp_list[cold_item_list]

    # Compute weighted exposure for warm and cold items
    warm_part = np.sum(warm_exp_list * warm_weight) / len(warm_item_list)
    cold_part = np.sum(cold_exp_list * cold_weight) / len(cold_item_list)

    # Compute TGF as the difference between warm and cold exposure
    user_tgf = warm_part - cold_part
    if user_tgf < 0:
        # Normalize TGF if it is negative
        user_tgf = user_tgf / (len(warm_item_list) / len(cold_item_list))
    return user_tgf


def get_stage_user_tgf(df, col, config, nc_weight=0.9):
    """
    Calculate the Top-K Group Fairness (TGF) and Novelty Coverage (NC) for users at a specific stage.

    Args:
        df (pd.DataFrame): DataFrame containing user data.
        col (str): Column name in the DataFrame representing the user's interaction history.
        config (dict): Configuration dictionary containing 'warm_item_list', 'cold_item_list', and 'item_num'.
        nc_weight (float): Weight for updating the Novelty Coverage (NC) value. Default is 0.9.

    Returns:
        pd.DataFrame: Updated DataFrame with columns 'exp_list', 'hist_tgf', and 'hist_nc'.
    """
    # Convert warm_item_list and cold_item_list to NumPy arrays
    warm_item_list = np.array(config['warm_item_list'])
    cold_item_list = np.array(config['cold_item_list'])

    # Precompute weights for warm and cold items
    warm_weight, cold_weight = get_weight(warm_item_list, cold_item_list)

    # Generate exposure list for each user
    df['exp_list'] = df[col].apply(lambda x: set_values_by_id(x, config['item_num']))

    # Calculate TGF (Top-K Group Fairness) for each user
    df['hist_tgf'] = df['exp_list'].apply(
        lambda x: get_user_hist_tgf(x, warm_item_list, cold_item_list, warm_weight, cold_weight)
    )

    # Convert cold_item_list to a set for efficient membership checking
    cold_item_set = set(cold_item_list)

    # Calculate Novelty Coverage (NC) for each user
    new_stage_nc = df[col].apply(lambda x: calculate_nc(x, cold_item_set))

    # Update the historical Novelty Coverage (NC) with the new value
    df['hist_nc'] = nc_weight * df['hist_nc'] + (1 - nc_weight) * new_stage_nc

    return df


# ====================================================
# Train Baseline (PD+Pearson+CNIF)
# ====================================================

def pcc_train(model_here, train_data, item_pop_dict, config):
    """
    Train the model using Pearson Correlation Coefficient (PCC) as a loss component.

    Args:
        model_here: The model to be trained.
        train_data (pd.DataFrame): Training data containing 'user' and 'item' columns.
        item_pop_dict (dict): Dictionary mapping item IDs to their popularity counts.
        config (dict): Configuration dictionary containing 'algo_name'.

    Returns:
        pcc (torch.Tensor): Pearson Correlation Coefficient between predictions and item popularity.
    """
    data2 = train_data.copy()
    
    # Convert user and item columns to integers
    data2['user'] = data2['user'].apply(lambda x: int(x))
    data2['item'] = data2['item'].apply(lambda x: int(x))    
        
    # Filter users with more than one interaction
    filter_users = data2.user.value_counts()[data2.user.value_counts() > 1].index
    data2 = data2[data2.user.isin(filter_users)]
    data2 = data2.reset_index()[['user', 'item']]
    
    user_num = len(data2.user.unique())
    data_len = data2.shape[0]
    frac = 50  # Number of chunks to split the data into
    frac_user_num = int(data_len / frac)
    
    predictions_list = torch.tensor([]).cuda()  # Initialize an empty tensor for predictions
    model_here.cuda()  # Move the model to GPU
    
    # Process data in chunks
    for itr in range(frac):
        tmp = data2.iloc[(frac_user_num * itr): (frac_user_num * (itr + 1))].values    
        user = tmp[:, 0]
        user = user.astype(np.int64)        
        user = torch.from_numpy(user).cuda()
        item = tmp[:, 1]
        item = item.astype(np.int64)        
        item = torch.from_numpy(item).cuda()
        
        # Get predictions based on the model type
        if config['algo_name'] == 'mf':
            predictions = model_here(user, item)
        elif config['algo_name'] == 'lightgcn':
            user_emb_all, item_emb_all = model_here.forward()
            user_emb = user_emb_all[user]
            item_emb = item_emb_all[item]
            predictions = (user_emb * item_emb).sum(dim=-1)
        
        predictions_list = torch.hstack((predictions_list, predictions))  # Append predictions

        # Handle the last chunk
        if itr + 1 == frac:
            tmp = data2.iloc[(frac_user_num * (itr + 1)):].values    
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

    # Map item popularity counts to the training data
    data2['item_pop_count'] = data2['item'].map(item_pop_dict)            
        
    # Reshape predictions and item popularity counts
    values = predictions_list.reshape(-1, 1)
    item_pop_count = data2.item_pop_count.values
    item_pop_count = item_pop_count.astype(np.int32)
    item_pop_count = torch.from_numpy(item_pop_count).float().cuda()
    
    # Compute Pearson Correlation Coefficient (PCC)
    X = values
    Y = item_pop_count  # Item popularity
    X = X.view([1, -1])
    Y = Y.view([1, -1])
    pcc = ((X - X.mean()) * (Y - Y.mean())).sum() / ((X - X.mean()) * (X - X.mean())).sum().sqrt() / ((Y - Y.mean()) * (Y - Y.mean())).sum().sqrt()    
    
    return pcc


def calculate_loss(sample, pos_scores, neg_scores, pos, neg, config):
    """
    Calculate the loss based on the sampling method.

    Args:
        sample (str): Sampling method (e.g., 'bpr', 'posneg', 'pd', 'pearson').
        pos_scores (torch.Tensor): Scores for positive items.
        neg_scores (torch.Tensor): Scores for negative items.
        pos (torch.Tensor): Positive items.
        neg (torch.Tensor): Negative items.
        config (dict): Configuration dictionary containing 'item_pop_train_dict', 'warm_item_list',
                       'cold_item_list', 'weight', etc.

    Returns:
        loss (torch.Tensor): Computed loss.
        extra_loss_info (tuple): Additional loss information (e.g., accuracy loss, popularity loss).
    """
    item_pop_train_dict = config['item_pop_train_dict']
    warm_item_train = config['warm_item_list']
    cold_item_train = config['cold_item_list']
    acc_w = config['weight']
    pop_w = 1.0 - acc_w

    if sample == 'bpr':
        # Bayesian Personalized Ranking (BPR) loss
        acc_loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()
        return acc_loss, (acc_loss.item(), 0)
    elif sample in ['pd']:
        # Popularity Debiasing (PD) loss
        pos_weight = torch.tensor([item_pop_train_dict[item.item()] for item in pos], device=pos.device)
        neg_weight = torch.tensor([item_pop_train_dict[item.item()] for item in neg], device=neg.device)
        m = nn.ELU()
        return -torch.log(torch.sigmoid((m(pos_scores) + 1.0) * pos_weight ** acc_w - (m(neg_scores) + 1.0) * neg_weight ** acc_w)).mean(), None
    elif sample == 'pearson':
        # Pearson Correlation Coefficient (PCC) loss
        return -torch.log(torch.sigmoid(pos_scores - neg_scores)).mean(), None
    else:
        raise ValueError(f"Unsupported sample method: {sample}")


def train_baseline_model(model, train_loader, train_set, config, device='cuda'):
    """
    Train the baseline model.

    Args:
        model: The model to be trained.
        train_loader (DataLoader): DataLoader for the training data.
        train_set (pd.DataFrame): Training data.
        config (dict): Configuration dictionary containing hyperparameters.
        device (str): Device to use for training (default: 'cuda').

    Returns:
        model: Trained model.
    """
    item_pop_train_dict = config['item_pop_train_dict']
    warm_item_train = config['warm_item_list']
    cold_item_train = config['cold_item_list']
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    weight = config['weight']
    
    # Training loop
    for epoch in range(config['epochs']):
        print(f'Epoch {epoch + 1}/{config["epochs"]}')
        model.train()
        model.to(device)
        
        if config['debias_method'] in ['cnif', 'posneg']:
            config['burnin'] = 'yes'
        
        # Dynamic sampling method selection
        sample = 'bpr' if (epoch < (config['epochs'] / 2) and config['burnin'] == 'yes') else config['debias_method']

        if sample == 'cnif':
            # Train using CNIF (Counterfactual Negative Item Frequency) loss
            for epoch in range(10):  # Train for 10 epochs
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
                    loss = weight * ((cgf_loss - 0.3430) ** 2)
                    if count_cnif % 5 == 0:
                        print(f'cgf_loss of batch {batch_idx + 1}/{num_batches} in epoch {epoch + 1}: {cgf_loss.item()}')
                        print(f'loss of batch {batch_idx + 1}/{num_batches} in epoch {epoch + 1}: {loss.item()}')
                    count_cnif += 1
                    loss.backward()
                    optimizer.step()
        else:
            # Train using the selected sampling method
            count = 0
            for user, pos1, neg1 in tqdm(train_loader, desc='Training'):
                user = user.to(device)
                pos = pos1.to(device)
                neg = neg1.to(device)

                optimizer.zero_grad()

                if config['algo_name'] == 'mf':
                    user_emb = model.embed_user(user)
                    pos_emb = model.embed_item(pos)
                    neg_emb = model.embed_item(neg)
                    pos_scores = model(user, pos)
                    neg_scores = model(user, neg)
                elif config['algo_name'] == 'lightgcn':
                    user_emb_all, item_emb_all = model.forward()
                    user, pos, neg = to_long_tensor(user, pos, neg)
                    user_emb = user_emb_all[user]
                    pos_emb = item_emb_all[pos]
                    neg_emb = item_emb_all[neg]
                    pos_scores = (user_emb * pos_emb).sum(dim=-1)
                    neg_scores = (user_emb * neg_emb).sum(dim=-1)

                loss, extra_loss_info = calculate_loss(sample, pos_scores, neg_scores, pos, neg, config)

                if config['add_reg'] == 'yes':
                    reg = (torch.norm(user_emb) ** 2 + torch.norm(pos_emb) ** 2 + torch.norm(neg_emb) ** 2) / 3 / config['batch_size']
                    loss += config['reg_2'] * reg

                loss.backward()
                optimizer.step()

                if count % 1000 == 0:
                    if extra_loss_info:
                        print(f'Loss: {loss.item()}, Extra acc Loss: {extra_loss_info[0]}, Extra pop Loss: {extra_loss_info[1]}')
                count += 1

            if sample == 'pearson':
                pcc = pcc_train(model, train_set, item_pop_train_dict, config) 
                loss = weight * (pcc ** 2)
                print(f'pcc loss of epoch: {pcc.item()}')
                print(f'loss of epoch: {loss.item()}')
                loss.backward()
                optimizer.step()

    return model


def cgf_topk_loss(model_cgf, users, items, config):
    """
    Calculate the Counterfactual Group Fairness (CGF) loss.

    Args:
        model_cgf: The model.
        users (np.array): Array of user IDs.
        items (np.array): Array of item IDs.
        config (dict): Configuration dictionary containing 'warm_item_list', 'cold_item_list', 'topk'.

    Returns:
        cgf_loss (torch.Tensor): CGF loss.
    """
    warm_items = config['warm_item_list']
    cold_items = config['cold_item_list']
    item_num = len(items)
    warm_nums = len(warm_items)
    cold_nums = len(cold_items)
    top_score = config['topk']
    
    model_cgf.cuda()
    users = torch.tensor(users).cuda()
    items = torch.tensor(items).cuda()
    
    if config['algo_name'] == 'mf':
        user_embed = model_cgf.embed_user(users)
        item_embed = model_cgf.embed_item(items)
    elif config['algo_name'] == 'lightgcn':
        user_emb_all, item_emb_all = model_cgf.forward()
        user_embed = user_emb_all[users]
        item_embed = item_emb_all[items]
    
    score_mat = torch.mm(item_embed, user_embed.t())
    _, item_index = score_mat.topk(top_score, dim=0)
    
    mask = torch.zeros_like(score_mat).cuda()
    mask.scatter_(0, item_index, 1)
    score_top = score_mat * mask
    
    exp_item_list = score_top.mean(dim=1)
    exp_item_list = exp_item_list / exp_item_list.sum()
    
    weight_list = torch.zeros(item_num).cuda()
    weight_list[:warm_nums] = (warm_nums - torch.arange(warm_nums).cuda()) / warm_nums
    weight_list[warm_nums:] = -(1 + (torch.arange(cold_nums).cuda() * (warm_nums - 1) / (cold_nums - 1))) / cold_nums
    
    cgf_loss = (exp_item_list * weight_list).sum()
    
    return cgf_loss


# ====================================================
# Train ALDI and Evaluation
# ====================================================

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

def get_test_predict(test_data, gen_user_emb, gen_item_emb, warm_items, cold_items):
    """
    Generate predictions for test data using user and item embeddings.

    Args:
        test_data (pd.DataFrame): Test data containing 'user', 'item', and 'label' columns.
        gen_user_emb (list): List of user embeddings (warm and cold).
        gen_item_emb (np.array): Item embeddings.
        warm_items (list): List of warm item IDs.
        cold_items (list): List of cold item IDs.

    Returns:
        pd.DataFrame: DataFrame containing predictions for warm and cold items.
    """
    # Filter and preprocess test data
    test_data = test_data[['user', 'item', 'label']]
    test_data['user'] = test_data['user'].apply(lambda x: int(x))
    test_data['item'] = test_data['item'].apply(lambda x: int(x))
    test_users_num = test_data['user'].nunique()  # Number of unique users in the test data

    # Split test data into warm and cold items
    warm_data = test_data[test_data['item'].isin(warm_items)]
    cold_data = test_data[test_data['item'].isin(cold_items)]

    # Generate predictions for warm items
    warm_predict_list = predict_score_embed(warm_data, gen_user_emb[0], gen_item_emb)
    warm_data['pred'] = warm_predict_list

    # Generate predictions for cold items
    cold_predict_list = predict_score_embed(cold_data, gen_user_emb[1], gen_item_emb)
    cold_data['pred'] = cold_predict_list

    # Combine warm and cold predictions into a single DataFrame
    result_pd = pd.concat([warm_data, cold_data])
    return result_pd


def predict_score_embed(target_data, user_emb, item_emb, test_batch=10000):
    """
    Predict scores for target data using user and item embeddings.

    Args:
        target_data (pd.DataFrame): Target data containing 'user' and 'item' columns.
        user_emb (np.array): User embeddings.
        item_emb (np.array): Item embeddings.
        test_batch (int): Batch size for processing. Default is 10,000.

    Returns:
        list: List of predicted scores.
    """
    data_len = target_data.shape[0]  # Total number of rows in the target data
    frac_user_num = int(data_len / test_batch)  # Number of batches

    # Convert embeddings to numpy arrays
    user_emb = np.array(user_emb)
    item_emb = np.array(item_emb)

    predictions_list = []  # Initialize an empty list to store predictions
    i = 0

    # Process data in batches
    while i < frac_user_num:
        tmp = target_data.iloc[(test_batch * i): (test_batch * (i + 1))].values
        user = tmp[:, 0].astype(np.int32)  # Extract user IDs and convert to int32
        item = tmp[:, 1].astype(np.int32)  # Extract item IDs and convert to int32

        # Select embeddings for the current batch
        select_user_emb = user_emb[user]
        select_item_emb = item_emb[item]

        # Compute predictions as the dot product of user and item embeddings
        predictions_tmp = select_user_emb * select_item_emb
        predictions_tmp = list(np.sum(predictions_tmp, axis=1))  # Sum over embedding dimensions
        predictions_list += predictions_tmp  # Append predictions to the list
        i += 1

    # Process the remaining data (if any)
    if frac_user_num * test_batch < data_len:
        tmp = target_data.iloc[(test_batch * i):].values
        user = tmp[:, 0].astype(np.int32)
        item = tmp[:, 1].astype(np.int32)

        select_user_emb = user_emb[user]
        select_item_emb = item_emb[item]

        predictions_tmp = select_user_emb * select_item_emb
        predictions_tmp = list(np.sum(predictions_tmp, axis=1))
        predictions_list += predictions_tmp

    return predictions_list


def get_rec_result_aldi(df, max_item, topk=50):
    """
    Generate top-k recommendations for each user.

    Args:
        df (pd.DataFrame): DataFrame containing 'user', 'item', and 'pred' columns.
        max_item (int): Maximum item ID to consider.
        topk (int): Number of top recommendations to generate. Default is 50.

    Returns:
        tuple: A tuple containing:
            - List of unique user IDs.
            - List of top-k item recommendations for each user.
            - DataFrame containing the top-k recommendations.
    """
    # Filter items with IDs less than or equal to max_item
    df = df[df['item'] <= max_item]

    # Rank predictions within each user group
    df['rank'] = df.groupby('user')['pred'].rank(ascending=False, method='first')

    # Sort by user and rank
    df = df.sort_values(by=['user', 'rank'])

    # Select the top-k items for each user
    result_df = df.groupby('user').head(topk)

    # Extract unique user IDs
    test_u = result_df['user'].unique()

    # Extract top-k item recommendations for each user
    preds_topk = result_df.groupby('user')['item'].apply(lambda x: x.head(topk).tolist()).tolist()

    return test_u.tolist(), preds_topk, result_df

# ====================================================
# Utility Functions
# ====================================================

def get_config():
    """
    Get model, metric, and hyperparameter configurations.

    Returns:
        model_config: Dictionary of model configurations.
        metrics_config: Dictionary of metric configurations.
        tune_params_config: Dictionary of hyperparameter configurations.
        param_type_config: Dictionary of parameter types.
    """
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

def to_long_tensor(*args, device='cuda'):
    """
    Convert input variables (numpy arrays or PyTorch tensors) to torch.long type and move them to the specified device.

    Args:
        *args: Input variables (numpy arrays or PyTorch tensors).
        device (str): Target device, default is 'cuda'.

    Returns:
        torch.Tensor or tuple: Converted torch.long tensor(s). If only one input is provided, a single tensor is returned.
                              If multiple inputs are provided, a tuple of tensors is returned.
    """
    results = []
    for arg in args:
        if isinstance(arg, np.ndarray):  # If the input is a numpy array
            arg = arg.astype(np.int64)  # Convert to 64-bit integer
            arg = torch.from_numpy(arg).to(device)  # Convert to PyTorch tensor and move to the specified device
        elif isinstance(arg, torch.Tensor):  # If the input is a PyTorch tensor
            arg = arg.long().to(device)  # Convert to long type and move to the specified device
        else:
            raise TypeError(f"Unsupported type: {type(arg)}. Expected numpy array or PyTorch tensor.")
        results.append(arg)
    
    # Return a single tensor if there's only one input, otherwise return a tuple of tensors
    return results[0] if len(results) == 1 else tuple(results)




