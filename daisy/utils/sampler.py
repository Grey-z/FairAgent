import numpy as np
import pandas as pd
from collections import defaultdict
from joblib import Parallel, delayed
from functools import lru_cache

class AbstractSampler(object):
    def __init__(self, config):
        self.uid_name = config['UID_NAME']
        self.iid_name = config['IID_NAME']
        self.item_num = config['item_num']
        self.ur = config['train_ur']

    def sampling(self):
        raise NotImplementedError

class BasicNegtiveSampler(AbstractSampler):
    def __init__(self, df, config):
        """
        negative sampling class for <u, pos_i, neg_i> or <u, pos_i, r>
        Parameters
        ----------
        df : pd.DataFrame, the raw <u, i, r> dataframe
        user_num: int, the number of users
        item_num: int, the number of items
        num_ng : int, No. of nagative sampling per sample, default is 4
        sample_method : str, sampling method, default is 'uniform',
                        'uniform' discrete uniform sampling
                        'high-pop' sample items with high popularity as priority
                        'low-pop' sample items with low popularity as prority
        sample_ratio : float, scope [0, 1], it determines the ratio that the other sample method except 'uniform' occupied, default is 0
        """
        super(BasicNegtiveSampler, self).__init__(config)
        self.user_num = config['user_num']
        self.num_ng = config['num_ng']
        self.inter_name = config['INTER_NAME']
        self.sample_method = config['sample_method']
        self.sample_ratio = config['sample_ratio']
        self.loss_type = config['loss_type'].upper()

        assert self.sample_method in ['uniform', 'low-pop', 'high-pop'], f'Invalid sampling method: {self.sample_method}'
        assert 0 <= self.sample_ratio <= 1, 'Invalid sample ratio value'

        self.df = df
        self.pop_prob = None
        
        if self.sample_method in ['high-pop', 'low-pop']:
            pop = df.groupby(self.iid_name).size()
            # rescale to [0, 1]
            pop /= pop.sum()
            if self.sample_method == 'high-pop':
                norm_pop = np.zeros(self.item_num)
                norm_pop[pop.index] = pop.values
            if self.sample_method == 'low-pop':
                norm_pop = np.ones(self.item_num)
                norm_pop[pop.index] = (1 - pop.values)
            self.pop_prob = norm_pop / norm_pop.sum()

    def sampling(self):
        if self.num_ng == 0:
            self.df['neg_set'] = self.df.apply(lambda x: [], axis=1)
            if self.loss_type in ['CL', 'SL']:
                return self.df[[self.uid_name, self.iid_name, self.inter_name]].values.astype(np.int32)
            else:
                raise NotImplementedError('loss function (BPR, TL, HL) need num_ng > 0')

        js = np.zeros((self.user_num, self.num_ng), dtype=np.int32)
        if self.sample_method in ['low-pop', 'high-pop']:
            other_num = int(self.sample_ratio * self.num_ng)
            uniform_num = self.num_ng - other_num

            for u in range(self.user_num):
                past_inter = list(self.ur[u])

                uni_negs = np.random.choice(
                    np.setdiff1d(np.arange(self.item_num), past_inter), 
                    size=uniform_num
                )
                other_negs = np.random.choice(
                    np.arange(self.item_num),
                    size=other_num,
                    p=self.pop_prob
                )
                js[u] = np.concatenate((uni_negs, other_negs), axis=None)

        else:
            # all negative samples are sampled by uniform distribution
            for u in range(self.user_num):
                past_inter = list(self.ur[u])
                js[u] = np.random.choice(          # get random choice
                    np.setdiff1d(np.arange(self.item_num), past_inter),  # list1 - list2
                    size=self.num_ng
                )

        self.df['neg_set'] = self.df[self.uid_name].agg(lambda u: js[u])

        if self.loss_type.upper() in ['CL', 'SL']:
            point_pos = self.df[[self.uid_name, self.iid_name, self.inter_name]]
            point_neg = self.df[[self.uid_name, 'neg_set', self.inter_name]].copy()
            point_neg[self.inter_name] = 0
            point_neg = point_neg.explode('neg_set')    
            return np.vstack([point_pos.values, point_neg.values]).astype(np.int32)
        elif self.loss_type.upper() in ['BPR', 'HL', 'TL']:
            self.df = self.df[[self.uid_name, self.iid_name, 'neg_set']].explode('neg_set')
            return self.df.values.astype(np.int32)
        else:
            raise NotImplementedError


# class UniqueNegativeSampler(AbstractSampler):
#     def __init__(self, df, config):
#         """
#         Unique negative sampling class for <u, pos_i, neg_i> or <u, pos_i, r>
#         Parameters
#         ----------
#         df : pd.DataFrame, the raw <u, i, r> dataframe
#         config : dict, configuration parameters
#         """
#         super(UniqueNegativeSampler, self).__init__(config)
#         self.user_num = config['user_num']
#         self.num_ng = config['num_ng']
#         self.inter_name = config['INTER_NAME']
#         self.sample_method = config['sample_method']
#         self.sample_ratio = config['sample_ratio']
#         self.item_name = config['IID_NAME']
#         self.loss_type = config['loss_type'].upper()

#         assert self.sample_method in ['uniform', 'low-pop', 'high-pop'], f'Invalid sampling method: {self.sample_method}'
#         assert 0 <= self.sample_ratio <= 1, 'Invalid sample ratio value'

#         self.df = df
#         self.pop_prob = None
        
#         if self.sample_method in ['high-pop', 'low-pop']:
#             pop = df.groupby(self.iid_name).size()
#             pop /= pop.sum()
#             if self.sample_method == 'high-pop':
#                 norm_pop = np.zeros(self.item_num)
#                 norm_pop[pop.index] = pop.values
#             if self.sample_method == 'low-pop':
#                 norm_pop = np.ones(self.item_num)
#                 norm_pop[pop.index] = (1 - pop.values)
#             self.pop_prob = norm_pop / norm_pop.sum()

#     def sampling(self):
#         if self.num_ng == 0:
#             self.df['neg_set'] = self.df.apply(lambda x: [], axis=1)
#             if self.loss_type in ['CL', 'SL']:
#                 return self.df[[self.uid_name, self.iid_name, self.inter_name]].values.astype(np.int32)
#             else:
#                 raise NotImplementedError('loss function (BPR, TL, HL) need num_ng > 0')

#         # Generate unique negative samples for each interaction record
#         self.df['neg_set'] = self.df.apply(self._sample_negatives, axis=1)

#         if self.loss_type.upper() in ['CL', 'SL']:
#             point_pos = self.df[[self.uid_name, self.iid_name, self.inter_name]]
#             point_neg = self.df[[self.uid_name, 'neg_set', self.inter_name]].copy()
#             point_neg[self.inter_name] = 0
#             point_neg = point_neg.explode('neg_set')    
#             return np.vstack([point_pos.values, point_neg.values]).astype(np.int32)
#         elif self.loss_type.upper() in ['BPR', 'HL', 'TL']:
#             self.df = self.df[[self.uid_name, self.iid_name, 'neg_set']].explode('neg_set')
#             return self.df.values.astype(np.int32)
#         else:
#             raise NotImplementedError

#     def _sample_negatives(self, row):
#         user_id = row[self.uid_name]
#         pos_item_id = row[self.iid_name]
#         past_inter = list(self.ur[user_id])
#         past_inter.append(pos_item_id)

#         # Create candidate set by removing the user's past interactions
#         # candidate_set = np.setdiff1d(np.arange(self.item_num), past_inter)
#         candidate_set = np.setdiff1d(self.df[self.item_name].unique(), past_inter)
        
#         if self.sample_method in ['low-pop', 'high-pop']:
#             other_num = int(self.sample_ratio * self.num_ng)
#             uniform_num = self.num_ng - other_num

#             uni_negs = np.random.choice(
#                 candidate_set, 
#                 size=uniform_num
#             )
#             other_negs = np.random.choice(
#                 candidate_set,
#                 size=other_num,
#                 p=self.pop_prob[candidate_set] / self.pop_prob[candidate_set].sum()
#             )
#             negs = np.concatenate((uni_negs, other_negs), axis=None)
#         else:
#             negs = np.random.choice(
#                 candidate_set, 
#                 size=self.num_ng
#             )

#         return negs.tolist()



# class UniqueNegativeSampler(AbstractSampler):
#     def __init__(self, df, config):
#         """
#         Unique negative sampling class for <u, pos_i, neg_i> or <u, pos_i, r>
#         Parameters
#         ----------
#         df : pd.DataFrame, the raw <u, i, r> dataframe
#         config : dict, configuration parameters
#         """
#         super(UniqueNegativeSampler, self).__init__(config)
#         self.user_num = config['user_num']
#         self.num_ng = config['num_ng']
#         self.inter_name = config['INTER_NAME']
#         self.sample_method = config['sample_method']
#         self.sample_ratio = config['sample_ratio']
#         self.item_name = config['IID_NAME']
#         self.loss_type = config['loss_type'].upper()

#         assert self.sample_method in ['uniform', 'low-pop', 'high-pop'], f'Invalid sampling method: {self.sample_method}'
#         assert 0 <= self.sample_ratio <= 1, 'Invalid sample ratio value'

#         self.df = df
#         self.pop_prob = None
#         self.ur = self._get_user_interactions(df)  # 提前计算用户交互记录

#         if self.sample_method in ['high-pop', 'low-pop']:
#             self.pop_prob = self._calculate_popularity_prob(df)
#         print('begin sampling using Parallel method')

#     def _get_user_interactions(self, df):
#         """计算用户交互记录"""
#         ur = defaultdict(set)
#         for user, item in zip(df[self.uid_name], df[self.item_name]):
#             ur[user].add(item)
#         return ur

#     def _calculate_popularity_prob(self, df):
#         """计算物品流行度概率"""
#         pop = df.groupby(self.item_name).size()
#         pop /= pop.sum()
#         if self.sample_method == 'high-pop':
#             norm_pop = np.zeros(self.user_num)
#             norm_pop[pop.index] = pop.values
#         elif self.sample_method == 'low-pop':
#             norm_pop = np.ones(self.user_num)
#             norm_pop[pop.index] = (1 - pop.values)
#         return norm_pop / norm_pop.sum()

#     @lru_cache(maxsize=None)
#     def _get_candidate_set(self, user_id, pos_item_id):
#         """缓存候选集"""
#         past_inter = self.ur[user_id].copy()
#         past_inter.add(pos_item_id)
#         return np.setdiff1d(self.df[self.item_name].unique(), list(past_inter))

#     def _sample_negatives_vectorized(self, user_id, pos_item_id):
#         """向量化的负样本采样方法"""
#         candidate_set = self._get_candidate_set(user_id, pos_item_id)

#         if self.sample_method in ['low-pop', 'high-pop']:
#             other_num = int(self.sample_ratio * self.num_ng)
#             uniform_num = self.num_ng - other_num

#             uni_negs = np.random.choice(candidate_set, size=uniform_num)
#             other_negs = np.random.choice(
#                 candidate_set,
#                 size=other_num,
#                 p=self.pop_prob[candidate_set] / self.pop_prob[candidate_set].sum()
#             )
#             negs = np.concatenate((uni_negs, other_negs), axis=None)
#         else:
#             negs = np.random.choice(candidate_set, size=self.num_ng)

#         return negs.tolist()

#     def sampling(self):
#         """生成采样数据"""
#         print('begin sampling using Parallel method')
#         if self.num_ng == 0:
#             self.df['neg_set'] = self.df.apply(lambda x: [], axis=1)
#             if self.loss_type in ['CL', 'SL']:
#                 return self.df[[self.uid_name, self.item_name, self.inter_name]].values.astype(np.int32)
#             else:
#                 raise NotImplementedError('loss function (BPR, TL, HL) need num_ng > 0')

#         # 并行采样
#         neg_sets = Parallel(n_jobs=-1)(
#             delayed(self._sample_negatives_vectorized)(row[self.uid_name], row[self.item_name])
#             for _, row in self.df.iterrows()
#         )

#         self.df['neg_set'] = neg_sets

#         if self.loss_type.upper() in ['CL', 'SL']:
#             point_pos = self.df[[self.uid_name, self.item_name, self.inter_name]]
#             point_neg = self.df[[self.uid_name, 'neg_set', self.inter_name]].copy()
#             point_neg[self.inter_name] = 0
#             point_neg = point_neg.explode('neg_set')
#             return np.vstack([point_pos.values, point_neg.values]).astype(np.int32)
#         elif self.loss_type.upper() in ['BPR', 'HL', 'TL']:
#             self.df = self.df[[self.uid_name, self.item_name, 'neg_set']].explode('neg_set')
#             return self.df.values.astype(np.int32)
#         else:
#             raise NotImplementedError

# from collections import defaultdict
# import numpy as np
# import pandas as pd
# from joblib import Parallel, delayed
# from functools import lru_cache
# from tqdm import tqdm  # 导入 tqdm 库

# class UniqueNegativeSampler(AbstractSampler):
#     def __init__(self, df, config):
#         """
#         Unique negative sampling class for <u, pos_i, neg_i> or <u, pos_i, r>
#         Parameters
#         ----------
#         df : pd.DataFrame, the raw <u, i, r> dataframe
#         config : dict, configuration parameters
#         """
#         super(UniqueNegativeSampler, self).__init__(config)
#         self.user_num = config['user_num']
#         self.num_ng = config['num_ng']
#         self.inter_name = config['INTER_NAME']
#         self.sample_method = config['sample_method']
#         self.sample_ratio = config['sample_ratio']
#         self.item_name = config['IID_NAME']
#         self.loss_type = config['loss_type'].upper()

#         assert self.sample_method in ['uniform', 'low-pop', 'high-pop'], f'Invalid sampling method: {self.sample_method}'
#         assert 0 <= self.sample_ratio <= 1, 'Invalid sample ratio value'

#         self.df = df
#         self.pop_prob = None
#         self.ur = self._get_user_interactions(df)  # 提前计算用户交互记录

#         if self.sample_method in ['high-pop', 'low-pop']:
#             self.pop_prob = self._calculate_popularity_prob(df)
#         print('begin sampling using Parallel method')

#     def _get_user_interactions(self, df):
#         """计算用户交互记录"""
#         ur = defaultdict(set)
#         for user, item in zip(df[self.uid_name], df[self.item_name]):
#             ur[user].add(item)
#         return ur

#     def _calculate_popularity_prob(self, df):
#         """计算物品流行度概率"""
#         pop = df.groupby(self.item_name).size()
#         pop /= pop.sum()
#         if self.sample_method == 'high-pop':
#             norm_pop = np.zeros(self.user_num)
#             norm_pop[pop.index] = pop.values
#         elif self.sample_method == 'low-pop':
#             norm_pop = np.ones(self.user_num)
#             norm_pop[pop.index] = (1 - pop.values)
#         return norm_pop / norm_pop.sum()

#     @lru_cache(maxsize=None)
#     def _get_candidate_set(self, user_id, pos_item_id):
#         """缓存候选集"""
#         past_inter = self.ur[user_id].copy()
#         past_inter.add(pos_item_id)
#         return np.setdiff1d(self.df[self.item_name].unique(), list(past_inter))

#     def _sample_negatives_vectorized(self, user_id, pos_item_id):
#         """向量化的负样本采样方法"""
#         candidate_set = self._get_candidate_set(user_id, pos_item_id)

#         if self.sample_method in ['low-pop', 'high-pop']:
#             other_num = int(self.sample_ratio * self.num_ng)
#             uniform_num = self.num_ng - other_num

#             uni_negs = np.random.choice(candidate_set, size=uniform_num)
#             other_negs = np.random.choice(
#                 candidate_set,
#                 size=other_num,
#                 p=self.pop_prob[candidate_set] / self.pop_prob[candidate_set].sum()
#             )
#             negs = np.concatenate((uni_negs, other_negs), axis=None)
#         else:
#             negs = np.random.choice(candidate_set, size=self.num_ng)

#         return negs.tolist()

#     def _sample_negatives_batch(self, user_ids, pos_item_ids):
#         """批量采样负样本"""
#         return [self._sample_negatives_vectorized(uid, pid) for uid, pid in zip(user_ids, pos_item_ids)]

#     def sampling(self):
#         """生成采样数据"""
#         print('begin sampling using Parallel method')
#         if self.num_ng == 0:
#             self.df['neg_set'] = self.df.apply(lambda x: [], axis=1)
#             if self.loss_type in ['CL', 'SL']:
#                 return self.df[[self.uid_name, self.item_name, self.inter_name]].values.astype(np.int32)
#             else:
#                 raise NotImplementedError('loss function (BPR, TL, HL) need num_ng > 0')

#         # 分批处理数据
#         batch_size = 1000  # 每批处理 1000 行数据
#         user_ids = self.df[self.uid_name].values
#         pos_item_ids = self.df[self.item_name].values

#         # 将数据分成多个批次
#         batches = [
#             (user_ids[i:i + batch_size], pos_item_ids[i:i + batch_size])
#             for i in range(0, len(self.df), batch_size)
#         ]

#         # 并行采样（带进度条）
#         neg_sets = Parallel(n_jobs=-1)(
#             delayed(self._sample_negatives_batch)(batch_users, batch_items)
#             for batch_users, batch_items in tqdm(batches, desc="Sampling Progress", unit="batch")
#         )

#         # 合并结果
#         self.df['neg_set'] = [item for sublist in neg_sets for item in sublist]

#         if self.loss_type.upper() in ['CL', 'SL']:
#             point_pos = self.df[[self.uid_name, self.item_name, self.inter_name]]
#             point_neg = self.df[[self.uid_name, 'neg_set', self.inter_name]].copy()
#             point_neg[self.inter_name] = 0
#             point_neg = point_neg.explode('neg_set')
#             return np.vstack([point_pos.values, point_neg.values]).astype(np.int32)
#         elif self.loss_type.upper() in ['BPR', 'HL', 'TL']:
#             self.df = self.df[[self.uid_name, self.item_name, 'neg_set']].explode('neg_set')
#             return self.df.values.astype(np.int32)
#         else:
#             raise NotImplementedError

from collections import defaultdict
import numpy as np
import pandas as pd
from functools import lru_cache
from tqdm import tqdm  # 导入 tqdm 库

class UniqueNegativeSampler(AbstractSampler):
    def __init__(self, df, config):
        """
        Unique negative sampling class for <u, pos_i, neg_i> or <u, pos_i, r>
        Parameters
        ----------
        df : pd.DataFrame, the raw <u, i, r> dataframe
        config : dict, configuration parameters
        """
        super(UniqueNegativeSampler, self).__init__(config)
        self.user_num = config['user_num']
        self.num_ng = config['num_ng']
        self.inter_name = config['INTER_NAME']
        self.sample_method = config['sample_method']
        self.sample_ratio = config['sample_ratio']
        self.item_name = config['IID_NAME']
        self.loss_type = config['loss_type'].upper()

        assert self.sample_method in ['uniform', 'low-pop', 'high-pop'], f'Invalid sampling method: {self.sample_method}'
        assert 0 <= self.sample_ratio <= 1, 'Invalid sample ratio value'

        self.df = df
        self.pop_prob = None
        self.ur = self._get_user_interactions(df)  # 提前计算用户交互记录
        self.all_items = set(df[self.item_name].unique())  # 所有物品集合

        if self.sample_method in ['high-pop', 'low-pop']:
            self.pop_prob = self._calculate_popularity_prob(df)
        print('begin sampling using apply method')

    def _get_user_interactions(self, df):
        """计算用户交互记录"""
        ur = defaultdict(set)
        for user, item in zip(df[self.uid_name], df[self.item_name]):
            ur[user].add(item)
        return ur

    def _calculate_popularity_prob(self, df):
        """计算物品流行度概率"""
        pop = df.groupby(self.item_name).size()
        pop /= pop.sum()
        if self.sample_method == 'high-pop':
            norm_pop = np.zeros(self.user_num)
            norm_pop[pop.index] = pop.values
        elif self.sample_method == 'low-pop':
            norm_pop = np.ones(self.user_num)
            norm_pop[pop.index] = (1 - pop.values)
        return norm_pop / norm_pop.sum()

    def _get_candidate_set(self, user_id):
        """获取用户的候选集"""
        return list(self.all_items - self.ur[user_id])

    def _sample_negatives_vectorized(self, user_id):
        """向量化的负样本采样方法"""
        candidate_set = self._get_candidate_set(user_id)

        if self.sample_method in ['low-pop', 'high-pop']:
            other_num = int(self.sample_ratio * self.num_ng)
            uniform_num = self.num_ng - other_num

            uni_negs = np.random.choice(candidate_set, size=uniform_num)
            other_negs = np.random.choice(
                candidate_set,
                size=other_num,
                p=self.pop_prob[candidate_set] / self.pop_prob[candidate_set].sum()
            )
            negs = np.concatenate((uni_negs, other_negs), axis=None)
        else:
            negs = np.random.choice(candidate_set, size=self.num_ng)

        return negs.tolist()

    def sampling(self):
        """生成采样数据"""
        print('begin sampling using apply method')
        if self.num_ng == 0:
            self.df['neg_set'] = self.df.apply(lambda x: [], axis=1)
            if self.loss_type in ['CL', 'SL']:
                return self.df[[self.uid_name, self.item_name, self.inter_name]].values.astype(np.int32)
            else:
                raise NotImplementedError('loss function (BPR, TL, HL) need num_ng > 0')

        # 使用 apply 方法采样（带进度条）
        tqdm.pandas(desc="Sampling Progress")  # 启用 tqdm 进度条
        self.df['neg_set'] = self.df.progress_apply(
            lambda row: self._sample_negatives_vectorized(row[self.uid_name]), axis=1
        )

        if self.loss_type.upper() in ['CL', 'SL']:
            point_pos = self.df[[self.uid_name, self.item_name, self.inter_name]]
            point_neg = self.df[[self.uid_name, 'neg_set', self.inter_name]].copy()
            point_neg[self.inter_name] = 0
            point_neg = point_neg.explode('neg_set')
            return np.vstack([point_pos.values, point_neg.values]).astype(np.int32)
        elif self.loss_type.upper() in ['BPR', 'HL', 'TL']:
            self.df = self.df[[self.uid_name, self.item_name, 'neg_set']].explode('neg_set')
            return self.df.values.astype(np.int32)
        else:
            raise NotImplementedError

class SkipGramNegativeSampler(AbstractSampler):
    def __init__(self, df, config, discard=False):
        '''
        skip-gram negative sampling class for <target_i, context_i, label>

        Parameters
        ----------
        df : pd.DataFrame
            training set
        rho : float, optional
            threshold to discard word in a sequence, by default 1e-5
        context_window: int, context range around target
        train_ur: dict, ground truth for each user in train set
        item_num: int, the number of items
        '''    
        super(SkipGramNegativeSampler, self).__init__(config)    
        self.context_window = config['context_window']

        word_frequecy = df[self.iid_name].value_counts()
        prob_discard = 1 - np.sqrt(config['rho'] / word_frequecy)

        if discard:
            rnd_p = np.random.uniform(low=0., high=1., size=len(df))
            discard_p_per_item = df[self.iid_name].map(prob_discard).values
            df = df[rnd_p >= discard_p_per_item]

        self.train_seqs = self._build_seqs(df)

    def sampling(self):
        sgns_samples = []

        for u, seq in self.train_seqs.iteritems():
            past_inter = list(self.ur[u])
            cands = np.setdiff1d(np.arange(self.item_num), past_inter)

            for i in range(len(seq)):
                target = seq[i]
                # generate positive sample
                context_list = []
                j = i - self.context_window
                while j <= i + self.context_window and j < len(seq):
                    if j >= 0 and j != i:
                        context_list.append(seq[j])
                        sgns_samples.append([target, seq[j], 1])
                    j += 1
                # generate negative sample
                num_ng = len(context_list)
                for neg_item in np.random.choice(cands, size=num_ng):
                    sgns_samples.append([target, neg_item, 0])
        
        return np.array(sgns_samples)

    def _build_seqs(self, df):
        train_seqs = df.groupby(self.uid_name)[self.iid_name].agg(list)

        return train_seqs
