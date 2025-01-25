import pandas as pd

class Preprocessor_fairagent(object):
    def __init__(self, config):
        """
        Method of loading certain raw data
        Parameters
        ----------
        src : str, the name of dataset
        prepro : str, way to pre-process raw data input, expect 'origin', f'{N}core', f'{N}filter', N is integer value
        binary : boolean, whether to transform rating to binary label as CTR or not as Regression
        pos_threshold : float, if not None, treat rating larger than this threshold as positive sample
        level : str, which level to do with f'{N}core' or f'{N}filter' operation (it only works when prepro contains 'core' or 'filter')

        Returns
        -------
        df : pd.DataFrame, rating information with columns: user, item, rating, (options: timestamp)
        """
        self.src = config['dataset']
        self.prepro = config['prepro']
        self.uid_name = config['UID_NAME']
        self.iid_name = config['IID_NAME']
        self.tid_name = config['TID_NAME']
        self.inter_name = config['INTER_NAME']
        self.binary = config['binary_inter']
        self.pos_threshold = config['positive_threshold']
        self.level = config['level'] # ui, u, i
        self.logger = config['logger']

        self.get_pop = True if 'popularity' in config['metrics'] else False

        self.user_num, self.item_num = None, None
        self.item_pop = None
        self.time_dict = dict()

    def process(self, df):
        df = self.__remove_duplication(df)
        df = self.__reserve_pos(df)
        df = self.__binary_inter(df)
        df = self.__core_filter(df)
        self.time_dict = self.__get_time_dict(df)
        df = self.__reindex_item_time(df, self.time_dict)
        self.user_num, self.item_num = self.__get_stats(df)
        df = self.__category_encoding(df)
        df = self.__sort_by_time(df)
        if self.get_pop:
            self.__get_item_popularity(df)

        self.logger.info(f'Finish loading [{self.src}]-[{self.prepro}] dataset')

        return df

    def __get_item_popularity(self, df):
        self.item_pop = np.zeros(self.item_num)
        pop = df.groupby(self.iid_name).size() / self.user_num
        self.item_pop[pop.index] = pop.values

    def __sort_by_time(self, df):
        df = df.sort_values(self.tid_name).reset_index(drop=True)

        return df

    def get_user_num(self):
        return self.user_num

    def get_item_num(self):
        return self.item_num

    def __remove_duplication(self, df): 
        return df.drop_duplicates([self.uid_name, self.iid_name], keep='last', ignore_index=True)

    def __category_encoding(self, df):
        # encoding user_id and item_id
        self.uid_token = pd.Categorical(df[self.uid_name]).categories.to_numpy()
        self.iid_token = pd.Categorical(df[self.iid_name]).categories.to_numpy()
        self.token_uid = {uid: token for token, uid in enumerate(self.uid_token)}
        self.token_iid = {iid: token for token, iid in enumerate(self.iid_token)}
        df[self.uid_name] = pd.Categorical(df[self.uid_name]).codes
        df[self.iid_name] = pd.Categorical(df[self.iid_name]).codes

        return df

    def __get_stats(self, df):
        user_num = df[self.uid_name].nunique()
        item_num = df[self.iid_name].nunique()

        return user_num, item_num

    def __get_illegal_ids_by_inter_num(self, df, field, inter_num, min_num):
        ids = set()
        for id_ in df[field].values:
            if inter_num[id_] < min_num:
                ids.add(id_)
        return ids

    def __core_filter(self, df):
        # which type of pre-dataset will use
        if self.prepro == 'origin':
            pass
        elif self.prepro.endswith('filter'):
            pattern = re.compile(r'\d+')
            filter_num = int(pattern.findall(self.prepro)[0])

            tmp1 = df.groupby([self.uid_name], as_index=False)[self.iid_name].count()
            tmp1.rename(columns={self.iid_name: 'cnt_item'}, inplace=True)
            tmp2 = df.groupby([self.iid_name], as_index=False)[self.uid_name].count()
            tmp2.rename(columns={self.uid_name: 'cnt_user'}, inplace=True)
            df = df.merge(tmp1, on=[self.uid_name]).merge(tmp2, on=[self.iid_name])
            if self.level == 'ui':    
                df = df.query(f'cnt_item >= {filter_num} and cnt_user >= {filter_num}').reset_index(drop=True).copy()
            elif self.level == 'u':
                df = df.query(f'cnt_item >= {filter_num}').reset_index(drop=True).copy()
            elif self.level == 'i':
                df = df.query(f'cnt_user >= {filter_num}').reset_index(drop=True).copy()        
            else:
                raise ValueError(f'Invalid level value: {self.level}')

            df.drop(['cnt_item', 'cnt_user'], axis=1, inplace=True)
            del tmp1, tmp2
            gc.collect()

        elif self.prepro.endswith('core'):
            pattern = re.compile(r'\d+')
            core_num = int(pattern.findall(self.prepro)[0])

            if self.level == 'ui':
                user_inter_num = Counter(df[self.uid_name].values)
                item_inter_num = Counter(df[self.iid_name].values)
                while True:
                    ban_users = self.__get_illegal_ids_by_inter_num(df, self.uid_name, user_inter_num, core_num)
                    ban_items = self.__get_illegal_ids_by_inter_num(df, self.iid_name, item_inter_num, core_num)

                    if len(ban_users) == 0 and len(ban_items) == 0:
                        break

                    dropped_inter = pd.Series(False, index=df.index)
                    user_inter = df[self.uid_name]
                    item_inter = df[self.iid_name]
                    dropped_inter |= user_inter.isin(ban_users)
                    dropped_inter |= item_inter.isin(ban_items)
                    
                    user_inter_num -= Counter(user_inter[dropped_inter].values)
                    item_inter_num -= Counter(item_inter[dropped_inter].values)

                    dropped_index = df.index[dropped_inter]
                    df.drop(dropped_index, inplace=True)

            elif self.level == 'u':
                tmp = df.groupby([self.uid_name], as_index=False)[self.iid_name].count()
                tmp.rename(columns={self.iid_name: 'cnt_item'}, inplace=True)
                df = df.merge(tmp, on=[self.uid_name])
                df = df.query(f'cnt_item >= {core_num}').reset_index(drop=True).copy()
                df.drop(['cnt_item'], axis=1, inplace=True)
            elif self.level == 'i':
                tmp = df.groupby([self.iid_name], as_index=False)[self.uid_name].count()
                tmp.rename(columns={self.uid_name: 'cnt_user'}, inplace=True)
                df = df.merge(tmp, on=[self.iid_name])
                df = df.query(f'cnt_user >= {core_num}').reset_index(drop=True).copy()
                df.drop(['cnt_user'], axis=1, inplace=True)
            else:
                raise ValueError(f'Invalid level value: {self.level}')

            gc.collect()

        else:
            raise ValueError('Invalid dataset preprocess type, origin/Ncore/Nfilter (N is int number) expected')
        
        df = df.reset_index(drop=True)

        return df

    def __reserve_pos(self, df): 
        # set rating >= threshold as positive samples
        if self.pos_threshold is not None:
            df = df.query(f'{self.inter_name} >= {self.pos_threshold}').reset_index(drop=True)
        return df

    def __binary_inter(self, df):
        # reset rating to interaction, here just treat all rating as 1
        if self.binary:
            df['rating'] = 1.0
        return df
    
    def __get_time_dict(self, df):
        df_item = df.sort_values(self.tid_name).reset_index(drop=True)
        df_item.drop_duplicates(subset=self.iid_name, keep='first', inplace=True)
        df_item = df_item[[self.iid_name,self.tid_name]]
        iid_time = df_item.reset_index(drop = True)
        iid_time = iid_time.reset_index()
        iid_time = iid_time[['index', self.iid_name]]
        iid_time.columns = ['new_iid', self.iid_name]
        time_dict = dict(zip(iid_time[self.iid_name], iid_time['new_iid']))
        return time_dict
        
    def __reindex_item_time(self, df, iid_dict):
        # reindex item id by the order of time
        df[self.iid_name] = df[self.iid_name].map(iid_dict).values
        return df
    