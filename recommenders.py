#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

# Для работы с матрицами
from scipy.sparse import csr_matrix

# Матричная факторизация
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import ItemItemRecommender  # нужен для одного трюка
from implicit.nearest_neighbours import bm25_weight, tfidf_weight


# In[2]:


class MainRecommender:
    """Рекоммендации, которые можно получить из ALS
    
    Input
    -----
    user_item_matrix: pd.DataFrame
        Матрица взаимодействий user-item
    """
    
    def __init__(self, data, weighting=True):
        
        # your_code. Это не обязательная часть. Но если вам удобно что-либо посчитать тут - можно это сделать
        
        self.user_item_matrix = self.prepare_matrix(data)  # pd.DataFrame
        self.id_to_itemid, self.id_to_userid,         self.itemid_to_id, self.userid_to_id = self.prepare_dicts(self.user_item_matrix)
        
        if weighting:
            self.user_item_matrix = bm25_weight(self.user_item_matrix.T).T 
        
        self.model = self.fit(self.user_item_matrix)
        self.own_recommender = self.fit_own_recommender(self.user_item_matrix)
     
    @staticmethod
    def prepare_matrix(data):
        user_item_matrix = pd.pivot_table(data, 
                                  index='user_id', columns='item_id', values='quantity', aggfunc='count', 
                                  fill_value=0)
        
        user_item_matrix = user_item_matrix.astype(float) 
        
        return user_item_matrix
    
    @staticmethod
    def prepare_dicts(user_item_matrix):
        """Подготавливает вспомогательные словари"""
        
        userids = user_item_matrix.index.values
        itemids = user_item_matrix.columns.values

        matrix_userids = np.arange(len(userids))
        matrix_itemids = np.arange(len(itemids))

        id_to_itemid = dict(zip(matrix_itemids, itemids))
        id_to_userid = dict(zip(matrix_userids, userids))

        itemid_to_id = dict(zip(itemids, matrix_itemids))
        userid_to_id = dict(zip(userids, matrix_userids))
        
        return id_to_itemid, id_to_userid, itemid_to_id, userid_to_id
     
    @staticmethod
    def fit_own_recommender(user_item_matrix):
        """Обучает модель, которая рекомендует товары, среди товаров, купленных юзером"""
    
        own_recommender = ItemItemRecommender(K=1, num_threads=4)
        own_recommender.fit(csr_matrix(user_item_matrix).T.tocsr())
        
        return own_recommender
    
    @staticmethod
    def fit(user_item_matrix, n_factors=20, regularization=0.001, iterations=15, num_threads=4):
        """Обучает ALS"""
        
        model = AlternatingLeastSquares(factors=n_factors, 
                                             regularization=regularization,
                                             iterations=iterations,  
                                             num_threads=num_threads)
        model.fit(csr_matrix(user_item_matrix).T.tocsr())
        
        return model

    def get_similar_items_recommendation(self, user, num=5):
        """Рекомендуем товары, похожие на топ-N купленных юзером товаров"""

        # your_code
        N=num+1
        similar_items = self.model.similar_items(self.userid_to_id[user], N)
        # we drop the first from recommendations, for it is the item itself
        res = list([self.id_to_itemid[item[0]] for item in similar_items])[1:]

        assert len(res) == num, 'Количество рекомендаций != {}'.format(num)
        return res
    
    def get_similar_users_recommendation(self, user, num=5):
        """Рекомендуем топ-N товаров, среди купленных похожими юзерами"""
    
        # your_code
        N=num+1
        similar_users = self.model.similar_users(self.userid_to_id[user], N)
        
        # we drop the first from recommendation, for it is the user himself
        usr_list = list([usr_id[0] for usr_id in similar_users])[1:]
        
        # taking user-item matrix with selected users only, than
        # summing up each item through these users and selecting N with greatest sum
        res_items = self.user_item_matrix.loc[self.user_item_matrix.index.isin(usr_list), :].sum(axis=0).sort_values(ascending=False).index[:num]
       
        # getting true id numbers of recommended items
        res = list([self.id_to_itemid[item] for item in res_items])
            

        assert len(res) == num, 'Количество рекомендаций != {}'.format(num)
        return res


# In[ ]:




