#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np


# In[4]:


def prefilter_items(data):
    # num_all_items = data['item_id'].nunique() # number of all items before filtering
    
    # Уберем самые популярные товары (их и так купят)
    popularity = data.groupby('item_id')['user_id'].nunique().reset_index() / data['user_id'].nunique()
    popularity.rename(columns={'user_id': 'share_unique_users'}, inplace=True)
    
    top_popular = popularity[popularity['share_unique_users'] > 0.5].item_id.tolist()
    data = data[~data['item_id'].isin(top_popular)]
    
    # Уберем самые НЕ популярные товары (их и так НЕ купят)
    top_notpopular = popularity[popularity['share_unique_users'] < 0.01].item_id.tolist()
    data = data[~data['item_id'].isin(top_notpopular)]
    
    # Уберем товары, которые не продавались за последние 12 месяцев
    # we take a 'day' column so 365 days will constitute a year
    old_items = data.loc[data['day']<(data['day'].max()-365), 'item_id']
    new_items = data.loc[data['day']>=(data['day'].max()-365), 'item_id']
    outdated_items = list(set(old_items)-set(new_items))
    
    data = data[~data['item_id'].isin(outdated_items)]     
    
       
    # count items
    num_filtered_items = data['item_id'].nunique()
    
    if num_filtered_items > 5000:
        popular = data.groupby('item_id')['quantity'].sum().reset_index()
        popular.rename(columns={'quantity': 'n_sold'}, inplace=True)
        top_5000 = popular.sort_values('n_sold', ascending=False).head(5000).item_id.tolist()
        
        data = data.loc[data['item_id'].isin(top_5000), :]
        
        return data
    
    else:
        return data


# In[5]:


def cold_user_recommend(prefiltered_data, n=5):
        
    popular = prefiltered_data.groupby('item_id')['quantity'].count().reset_index()
    popular.sort_values('quantity', ascending=False, inplace=True)
    
    recs = popular.head(n).item_id
    
    return recs.tolist()


# In[6]:


def get_user_matrix(data, user_features, user_factors):
    
    user_feat = user_features
    user_feat.set_index('user_id', inplace=True)

    
    # adding factors as features
    user_feat = user_feat.merge(user_factors, left_index=True, right_index=True, how='outer')
    
    # mean time of transaction
    time = data.groupby('user_id')['trans_time'].mean().reset_index()
    time.rename(columns={'trans_time': 'mean_time'}, inplace=True)
    time = time.astype(np.float32)
    time.set_index('user_id', inplace=True)
    user_feat = user_feat.merge(time, left_index=True, right_index=True, how='outer')
    

    # mean baskets a week
    user_expenses = data.groupby(['user_id'])['sales_value'].sum().reset_index()

    baskets_per_user = data.groupby('user_id')['basket_id'].count().reset_index()
    baskets_per_user.rename(columns={'basket_id': 'baskets_amount'}, inplace=True)

    average_basket = user_expenses.merge(baskets_per_user)

    average_basket['average_basket'] = average_basket.sales_value / average_basket.baskets_amount
    average_basket['baskets_per_week'] = average_basket.sales_value / data.week_no.nunique()

    average_basket = average_basket.drop(['sales_value', 'baskets_amount'], axis=1)
    average_basket.set_index('user_id', inplace=True)
    
    user_feat = user_feat.merge(average_basket, left_index=True, right_index=True, how='outer')
    
    
    user_feat_lightfm = pd.get_dummies(user_feat, columns=user_feat.columns.tolist())

    return user_feat_lightfm


# In[7]:


def get_item_matrix(data, item_features, item_factors):

    item_feat = item_features
    items_in_department = item_feat.groupby('department')['item_id'].count().reset_index().sort_values('item_id', ascending=False)
    
    item_feat.set_index('item_id', inplace=True)


    # adding item factors as new features
    item_feat = item_feat.merge(item_factors, left_index=True, right_index=True, how='outer')
    
    # item's mean discount
    mean_discount = data.groupby('item_id')['coupon_disc'].mean().reset_index().sort_values('coupon_disc')
    mean_discount.set_index('item_id', inplace=True)
    item_feat = item_feat.merge(mean_discount, right_index=True, left_index=True, how='outer')
       

    # mean item sales per week
    item_sales = data.groupby(['item_id'])['quantity'].count().reset_index()
    item_sales.rename(columns={'quantity': 'quantity_of_sales'}, inplace=True)
    
    item_sales['quantity_of_sales_per_week'] = item_sales['quantity_of_sales'] / data['week_no'].nunique()
    item_sales.set_index('item_id', inplace=True)
    item_feat = item_feat.merge(item_sales, right_index=True, left_index=True, how='outer')
    

    # number of items in department

    items_in_department.rename(columns={'item_id': 'items_in_department'}, inplace=True)
    
    data_department = data.set_index('item_id')
    data_department = data_department.merge(item_feat['department'], right_index=True, left_index=True, how="left") 

    # number of sales per department
    qnt_of_sales_per_dep = data_department.groupby(['department'])['quantity'].count().reset_index().sort_values(
        'quantity', ascending=False)
    qnt_of_sales_per_dep.rename(columns={'quantity': 'qnt_of_sales_per_dep'}, inplace=True)

    # mean number of sales per item per department
    items_in_department = items_in_department.merge(qnt_of_sales_per_dep, on='department')
    items_in_department['qnt_of_sales_per_item_per_dep_per_week'] = (
        items_in_department['qnt_of_sales_per_dep'] / 
        items_in_department['items_in_department'] / 
        data_department['week_no'].nunique()
    )
    items_in_department = items_in_department.drop(['items_in_department'], axis=1)
    item_feat = item_feat.merge(items_in_department, on=['department'], how='left')
    


    
    item_f_dummies = pd.get_dummies(item_feat, columns=item_feat.columns.tolist())
    
    return item_f_dummies


# In[8]:


# sub_commodity_desc

def sub_commodity_desc(user_item_matrix, item_features, item_factors):

    items_in_department = item_feat.groupby('sub_commodity_desc')['item_id'].count().reset_index().sort_values(
        'item_id', ascending=False
    )
    items_in_department.rename(columns={'item_id': 'items_in_sub_commodity_desc'}, inplace=True)

    qnt_of_sales_per_dep = new_item_features.groupby(['sub_commodity_desc'])['quantity'].count().reset_index().sort_values(
        'quantity', ascending=False
    )
    qnt_of_sales_per_dep.rename(columns={'quantity': 'qnt_of_sales_per_sub_commodity_desc'}, inplace=True)


    items_in_department = items_in_department.merge(qnt_of_sales_per_dep, on='sub_commodity_desc')
    items_in_department['qnt_of_sales_per_item_per_sub_commodity_desc_per_week'] = (
        items_in_department['qnt_of_sales_per_sub_commodity_desc'] / 
        items_in_department['items_in_sub_commodity_desc'] / 
        new_item_features['week_no'].nunique()
    )
    items_in_department = items_in_department.drop(['items_in_sub_commodity_desc'], axis=1)
    item_features = item_features.merge(items_in_department, on=['sub_commodity_desc'], how='left')


# In[9]:


def filter_items_by_dept(data, item_features, user_id):
      # Уберем не интересные для рекоммендаций категории (department)
    # that will be a separate set for each user which later can be somehow additionally processed
    # we will use additional parameters for the function: 
    # - to pass items_features dataset as an argument, which contains department feature
    # - to pass a user_id, for whom we choose relevant departments
    
    user = user_id   
    items_by_user = data.groupby('user_id')['item_id'].unique().reset_index() # each user has his bought items in list
    user_items = items_by_user.loc[items_by_user['user_id']==user, 'item_id'] # getting relevant user's list of items
    relevant_depatments = item_features.loc[item_features['item_id'].isin(list(user_items)[0]), 'department'].unique()
    
    relevant_items = item_features.loc[item_features['department'].isin(list(relevant_depatments)), 'item_id']
    
    data = data.loc[data['item_id'].isin(relevant_items), :]

