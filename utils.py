#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


def prefilter_items_price(data):
    data['price'] = data['sales_value'] / (np.maximum(data_train['quantity'], 1))

# Уберем слишком дешевые товары (на них не заработаем). 1 покупка из рассылок стоит 60 руб. 
    # using 'price' column assuming that prices there are in rubles
    
    cheap_items = data.loc[data['price']<60, 'item_id']
    data = data[~data['item_id'].isin(cheap_items)]
    
    # Уберем слишком дорогие товары
    # to exclude around 500 most expensive items from all 92353 we have to use 0,995 quantile
    
    expensive_items = data.loc[data['price']>(data['price'].quantile(0.995)), 'item_id']
    data = data[~data['item_id'].isin(expensive_items)]
    
    # using the price graph we can use 300 rubles margin to exclude most expensive items 
    
    expensive_items = data.loc[data['price']>300, 'item_id']
    data = data[~data['item_id'].isin(expensive_items)]
    
    return data


# In[3]:


def prefilter_items_on_department(data, item_features, user_id):
    
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
    
    return data


# In[ ]:




