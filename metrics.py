#!/usr/bin/env python
# coding: utf-8

# ### 1. Hit rate

# In[1]:


def hit_rate(recommended_list, bought_list):
    
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    
    flags = np.isin(bought_list, recommended_list)
    
    hit_rate = (flags.sum() > 0) * 1
    
    return hit_rate


# In[2]:


def hit_rate_at_k(recommended_list, bought_list, k=5):
    
    # your_code
    # assuming top-k items are in the end of the list
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    flags = np.isin(bought_list, recommended_list[-k:])
    
    hit_rate = (flags.sum() > 0) * 1
    
    return hit_rate


# ### 2. Precision

# In[3]:


def precision(recommended_list, bought_list):
    
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    
    flags = np.isin(bought_list, recommended_list)
    
    precision = flags.sum() / len(recommended_list)
    
    return precision


# In[4]:


def precision_at_k(recommended_list, bought_list, k=5):
    
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    
    #assuming top-k items are in the beginning of the list
    flags = np.isin(bought_list, recommended_list[:k])
    
    precision = flags.sum() / k
    
    
    return precision


# In[5]:


def money_precision_at_k(recommended_list, bought_list, prices_recommended, k=5):
        
    # your_code
    # Лучше считать через скалярное произведение, а не цикл
       
    flags = np.isin(recommended_list[:k], bought_list)
    
    bought_prices = flags*prices_recommended[:k]
    
    precision = bought_prices.sum()/ np.sum(prices_recommended[:k])
    
    
    return precision


# ### 3. Recall

# In[6]:


def recall(recommended_list, bought_list):
    
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    
    flags = np.isin(bought_list, recommended_list)
    
    recall = flags.sum() / len(bought_list)
    
    return recall


# In[7]:


def recall_at_k(recommended_list, bought_list, k=5):
    
    # your_code
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    
    flags = np.isin(bought_list, recommended_list[:k])
    
    recall = flags.sum() / len(bought_list)
    
    return recall


# In[8]:


def money_recall_at_k(recommended_list, bought_list, items_list, price_list, k=5):
    
    # your_code
    items_dict = dict(zip(items_list, price_list))
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    
    bought_prices = list(items_dict[key] for key in bought_list)
    relevant_items = (item for item in bought_list if item in recommended_list[:k])
    relevant_prices = list(items_dict[key] for key in relevant_items)
    
    recall = np.sum(relevant_prices)/ np.sum(bought_prices)
  
    
    return recall


# In[ ]:





# In[9]:


def ap_k(recommended_list, bought_list, k=5):
    
    bought_list = np.array(bought_list)
    recommended_list = np.array(recommended_list)
    
    flags = np.isin(recommended_list, bought_list)
    
    if sum(flags) == 0:
        return 0
    
    sum_ = 0
    for i in range(0, k):
        
        if flags[i] == True:
            p_k = precision_at_k(recommended_list, bought_list, k=i+1)
            
            sum_ += p_k
            
    result = sum_ / sum(flags) 
    
    return result


# In[ ]:




