#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt; plt.rcdefaults()
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.sparse import coo_matrix
import math as mt
from scipy.sparse.linalg import * #used for matrix multiplication
from scipy.sparse.linalg import svds
from scipy.sparse import csc_matrix
from scipy.stats import skew, norm, probplot
import seaborn as sns
sns.set(style="ticks", color_codes=True, font_scale=1.5)
color = sns.color_palette()
sns.set_style('darkgrid')



# In[2]:


song_df_1 = pd.read_csv('E:/Google Analytics/10000.txt', sep='\t', header=None, names=['user_id','song_id','listen_count'])


# In[3]:


song_df_2 = pd.read_csv('E:/Google Analytics/song_data.csv')


# In[4]:


user_song_list_count = pd.merge(song_df_1,song_df_2.drop_duplicates(['song_id']), on='song_id', how='left')


# In[5]:


user_song_list_count.head()


# In[6]:


user_song_list_count.rename(columns={'user_id':'user'},inplace=True)

user_song_list_count.rename(columns={'song_id':'song'},inplace=True)


# In[7]:


user_song_list_listen = user_song_list_count[['user','listen_count']].groupby('user').sum().reset_index()

user_song_list_listen.rename(columns={'listen_count':'total_listen_count'},inplace=True)
user_song_list_count_merged = pd.merge(user_song_list_count,user_song_list_listen)
user_song_list_count_merged['fractional_play_count'] = user_song_list_count_merged['listen_count']/user_song_list_count_merged['total_listen_count']


# In[8]:


user_codes = user_song_list_count_merged.user.drop_duplicates().reset_index()


# In[9]:


user_codes.rename(columns={'index':'user_index'}, inplace=True)
user_codes['us_index_value'] = list(user_codes.index)


# In[10]:


song_codes = user_song_list_count_merged.song.drop_duplicates().reset_index()


# In[11]:


song_codes.rename(columns={'index':'song_index'}, inplace=True)
song_codes['so_index_value'] = list(song_codes.index)


# In[12]:


small_set = pd.merge(user_song_list_count_merged,song_codes,how='left')


# In[13]:


small_set = pd.merge(small_set,user_codes,how='left')


# In[14]:


mat_candidate = small_set[['us_index_value','so_index_value','fractional_play_count']]


# In[15]:


data_array = mat_candidate.fractional_play_count.values


# In[16]:


row_array = mat_candidate.us_index_value.values
col_array = mat_candidate.so_index_value.values


# In[17]:


data_sparse = coo_matrix((data_array, (row_array, col_array)),dtype=float)


# In[18]:


recommendations = pd.read_csv('E:/Google Analytics/recommendations.csv')


# In[19]:


recommendations = recommendations.drop(columns=['Unnamed: 0'])


# In[20]:


name= []

name.append('43683da3c6c5a93c7938ff550faf0d039a9a639a')


# In[21]:


recommendations.to_dict()


# In[22]:


recommendations = recommendations['43683da3c6c5a93c7938ff550faf0d039a9a639a']


# In[23]:


for i in range(len(recommendations)):
    name.append(recommendations[i])


# In[24]:


num=[]

for i in range(len(name)):

    num.append(list(user_codes.loc[user_codes['user']==name[i]]['us_index_value'])[0])


# In[26]:


def compute_svd(urm, K):
    U, s, Vt = svds(urm, K)

    dim = (len(s), len(s))
    S = np.zeros(dim, dtype=np.float32)
    for i in range(0, len(s)):
        S[i,i] = mt.sqrt(s[i])

    U = csc_matrix(U, dtype=np.float32)
    S = csc_matrix(S, dtype=np.float32)
    Vt = csc_matrix(Vt, dtype=np.float32)
    
    return U, S, Vt


def compute_estimated_matrix(urm, U, S, Vt, uTest, K, test):
    rightTerm = S*Vt 
    max_recommendation = 250
    estimatedRatings = np.zeros(shape=(MAX_UID, MAX_PID), dtype=np.float16)
    recomendRatings = np.zeros(shape=(MAX_UID,max_recommendation ), dtype=np.float16)
    for userTest in uTest:
        prod = U[userTest, :]*rightTerm
        estimatedRatings[userTest, :] = prod.todense()
        recomendRatings[userTest, :] = (-estimatedRatings[userTest, :]).argsort()[:max_recommendation]
    return recomendRatings

def show_recomendations(uTest, num_recomendations = 10):
    xyz=[]
    for user in uTest:
        print('-'*70)
        print("Recommendation for user id {}".format(user))
        rank_value = 1
        i = 0
        while (rank_value <  num_recomendations + 1):
            so = uTest_recommended_items[user,i:i+1][0]
            if (small_set.user[(small_set.so_index_value == so) & (small_set.us_index_value == user)].count()==0):
                song_details = small_set[(small_set.so_index_value == so)].\
                    drop_duplicates('so_index_value')[['song']]
                xyz.append(list(song_details['song'])[0])
                rank_value+=1
            i += 1
    return xyz


# In[27]:


song_real={}

for i in range(len(num)):
    K=50
    urm = data_sparse
    MAX_PID = urm.shape[1]
    MAX_UID = urm.shape[0]

    U, S, Vt = compute_svd(urm, K)
    uTest = [num[i]]

    uTest_recommended_items = compute_estimated_matrix(urm, U, S, Vt, uTest, K, True)

    song_real[name[i]]=show_recomendations(uTest)


# In[28]:


for i in range(1,len(name),1):
    xyz=",".join(song_real.get(name[i]))
    ghi=name[i]+","+xyz
    recommendations.loc[i-1]=ghi


# In[29]:


recommendations = pd.read_csv('E:/Google Analytics/recommendations.csv')

recommendations = recommendations.drop(columns=['Unnamed: 0'])


# In[30]:


for i in range(1,len(name),1):
    xyz=",".join(song_real.get(name[i]))
    ghi=name[i]+","+xyz
    recommendations.loc[i-1]=ghi


# In[31]:


x= name[0] +","+",".join(song_real.get(name[0]))


# In[32]:


recommendations.rename(columns={'43683da3c6c5a93c7938ff550faf0d039a9a639a':x},inplace=True)


# In[33]:


recommendations


# In[ ]:




