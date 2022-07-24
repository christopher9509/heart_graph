#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")

import time
import datetime
start = time.time()
import gower
import os
import networkx as nx

from scipy import sparse
import pickle
from scipy.sparse import csr_matrix
from networkx.readwrite import json_graph
from tqdm.auto import tqdm
import json
import math


# In[4]:


train = pd.read_csv('/nas1/yongk/BK/val.csv') #sftp://hjae@165.132.195.253/home/hjae/heart/dataset/train_SMOTE_not.csv
train = train.iloc[:-1,:]
#train = train.sample(frac=1).reset_index(drop=True)

# In[6]:


list1 = []
def split_train(n, df0):
    for i in range(len(df0)//n):
        temp_x = df0.iloc[n*i:n*(1+i),:]
        #temp_1 = df1.iloc[403*i:403*(1+i),:]
        #temp_y = y_train.iloc[n*i:n*(1+i)]
        #temp = pd.concat([temp_0, temp_1], axis = 0, ignore_index = True)
        
       #temp = pd.concat([temp_x, temp_y], axis = 1)
        
        globals()['train_temp_{}'.format(i)] = temp_x
        
        list1.append('train_temp_{}'.format(i))
    
    return list1

# In[53]:


list1 = split_train(4096, train)


# In[7]:


list1


# In[8]:


#initial parts
count = 95
cnt = [count]
counts = [94]*4096

start = globals()[list1[0]]
node_features = start.iloc[:,1:]
node_labels = start.iloc[:,0]

start_mat = gower.gower_matrix(node_features)
start_mat_copy = start_mat[:, :] # start_mat 복사본 생성
start_mat_copy_sort = np.sort(start_mat_copy) # 상위 205번째 값 도출을 위해 sort
mat = np.identity(len(node_features))

for i in range(4096):
    x = start_mat_copy_sort[i, 41]
    for j in range(4096):
        if start_mat[i, j] < x:
            mat[i, j] = 1
        elif start_mat[i, j] >= x:
            mat[i, j] = 0
        else:
            print("error")
        if i == j:
            mat[i, j] = 1

start_graph = nx.from_numpy_array(mat)

node_ids = np.array(counts)

# In[ ]:

#start
for i in tqdm(range(1, 7)):
    
    cnt = [count]*4096
    
    temp = globals()[list1[i]]
    
    X = temp.iloc[:,1:]
    y = temp.iloc[:,0]
    
    go_mat = gower.gower_matrix(X)
    start_mat_copy = go_mat[:, :] # start_mat 복사본 생성
    start_mat_copy_sort = np.sort(start_mat_copy) # 상위 205번째 값 도출을 위해 sort
    mat_g = np.identity(len(X))
    
    for i in range(4096):
        x = start_mat_copy_sort[i, 41]
        for j in range(4096):
            if start_mat[i, j] < x:
                mat_g[i, j] = 1
            elif start_mat[i, j] >= x:
                mat_g[i, j] = 0
            else:
                print("error")
            if i == j:
                mat_g[i, j] = 1
    
    temp_graph = nx.from_numpy_array(mat_g)
    
    #graph_ids
    counts = counts + cnt
    count += 1
    
    #node_features
    node_features = pd.concat([node_features, X], axis = 0)
    
    #node_labels
    node_labels = pd.concat([node_labels, y], axis = 0)
    
    #node_links_dicts
    start_graph = nx.disjoint_union(start_graph, temp_graph)

node_ids = np.array(counts)


#sftp://hjae@165.132.195.253/nas3/hjae/heart/4096
#내보내기 - (1) : node-feature matrix
sparse_X = sparse.csr_matrix(node_features)
np.savez_compressed('/nas1/yongk/BK/dataset/top_1p_smt_s_valid_feats_4096.csr'.format(i), sparse_X)

with open('/nas1/yongk/BK/dataset/top_1p_smt_s_valid_feats_4096.csr'.format(i), 'wb') as f:
    pickle.dump(sparse_X, f, pickle.HIGHEST_PROTOCOL)

#내보내기 - (2) : node-labels
np.save('/nas1/yongk/BK/dataset/top_1p_smt_s_valid_labels_4096.npy'.format(i), node_labels)

with open('/nas1/yongk/BK/dataset/top_1p_smt_s_valid_labels_4096.npy'.format(i), 'wb') as f:
    pickle.dump(node_labels, f, pickle.HIGHEST_PROTOCOL)
    
#내보내기 - (3) :  adjacency_list_dict
#graph construction from np
composed_graph = json_graph.node_link_data(start_graph)

with open("/nas1/yongk/BK/dataset/top_1p_smt_s_valid_graph_4096.json", "w") as f:
    json.dump(composed_graph, f)

#내보내기 - (4) : graph_ids
np.save('/nas1/yongk/BK/dataset/top_1p_smt_s_valid_graph_id_4096.npy'.format(i), node_ids)

with open('/nas1/yongk/BK/dataset/top_1p_smt_s_valid_graph_id_4096.npy'.format(i), 'wb') as f:
    pickle.dump(node_ids, f, pickle.HIGHEST_PROTOCOL)


# %%
