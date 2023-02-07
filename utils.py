#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


class Graph():
    def __init__(self, dataset='hubei', strategy='distance'):
        self.A = self.get_adjacency(dataset, strategy)
        
    def get_adjacency(self, dataset, strategy, path='dataset/'):
        
        
        if strategy == 'distance' or 'correlation':
            A = np.genfromtxt("{}{}.{}.txt".format(path, dataset, strategy), dtype=np.float64)  
            if strategy == 'distance':
                A = np.where(A>0,1/A,0)
        else:
            raise ValueError("Do Not Exist This Strategy")
        
        return A
    
    


# In[3]:


def normalize_adjacency(adj, symmetric=True):
        adj = adj + np.eye(adj.shape[0])
        
        if symmetric:
            d = np.diag(np.power(np.array(adj.sum(1)), -0.5).flatten(), 0)
            a_norm = adj.dot(d).transpose().dot(d)
        else:
            d = np.diag(np.power(np.array(adj.sum(1)), -1).flatten(), 0)
            a_norm = d.dot(adj)

        return a_norm


# In[4]:


def to_level(x):
    x[np.where( (x>=0) & (x<1))]=0
    x[np.where( (x>=1) & (x<10))]=1
    x[np.where( (x>=10) & (x<100))]=2
    x[np.where( (x>=100) & (x<1000))]=3
    x[np.where( (x>=1000) & (x<10000))]=4
    x[np.where( (x>=10000))]=5
    
    return x    


# In[5]:


class Data():
    def __init__(self, dataset, in_steps, out_steps):
        self.dataset = dataset
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.adj_dist = Graph(dataset, 'distance').A
        self.adj_corr = Graph(dataset, 'correlation').A
    
    def get_sequences(self, path='dataset/'):
        cases = np.genfromtxt("{}{}.cases.txt".format(path, self.dataset), dtype=np.int32)
        #cases = to_level(cases)

        X = list()
        y = list()
        for i in range(len(cases)):
            end_ix = i + self.in_steps
            if end_ix + self.out_steps > len(cases)-1:
                break
            seq_x, seq_y = cases[i:end_ix, :], cases[end_ix:end_ix+self.out_steps, :]
            X.append(seq_x)
            y.append(seq_y)

        X = np.array(X)
        X = X.reshape(X.shape[0],X.shape[1],X.shape[2],1)   
        y = np.array(y)
        #y= y.reshape(y.shape[0],y.shape[1], 1) 
        
        A_dist = normalize_adjacency(self.adj_dist, symmetric=True)
        A_dist = np.expand_dims(A_dist, axis=0)
        A_dist = np.repeat(A_dist, X.shape[0], axis=0)
        
        A_corr = normalize_adjacency(self.adj_corr, symmetric=False)
        A_corr = np.expand_dims(A_corr, axis=0)
        A_corr = np.repeat(A_corr, X.shape[0], axis=0)
        
        return X, A_dist, A_corr, y
    


# In[6]:


def get_split(X, A, B, y):
        #train_indices = np.random.choice(len(X), round(len(X)*0.7), replace = False)
        train_indices = np.arange(round(len(X)*0.7))
        test_indices = np.array(list(set(range(len(X))) - set(train_indices)))
        X_train = X[train_indices]
        X_test = X[test_indices]
        y_train = y[train_indices]
        y_test = y[test_indices]
        A_train = A[train_indices]
        A_test = A[test_indices]
        B_train = B[train_indices]
        B_test = B[test_indices]
        
        return [X_train, A_train, B_train], y_train, [X_test, A_test, B_test], y_test


# In[ ]:




