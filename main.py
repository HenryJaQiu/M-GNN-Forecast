#!/usr/bin/env python
# coding: utf-8

# In[1]:


from model import *
from utils import *
from keras import Model
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt
from keras.models import load_model

# In[2]:


conv_units = 64
#num_nodes = 30 # Quanguo
num_nodes = 17 # Hubei
num_features = 1
encoder_timesteps = 6
decoder_timesteps = 3
encoder_hidden_units = 512
decoder_hidden_units = 512
dataset = 'hubei'
#model_name = 'Hubei_8.20_5000.h5'
epoch_amount = 100


# In[4]:


input_shape = (encoder_timesteps, num_nodes, num_features)
adjacency_shape = (num_nodes, num_nodes)

data = Data(dataset, encoder_timesteps, decoder_timesteps)
X, A_dist, A_corr, y = data.get_sequences()
X_train, y_train, X_test, y_test = get_split(X, A_dist, A_corr, y)

gcn_lstm = GCN_LSTM(conv_units, num_nodes, num_features, encoder_timesteps, decoder_timesteps, encoder_hidden_units, decoder_hidden_units)
model = gcn_lstm.train_model(input_shape, adjacency_shape)

#model.fit(X_train, y_train, batch_size=5, epochs=epoch_amount)
# None corr
model.fit(X_train[:2], y_train, batch_size=5, epochs=epoch_amount)


# save model
#model.save_weights(model_name)

# In[6]:

# NONE corr
#y_hat = model.predict(X_test, batch_size = 1)
y_hat = model.predict(X_test[:2], batch_size = 1)

# In[7]:


#print(y_hat)


# In[8]:


#print(y_test)


# In[10]:

for i in range(0, len(y_test)):
    
    #print(y_test[i][:,0:10])
    #print(y_test[i][:,10:17])

    #print(y_hat[i][:,0:10])
    #print(y_hat[i][:,10:17])
    rmse1 = mean_squared_error(y_hat[i][:,0:4]+y_hat[i][:,5:9], y_test[i][:,0:4]+y_test[i][:,5:9])
    rmse2 = mean_squared_error(y_hat[i][:,9:17], y_test[i][:,9:17])
    rmset = sqrt(rmse1 + rmse2)
    print('RMSE: %f' % rmset)

'''
for i in range(0, len(y_test)):
    
    print(y_test[i][:,0:10])
    print(y_test[i][:,10:20])
    print(y_test[i][:,20:30])
    print(y_hat[i][:,0:10])
    print(y_hat[i][:,10:20])
    print(y_hat[i][:,20:30])

    rmse1 = mean_squared_error(y_hat[i][:,0:10], y_test[i][:,0:10])
    rmse2 = mean_squared_error(y_hat[i][:,11:20], y_test[i][:,11:20])
    rmse3 = mean_squared_error(y_hat[i][:,20:30], y_test[i][:,20:30])
    rmset = sqrt(rmse1 + rmse2 + rmse3)
    print('RMSE: %f' % rmset)

'''
# In[ ]:




