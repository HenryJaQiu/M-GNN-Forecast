#!/usr/bin/env python
# coding: utf-8

# In[1]:


from model import *
from utils import *
from keras import Model
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt


# In[2]:


conv_units = 64
num_nodes = 30
num_features = 1
encoder_timesteps = 6
decoder_timesteps = 2
encoder_hidden_units = 512
decoder_hidden_units = 512
dataset = 'quanguo'

input_shape = (encoder_timesteps, num_nodes, num_features)
adjacency_shape = (num_nodes, num_nodes)

data = Data(dataset, encoder_timesteps)
X, A_dist, A_corr, y = data.get_sequences()
X_train, y_train, X_test, y_test = get_split(X, A_dist, A_corr, y)

gcn_lstm = GCN_LSTM(conv_units, num_nodes, num_features, encoder_timesteps, decoder_timesteps, encoder_hidden_units, decoder_hidden_units)
model = gcn_lstm.train_model(input_shape, adjacency_shape)

model.fit(X_train, y_train, batch_size=5, epochs=300)


# In[3]:


def evaluate(y_hat, y_truth):
    rmse = sqrt(mean_squared_error(y_hat, y_test))
    print('RMSE: %f' % rmse)
    
    for i in range(0,len(y_hat)):
        rmse = sqrt(mean_squared_error(y_hat[i], y_test[i]))
  
        x = np.arange(1,12)
        width = 0.2

        fig, ax = plt.subplots()
        rects1 = ax.bar(x, y_hat[i], width, color='#6495ED')
        rects2 = ax.bar(x+width, y_test[i], width, color='#F08080')
        
        rects1_x = [rect.get_x()+width/2 for rect in rects1]
        rects2_x = [rect.get_x()+width/2 for rect in rects2]
        rects_x = [(x1 + x2)/2 for x1,x2 in zip(rects1_x, rects2_x)]
        
        ax.set_ylabel('Level')
        ax.set_xlabel('Node Number')
        ax.set_title('RMSE = %f' % rmse)
        ax.set_xticks(rects_x)
        ax.set_xticklabels(('1', '2', '3', '4', '5','6','7','8','9','10','11'))

        ax.legend((rects1[0], rects2[0]), ('y_hat', 'y_truth'))
        plt.show()


# In[4]:


y_hat = model.predict(X_test, batch_size = 1)


# In[5]:


evaluate(y_hat, y_test)


# In[ ]:




