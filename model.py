#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.layers import Input,Dense,LSTM, Lambda, TimeDistributed, RNN, GRU, SimpleRNN
from layers import GraphFusion, GraphConvolution
from keras.models import Model
import keras.backend as K
from keras.utils import plot_model


# In[2]:


def slice(x,index):
    return x[:,index:,:]


# In[5]:


class GCN_LSTM(object):
    ''' train LSTM encoder-decoder and make predictions '''
    
    def __init__(self, conv_units, num_nodes, num_features, encoder_timesteps, decoder_timesteps, encoder_hidden_units, decoder_hidden_units):
        '''
        : param input_size:     the number of expected features in the input X
        : param hidden_size:    the number of features in the hidden state h
        '''
        super(GCN_LSTM, self).__init__()
        
        self.conv_units = conv_units
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.encoder_timesteps = encoder_timesteps
        self.decoder_timesteps = decoder_timesteps
        self.encoder_hidden_units = encoder_hidden_units
        self.decoder_hidden_units = decoder_hidden_units
        
        self.graph_fusion = GraphFusion()
        self.graph_convolution = GraphConvolution(conv_units, activation='relu')
        self.expand_dims = Lambda(lambda x: K.expand_dims(x, axis=1))
        self.tile = Lambda(lambda x: K.tile(x, (1, encoder_timesteps, 1, 1)))
        self.batch_dot = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[0, 1]))
        self.reshape = Lambda(lambda x: K.reshape(x, (-1, self.encoder_timesteps, self.num_nodes*self.conv_units)))
        self.slice = Lambda(slice,output_shape=(self.decoder_timesteps, self.num_nodes*self.conv_units),arguments={'index':self.encoder_timesteps-self.decoder_timesteps})
        self.encoder_lstm = LSTM(self.encoder_hidden_units, return_state=True)
        #self.encoder_gru = GRU(self.encoder_hidden_units, return_state=True)
        #self.encoder_rnn = SimpleRNN(self.encoder_hidden_units, return_state=True)
        self.decoder_lstm = LSTM(self.decoder_hidden_units, return_sequences=True, return_state=True)
        #self.decoder_gru = GRU(self.decoder_hidden_units, return_sequences=True, return_state=True)
        #self.decoder_rnn = SimpleRNN(self.decoder_hidden_units, return_sequences=True, return_state=True)
        #self.decoder_lstm = LSTM(self.decoder_hidden_units, return_state=True)
        self.dense = Dense(self.num_nodes*self.num_features)
        
    def train_model(self, input_shape, adjacency_shape):
        H_0 = Input(shape=(input_shape))                #[batch_size, t, N, F]
        A_0 = Input(shape=(adjacency_shape))
        #A_1 = Input(shape=(adjacency_shape))
        
        # Graph Fusion
        #F = self.graph_fusion([A_0, A_1])               #[batch_size, N, N]
        F = A_0
        F = self.expand_dims(F)                          #[batch_size, 1, N, N]
        F = self.tile(F)                                 #[batch_size, t, N, N]
        
        # Graph Convolution
        H_1 = TimeDistributed(self.graph_convolution)(self.batch_dot([F, H_0]))
        H_1 = self.reshape(H_1)         #[batch_size, t, N*conv_units]
        
        #NONE FUSION
        #H_1 = A_0
       
        
        encoder_outputs, self.encoder_state_h, self.encoder_state_c = self.encoder_lstm(H_1) 
        #encoder_outputs, self.encoder_state_h = self.encoder_gru(H_1) 
        #encoder_outputs, self.encoder_state_h = self.encoder_rnn(H_1) 
        encoder_states = [self.encoder_state_h, self.encoder_state_c]
        #encoder_states = [self.encoder_state_h]
        decoder_inputs = self.slice(H_1)
        decoder_outputs, self.decoder_state_h, self.decoder_state_c = self.decoder_lstm(decoder_inputs, initial_state=encoder_states)
        #decoder_outputs, self.decoder_state_h = self.decoder_gru(decoder_inputs, initial_state=encoder_states)
        #decoder_outputs, self.decoder_state_h = self.decoder_rnn(decoder_inputs, initial_state=encoder_states)
        decoder_outputs = TimeDistributed(self.dense)(decoder_outputs) # TimeDistributed added!
        
        #model = Model([H_0, A_0, A_1], decoder_outputs)
        model = Model([H_0, A_0], decoder_outputs)
        model.compile(optimizer='adam', loss='mse')
        
        return model






# In[ ]:




