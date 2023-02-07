#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.layers.merge import _Merge
from keras import activations, initializers, constraints, regularizers
from keras.engine import Layer
from keras.layers import *
import keras.backend as K
import numpy as np

# In[2]:


class GraphFusion(_Merge):
    """
    Layer that performs weighted sum on the input tensor with a learnable scaling parameter.
    
    Arguments:
        weight_shape: (N, N) the same as adjacency matrix
        weight_num: K the number of weight matrix 
    
    Input shape:
        a list of tensors, all of the same shape (batch size, N, N)
    
    Returns:
        a single scaled tensor (also of the same shape)
    
    """
    def __init__(self, activation='softmax', **kwargs):
        super(GraphFusion, self).__init__(**kwargs)
        self.activation = activations.get(activation)
        
    def build(self, input_shape):
        super(GraphFusion, self).build(input_shape)
        weight_shape = input_shape[0][1:]
        weight_num = len(input_shape)
        #self.weights = [self.activation(K.variable(np.ones(weight_shape), name='weight_{}'.format(str(i)))) for i in range(0,weight_num)]
        #self.trainable_weights = self.weights
        self.weight_0 = K.variable(np.ones(weight_shape), name='fusion_weight_0') 
        self.weight_1 = K.variable(np.ones(weight_shape), name='fusion_weight_1')
        self.trainable_weights = [self.weight_0, self.weight_1]
        
    def _merge_function(self, inputs):
        output=self.activation(self.weight_0)*inputs[0]+self.activation(self.weight_1)*inputs[1]
        return output


# In[3]:


class GraphConvolution(Layer):
    def __init__(self, units, 
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

    def compute_output_shape(self, input_shape):
        
        output_shape = (input_shape[0], input_shape[1], self.units)
        return output_shape
    
    def build(self, input_shape):
        self.feature_dim = input_shape[-1]

        self.kernel = self.add_weight(shape=(self.feature_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.built = True
    
    def call(self, inputs, mask=None):  
        output = K.dot(inputs, self.kernel)
        if self.bias:
            output += self.bias
        return self.activation(output)
    
    def get_config(self):
        config = {'units': self.units,
                  #'support': self.support,
                  'activation': activations.serialize(self.activation),
                  'use_bias': self.use_bias,
                  'kernel_initializer': initializers.serialize(
                      self.kernel_initializer),
                  'bias_initializer': initializers.serialize(
                      self.bias_initializer),
                  'kernel_regularizer': regularizers.serialize(
                      self.kernel_regularizer),
                  'bias_regularizer': regularizers.serialize(
                      self.bias_regularizer),
                  'activity_regularizer': regularizers.serialize(
                      self.activity_regularizer),
                  'kernel_constraint': constraints.serialize(
                      self.kernel_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint)
        }

        base_config = super(GraphConvolution, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# In[ ]:




