#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 09:47:01 2017

@author: soujanyaporia
"""

import sys
import numpy as np
from sklearn.cross_validation import StratifiedKFold
import io
import re
import pandas as pd
import csv
import scipy.stats as stats
from keras import backend as K
from keras.layers.core import Lambda
from keras.engine.topology import Layer
from keras.models import Sequential
from keras.layers import Input,Dense,GRU,LSTM,Concatenate,Dropout,Activation,Add, Masking, Concatenate, Dot, RepeatVector, Permute, Multiply
from keras.layers.pooling import AveragePooling1D,MaxPooling1D
from keras.layers.core import Flatten
from keras.callbacks import EarlyStopping
from keras.layers.convolutional import Conv1D
from keras.models import Model
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.layers.core import Reshape
from keras.backend import shape
from keras.utils import plot_model
from keras.layers.merge import Multiply,Concatenate
from keras.optimizers import RMSprop,Adadelta,Adam
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

#from keras import initializations

MAX_LENGTH=80
EMBEDDING_SIZE=300
n_folds = 10
np.random.seed(int(sys.argv[2]))

def load_data_multitask():
    data=pd.read_csv("../data/text_and_annorations.csv")
    sentence=data["Text"]
    label_sentiment=data["Default_Polarity"]
    label_sarcasm=data["Sarcasm"]
    return sentence,label_sentiment,label_sarcasm
def load_data_multilabel():
    data=pd.read_csv("../data/text_and_annorations-4-way.csv")
    sentence=data["Text"]
    labels=data["Multi_Labels"]
    return sentence,labels
def get_embeddings(file2):
    embeddings = {}
    with io.open(file2, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            values = line.split(' ')
            word = values[0]
            embedding = np.asarray(values[1:], dtype='float32')
            embeddings[word] = embedding
    print "embeddings loaded...."
    return embeddings

def preprocessing(embeddings,text,labels_senti,labels_sarcasm):
    
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(text)
    
    mod_text = tokenizer.texts_to_sequences(text)
    word_index = tokenizer.word_index
    nb_words = len(word_index)
    
    we = np.zeros((nb_words + 1, EMBEDDING_SIZE))
    
    for word, i in word_index.items():
        embedding = embeddings.get(word)
        if embedding is not None:
            we[i] = embedding
        else:
            we[i] = np.random.uniform(-0.25,0.25,EMBEDDING_SIZE)
            
    mod_text = pad_sequences(mod_text, maxlen=MAX_LENGTH)
    
    labels_senti = np.array(labels_senti, dtype=int)
    labels_sarcasm = np.array(labels_sarcasm, dtype=int)

    return mod_text,labels_senti,labels_sarcasm,we,nb_words

def preprocessing_multilabel(embeddings,text,labels):
    
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(text)
    
    mod_text = tokenizer.texts_to_sequences(text)
    word_index = tokenizer.word_index
    nb_words = len(word_index)
    
    we = np.zeros((nb_words + 1, EMBEDDING_SIZE))
    
    for word, i in word_index.items():
        embedding = embeddings.get(word)
        if embedding is not None:
            we[i] = embedding
        else:
            we[i] = np.random.uniform(-0.25,0.25,EMBEDDING_SIZE)

    mod_text = pad_sequences(mod_text, maxlen=MAX_LENGTH)

    labels = np.array(labels, dtype=int)

    return mod_text,labels,we,nb_words

class NeuralTensorLayer(Layer):

  def __init__(self, output_dim, input_dim=None, **kwargs):
    self.output_dim = output_dim #k
    self.input_dim = input_dim   #d
    if self.input_dim:
      kwargs['input_shape'] = (self.input_dim,)
    super(NeuralTensorLayer, self).__init__(**kwargs)

  def build(self, input_shape):
    mean = 0.0
    std = 1.0
    # W : k*d*d
    k = self.output_dim
    d = self.input_dim
    initial_W_values = stats.truncnorm.rvs(-2 * std, 2 * std, loc=mean, scale=std, size=(k,d,d))
    initial_V_values = stats.truncnorm.rvs(-2 * std, 2 * std, loc=mean, scale=std, size=(2*d,k))
    self.W = K.variable(initial_W_values)
    self.V = K.variable(initial_V_values)
    self.b = K.zeros((self.input_dim,))
    self.trainable_weights = [self.W, self.V, self.b]

  def call(self, inputs, mask=None):
    if type(inputs) is not list or len(inputs) <= 1:
      raise Exception('BilinearTensorLayer must be called on a list of tensors '
                      '(at least 2). Got: ' + str(inputs))
    e1 = inputs[0]
    e2 = inputs[1]
    batch_size = K.shape(e1)[0]
    k = self.output_dim

    feed_forward_product = K.dot(K.concatenate([e1,e2], axis=1), self.V)
    bilinear_tensor_products = []
    for i in range(k):
      btp = K.sum((e2 * K.dot(e1, self.W[i])) + self.b, axis=1)
      bilinear_tensor_products.append(btp)
    result = K.tanh(K.reshape(K.concatenate(bilinear_tensor_products, axis=0), (batch_size, k)) + feed_forward_product)
    return result

  def compute_output_shape(self, input_shape):
    batch_size = input_shape[0][0]
    return (batch_size, self.output_dim)

def create_model_multitask(we,nb_words):
    """
    Simple multitask with tensor fusion
    """
    text = Input(shape=(MAX_LENGTH,))
    e1 = Embedding(nb_words + 1, 
                 EMBEDDING_SIZE, 
                 weights=[we], 
                 input_length=MAX_LENGTH, 
                 trainable=False)(text)
    lstm1 = GRU(500, activation='tanh', return_sequences = False, dropout=0.25, name='bi_lstm1')(e1)
    dropout1=Dropout(0.4)(lstm1)
    dense1=Dense(300,activation='relu',name='dense1')(dropout1)
    dense2=Dense(300,activation='relu',name='dense2')(dropout1)
    
    ntn_output = NeuralTensorLayer(output_dim=100, input_dim=300)([dense1, dense2])

    # What to do with NTN output? For now, just appending to sarcasm dense
    #merged1 = Concatenate(axis=1)([dense1, ntn_output])
    merged2 = Concatenate(axis=1)([dense2, ntn_output])


    #print "HERE" , ntn_output._keras_shape
    predictions1 = Dense(2, activation='softmax')(dense1)
    predictions2 = Dense(2, activation='softmax')(merged2)
    
    model = Model(inputs=text, outputs=[predictions1, predictions2])
    model.compile(optimizer='adam', loss='categorical_crossentropy',
              loss_weights=[1., 1.],metrics=['accuracy'])
    
    return model
def create_model_multitask_varriant13(we,nb_words):
    """
    Simple multitask without tensor fusion
    """
    text = Input(shape=(MAX_LENGTH,))
    e1 = Embedding(nb_words + 1, 
                 EMBEDDING_SIZE, 
                 weights=[we], 
                 input_length=MAX_LENGTH, 
                 trainable=False)(text)
    lstm1 = GRU(500, activation='tanh', return_sequences = False, dropout=0.25, name='bi_lstm1')(e1)
    dropout1=Dropout(0.4)(lstm1)
    dense1=Dense(300,activation='relu',name='dense1')(dropout1)
    dense2=Dense(300,activation='relu',name='dense2')(dropout1)
    
    #print "HERE" , ntn_output._keras_shape
    predictions1 = Dense(2, activation='softmax')(dense1)
    predictions2 = Dense(2, activation='softmax')(dense2)
    
    model = Model(inputs=text, outputs=[predictions1, predictions2])
    model.compile(optimizer='adam', loss='categorical_crossentropy',
              loss_weights=[1., 1.],metrics=['accuracy'])
    
    return model
def create_model_multitask_varriant14(we,nb_words):
    """
    Simple multitask without tensor fusion to test the effect of NTN
    """
    text = Input(shape=(MAX_LENGTH,))
    e1 = Embedding(nb_words + 1, 
                 EMBEDDING_SIZE, 
                 weights=[we], 
                 input_length=MAX_LENGTH, 
                 trainable=False)(text)
    lstm1 = GRU(500, activation='tanh', return_sequences = False, dropout=0.25, name='bi_lstm1')(e1)
    dropout1=Dropout(0.4)(lstm1)
    dense1=Dense(300,activation='relu',name='dense1')(dropout1)
    dense2=Dense(300,activation='relu',name='dense2')(dropout1)
    
    dense3=Dense(300,activation='tanh',name='dense3')

    out1=dense3(dense1)
    out2=dense3(dense2)

    dropout3=Dropout(0.4)(out1)
    dropout4=Dropout(0.4)(out2)
    #print "HERE" , ntn_output._keras_shape
    predictions1 = Dense(2, activation='softmax')(dropout3)
    predictions2 = Dense(2, activation='softmax')(dropout4)
    
    model = Model(inputs=text, outputs=[predictions1, predictions2])
    model.compile(optimizer='adam', loss='categorical_crossentropy',
              loss_weights=[1., 1.],metrics=['accuracy'])
    
    return model
def create_model_multitask_varriant15(we,nb_words):
    """
    Simple multitask with tensor fusion but LSTM is non-shared. This function tests the effect of shared LSTM
    """
    text = Input(shape=(MAX_LENGTH,))
    e1 = Embedding(nb_words + 1, 
                 EMBEDDING_SIZE, 
                 weights=[we], 
                 input_length=MAX_LENGTH, 
                 trainable=False)(text)
    lstm1 = GRU(500, activation='tanh', return_sequences = False, dropout=0.25, name='bi_lstm1')(e1)
    lstm2 = GRU(500, activation='tanh', return_sequences = False, dropout=0.25, name='bi_lstm2')(e1)
    
    dropout1=Dropout(0.4)(lstm1)
    dropout2=Dropout(0.4)(lstm2)
    
    dense1=Dense(300,activation='relu',name='dense1')(dropout1)
    dense2=Dense(300,activation='relu',name='dense2')(dropout2)
    
    ntn_output = NeuralTensorLayer(output_dim=100, input_dim=300)([dense1, dense2])

    # What to do with NTN output? For now, just appending to sarcasm dense
    #merged1 = Concatenate(axis=1)([dense1, ntn_output])
    merged2 = Concatenate(axis=1)([dense2, ntn_output])


    #print "HERE" , ntn_output._keras_shape
    predictions1 = Dense(2, activation='softmax')(dense1)
    predictions2 = Dense(2, activation='softmax')(merged2)
    
    model = Model(inputs=text, outputs=[predictions1, predictions2])
    model.compile(optimizer='adam', loss='categorical_crossentropy',
              loss_weights=[1., 1.],metrics=['accuracy'])
    
    return model
def create_model_multitask_varriant2(we,nb_words):
    """
    Multitask with attention and private/public layer
    """
    text = Input(shape=(MAX_LENGTH,))
    e1 = Embedding(nb_words + 1, 
                 EMBEDDING_SIZE, 
                 weights=[we], 
                 input_length=MAX_LENGTH, 
                 trainable=False)(text)
    lstm1 = GRU(500, activation='tanh', return_sequences = True, dropout=0.25, name='bi_lstm1')(e1)
    lstm2 = GRU(500, activation='tanh', return_sequences = True, dropout=0.25, name='bi_lstm2')(e1)
    lstm3 = GRU(500, activation='tanh', return_sequences = True, dropout=0.25, name='bi_lstm3')(e1)

    dropout1=Dropout(0.1)(lstm1)
    dropout2=Dropout(0.1)(lstm2)
    dropout3=Dropout(0.1)(lstm3)
    
    dense1=TimeDistributed(Dense(300,activation='relu',name='dense1'))(dropout1)
    dense2=TimeDistributed(Dense(300,activation='relu',name='dense2'))(dropout2)
    dense3=TimeDistributed(Dense(300,activation='relu',name='dense3'))(dropout3)

    merged13 = Concatenate(axis=1)([dense1, dense3])
    merged23 = Concatenate(axis=1)([dense2, dense3])

    M1 = Activation('tanh', name='M_tanh_activation1')(merged13)
    M1 = Dropout(0.5)(M1)
    w_M1 = TimeDistributed(Dense(1, name='fc1'))(M1)
    alpha1 = Activation('softmax', name = 'alpha_layer1')(w_M1)
    r1 = Dot(axes = 1)([alpha1, merged13])
    r1 = Flatten(name='r_flatten1')(r1)

    # Wp_r = Wp * r (matrix * matrix)
    Wp_r1 = Dense(300, use_bias=False, name='fc2')(r1)

    M2 = Activation('tanh', name='M_tanh_activation2')(merged23)
    M2 = Dropout(0.5)(M2)
    w_M2 = TimeDistributed(Dense(1, name='fc3'))(M2)
    alpha2 = Activation('softmax', name = 'alpha_layer2')(w_M2)
    r2 = Dot(axes = 1)([alpha2, merged23])
    r2 = Flatten(name='r_flatten2')(r2)

    # Wp_r = Wp * r (matrix * matrix)
    Wp_r2 = Dense(300, use_bias=False, name='fc4')(r2)
    
    ntn_output = NeuralTensorLayer(output_dim=100, input_dim=300)([Wp_r1, Wp_r2])

    # What to do with NTN output? For now, just appending to sarcasm dense
    #merged1 = Concatenate(axis=1)([dense1, ntn_output])
    merged = Concatenate(axis=1)([Wp_r2, ntn_output])


    #print "HERE" , ntn_output._keras_shape
    predictions1 = Dense(2, activation='softmax')(Wp_r1)
    predictions2 = Dense(2, activation='softmax')(merged)
    
    model = Model(inputs=text, outputs=[predictions1, predictions2])
    model.compile(optimizer='adam', loss='categorical_crossentropy',
              loss_weights=[1., 1.],metrics=['accuracy'])
    
    return model
def create_model_multitask_varriant3(we,nb_words):
    """
    Multitask with attention and private/public layer and ATAE type model
    """
    text = Input(shape=(MAX_LENGTH,))
    e1 = Embedding(nb_words + 1, 
                 EMBEDDING_SIZE, 
                 weights=[we], 
                 input_length=MAX_LENGTH, 
                 trainable=False)(text)
    lstm1 = GRU(500, activation='tanh', return_sequences = True, dropout=0.25, name='bi_lstm1')(e1)
    lstm2 = GRU(500, activation='tanh', return_sequences = True, dropout=0.25, name='bi_lstm2')(e1)
    lstm3 = GRU(500, activation='tanh', return_sequences = True, dropout=0.25, name='bi_lstm3')(e1)

    dropout1=Dropout(0.1)(lstm1)
    dropout2=Dropout(0.1)(lstm2)
    dropout3=Dropout(0.1)(lstm3)
    
    dense1=TimeDistributed(Dense(300,activation='relu',name='dense1'))(dropout1)
    dense2=TimeDistributed(Dense(300,activation='relu',name='dense2'))(dropout2)
    dense3=TimeDistributed(Dense(300,activation='relu',name='dense3'))(dropout3)

    merged13 = Concatenate(axis=1)([dense1, dense3])
    merged23 = Concatenate(axis=1)([dense2, dense3])

    M1 = Activation('tanh', name='M_tanh_activation1')(merged13)
    M1 = Dropout(0.5)(M1)
    w_M1 = TimeDistributed(Dense(1, name='fc1'))(M1)
    alpha1 = Activation('softmax', name = 'alpha_layer1')(w_M1)
    r1 = Dot(axes = 1)([alpha1, merged13])
    r1 = Flatten(name='r_flatten1')(r1)

    # Wp_r = Wp * r (matrix * matrix)
    Wp_r1 = Dense(300, use_bias=False, name='fc2')(r1)

    M2 = Activation('tanh', name='M_tanh_activation2')(merged23)
    M2 = Dropout(0.5)(M2)
    w_M2 = TimeDistributed(Dense(1, name='fc3'))(M2)
    alpha2 = Activation('softmax', name = 'alpha_layer2')(w_M2)
    r2 = Dot(axes = 1)([alpha2, merged23])
    r2 = Flatten(name='r_flatten2')(r2)

    # Wp_r = Wp * r (matrix * matrix)
    Wp_r2 = Dense(300, use_bias=False, name='fc4')(r2)
    
    def slice(input):
        return input[:,-1:, :]

    def slice_dims(input_shape):
        timesteps = 1
        return (input_shape[0], 1, input_shape[2])

    Hn1 = Lambda(slice, output_shape=slice_dims)(dense1)
    Hn1 = Flatten(name='hn_flatten1')(Hn1)

    # Wx_Hn = Wx * Hn (matrix * vector)
    Wx_Hn1 = Dense(300, use_bias=False, name='fc5')(Hn1)

    # h_star = tanh(Wp_r + WxHn)
    Wp_r_plus_Wx_Hn1 = Add()([Wp_r1, Wx_Hn1])
    h_star1 = Activation('tanh', name='tanh_activation1')(Wp_r_plus_Wx_Hn1)
    h_star1 = Dropout(0.5)(h_star1)

    Hn2 = Lambda(slice, output_shape=slice_dims)(dense2)
    Hn2 = Flatten(name='hn_flatten2')(Hn2)

    # Wx_Hn = Wx * Hn (matrix * vector)
    Wx_Hn2 = Dense(300, use_bias=False, name='fc6')(Hn2)

    # h_star = tanh(Wp_r + WxHn)
    Wp_r_plus_Wx_Hn2 = Add()([Wp_r2, Wx_Hn2])
    h_star2 = Activation('tanh', name='tanh_activation2')(Wp_r_plus_Wx_Hn2)
    h_star2 = Dropout(0.5)(h_star2)

    ntn_output = NeuralTensorLayer(output_dim=100, input_dim=300)([h_star1, h_star2])

    # What to do with NTN output? For now, just appending to sarcasm dense
    #merged1 = Concatenate(axis=1)([dense1, ntn_output])
    merged = Concatenate(axis=1)([h_star2, ntn_output])


    #print "HERE" , ntn_output._keras_shape
    predictions1 = Dense(2, activation='softmax')(h_star1)
    predictions2 = Dense(2, activation='softmax')(merged)
    
    model = Model(inputs=text, outputs=[predictions1, predictions2])
    model.compile(optimizer='adam', loss='categorical_crossentropy',
              loss_weights=[1., 1.],metrics=['accuracy'])
    
    return model
def create_model_multitask_varriant4(we,nb_words):
    """
    Multitask with attention and ATAE type model
    """
    text = Input(shape=(MAX_LENGTH,))
    e1 = Embedding(nb_words + 1, 
                 EMBEDDING_SIZE, 
                 weights=[we], 
                 input_length=MAX_LENGTH, 
                 trainable=False)(text)
    lstm1 = GRU(500, activation='tanh', return_sequences = True, dropout=0.25, name='bi_lstm1')(e1)

    dropout1=Dropout(0.1)(lstm1)
    
    dense1=TimeDistributed(Dense(300,activation='relu',name='dense1'))(dropout1)
    dense2=TimeDistributed(Dense(300,activation='relu',name='dense2'))(dropout1)


    M1 = Activation('tanh', name='M_tanh_activation1')(dense1)
    M1 = Dropout(0.5)(M1)
    w_M1 = TimeDistributed(Dense(1, name='fc1'))(M1)
    alpha1 = Activation('softmax', name = 'alpha_layer1')(w_M1)
    r1 = Dot(axes = 1)([alpha1, dense1])
    r1 = Flatten(name='r_flatten1')(r1)

    # Wp_r = Wp * r (matrix * matrix)
    Wp_r1 = Dense(300, use_bias=False, name='fc2')(r1)

    M2 = Activation('tanh', name='M_tanh_activation2')(dense2)
    M2 = Dropout(0.5)(M2)
    w_M2 = TimeDistributed(Dense(1, name='fc3'))(M2)
    alpha2 = Activation('softmax', name = 'alpha_layer2')(w_M2)
    r2 = Dot(axes = 1)([alpha2, dense2])
    r2 = Flatten(name='r_flatten2')(r2)

    # Wp_r = Wp * r (matrix * matrix)
    Wp_r2 = Dense(300, use_bias=False, name='fc4')(r2)
    
    def slice(input):
        return input[:,-1:, :]

    def slice_dims(input_shape):
        timesteps = 1
        return (input_shape[0], 1, input_shape[2])

    Hn1 = Lambda(slice, output_shape=slice_dims)(dense1)
    Hn1 = Flatten(name='hn_flatten1')(Hn1)

    # Wx_Hn = Wx * Hn (matrix * vector)
    Wx_Hn1 = Dense(300, use_bias=False, name='fc5')(Hn1)

    # h_star = tanh(Wp_r + WxHn)
    Wp_r_plus_Wx_Hn1 = Add()([Wp_r1, Wx_Hn1])
    h_star1 = Activation('tanh', name='tanh_activation1')(Wp_r_plus_Wx_Hn1)
    h_star1 = Dropout(0.5)(h_star1)

    Hn2 = Lambda(slice, output_shape=slice_dims)(dense2)
    Hn2 = Flatten(name='hn_flatten2')(Hn2)

    # Wx_Hn = Wx * Hn (matrix * vector)
    Wx_Hn2 = Dense(300, use_bias=False, name='fc6')(Hn2)

    # h_star = tanh(Wp_r + WxHn)
    Wp_r_plus_Wx_Hn2 = Add()([Wp_r2, Wx_Hn2])
    h_star2 = Activation('tanh', name='tanh_activation2')(Wp_r_plus_Wx_Hn2)
    h_star2 = Dropout(0.5)(h_star2)

    ntn_output = NeuralTensorLayer(output_dim=100, input_dim=300)([h_star1, h_star2])

    # What to do with NTN output? For now, just appending to sarcasm dense
    #merged1 = Concatenate(axis=1)([dense1, ntn_output])
    merged = Concatenate(axis=1)([h_star2, ntn_output])


    #print "HERE" , ntn_output._keras_shape
    predictions1 = Dense(2, activation='softmax')(h_star1)
    predictions2 = Dense(2, activation='softmax')(merged)
    
    model = Model(inputs=text, outputs=[predictions1, predictions2])
    model.compile(optimizer='adam', loss='categorical_crossentropy',
              loss_weights=[1., 1.],metrics=['accuracy'])
    
    return model
def attention(inputs):
    '''
    input : (None, timesteps, dimension)
    output : (None, 1, dimension)
    alpha vector : (None, 1, timestep)
    '''

    dim = inputs._keras_shape[2]
    timesteps = inputs._keras_shape[1]

    inp = Permute((2,1))(inputs) #(none, dim, timesteps)
    M = Dense(1, activation='tanh')(inp) # (None, dim , 1) -> middle hidden layer in the aux attention network
    M = Permute((2,1))(M) # (None,1, dim)
    alpha = Dense(timesteps)(M) # (None, 1, timesteps) --> ALPHA ATTENTION weight vector
    alpha = Activation('softmax')(alpha) # (None, 1, timestep) --> ALPHA ATTENTION weight vector 
    print alpha._keras_shape
                
    r = Dot((2,1))([alpha, inputs])          # weighing input as per attention
    return r, alpha
def create_model_multitask_varriant11(we,nb_words):
    """
    Multitask with non-shared-attention, different attention implementation
    """

    text = Input(shape=(MAX_LENGTH,))
    e1 = Embedding(nb_words + 1, 
                 EMBEDDING_SIZE, 
                 weights=[we], 
                 input_length=MAX_LENGTH, 
                 trainable=False)(text)
    lstm1 = GRU(500, activation='tanh', return_sequences = True, dropout=0.25, name='bi_lstm1')(e1)

    dropout1=Dropout(0.4)(lstm1)
    
    dense1=TimeDistributed(Dense(300,activation='relu',name='dense1'))(dropout1)
    dense2=TimeDistributed(Dense(300,activation='relu',name='dense2'))(dropout1)

    r1,alpha1=attention(dense1)
    r1 = Flatten()(r1)
    r2,alpha2=attention(dense2)
    r2 = Flatten()(r2)
    ntn_output = NeuralTensorLayer(output_dim=100, input_dim=300)([r1, r2])

    # What to do with NTN output? For now, just appending to sarcasm dense
    #merged1 = Concatenate(axis=1)([Wp_r1, ntn_output])
    merged2 = Concatenate(axis=1)([r2, ntn_output])


    #print "HERE" , ntn_output._keras_shape
    predictions1 = Dense(2, activation='softmax')(r1)
    predictions2 = Dense(2, activation='softmax')(merged2)
    
    model = Model(inputs=text, outputs=[predictions1, predictions2])
    model.compile(optimizer='adam', loss='categorical_crossentropy',
              loss_weights=[1., 1.],metrics=['accuracy'])
    model.summary()
    return model
def create_model_multitask_varriant12(we,nb_words):
    """
    Multitask with shared-attention, different attention implementation
    """

    text = Input(shape=(MAX_LENGTH,))
    e1 = Embedding(nb_words + 1, 
                 EMBEDDING_SIZE, 
                 weights=[we], 
                 input_length=MAX_LENGTH, 
                 trainable=False)(text)
    lstm1 = GRU(500, activation='tanh', return_sequences = True, dropout=0.25, name='bi_lstm1')(e1)

    dropout1=Dropout(0.4)(lstm1)
    
    dense1=TimeDistributed(Dense(300,activation='relu',name='dense1'))(dropout1)
    dense2=TimeDistributed(Dense(300,activation='relu',name='dense2'))(dropout1)

    dim = dense1._keras_shape[2]
    timesteps = dense1._keras_shape[1]

    op1 = Permute((2,1)) #(none, dim, timesteps)
    op2 = Dense(1, activation='tanh') # (None, dim , 1) -> middle hidden layer in the aux attention network
    op3 = Permute((2,1)) # (None,1, dim)
    op4 = Dense(timesteps) # (None, 1, timesteps) --> ALPHA ATTENTION weight vector
    op5 = Activation('softmax') # (None, 1, timestep) --> ALPHA ATTENTION weight vector 
                

    M1 = op1(dense1)
    M1 = op1(M1)
    M1 = op2(M1)
    M1 = op3(M1)
    M1 = op4(M1)
    alpha1 = op5(M1)
    r1 = Dot((2,1))([alpha1, dense1])  
    r1 = Flatten()(r1)

    M2 = op1(dense2)
    M2 = op1(M2)
    M2 = op2(M2)
    M2 = op3(M2)
    M2 = op4(M2)
    alpha2 = op5(M2)
    r2 = Dot((2,1))([alpha2, dense2])  
    r2 = Flatten()(r2)
    
    ntn_output = NeuralTensorLayer(output_dim=100, input_dim=300)([r1, r2])

    # What to do with NTN output? For now, just appending to sarcasm dense
    #merged1 = Concatenate(axis=1)([Wp_r1, ntn_output])
    merged2 = Concatenate(axis=1)([r2, ntn_output])


    #print "HERE" , ntn_output._keras_shape
    predictions1 = Dense(2, activation='softmax')(r1)
    predictions2 = Dense(2, activation='softmax')(merged2)
    
    model = Model(inputs=text, outputs=[predictions1, predictions2])
    model.compile(optimizer='adam', loss='categorical_crossentropy',
              loss_weights=[1., 1.],metrics=['accuracy'])
    model.summary()
    return model
def create_model_multitask_varriant5(we,nb_words):
    """
    Multitask with shared-attention
    """

    text = Input(shape=(MAX_LENGTH,))
    e1 = Embedding(nb_words + 1,
                 EMBEDDING_SIZE,
                 weights=[we],
                 input_length=MAX_LENGTH,
                 trainable=False)(text)
    lstm1 = GRU(500, activation='tanh', return_sequences = True, dropout=0.25, name='bi_lstm1')(e1)

    dropout1=Dropout(0.4)(lstm1)

    dense1=TimeDistributed(Dense(300,activation='relu',name='dense1'))(dropout1)
    dense2=TimeDistributed(Dense(300,activation='relu',name='dense2'))(dropout1)

    op1 = Permute((2, 1))
    op2 = Dense(MAX_LENGTH, activation='softmax')
    op3 = Permute((2, 1))
    op4 = Multiply()

    M1 = Activation('tanh', name='M_tanh_activation1')(dense1)
    M1 = Dropout(0.4)(dense1)
    a = op1(M1)
    a = op2(a)
    a = op3(a)
    out1 = op4([a,dense1])

    print out1._keras_shape
    r1 = Lambda(lambda x: K.mean(x, axis=1), name='sentence_vec_1', output_shape=(300,))(out1)

    print r1._keras_shape

    # Wp_r = Wp * r (matrix * matrix)
    Wp_r1 = Dense(300, name='fc2')(r1)

    M2 = Activation('tanh', name='M_tanh_activation2')(dense2)
    M2 = Dropout(0.4)(M2)
    a1 = op1(dense2)
    a1 = op2(a1)
    a1 = op3(a1)
    out2 = op4([a1,dense2])
    r2 = Lambda(lambda x: K.mean(x, axis=1), name='sentence_vec_2',output_shape=(300,))(out2)

    print r2._keras_shape

    # Wp_r = Wp * r (matrix * matrix)
    Wp_r2 = Dense(300, name='fc4')(r2)

    #print Wp_r1._keras_shape
    #print Wp_r2._keras_shape
    ntn_output = NeuralTensorLayer(output_dim=100, input_dim=300)([Wp_r1, Wp_r2])

    # What to do with NTN output? For now, just appending to sarcasm dense
    #merged1 = Concatenate(axis=1)([Wp_r1, ntn_output])
    merged2 = Concatenate(axis=1)([r2, ntn_output])


    #print "HERE" , ntn_output._keras_shape
    predictions1 = Dense(2, activation='softmax')(Wp_r1)
    predictions2 = Dense(2, activation='softmax')(merged2)

    model = Model(inputs=text, outputs=[predictions1, predictions2])
    model.compile(optimizer='adam', loss='categorical_crossentropy',
              loss_weights=[1., 1.],metrics=['accuracy'])
    model.summary()
    return model

def create_model_multitask_varriant6(we,nb_words):
    
    """
    Multitask with attention
    """
    text = Input(shape=(MAX_LENGTH,))
    e1 = Embedding(nb_words + 1, 
                 EMBEDDING_SIZE, 
                 weights=[we], 
                 input_length=MAX_LENGTH, 
                 trainable=False)(text)
    lstm1 = GRU(500, activation='tanh', return_sequences = True, dropout=0.25, name='bi_lstm1')(e1)

    dropout1=Dropout(0.4)(lstm1)
    
    dense1=TimeDistributed(Dense(300,activation='relu',name='dense1'))(dropout1)
    dense2=TimeDistributed(Dense(300,activation='relu',name='dense2'))(dropout1)

    op11 = Permute((2, 1))
    op12 = Dense(MAX_LENGTH, activation='softmax')
    op13 = Permute((2, 1))
    op14 = Multiply()

    op21 = Permute((2, 1))
    op22 = Dense(MAX_LENGTH, activation='softmax')
    op23 = Permute((2, 1))
    op24 = Multiply()

    M1 = Activation('tanh', name='M_tanh_activation1')(dense1)
    M1 = Dropout(0.4)(dense1)
    a = op11(M1)
    a = op12(a)
    a = op13(a)
    out1 = op24([a,dense1])

    print out1._keras_shape
    r1 = Lambda(lambda x: K.mean(x, axis=1), name='sentence_vec_1', output_shape=(300,))(out1)

    print r1._keras_shape

    # Wp_r = Wp * r (matrix * matrix)
    Wp_r1 = Dense(300, use_bias=False, name='fc2')(r1)

    M2 = Activation('tanh', name='M_tanh_activation2')(dense2)
    M2 = Dropout(0.4)(M2)
    a1 = op21(dense2)
    a1 = op22(a1)
    a1 = op23(a1)
    out2 = op24([a1,dense2])
    r2 = Lambda(lambda x: K.mean(x, axis=1), name='sentence_vec_2',output_shape=(300,))(out2)

    print r2._keras_shape

    # Wp_r = Wp * r (matrix * matrix)
    Wp_r2 = Dense(300, use_bias=False, name='fc4')(r2)

    #print Wp_r1._keras_shape
    #print Wp_r2._keras_shape
    ntn_output = NeuralTensorLayer(output_dim=100, input_dim=300)([Wp_r1, Wp_r2])

    # What to do with NTN output? For now, just appending to sarcasm dense
    #merged1 = Concatenate(axis=1)([dense1, ntn_output])
    merged = Concatenate(axis=1)([Wp_r2, ntn_output])


    #print "HERE" , ntn_output._keras_shape
    predictions1 = Dense(2, activation='softmax')(Wp_r1)
    predictions2 = Dense(2, activation='softmax')(merged)
    
    model = Model(inputs=text, outputs=[predictions1, predictions2])
    model.compile(optimizer='adam', loss='categorical_crossentropy',
              loss_weights=[1., 1.],metrics=['accuracy'])
    
    return model    
def create_model_multitask_varriant7(we,nb_words):
    """
    Multitask with shared-attention and ATAE type
    """
    text = Input(shape=(MAX_LENGTH,))
    e1 = Embedding(nb_words + 1, 
                 EMBEDDING_SIZE, 
                 weights=[we], 
                 input_length=MAX_LENGTH, 
                 trainable=False)(text)
    lstm1 = GRU(500, activation='tanh', return_sequences = True, dropout=0.25, name='bi_lstm1')(e1)

    dropout1=Dropout(0.4)(lstm1)

    dense1=TimeDistributed(Dense(300,activation='relu',name='dense1'))(dropout1)
    dense2=TimeDistributed(Dense(300,activation='relu',name='dense2'))(dropout1)

    op1 = Permute((2, 1))
    op2 = Dense(MAX_LENGTH, activation='softmax')
    op3 = Permute((2, 1))
    op4 = Multiply()

    M1 = Activation('tanh', name='M_tanh_activation1')(dense1)
    M1 = Dropout(0.4)(dense1)
    a = op1(M1)
    a = op2(a)
    a = op3(a)
    out1 = op4([a,dense1])

    print out1._keras_shape
    r1 = Lambda(lambda x: K.mean(x, axis=1), name='sentence_vec_1', output_shape=(300,))(out1)

    print r1._keras_shape

    # Wp_r = Wp * r (matrix * matrix)
    Wp_r1 = Dense(300, use_bias=False, name='fc2')(r1)

    M2 = Activation('tanh', name='M_tanh_activation2')(dense2)
    M2 = Dropout(0.4)(M2)
    a1 = op1(dense2)
    a1 = op2(a1)
    a1 = op3(a1)
    out2 = op4([a1,dense2])
    r2 = Lambda(lambda x: K.mean(x, axis=1), name='sentence_vec_2',output_shape=(300,))(out2)

    print r2._keras_shape

    # Wp_r = Wp * r (matrix * matrix)
    Wp_r2 = Dense(300, use_bias=False, name='fc4')(r2)
    
    def slice(input):
        return input[:,-1:, :]

    def slice_dims(input_shape):
        timesteps = 1
        return (input_shape[0], 1, input_shape[2])

    Hn1 = Lambda(slice, output_shape=slice_dims)(dense1)
    Hn1 = Flatten(name='hn_flatten1')(Hn1)

    # Wx_Hn = Wx * Hn (matrix * vector)
    Wx_Hn1 = Dense(300, use_bias=False, name='fc5')(Hn1)

    # h_star = tanh(Wp_r + WxHn)
    Wp_r_plus_Wx_Hn1 = Add()([Wp_r1, Wx_Hn1])
    h_star1 = Activation('tanh', name='tanh_activation1')(Wp_r_plus_Wx_Hn1)
    h_star1 = Dropout(0.4)(h_star1)

    Hn2 = Lambda(slice, output_shape=slice_dims)(dense2)
    Hn2 = Flatten(name='hn_flatten2')(Hn2)

    # Wx_Hn = Wx * Hn (matrix * vector)
    Wx_Hn2 = Dense(300, use_bias=False, name='fc6')(Hn2)

    # h_star = tanh(Wp_r + WxHn)
    Wp_r_plus_Wx_Hn2 = Add()([Wp_r2, Wx_Hn2])
    h_star2 = Activation('tanh', name='tanh_activation2')(Wp_r_plus_Wx_Hn2)
    h_star2 = Dropout(0.4)(h_star2)

    ntn_output = NeuralTensorLayer(output_dim=100, input_dim=300)([h_star1, h_star2])

    # What to do with NTN output? For now, just appending to sarcasm dense
    merged1 = Concatenate(axis=1)([h_star1, ntn_output])
    merged2 = Concatenate(axis=1)([h_star2, ntn_output])


    #print "HERE" , ntn_output._keras_shape
    predictions1 = Dense(2, activation='softmax')(merged1)
    predictions2 = Dense(2, activation='softmax')(merged2)
    
    model = Model(inputs=text, outputs=[predictions1, predictions2])
    model.compile(optimizer='adam', loss='categorical_crossentropy',
              loss_weights=[1., 1.],metrics=['accuracy'])
    
    return model
def create_model_multitask_varriant8(we,nb_words):
    """
    Multitask with shared-attention and private/public layer
    """
    text = Input(shape=(MAX_LENGTH,))
    e1 = Embedding(nb_words + 1, 
                 EMBEDDING_SIZE, 
                 weights=[we], 
                 input_length=MAX_LENGTH, 
                 trainable=False)(text)
    lstm1 = GRU(500, activation='tanh', return_sequences = True, dropout=0.25, name='bi_lstm1')(e1)
    lstm2 = GRU(500, activation='tanh', return_sequences = True, dropout=0.25, name='bi_lstm2')(e1)
    lstm3 = GRU(500, activation='tanh', return_sequences = True, dropout=0.25, name='bi_lstm3')(e1)

    dropout1=Dropout(0.1)(lstm1)
    dropout2=Dropout(0.1)(lstm2)
    dropout3=Dropout(0.1)(lstm3)
    
    dense1=TimeDistributed(Dense(300,activation='relu',name='dense1'))(dropout1)
    dense2=TimeDistributed(Dense(300,activation='relu',name='dense2'))(dropout2)
    dense3=TimeDistributed(Dense(300,activation='relu',name='dense3'))(dropout3)

    merged13 = Concatenate(axis=1)([dense1, dense3])
    merged23 = Concatenate(axis=1)([dense2, dense3])

    M = Activation('tanh', name='M_tanh_activation1')
    w_M = TimeDistributed(Dense(1, name='fc1'))
    alpha = Activation('softmax', name = 'alpha_layer')
    Wp_r =  Dense(300, use_bias=False, name='fc2')
    
    M1 = M(merged13)
    M1 = Dropout(0.1)(M1)
    w_M1 = w_M(M1)
    alpha1 = alpha(w_M1)
    r1 = Dot(axes = 1)([alpha1, merged13])
    r1 = Flatten(name='r_flatten1')(r1)

    # Wp_r = Wp * r (matrix * matrix)
    Wp_r1 = Wp_r(r1)

    M2 = M(merged23)
    M2 = Dropout(0.1)(M2)
    w_M2 = w_M(M2)
    alpha2 = alpha(w_M2)
    r2 = Dot(axes = 1)([alpha2, merged23])
    r2 = Flatten(name='r_flatten2')(r2)

    # Wp_r = Wp * r (matrix * matrix)
    Wp_r2 = Wp_r(r2)
    
    ntn_output = NeuralTensorLayer(output_dim=100, input_dim=300)([Wp_r1, Wp_r2])

    # What to do with NTN output? For now, just appending to sarcasm dense
    #merged1 = Concatenate(axis=1)([dense1, ntn_output])
    merged = Concatenate(axis=1)([Wp_r2, ntn_output])


    #print "HERE" , ntn_output._keras_shape
    predictions1 = Dense(2, activation='softmax')(Wp_r1)
    predictions2 = Dense(2, activation='softmax')(merged)
    
    model = Model(inputs=text, outputs=[predictions1, predictions2])
    model.compile(optimizer='adam', loss='categorical_crossentropy',
              loss_weights=[1., 1.],metrics=['accuracy'])
    
    return model
def create_model_multitask_varriant9(we,nb_words):
    """
    Multitask with shared-attention and private/public layer and ATAE type model
    """
    text = Input(shape=(MAX_LENGTH,))
    e1 = Embedding(nb_words + 1, 
                 EMBEDDING_SIZE, 
                 weights=[we], 
                 input_length=MAX_LENGTH, 
                 trainable=False)(text)
    lstm1 = GRU(500, activation='tanh', return_sequences = True, dropout=0.25, name='bi_lstm1')(e1)
    lstm2 = GRU(500, activation='tanh', return_sequences = True, dropout=0.25, name='bi_lstm2')(e1)
    lstm3 = GRU(500, activation='tanh', return_sequences = True, dropout=0.25, name='bi_lstm3')(e1)

    dropout1=Dropout(0.1)(lstm1)
    dropout2=Dropout(0.1)(lstm2)
    dropout3=Dropout(0.1)(lstm3)
    
    dense1=TimeDistributed(Dense(300,activation='relu',name='dense1'))(dropout1)
    dense2=TimeDistributed(Dense(300,activation='relu',name='dense2'))(dropout2)
    dense3=TimeDistributed(Dense(300,activation='relu',name='dense3'))(dropout3)

    merged13 = Concatenate(axis=1)([dense1, dense3])
    merged23 = Concatenate(axis=1)([dense2, dense3])

    M = Activation('tanh', name='M_tanh_activation1')
    w_M = TimeDistributed(Dense(1, name='fc1'))
    alpha = Activation('softmax', name = 'alpha_layer')
    Wp_r =  Dense(300, use_bias=False, name='fc2')
    
    M1 = M(merged13)
    M1 = Dropout(0.1)(M1)
    w_M1 = w_M(M1)
    alpha1 = alpha(w_M1)
    r1 = Dot(axes = 1)([alpha1, merged13])
    r1 = Flatten(name='r_flatten1')(r1)

    # Wp_r = Wp * r (matrix * matrix)
    Wp_r1 = Wp_r(r1)

    M2 = M(merged23)
    M2 = Dropout(0.1)(M2)
    w_M2 = w_M(M2)
    alpha2 = alpha(w_M2)
    r2 = Dot(axes = 1)([alpha2, merged23])
    r2 = Flatten(name='r_flatten2')(r2)

    # Wp_r = Wp * r (matrix * matrix)
    Wp_r2 = Wp_r(r2)
    
    def slice(input):
        return input[:,-1:, :]

    def slice_dims(input_shape):
        timesteps = 1
        return (input_shape[0], 1, input_shape[2])

    Hn1 = Lambda(slice, output_shape=slice_dims)(dense1)
    Hn1 = Flatten(name='hn_flatten1')(Hn1)

    # Wx_Hn = Wx * Hn (matrix * vector)
    Wx_Hn1 = Dense(300, use_bias=False, name='fc5')(Hn1)

    # h_star = tanh(Wp_r + WxHn)
    Wp_r_plus_Wx_Hn1 = Add()([Wp_r1, Wx_Hn1])
    h_star1 = Activation('tanh', name='tanh_activation1')(Wp_r_plus_Wx_Hn1)
    h_star1 = Dropout(0.5)(h_star1)

    Hn2 = Lambda(slice, output_shape=slice_dims)(dense2)
    Hn2 = Flatten(name='hn_flatten2')(Hn2)

    # Wx_Hn = Wx * Hn (matrix * vector)
    Wx_Hn2 = Dense(300, use_bias=False, name='fc6')(Hn2)

    # h_star = tanh(Wp_r + WxHn)
    Wp_r_plus_Wx_Hn2 = Add()([Wp_r2, Wx_Hn2])
    h_star2 = Activation('tanh', name='tanh_activation2')(Wp_r_plus_Wx_Hn2)
    h_star2 = Dropout(0.5)(h_star2)

    ntn_output = NeuralTensorLayer(output_dim=100, input_dim=300)([h_star1, h_star2])

    # What to do with NTN output? For now, just appending to sarcasm dense
    #merged1 = Concatenate(axis=1)([dense1, ntn_output])
    merged = Concatenate(axis=1)([h_star2, ntn_output])


    #print "HERE" , ntn_output._keras_shape
    predictions1 = Dense(2, activation='softmax')(h_star1)
    predictions2 = Dense(2, activation='softmax')(merged)
    
    model = Model(inputs=text, outputs=[predictions1, predictions2])
    model.compile(optimizer='adam', loss='categorical_crossentropy',
              loss_weights=[1., 1.],metrics=['accuracy'])
    
    return model

def create_model_multitask_varriant10(we,nb_words):
    """
    Multitask with shared-attention and ATAE type
    """
    text = Input(shape=(MAX_LENGTH,))
    e1 = Embedding(nb_words + 1, 
                 EMBEDDING_SIZE, 
                 weights=[we], 
                 input_length=MAX_LENGTH, 
                 trainable=False)(text)
    lstm1 = GRU(500, activation='tanh', return_sequences = True, dropout=0.25, name='bi_lstm1')(e1)

    dropout1=Dropout(0.4)(lstm1)

    dense1=TimeDistributed(Dense(300,activation='relu',name='dense1'))(dropout1)
    dense2=TimeDistributed(Dense(300,activation='relu',name='dense2'))(dropout1)

    op1 = Permute((2, 1))
    op2 = Dense(MAX_LENGTH, activation='softmax')
    op3 = Permute((2, 1))
    op4 = Multiply()

    M1 = Activation('tanh', name='M_tanh_activation1')(dense1)
    M1 = Dropout(0.4)(dense1)
    a = op1(M1)
    a = op2(a)
    a = op3(a)
    out1 = op4([a,dense1])

    print out1._keras_shape
    r1 = Lambda(lambda x: K.mean(x, axis=1), name='sentence_vec_1', output_shape=(300,))(out1)

    print r1._keras_shape

    # Wp_r = Wp * r (matrix * matrix)
    Wp_r1 = Dense(300, use_bias=False, name='fc2')(r1)

    M2 = Activation('tanh', name='M_tanh_activation2')(dense2)
    M2 = Dropout(0.4)(M2)
    a1 = op1(dense2)
    a1 = op2(a1)
    a1 = op3(a1)
    out2 = op4([a1,dense2])
    r2 = Lambda(lambda x: K.mean(x, axis=1), name='sentence_vec_2',output_shape=(300,))(out2)

    print r2._keras_shape

    # Wp_r = Wp * r (matrix * matrix)
    Wp_r2 = Dense(300, use_bias=False, name='fc4')(r2)
    
    def slice(input):
        return input[:,-1:, :]

    def slice_dims(input_shape):
        timesteps = 1
        return (input_shape[0], 1, input_shape[2])

    Hn1 = Lambda(slice, output_shape=slice_dims)(dense1)
    Hn1 = Flatten(name='hn_flatten1')(Hn1)

    # Wx_Hn = Wx * Hn (matrix * vector)
    Wx_Hn1 = Dense(300, use_bias=False, name='fc5')(Hn1)

    # h_star = tanh(Wp_r + WxHn)
    Wp_r_plus_Wx_Hn1 = Add()([Wp_r1, Wx_Hn1])
    h_star1 = Activation('tanh', name='tanh_activation1')(Wp_r_plus_Wx_Hn1)
    h_star1 = Dropout(0.4)(h_star1)

    Hn2 = Lambda(slice, output_shape=slice_dims)(dense2)
    Hn2 = Flatten(name='hn_flatten2')(Hn2)

    # Wx_Hn = Wx * Hn (matrix * vector)
    Wx_Hn2 = Dense(300, use_bias=False, name='fc6')(Hn2)

    # h_star = tanh(Wp_r + WxHn)
    Wp_r_plus_Wx_Hn2 = Add()([Wp_r2, Wx_Hn2])
    h_star2 = Activation('tanh', name='tanh_activation2')(Wp_r_plus_Wx_Hn2)
    h_star2 = Dropout(0.4)(h_star2)

    # cross-attention
    def broadcast(input):
        return input[:,None,:]

    def bcast_shape(input_shape):
        return (input_shape[0], 1, input_shape[1])

    bcast = Lambda(broadcast, output_shape=bcast_shape)
    cross_feat = Concatenate(axis=1)([bcast(h_star1), bcast(h_star2)])

    catt = Permute((2, 1))(cross_feat)
    catt = Dense(2, activation='softmax')(catt)
    catt = Permute((2, 1))(catt)
    rr = Multiply()([catt,cross_feat])
    ntn_output = Lambda(lambda x: K.mean(x, axis=1), name='cross_vec',output_shape=(300,))(rr)
    # ntn_output = NeuralTensorLayer(output_dim=100, input_dim=300)([h_star1, h_star2])

    # What to do with NTN output? For now, just appending to sarcasm dense
    merged1 = Concatenate(axis=1)([h_star1, ntn_output])
    merged2 = Concatenate(axis=1)([h_star2, ntn_output])
    # merged1 = h_star1
    # merged2 = h_star2


    #print "HERE" , ntn_output._keras_shape
    predictions1 = Dense(2, activation='softmax')(merged1)
    predictions2 = Dense(2, activation='softmax')(merged2)
    
    model = Model(inputs=text, outputs=[predictions1, predictions2])
    model.compile(optimizer='adam', loss='categorical_crossentropy',
              loss_weights=[1., 1.],metrics=['accuracy'])
    
    return model
def create_model(we,nb_words,nb_class):
    text = Input(shape=(MAX_LENGTH,))
    e1 = Embedding(nb_words + 1, 
                 EMBEDDING_SIZE, 
                 weights=[we], 
                 input_length=MAX_LENGTH, 
                 trainable=False)(text)
    lstm1 = GRU(500, activation='tanh', return_sequences = False, dropout=0.25, name='bi_lstm1')(e1)
    dropout1=Dropout(0.4)(lstm1)
    dense1=Dense(300,activation='relu',name='dense1')(dropout1)
    
    predictions1 = Dense(nb_class, activation='softmax')(dense1)
    
    model = Model(inputs=text, outputs=predictions1)
    model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
    
    return model 
def train_and_evaluate_model_multitask(model, data_train, labels_senti_train, labels_sarcasm_train, data_test, labels_senti_test, labels_sarcasm_test):
        early_stopping = EarlyStopping(monitor='val_loss', patience=10)
        model.fit(data_train,[labels_senti_train,labels_sarcasm_train],epochs=50, batch_size=30, callbacks=[early_stopping], validation_split=0.1,shuffle=True)
        prediction_senti_test,prediction_sarcasm_test=model.predict(data_test)
        labels=lambda true_labels: [np.argmax(a) for a in true_labels]
        #print prediction_senti_test,prediction_sarcasm_test
        #print labels(prediction_senti_test)
        precision_senti, recall_senti, fscore_senti, support_senti = precision_recall_fscore_support(labels(labels_senti_test), labels(prediction_senti_test), average='weighted')
        precision_sarcasm, recall_sarcasm, fscore_sarcasm, support_sarcasm = precision_recall_fscore_support(labels(labels_sarcasm_test), labels(prediction_sarcasm_test), average='weighted')
        #accuracy_sentiment=accuracy_score(labels(labels_senti_test), labels(prediction_senti_test))
        #print accuracy_sentiment
        accuracy = model.evaluate(data_test,[labels_senti_test,labels_sarcasm_test])
        return precision_senti, recall_senti, fscore_senti, precision_sarcasm, recall_sarcasm, fscore_sarcasm, accuracy

def train_and_evaluate_model(model, data_train,labels_train, data_test, labels_test):
        early_stopping = EarlyStopping(monitor='val_loss', patience=10)
        model.fit(data_train,labels_train,epochs=50, batch_size=30, callbacks=[early_stopping], validation_split=0.1,shuffle=True)
        return model.evaluate(data_test,labels_test)

def multitask():
    data, labels_senti, labels_sarcasm = load_data_multitask()
    embeddings=get_embeddings("/Users/soujanyaporia/Documents/glove.840B.300d.txt")
    data,labels_senti,labels_sarcasm,we,nb_words=preprocessing(embeddings,data,labels_senti,labels_sarcasm)
    skf = StratifiedKFold(labels_sarcasm, n_folds=n_folds, shuffle=True)
    return data,labels_senti,labels_sarcasm,we,nb_words,skf
def multilabel():

    data, labels = load_data_multilabel()
    embeddings=get_embeddings("/Users/soujanyaporia/Documents/glove.840B.300d.txt")
    data,labels,we,nb_words=preprocessing_multilabel(embeddings,data,labels)
    skf = StratifiedKFold(labels, n_folds=n_folds, shuffle=True)
    return data,labels,we,nb_words,skf
def convert_labels(labels):
    new_label=[]
    for item in labels:
        if item==1:
            new_label.append([0,1])
        else:
            new_label.append([1,0])
    return new_label

def convert_labels_multi(labels):
    new_label=[]
    for item in labels:
        if item==0:
            new_label.append([1,0,0,0])
        if item==1:
            new_label.append([0,1,0,0])
        if item==2:
            new_label.append([0,0,1,0])
        if item==3:
            new_label.append([0,0,0,1])
    return new_label

def find_dist(labels_train,labels_test):
    train_pos = np.sum(labels_train)
    train_neg = len(labels_train) - train_pos
    test_pos = np.sum(labels_test)
    test_neg = len(labels_test) - test_pos
    return train_pos,train_neg,test_pos,test_neg

model_func = {1: create_model_multitask,
        2: create_model_multitask_varriant2,
        3: create_model_multitask_varriant3,
        4: create_model_multitask_varriant4,
        5: create_model_multitask_varriant5,
        6: create_model_multitask_varriant6,
        7: create_model_multitask_varriant7,
        8: create_model_multitask_varriant8,
        9: create_model_multitask_varriant9,
        10: create_model_multitask_varriant10,
        11: create_model_multitask_varriant11,
        12: create_model_multitask_varriant12,
        13: create_model_multitask_varriant13,
        14: create_model_multitask_varriant14,
        15: create_model_multitask_varriant15}

def run_multitask(data,labels_senti,labels_sarcasm,we,nb_words,skf):
    prec_sarcasm=[]
    rec_sarcasm=[]
    f_score_sarcasm=[]
    acc_sarcasm=[]
    prec_sentiment=[]
    rec_sentiment=[]
    f_score_sentiment=[]
    acc_sentiment=[]
    for i, (train, test) in enumerate(skf):
        print "Running Fold", i+1, "/", n_folds
        model = None # Clearing the NN.
        model = model_func[int(sys.argv[1])](we,nb_words)
        sarcasm_distribution=find_dist(labels_sarcasm[train],labels_sarcasm[test])
        sentiment_distribution=find_dist(labels_senti[train],labels_senti[test])
        print "sarcasm distribution", sarcasm_distribution
        print "sentiment distribution", sentiment_distribution
        new_label_senti_train=np.asarray(convert_labels(labels_senti[train]))
        new_label_sarcasm_train=np.asarray(convert_labels(labels_sarcasm[train]))
        new_label_senti_test=np.asarray(convert_labels(labels_senti[test]))
        new_label_sarcasm_test=np.asarray(convert_labels(labels_sarcasm[test]))
        performance=train_and_evaluate_model_multitask(model, data[train], new_label_senti_train, new_label_sarcasm_train, data[test], new_label_senti_test, new_label_sarcasm_test)
        print "precision_senti, recall_senti, fscore_senti, precision_sarcasm, recall_sarcasm, fscore_sarcasm, ", model.metrics_names
        del model
        print "Fold", i+1, "performance", performance
        #print type(performance)
        prec_sentiment.append(performance[0])
        rec_sentiment.append(performance[1])
        f_score_sentiment.append(performance[2])
        acc_sentiment.append(performance[6][3])

        prec_sarcasm.append(performance[3])
        rec_sarcasm.append(performance[4])
        f_score_sarcasm.append(performance[5])
        acc_sarcasm.append(performance[6][4])
    print "precision for sentiment task", np.mean(prec_sentiment)
    print "recall for sentiment task", np.mean(rec_sentiment)
    print "f_score for sentiment task", np.mean(f_score_sentiment)
    print "accuracy for sentiment task", np.mean(acc_sentiment)

    print "precision for sarcasm task", np.mean(prec_sarcasm)
    print "recall for sarcasm task", np.mean(rec_sarcasm)
    print "f_score for sarcasm task", np.mean(f_score_sarcasm)
    print "accuracy for sarcasm task", np.mean(acc_sarcasm)

def run_sarcasm(data,labels_senti,labels_sarcasm,we,nb_words,skf):
    acc_sarcasm=[]
    for i, (train, test) in enumerate(skf):
        print "Running Fold", i+1, "/", n_folds
        model = None # Clearing the NN.
        model = create_model(we,nb_words,2)
        new_label_sarcasm_train=np.asarray(convert_labels(labels_sarcasm[train]))
        new_label_sarcasm_test=np.asarray(convert_labels(labels_sarcasm[test]))
        accuracy=train_and_evaluate_model(model, data[train], new_label_sarcasm_train, data[test], new_label_sarcasm_test)
        print model.metrics_names
        del model
        print "Fold", i+1, "accuracy", accuracy
        acc_sarcasm.append(accuracy[1])
    print "accuracy for sarcasm task", np.mean(acc_sarcasm)

def run_sentiment(data,labels_senti,labels_sarcasm,we,nb_words,skf):
    acc_sentiment=[]
    for i, (train, test) in enumerate(skf):
        print "Running Fold", i+1, "/", n_folds
        model = None # Clearing the NN.
        model = create_model(we,nb_words,2)
        new_label_senti_train=np.asarray(convert_labels(labels_senti[train]))
        new_label_senti_test=np.asarray(convert_labels(labels_senti[test]))
        accuracy=train_and_evaluate_model(model, data[train], new_label_senti_train, data[test], new_label_senti_test)
        print model.metrics_names
        del model
        print "Fold", i+1, "accuracy", accuracy
        acc_sentiment.append(accuracy[1])
    print "accuracy for sentiment task", np.mean(acc_sentiment)
def run_multi_labels_baseline(data,labels,we,nb_words,skf):
    acc=[]
    for i, (train, test) in enumerate(skf):
        print "Running Fold", i+1, "/", n_folds
        model = None # Clearing the NN.
        model = create_model(we,nb_words,4)
        new_label_train=np.asarray(convert_labels_multi(labels[train]))
        new_label_test=np.asarray(convert_labels_multi(labels[test]))
        accuracy=train_and_evaluate_model(model, data[train], new_label_train, data[test], new_label_test)
        print model.metrics_names
        print "Fold", i+1, "accuracy", accuracy
        acc.append(accuracy[1])
    print "accuracy for multilabel task", np.mean(acc)
data,labels_senti,labels_sarcasm,we,nb_words,skf=multitask()
#data,labels,we,nb_words,skf=multilabel()
run_multitask(data,labels_senti,labels_sarcasm,we,nb_words,skf)
multitask_func={1: run_sarcasm,
                2: run_sentiment}
# multitask_func[int(sys.argv[1])](data,labels_senti,labels_sarcasm,we,nb_words,skf)
