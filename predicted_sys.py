# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import sklearn
import numpy as np
import pickle
from tensorflow import keras
from keras.utils.vis_utils import plot_model
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, RepeatVector, TimeDistributed, Flatten, Dropout
from keras.models import load_model




load_data = pd.read_csv('C:/Users/nakul/OneDrive/Desktop/Parth/Final Submission/deployment/loading_dataset.csv',index_col=False)


loaded_model = load_model('C:/Users/nakul/OneDrive/Desktop/Parth/Final Submission/deployment/model_lstm.h5')
#loaded_model = pickle.load(open('C:/Users/nakul/OneDrive/Desktop/Parth/Final Submission/deployment/trained_model_xgbregressor.pkl', 'rb'))
#input_data = ( 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,0., 0., 0., 0., 0., 0., 2., 2., 0., 0., 0., 1., 1., 0., 5., 4., 0.)



if (any(load_data["shop_id"].isin([shop_id]))) & (any(load_data["item_id"].isin([item_id]))):
    ip_data_mid = load_data[(load_data["shop_id"]==shop_id) & (load_data["item_id"]==item_id)]
    values_temp = ip_data_mid.iloc[:,3:]
    values = np.asarray(values_temp.values)
else:
    values = [0]* 33
         
print(values)
values.shape

#ip = np.expand_dims(input_data,axis = 1)
tp = np.asarray(values)
ip = tp.reshape(1,33,1)

# changing the input_data to numpy array
#input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
#input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
#input_data_reshaped

prediction = loaded_model.predict(ip)
list(prediction[:,:])
print(np.round(prediction[:,:]))



