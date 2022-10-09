# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 17:23:57 2022

@author: Parth
"""

import numpy as np
import pandas as pd
import streamlit as st
import sklearn
import numpy as np
import pickle
from tensorflow import keras
from keras.layers import Dense, LSTM, RepeatVector, TimeDistributed, Flatten, Dropout
from keras.models import load_model


loaded_model = load_model('C:/Users/nakul/OneDrive/Desktop/Parth/Final Submission/deployment/model_lstm.h5')

load_data = pd.read_csv('C:/Users/nakul/OneDrive/Desktop/Parth/Final Submission/deployment/loading_dataset.csv',index_col=False)

#create function
def time_series(shop_id,item_id):
    shop_id = np.int(shop_id)
    #print(shop_id)
    item_id = np.int(item_id)
    #print(item_id)
    if (any(load_data["shop_id"].isin([shop_id]))) & (any(load_data["item_id"].isin([item_id]))):
        print("inside")
        ip_data_mid = load_data[(load_data["shop_id"]==shop_id) & (load_data["item_id"]==item_id)]
        values_temp = ip_data_mid.iloc[:,3:]
        final_values = np.asarray(values_temp.values)
    else:
        final_values = np.zeros(33) # need to change 33 as models change with time
    
    #print(final_values)
    #input_data = [  0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,0., 0., 0., 0., 0., 0., 2., 2., 0., 0., 0., 1., 1., 0., 5., 4., 0.]

    #ip = np.expand_dims(input_data,axis = 1)
    tp = np.asarray(final_values)
    ip = tp.reshape(1,33,1)


    prediction = loaded_model.predict(ip)
    #pred_value = prediction[[0]]
    return (np.round(prediction))

time_series(38, 5233)

def main():
    
    #giving a title
    st.title("Future sales Time series forecasting")
    
    # getting input data from user
    shop_id = st.text_input("shop id")
    item_id = st.text_input("item id")
    
    #code for prediction
    prediction_value = ''
    
    #creating a button
    if st.button("item_count_predicton_monthly"):
        prediction_value = time_series(shop_id,item_id)
    
    st.success(prediction_value)
    
if __name__=='__main__':
    main()
    
    