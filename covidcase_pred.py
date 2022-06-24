# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 09:25:59 2022

@author: KTong
"""

import os
import pickle
import datetime
import numpy as np
import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt
from custom_module import Visuals as vs
from custom_module import NN_modules as nn
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import TensorBoard
from sklearn.metrics import mean_squared_error,mean_absolute_error,mean_absolute_percentage_error

#%% STATIC
log_dir=datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
LOG_PATH = os.path.join(os.getcwd(),'logs',log_dir)
DS_FILE_PATH=os.path.join(os.getcwd(),'dataset','cases_malaysia_train.csv')
DS_TEST_FILE_PATH=os.path.join(os.getcwd(),'dataset','cases_malaysia_test.csv')
MMS_PICKLE_PATH=os.path.join(os.getcwd(),'models','mms.pkl')
MODEL_SAVE_PATH=os.path.join(os.getcwd(),'models','model.h5')
win_size=30

#%% DATA LOADING
df=pd.read_csv(DS_FILE_PATH)
df_test=pd.read_csv(DS_TEST_FILE_PATH)

#%% DATA INSPECTION
# Target feature: 'cases_new'
df.info()

# df['cases_new'] contain object

# Numerize 'cases_new'
df['cases_new']=pd.to_numeric(df['cases_new'],errors='coerce')

# Check for null values
df['cases_new'].isna().sum()
df['cases_new'][df['cases_new'].isna()]

# 'cases_new' contain 12 NaNs

df['cases_new'].describe()

# Check for duplicate observations
df.duplicated().sum()

#%% DATA VISUALIZATION
visuals=vs.Visualisation()
visuals.line_plot(df,xlen=680,column='cases_new',linewidth=1)
msno.matrix(df)

# MSNO module has better visualisation of null values, however line plot
# visualizes the trend of target feature better.

#%% DATA CLEANING
# Impute null values by interpolation on train dataset
df['cases_new']=df['cases_new'].interpolate(method='linear')
visuals.line_plot(df,xlen=680,column='cases_new',linewidth=1)

#%% PREPROCESSING
mms=MinMaxScaler()
df_scaled=mms.fit_transform(np.expand_dims(df['cases_new'],axis=-1))

with open(MMS_PICKLE_PATH,'wb') as file:
    pickle.dump(mms,file)

# Create train dataset
x_train=[]
y_train=[]

for i in range(win_size,np.shape(df_scaled)[0]):
    x_train.append(df_scaled[i-win_size:i])
    y_train.append(df_scaled[i])
    
# Increase dimension to fit into RNN model
x_train=np.array(x_train)
y_train=np.array(y_train)

#%% MODEL BUILDING
nn_mod=nn.NeuralNetworkModel()
model=nn_mod.simple_prediction_model(x_train,y_train,node1=64,node2=32,droprate=0.2)
plot_model(model,show_shapes=(True))

model.compile(optimizer='adam',loss='mse',metrics=['mse','mae','mape'])

tb=TensorBoard(log_dir=LOG_PATH)

hist=model.fit(x_train,y_train,batch_size=64,epochs=100,callbacks=tb,verbose=1)

# Evaluation plots
nn_mod.eval_plot(hist)

#%% TEST DATASET PREPARATION
# Inspect test data
df_test.info()

# df_test['cases_new'] contain 1 missing value

df_test['cases_new'].isna().sum()
df_test['cases_new'][df_test['cases_new'].isna()]

# Check for duplicated rows
df_test.duplicated().sum()

# Impute null value by interpolation on test dataset
df_test['cases_new']=df_test['cases_new'].interpolate(method='linear')
df_test_ori=np.expand_dims(df_test['cases_new'],axis=-1)

# mms_test=MinMaxScaler()
df_test_scaled=mms.transform(df_test_ori)

# Concatenate scaled train and test dataset
concat_df=np.concatenate([df_scaled,df_test_scaled],axis=0)

# Create test dataset
concat_test=concat_df[-130:]

x_test=[]

for i in range(win_size,np.shape(concat_test)[0]):
    x_test.append(concat_test[i-win_size:i])

x_test=np.array(x_test)

#%% PREDICTION PLOT
y_pred=model.predict(x_test)

plt.figure()
plt.plot(df_test_scaled)
plt.plot(y_pred)
plt.legend(['Actual Cases','Predicted Cases'])
plt.show()

#%% TEST DATA EVALUATION
print('MSE:',mean_squared_error(df_test_scaled,y_pred))
print('MAE:',mean_absolute_error(df_test_scaled,y_pred))
print('MAPE:',mean_absolute_percentage_error(df_test_scaled,y_pred))

pred_inv=mms.inverse_transform(y_pred)

print('ORI MSE:',mean_squared_error(df_test_ori,pred_inv))
print('ORI MAE:',mean_absolute_error(df_test_ori,pred_inv))
print('ORI MAPE:',mean_absolute_percentage_error(df_test_ori,pred_inv))
print((mean_absolute_error(df_test_ori,pred_inv)/sum(abs(df_test_ori))) *100)

#%% EXPORT MODEL
model.save(MODEL_SAVE_PATH)
