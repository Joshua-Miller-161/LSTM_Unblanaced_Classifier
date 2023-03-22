import os
import datetime
# import IPython
# import IPython.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time as TM
import random
import pickle
import copy


import tensorflow as tf
from tensorflow import keras
print('tf.__version__', tf.__version__)
print(f"TensorFlow has access to the following devices:\n{tf.config.list_physical_devices()}")

from keras import layers
from keras.layers import PReLU, LeakyReLU
from keras.models import Sequential
from keras.layers import *
from keras.callbacks import ModelCheckpoint
from keras.losses import MeanSquaredError
from keras.metrics import RootMeanSquaredError
from keras.optimizers import Adam


from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import StandardScaler
#import keras_tuner as kt



# from BASIC_FUNCTIONS import *
import tensorflow_addons as tfa

#f1_score = tfa.metrics.F1Score(num_classes=2, threshold=None)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

para_relu = PReLU()
leaky_relu = LeakyReLU(alpha=0.01)

wd = '/home/r62/repos/River_Ice_Break-Up'
data_filepath = '/mnt/locutus/remotesensing/r62/River_Ice/LSTM_Data.pkl'
os.chdir(wd)

####################################  PARAMETERS #############################################################
SEED = 123
EPOCHS = (100, 20)
CLEAR_SESSION = True
WINDOW_SIZE = 450
BATCH_SIZE = 450
TUNER = 'HYPERBAND'
####################################  FUNCTIONS  ##############################################################

def set_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

set_seeds(SEED)

def normalize_df(df, col_list, convert):
	''' Only works on float type columns unless you 
	set convert == True'''
	
	DF = copy.copy(df)
	if type(col_list) == list:
		pass
	elif type(col_list) == str:
		if col_list.upper() == 'ALL':
			col_list = list(DF.columns)
		else:
			print("COL_LIST SHOULD BE EITHER 'ALL' OR A LIST OF YOUR COLUMNS")
		
	for col in col_list:
		
		if (convert == True) and (np.nanmax(DF[col]) != 0) and (np.nanmax(DF[col]) != np.nanmin(DF[col])):
			DF[col] = DF[col].astype('float32')
			DF[col] = DF[col] - np.nanmin(DF[col])
			DF[col] = DF[col]/np.nanmax(DF[col])
		
		elif (convert == False) and (DF[col].dtype == 'float') and (np.nanmax(DF[col]) != 0) and (np.nanmax(DF[col]) != np.nanmin(DF[col])):
			DF[col] = DF[col] - np.nanmin(DF[col])
			DF[col] = DF[col]/np.nanmax(DF[col])
		else:
			print('ERROR SOMETHING IS WRONG HERE')
	return DF

def df_to_X_y2(df, window_size):
    
    '''Assumes target variable is the first column'''
    
    df_as_np = df.to_numpy()
    X = []
    for i in range(len(df_as_np)-window_size):
        row = [r for r in df_as_np[i:i+window_size]]
        X.append(row)
    return np.array(X)


if CLEAR_SESSION == True:
    tf.keras.backend.clear_session()
    
data = pd.read_pickle(data_filepath)
date = data.index

data = normalize_df(data, col_list=['prcp(mm/day)', 'srad(W/m^2)', 'swe(kg/m^2)', 'tmax(deg c)', 'tmin(deg c)', 'vp(Pa)', 'Radians', 'Year'], convert=True)

neg = len(data['Predictor'][data['Predictor'] == 0])
pos = len(data['Predictor'][data['Predictor'] == 1])

train_df = data.loc[:'2010-04-29']
val_df = data.loc['2010-04-30':'2016-04-23']
test_df = data.loc['2016-04-24':]

train_y = np.array(train_df.pop('Predictor'))
val_y = np.array(val_df.pop('Predictor'))
test_y = np.array(test_df.pop('Predictor'))

train_y = train_y[450:]
val_y = val_y[450:]
test_y = test_y[450:]

train_X = df_to_X_y2(train_df, window_size=WINDOW_SIZE)
val_X = df_to_X_y2(val_df, window_size=WINDOW_SIZE)
test_X = df_to_X_y2(test_df, window_size=WINDOW_SIZE)

print(f'Average class probability in training set:   {train_y.mean():.4f}')
print(f'Average class probability in validation set: {val_y.mean():.4f}')
print(f'Average class probability in test set:       {test_y.mean():.4f}')


METRICS = [
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'),
      # tfa.metrics.F1Score(num_classes=2, name = 'f1_score'),
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
      keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
]

initial_bias = np.log([pos/neg])

def build_model(hp, output_bias=initial_bias):
    
    output_bias = tf.keras.initializers.Constant(initial_bias)
    model = keras.Sequential()
    model.add(InputLayer((WINDOW_SIZE, train_df.shape[1])))
    model.add(LSTM(64))

    for i in range(hp.Int("num_layers", 1, 5)):
        model.add(layers.Dense(units=hp.Int(f"units_{i}", min_value=32, max_value=544, step=32),
                activation=hp.Choice("activation", ["leaky_relu", "elu", 'relu', 'tanh']),)
        )
        
    if hp.Boolean("dropout"):
        model.add(layers.Dropout(rate=0.25))
        
    model.add(keras.layers.Dense(1, 'sigmoid', bias_initializer=output_bias))

    hp_learning_rate = hp.Choice('learning_rate', values = [1e-2, 1e-3, 1e-4])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate = hp_learning_rate),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=METRICS)
    
    return model

if TUNER == 'HYPERBAND':
# Hyperband determines the number of models to train in a bracket by computing 1 + log_factor(max_epochs) and rounding it up to the nearest integer.

    tuner = kt.Hyperband(build_model,
                         objective = kt.Objective('val_recall', direction="max"),
                         max_epochs=EPOCHS[1],
                         overwrite=CLEAR_SESSION,
                         factor=3,
                         directory = '/home/r62/repos/River_Ice_Yukon/LSTM_Results2/',
                         project_name = 'TUNING_LSTM_HYPERBAND2')
    
elif TUNER == 'BAYESIAN':
    tuner = kt.BayesianOptimization(build_model,
                         objective = kt.Objective('val_recall', direction="max"), 
                         max_trials = 15,
                         overwrite=CLEAR_SESSION,
                         directory = '/home/r62/repos/River_Ice_Yukon/LSTM_Results2/',
                         project_name = 'TUNING_LSTM_BAYESIAN2')
    

start = TM.time()

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_recall', # early stopping doesn't work with f1_score 
    verbose=1,
    patience=10,
    mode='max',
    restore_best_weights=True)

tuner.search(train_X, train_y, batch_size=BATCH_SIZE, epochs=EPOCHS[0], validation_data=(val_X, val_y), callbacks=[early_stopping])

# Get the optimal hyperparameters
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
model = tuner.hypermodel.build(best_hps)

hist = model.fit(train_X, train_y, batch_size=BATCH_SIZE, epochs=EPOCHS[0], validation_data=(val_X, val_y), callbacks=[early_stopping])

val_recall_per_epoch = hist.history['val_recall']
best_epoch = val_recall_per_epoch.index(max(val_recall_per_epoch)) + 1
print('Best epoch: {best_epoch}'.format(best_epoch=best_epoch))

hypermodel = tuner.hypermodel.build(best_hps)

# Retrain the model
hypermodel.fit(train_X, train_y, epochs=best_epoch, validation_data=(val_X, val_y))

if TUNER == 'HYPERBAND':
    hypermodel.save('LSTM_HYPERBAND_MODEL2') 
    
elif TUNER == 'BAYESIAN':
    hypermodel.save('LSTM_BAYESIAN_MODEL2') 

stop = TM.time()    
complete = stop - start

print('Process complete! Took ', complete, 'seconds')
