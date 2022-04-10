# -*- coding: utf-8 -*-
'''
论文题目：基于深度学习的PSA过程优化与控制
作    者：余秀鑫
单    位：天津大学化工学院
'''

import numpy as np
import time
import uuid
import os
import sys

from tqdm.keras import TqdmCallback
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score

import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense 

import optuna
import neptune.new as neptune
import neptune.new.integrations.optuna as optuna_utils

import pandas as pd
import plotly

# Study properties. 
STUDY_NAME_ROOT = 'LSTM_thesis_3'
data_label = 'H2Purity'
nb_epoch = 200
reduce_lr_patience = 10
early_stopping_patience = 30

def get_data():
  Train_X = np.load('Train_Test_Data/Train_X.npy').reshape((1999, 2))
  Train_y = np.load('Train_Test_Data/Train_y.npy').reshape((1999, 1))

  scaler_X = MinMaxScaler(feature_range=(0, 1)).fit(Train_X)
  Train_X_scaled = scaler_X.transform(Train_X)
  scaler_y = MinMaxScaler(feature_range=(0, 1)).fit(Train_y)
  Train_y_scaled = scaler_y.transform(Train_y)

  # 使x_train符合RNN输入要求：[送入样本数， 循环核时间展开步数， 每个时间步输入特征个数]。
  # 此处整个数据集送入，送入样本数为x_train.shape[0]即1999组数据
  Train_X_scaled = np.reshape(Train_X_scaled, (Train_X_scaled.shape[0], 1, 2))

  # 准备验证集
  # Make the train/test splits
  def make_train_test_splits(windows, labels, test_split=0.2):
    """
    Splits matching pairs of windows and labels into train and test splits.
    """
    split_size = int(len(windows) * (1-test_split)) # this will default to 80% train/20% test
    train_windows = windows[:split_size]
    train_labels = labels[:split_size]
    test_windows = windows[split_size:]
    test_labels = labels[split_size:]
    return train_windows, test_windows, train_labels, test_labels

  train_windows, test_windows, train_labels, test_labels = make_train_test_splits(Train_X_scaled, Train_y_scaled)

  return train_windows, test_windows, train_labels, test_labels, Train_X_scaled, Train_y_scaled

def get_accuracy(y,y_pred, n_samples, n_features):
    n, k = n_samples, n_features
    R2 = r2_score(y, y_pred)
    R2_Adj = 1 - (1 - R2) * (n - 1) / (n - k - 1)
    return R2_Adj

def get_model(nb_hidden_nodes,
              nb_hidden_layers, 
              reg_dropout=0):
    
    model = tf.keras.Sequential()
    # Add Input layer:
    model.add(Input(shape=(1, 2)))
    # Add LSTM hidden layers:
    for i in range(nb_hidden_layers):
        model.add(LSTM(nb_hidden_nodes, return_sequences=True))
        model.add(Dropout(reg_dropout))
    # Add output layer:
    model.add(Dense(units=1))
    # Compile model:
    adam = tf.keras.optimizers.Adam()
    model.compile(optimizer=adam,loss='mean_squared_error') 

    return model

# BO objective function.
def objective(trial):

    # Random run ID.
    RUN_ID = str(uuid.uuid1())
    print("RUN_ID: ", RUN_ID)

    PARAMS = {
        'nb_epoch': nb_epoch,
        'batch_size': 64,
        'nb_hidden_layers': trial.suggest_int('nb_hidden_layers', 1, 3),
        'nb_hidden_nodes': trial.suggest_categorical('nb_hidden_nodes', [10, 50, 100, 200]),
        'reg_dropout': trial.suggest_categorical('reg_dropout', [0, 0.05, 0.1, 0.2, 0.3])
    }
    TB_LOG_DIR = 'logs/' + STUDY_NAME_ROOT + '/' + RUN_ID
    CHECKPOINT_DIR = 'trained_models/' + STUDY_NAME_ROOT + '/' + RUN_ID

    # Get data.
    train_windows, test_windows, train_labels, test_labels, Train_X_scaled, Train_y_scaled = get_data()

    # Initialize model.
    model = get_model(nb_hidden_layers = PARAMS['nb_hidden_layers'],
                    nb_hidden_nodes = PARAMS['nb_hidden_nodes'],
                    reg_dropout = PARAMS['reg_dropout'])

    # Define callbacks.
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=TB_LOG_DIR,
                                                histogram_freq=1)
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                            patience=early_stopping_patience,
                                                            verbose=1)
    reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                            factor=0.5,
                                                            patience=reduce_lr_patience,
                                                            min_lr=1e-10,
                                                            verbose=1)
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=CHECKPOINT_DIR,
                                                                save_weights_only=True,
                                                                monitor='val_loss',
                                                                mode='min',
                                                                save_best_only=True)
    tqdm_callback = TqdmCallback()

    print('\n###############################\n',PARAMS,'\n###############################\n')

    # Train model. 
    history = model.fit(train_windows,
                        train_labels,
                        validation_data=(test_windows, test_labels),
                        epochs=PARAMS['nb_epoch'],
                        batch_size=PARAMS['batch_size'],
                        verbose=0,
                        callbacks=[tensorboard,
                                   early_stopping_callback,
                                   reduce_lr_callback,
                                   model_checkpoint_callback,
                                   tqdm_callback])
        
    accuracy = get_accuracy(y=Train_y_scaled.squeeze(),y_pred=model.predict(Train_X_scaled).squeeze(),n_samples=Train_X_scaled.shape[0],n_features=Train_X_scaled.shape[1])
    return accuracy


# create a study-level Run
run = neptune.init(
    project="xxxxxxx/LSTM-Thesis",
    tags=[STUDY_NAME_ROOT],
    api_token="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
)
# Create a NeptuneCallback for Optuna
neptune_callback = optuna_utils.NeptuneCallback(run)

# Turn on memory growth: allocate only as much GPU memory as required during runtime. 
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

search_space = {
        'nb_hidden_layers': [1, 2, 3],
        'nb_hidden_nodes': [10, 50, 100, 200],
        'reg_dropout': [0, 0.05, 0.1, 0.2, 0.3]
    }

study = optuna.create_study(study_name=STUDY_NAME_ROOT,
                                direction='maximize',
                                load_if_exists=True,
                                sampler=optuna.samplers.GridSampler(search_space))

t0 = time.time()
study.optimize(objective,
                callbacks=[neptune_callback])

t1 = time.time()
print('Runtime: %.2f s' %(t1-t0))
print("Number of finished trials: ", len(study.trials))

print("Best trial:")
trial = study.best_trial

print("  Value: ", trial.value)

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

# Stop logging to a Neptune Run
run.stop()

