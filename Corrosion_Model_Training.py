#/ coding: utf-8
# authour: Yi Yao && YG 
# Spring 2022

# Use this file after the HP tuning is done (Corrosion_Model_HP_Tuning.py). 
# Use the HPs from TensorBoard (from Corrosion_Model_HP_Tuning.py) to train a model.

import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt
import numpy as np
import pandas as pd
import os

# =============================================================================================================================================
# helper function

def feature_normalize(Feature, mean_value, range_value):
  Feature = Feature - mean_value
  Feature = Feature / range_value
  return Feature

def getNormPara(Lx):
    meanNorm = np.mean(Lx, axis=0)
    rangeNorm = np.std(Lx, axis=0)+ (10 ** -8)
    return meanNorm, rangeNorm

def selectSampleData(df, i, a1, a2):
    X = df.iloc[i*101:(i+1)*101, 1:-1]
    X = feature_normalize(X, a1, a2)
    X = np.asarray(X).astype(np.float32)
    Y = df.iloc[i*101:(i+1)*101, -1]
    Y = np.asarray(Y).astype(np.float32)
    return X, Y

def removeSampleData(df, i, a1, a2):
    df = df.loc[~df["Group"].isin([i])]
    X = df.iloc[:, 1:-1]
    X = feature_normalize(X, a1, a2)
    X = np.asarray(X).astype(np.float32)
    Y = df.iloc[:,-1]
    Y = np.asarray(Y).astype(np.float32)
    return X, Y


# ===============================================================================================================pip==============
# Main function: 

# load data
path = os.getcwd()
df = pd.read_excel(path+"/input file.xlsx", index_col=False)

X = df.iloc[:, 1:-1]

[a1, a2] = getNormPara(X)
print(a1)
print(a2)    

# lists to save data
L_MAAE = []
L_pred = []

# adjust range and model layers based on inputs
# also remeber to change the input sharp based on you input file
# !strain, wear, and corrosion input files have different shape!

for i in range(100):

    X_train, Y_train = removeSampleData(df, i, a1, a2)
    print(X_train.shape)
    print(Y_train.shape)
    X_train = X_train.reshape(9999,1,12)
    X_test, Y_test = selectSampleData(df, i, a1, a2)
    print(X_test.shape)
    print(Y_test.shape)
    X_test = X_test.reshape(101,1,12)
    # Define the model architecture
    model = tf.keras.models.Sequential([
    tf.keras.Input(shape=(1,12)),
    tf.keras.layers.Dense(190, activation='relu'),
    #tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(70, activation='relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(190, activation='relu'),
    #tf.keras.layers.Dropout(0.1),
    #tf.keras.layers.Dense(140, activation='relu'),
    tf.keras.layers.Dense(1)])
    print(X_train.shape)
    print(Y_train.shape)
    # Compile the model
    opt = keras.optimizers.Adam(learning_rate=0.012545)
    model.compile(
        optimizer=opt,
        loss=['huber_loss'],
        metrics=[tf.keras.metrics.RootMeanSquaredError(),"MAPE"]
    )

    # Fit data to model
    history = model.fit(X_train, Y_train,
                batch_size=1,
                epochs=35,
                verbose=1)


    scores = model.evaluate(X_test, Y_test, verbose=0)
    print(f'Score for sample {i}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]}')
    L_MAAE.append(scores[1])

    pred_Fs = model(X_test)
    pred_Fs = pred_Fs.numpy()
    L_pred.append(pred_Fs)

#  remeber to change the output sharp based on you input file
# !strain, wear, and corrosion output files have different shape!

pd.DataFrame(L_MAAE).to_excel(path + "/model performance.xlsx", index=False, header=False)
L_pred = np.array(L_pred)
L_pred = L_pred.reshape([100, 101])
pd.DataFrame(L_pred).T.to_excel(path + "/predicted Surface profile.xlsx", index=False, header=False)

# Save the entire model as a SavedModel.
model.save('saved_model/model name')