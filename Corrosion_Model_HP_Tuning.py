# YG 
# Oct 2022
# First, use this file to tune the hyperparameters (HPs). 
# Then, utilize TensorBoard to analyze the results and identify the most desirable combinations. 
# Finally, apply these combinations in the 'Corrosion_Model_Training' file to train and save the models.

import tensorflow as tf
from tensorflow import keras 
import keras_tuner 
import numpy as np
import pandas as pd
import os
from tensorflow.keras import layers

# =================================================================================================
# Helper functions

# Normalize features using mean and range.
def feature_normalize(feature, mean_value, range_value):
    feature = (feature - mean_value) / range_value
    return feature

# Compute mean and range for normalization.
def get_norm_params(data):
    mean_norm = np.mean(data, axis=0)
    range_norm = np.std(data, axis=0) + (10 ** -8)
    return mean_norm, range_norm

#  Select and normalize data for a specific group.
def select_sample_data(df, group_id, mean_value, range_value):
    X = df.iloc[group_id * 101:(group_id + 1) * 101, 1:-1]
    X = feature_normalize(X, mean_value, range_value)
    X = np.asarray(X).astype(np.float32)
    Y = df.iloc[group_id * 101:(group_id + 1) * 101, -1]
    Y = np.asarray(Y).astype(np.float32)
    return X, Y

# Remove data from a specific group and normalize the rest.
def remove_sample_data(df, group_id, mean_value, range_value):

    df = df.loc[~df["Group"].isin([group_id])]
    X = df.iloc[:, 1:-1]
    X = feature_normalize(X, mean_value, range_value)
    X = np.asarray(X).astype(np.float32)
    Y = df.iloc[:, -1]
    Y = np.asarray(Y).astype(np.float32)
    return X, Y

#
def build_model(hp):
    model = tf.keras.models.Sequential()

    model.add(layers.Dense(
        units=hp.Int("units1", min_value=50, max_value=200, step=10),
        activation=hp.Choice("activation1", ["relu", "sigmoid", "elu", "softmax"])
    ))
    if hp.Boolean("dropout1"):
        model.add(layers.Dropout(rate=0.1))

    model.add(layers.Dense(
        units=hp.Int("units2", min_value=30, max_value=200, step=10),
        activation=hp.Choice("activation2", ["relu", "sigmoid", "elu", "softmax"])
    ))
    if hp.Boolean("dropout2"):
        model.add(layers.Dropout(rate=0.1))

    model.add(layers.Dense(
        units=hp.Int("units3", min_value=10, max_value=200, step=10),
        activation=hp.Choice("activation3", ["relu", "sigmoid", "elu", "softmax"])
    ))

    model.add(layers.Dense(1))

    learning_rate = hp.Float("lr", min_value=1e-5, max_value=1e-2, sampling="log")
    optimizer = keras.optimizers.Adam(learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="mean_squared_error",
        metrics=[tf.keras.metrics.RootMeanSquaredError(), "MAPE"]
    )
    return model

# =================================================================================================
# Main function

# Load data
path = os.getcwd()
data_file = path + "/input_data.xlsx"  # Change the input file to yours
df = pd.read_excel(data_file, index_col=False)

# Normalize features
X = df.iloc[:, 1:-1]
mean_norm, range_norm = get_norm_params(X)

# Lists to store results
maae_list = []
predictions_list = []
callbacks_list = [
    keras.callbacks.TensorBoard(log_dir=path + "/logs/tmp/", histogram_freq=1),
    tf.keras.callbacks.EarlyStopping(monitor="mean_squared_error", patience=3),
]

# change the range based on you input

for group_id in range(106):
    # Prepare training and testing data
    X_train, Y_train = remove_sample_data(df, group_id, mean_norm, range_norm)
    X_test, Y_test = select_sample_data(df, group_id, mean_norm, range_norm)

    X_train = X_train.reshape(-1, 1, 10)
    X_test = X_test.reshape(-1, 1, 10)

    # Hyperparameter tuning
    tuner = keras_tuner.Hyperband(
        hypermodel=build_model,
        objective=keras_tuner.Objective("root_mean_squared_error", "min"),
        max_epochs=35,
        factor=3,
        overwrite=True,
        directory=path + "/hyperparam_tuning/",
        project_name="model_tuning"
    )

    tuner.search(
        X_train, Y_train,
        batch_size=1,
        epochs=50,
        callbacks=callbacks_list,
        verbose=0
    )

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    model = build_model(best_hps)

    # Train the model with combined data
    X_all = np.concatenate((X_train, X_test))
    Y_all = np.concatenate((Y_train, Y_test))
    model.fit(X_all, Y_all, epochs=100, callbacks=callbacks_list)

    # Evaluate the model
    scores = model.evaluate(X_test, Y_test, verbose=0)
    print(f"Group {group_id}: Loss={scores[0]}, RMSE={scores[1]}")

    maae_list.append(scores[1])
    predictions_list.append(model.predict(X_test).flatten())

# Save results
output_path = path + "/output/"
os.makedirs(output_path, exist_ok=True)
pd.DataFrame(maae_list).to_excel(output_path + "performance.xlsx", index=False, header=False)
pd.DataFrame(np.array(predictions_list).reshape(106, 101)).T.to_excel(output_path + "predictions.xlsx", index=False, header=False)
