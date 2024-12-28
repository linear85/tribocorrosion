# coding: utf-8
# Author: YG
# Dec 2022

import json
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import os
import random
import matplotlib.pyplot as plt

# =============================================================================================================================================

# List to store results
Results = []

# Normalize a feature using mean and range
def feature_normalize(Feature, mean_value, range_value):
    
    Feature = Feature - mean_value
    Feature = Feature / range_value
    return Feature

# Calculate normalization parameters (mean and range)
def getNormPara(Lx):
    
    meanNorm = np.mean(Lx, axis=0)
    rangeNorm = np.std(Lx, axis=0) + (10 ** -8)
    return meanNorm, rangeNorm

# Select data
def selectSampleData(df, a1, a2):
   
    X = df
    X = feature_normalize(X, a1, a2)
    X = np.asarray(X).astype(np.float32)
    return X

# Load trained models
PS1_model1 = tf.keras.models.load_model('saved_model/Model_PS1_Version1')
PS2_model1 = tf.keras.models.load_model('saved_model/Model_PS2_Version1')
PS3_model1 = tf.keras.models.load_model('saved_model/Model_PS3_Version1')
Wear_model1 = tf.keras.models.load_model('saved_model/Model_Wear_Version1')
Corr_model1 = tf.keras.models.load_model('saved_model/Model_Corr_Version1')

PS1_model2 = tf.keras.models.load_model('saved_model/Model_PS1_Version2')
PS2_model2 = tf.keras.models.load_model('saved_model/Model_PS2_Version2')
PS3_model2 = tf.keras.models.load_model('saved_model/Model_PS3_Version2')
Wear_model2 = tf.keras.models.load_model('saved_model/Model_Wear_Version2')
Corr_model2 = tf.keras.models.load_model('saved_model/Model_Corr_Version2')

PS1_model3 = tf.keras.models.load_model('saved_model/Model_PS1_Version3')
PS2_model3 = tf.keras.models.load_model('saved_model/Model_PS2_Version3')
PS3_model3 = tf.keras.models.load_model('saved_model/Model_PS3_Version3')
Wear_model3 = tf.keras.models.load_model('saved_model/Model_Wear_Version3')
Corr_model3 = tf.keras.models.load_model('saved_model/Model_Corr_Version3')

# Load orginal dataset
PathOr = os.getcwd()
df2 = pd.read_excel(PathOr + "/Input_Data_File.xlsx", index_col=False)

# Indenter info
location = np.array([0,0.02,0.04,0.06,0.08,0.1,0.12,0.14,0.16,0.18,0.2,0.22,0.24,0.26,0.28,0.3,0.32,0.34,0.36,0.38,0.4,0.42,0.44,0.46,0.48,0.5,0.52,0.54,0.56,0.58,0.6,0.62,0.64,0.66,0.68,0.7,0.72,0.74,0.76,0.78,0.8,0.82,0.84,0.86,0.88,0.9,0.92,0.94,0.96,0.98,1,1.02,1.04,1.06,1.08,1.1,1.12,1.14,1.16,1.18,1.2,1.22,1.24,1.26,1.28,1.3,1.32,1.34,1.36,1.38,1.4,1.42,1.44,1.46,1.48,1.5,1.52,1.54,1.56,1.58,1.6,1.62,1.64,1.66,1.68,1.7,1.72,1.74,1.76,1.78,1.8,1.82,1.84,1.86,1.88,1.9,1.92,1.94,1.96,1.98,2]).reshape(-1, 1)
shape = np.array([0,0.000100002500124985,0.000400040008001978,0.000900202591176402,0.00160064051251263,0.0025015644561821,0.00360324584515515,0.00490601725131756,0.00641027289966201,0.00811646926834597,0.0100251257867601,0.0121368256341183,0.0144522166414629,0.0169720123003811,0.0196969928821498,0.022628006671481,0.0257659713195093,0.029111875321179,0.03266677962273,0.0364318193655715,0.0404082057734576,0.0445972281905704,0.0490002562788483,0.0536187423837025,0.0584542240781445,0.0635083268962915,0.0687827672682702,0.0742793556696755,0.0800000000000001,0.0859467092058277,0.0921215971661087,0.0985268868585072,0.105164914827678,0.112038135978377,0.1191491287186,0.126500600480481,0.134095393649504,0.141936491935757,0.150027027224452,0.158370286946912,0.166969722017664,0.175828955388229,0.184951791273852,0.194342225115734,0.204004454348508,0.213942890050825,0.224162169566151,0.234667170191411,0.245463024043095,0.256555134224199,0.267949192431123,0.279651198157769,0.29166747967499,0.304004716987692,0.316669966999935,0.329670691150993,0.343014785823362,0.356710615868282,0.370767051646696,0.385193510045244,0.4,0.415197173147398,0.430796380325358,0.446809734771686,0.463250183016116,0.480131584642934,0.497468802320564,0.515277803762603,0.533575777614131,0.552381265664194,0.57171431429143,0.59159664868334,0.612051874168202,0.633105710012658,0.654786262335981,0.677124344467705,0.700153855258246,0.723912228724058,0.748440972227039,0.773786315522454,0.8,0.82714024708834,0.855272958299665,0.884473218609253,0.914827202699957,0.946434624714726,0.979411934226154,1.01389655715032,1.05005263303697,1.08807895078576,1.12822021129187,1.17078350233489,1.21616328230938,1.26488096202044,1.31765111563072,1.37550020016016,1.44,1.51379016875427,1.60200502515735,1.71786528040668,2]).reshape(-1, 1)
LS = np.concatenate((location, shape), axis=1)

# Define input ranges for genetic algorithm
input_ranges = [(55, 95), (1, 5), (-1.21761, -0.9511), (2.74e-8, 2.4e-7)]

# Genetic algorithm parameters
population_size = 100
generations = 100
mutation_rate = 0.1
num_predictions = 1

# Initialize population randomly
population = []
for i in range(population_size):
    individual = []
    for j in range(4):
        value = np.random.uniform(*input_ranges[j])
        individual.append(np.full(101, value))
    population.append(individual)
  
# Function to evaluate an individual
def evaluate_individual(individual):
    output1 = []  # Store material loss outputs
    output2 = []  # Store wear outputs

    for i in range(num_predictions):
        # Extract features from the dataframe and normalize
        X = df2.iloc[:, 1:-5].values
        [a1, a2] = getNormPara(X)

        X = df2.iloc[:, 1:-2].values
        [a3, a4] = getNormPara(X)

        X = df2.iloc[:, 1:-1].values
        [a5, a6] = getNormPara(X)

        # Prepare input data for prediction
        input_data0 = np.array(individual).T
        input_data = np.expand_dims(np.concatenate((input_data0, LS), axis=1), axis=1)

        input_data11 = input_data
        input_data22 = input_data

        # Step 1: Predict outputs using PS1, PS2, PS3 models
        input_data0 = selectSampleData(input_data, a1, a2)
        outputs = PS1_model3.predict(input_data0, verbose=0)
        input_data = np.concatenate((input_data, outputs), axis=2)
        outputs = PS2_model3.predict(input_data0, verbose=0)
        input_data = np.concatenate((input_data, outputs), axis=2)
        outputs = PS3_model3.predict(input_data0, verbose=0)
        input_data = np.concatenate((input_data, outputs), axis=2)

        # Step 2: Predict wear outputs
        input_data1 = selectSampleData(input_data, a3, a4)
        outputs = Wear_model3.predict(input_data1, verbose=0)
        wearOut = outputs
        input_data = np.concatenate((input_data, outputs), axis=2)

        # Step 3: Predict corrosion outputs
        input_data1 = selectSampleData(input_data, a5, a6)
        outputs = Corr_model3.predict(input_data1, verbose=0)
        input_data = np.concatenate((input_data, outputs), axis=2)

        # Calculate material loss areas
        x_values = location.reshape(-1)
        y_values = outputs.reshape(101,)
        area_above_below = np.trapz(np.maximum(y_values, 0), x_values)
        area_below_above = np.trapz(np.maximum(-y_values, 0), x_values)
        outputs = area_above_below + area_below_above
        output1.append(outputs)

        # Calculate wear loss areas
        wear_values = wearOut.reshape(101,)
        wear_area_above_below = np.trapz(np.maximum(wear_values, 0), x_values)
        wear_area_below_above = np.trapz(np.maximum(-wear_values, 0), x_values)
        wear_outputs = wear_area_above_below + wear_area_below_above
        output2.append(wear_outputs)

        # Repeat predictions for input_data11 with PS1-2, PS2-2, PS3-2 models
        input_data111 = selectSampleData(input_data11, a1, a2)
        outputs = PS1_model1.predict(input_data111, verbose=0)
        input_data11 = np.concatenate((input_data11, outputs), axis=2)
        outputs = PS2_model1.predict(input_data111, verbose=0)
        input_data11 = np.concatenate((input_data11, outputs), axis=2)
        outputs = PS3_model1.predict(input_data111, verbose=0)
        input_data11 = np.concatenate((input_data11, outputs), axis=2)

        # Predict wear and corrosion outputs 
        input_data111 = selectSampleData(input_data11, a3, a4)
        outputs = Wear_model1.predict(input_data111, verbose=0)
        wearOut = outputs
        input_data11 = np.concatenate((input_data11, outputs), axis=2)

        input_data111 = selectSampleData(input_data11, a5, a6)
        outputs = Corr_model1.predict(input_data111, verbose=0)

        # Calculate material loss areas 
        x_values = location.reshape(-1)
        y_values = outputs.reshape(101,)
        area_above_below = np.trapz(np.maximum(y_values, 0), x_values)
        area_below_above = np.trapz(np.maximum(-y_values, 0), x_values)
        outputs = area_above_below + area_below_above
        output1.append(outputs)

        # Calculate wear loss areas 
        wear_values = wearOut.reshape(101,)
        wear_area_above_below = np.trapz(np.maximum(wear_values, 0), x_values)
        wear_area_below_above = np.trapz(np.maximum(-wear_values, 0), x_values)
        wear_outputs = wear_area_above_below + wear_area_below_above
        output2.append(wear_outputs)

        # Repeat predictions for input_data22 with PS1-3, PS2-3, PS3-3 models
        input_data222 = selectSampleData(input_data22, a1, a2)
        outputs = PS1_model2.predict(input_data222, verbose=0)
        input_data22 = np.concatenate((input_data22, outputs), axis=2)
        outputs = PS2_model2.predict(input_data222, verbose=0)
        input_data22 = np.concatenate((input_data22, outputs), axis=2)
        outputs = PS3_model2.predict(input_data222, verbose=0)
        input_data22 = np.concatenate((input_data22, outputs), axis=2)

        # Predict wear and corrosion outputs 
        input_data222 = selectSampleData(input_data22, a3, a4)
        outputs = Wear_model2.predict(input_data222, verbose=0)
        wearOut = outputs
        input_data22 = np.concatenate((input_data22, outputs), axis=2)

        input_data222 = selectSampleData(input_data22, a5, a6)
        outputs = Corr_model2.predict(input_data222, verbose=0)

        # Calculate material loss areas 
        x_values = location.reshape(-1)
        y_values = outputs.reshape(101,)
        area_above_below = np.trapz(np.maximum(y_values, 0), x_values)
        area_below_above = np.trapz(np.maximum(-y_values, 0), x_values)
        outputs = area_above_below + area_below_above
        output1.append(outputs)

        # Calculate wear loss areas 
        wear_values = wearOut.reshape(101,)
        wear_area_above_below = np.trapz(np.maximum(wear_values, 0), x_values)
        wear_area_below_above = np.trapz(np.maximum(-wear_values, 0), x_values)
        wear_outputs = wear_area_above_below + wear_area_below_above
        output2.append(wear_outputs)

    # Calculate average and standard deviation of outputs
    avgoutput = np.mean(output1)
    avgwear = np.mean(output2)
    std1 = np.std(output1)
    std2 = np.std(output2)

    return avgoutput, std1, avgwear, std2
  
# Evaluate the initial population
scores1 = [evaluate_individual(individual) for individual in population]
scores = [result[0] for result in scores1]
Stds = [result[1] for result in scores1]
wearloss = [result[2] for result in scores1]
wearstd = [result[3] for result in scores1]
fscores = np.add(scores, Stds)  # Combined  score including standard deviations

# Start the genetic algorithm loop
for i in range(generations):
    print("Case #:", i + 1)

    # Select the best individuals for mating
    sorted_population = [x for _, x in sorted(zip(fscores, population))]  # Sort population by  score
    top_individuals = random.sample(sorted_population, k=int(population_size / 2))  # Select top half randomly

    # Mate the top individuals to generate the next generation
    offspring = []
    for j in range(population_size - len(top_individuals)):
        parent1 = top_individuals[np.random.randint(0, len(top_individuals))]  # Random parent selection
        offspring.append(np.mean([parent1], axis=0))  # Generate offspring by averaging

    # Add some random mutation to the offspring
    for j in range(len(offspring)):
        for k in range(4):  # Apply mutation to the first 4 parameters
            original_value = offspring[j][k][0]
            change_percentage = np.random.normal(0, mutation_rate)  # Mutation rate as percentage
            change_amount = original_value * change_percentage
            new_value = original_value + change_amount
            # Ensure that the new value is within the input range
            new_value = max(new_value, input_ranges[k][0])
            new_value = min(new_value, input_ranges[k][1])
            offspring[j][k] = np.full(101, new_value)

    # Evaluate the new population
    population = top_individuals + offspring
    scores1 = [evaluate_individual(individual) for individual in population]
    scores = [result[0] for result in scores1]
    Stds = [result[1] for result in scores1]
    wearloss = [result[2] for result in scores1]
    wearstd = [result[3] for result in scores1]
    fscores = np.add(scores, Stds)  # Recalculate fitness scores

    # Record results for each individual
    for individual in population:
        first_elements = [col[0] for col in individual]  # Extract first element of each column
        score, Std, wearloss, wearstd = evaluate_individual(individual)  # Evaluate individual
        row = first_elements + [score] + [Std] + [wearloss] + [wearstd]  # Combine data into a row
        Results.append(row)  # Append row to results

    # Replace the old population with the new offspring
    population = offspring

# Save the results to files
df = pd.DataFrame(Results, columns=["col1", "col2", "col3", "col4", "score", "Std", "wearloss", "wearstd"])
para=np.array(population)
para=para[0,:,:]
para=np.array(para)

#mloss=np.array(scores)
#mloss=mloss[0,:,:]
#mloss=np.array(mloss).T

# Save results to an Excel file
path = os.getcwd()
df.to_excel(path + "/results.xlsx", index=False, header=False)  # Save results
pd.DataFrame(para).T.to_excel(path + "/parameters.xlsx", index=False, header=False)  # Save parameters
#pd.DataFrame(mloss).T.to_excel(path + "/leave-one-out prediction outs.xlsx", index=False, header=False)