#/ coding: utf-8
# authour:  YG et al.
# Nov 22

import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import os

# ============================================================================
# Helper functions

# Normalize a feature using given mean and range

def feature_normalize(feature, mean_value, range_value):
    feature = (feature - mean_value) / range_value
    return feature

# Compute normalization parameters (mean and standard deviation)
def get_norm_parameters(data):
    mean_value = np.mean(data, axis=0)
    range_value = np.std(data, axis=0) + 1e-8
    return mean_value, range_value

# Select sample data for a specific group and normalize it
def select_sample_data(df, group, mean, range_):
    data = df.iloc[group * 101:(group + 1) * 101, 1:-1]
    normalized_data = feature_normalize(data, mean, range_)
    return np.asarray(normalized_data).astype(np.float32)

# Remove specific group data and normalize remaining data
def remove_sample_data(df, group, mean, range_):
    filtered_df = df.loc[~df["Group"].isin([group])]
    data = filtered_df.iloc[:, 1:-1]
    normalized_data = feature_normalize(data, mean, range_)
    return np.asarray(normalized_data).astype(np.float32)

# ============================================================================
# Load pre-trained models
model_1 = tf.keras.models.load_model('saved_model/model_Strain1')
model_2 = tf.keras.models.load_model('saved_model/model_Strain2')
model_3 = tf.keras.models.load_model('saved_model/model_Strain3')
model_wear = tf.keras.models.load_model('saved_model/model_wear')
model_corr = tf.keras.models.load_model('saved_model/model_corr')

# Print model summaries
model_1.summary()
model_2.summary()
model_3.summary()
model_wear.summary()
model_corr.summary()

# ============================================================================
# Setup parameters

# Define location and shape arrays
location = np.array([0,0.02,0.04,0.06,0.08,0.1,0.12,0.14,0.16,0.18,0.2,0.22,0.24,0.26,0.28,0.3,0.32,0.34,0.36,0.38,0.4,0.42,0.44,0.46,0.48,0.5,0.52,0.54,0.56,0.58,0.6,0.62,0.64,0.66,0.68,0.7,0.72,0.74,0.76,0.78,0.8,0.82,0.84,0.86,0.88,0.9,0.92,0.94,0.96,0.98,1,1.02,1.04,1.06,1.08,1.1,1.12,1.14,1.16,1.18,1.2,1.22,1.24,1.26,1.28,1.3,1.32,1.34,1.36,1.38,1.4,1.42,1.44,1.46,1.48,1.5,1.52,1.54,1.56,1.58,1.6,1.62,1.64,1.66,1.68,1.7,1.72,1.74,1.76,1.78,1.8,1.82,1.84,1.86,1.88,1.9,1.92,1.94,1.96,1.98,2]).reshape(-1, 1)
shape = np.array([0,0.000100002500124985,0.000400040008001978,0.000900202591176402,0.00160064051251263,0.0025015644561821,0.00360324584515515,0.00490601725131756,0.00641027289966201,0.00811646926834597,0.0100251257867601,0.0121368256341183,0.0144522166414629,0.0169720123003811,0.0196969928821498,0.022628006671481,0.0257659713195093,0.029111875321179,0.03266677962273,0.0364318193655715,0.0404082057734576,0.0445972281905704,0.0490002562788483,0.0536187423837025,0.0584542240781445,0.0635083268962915,0.0687827672682702,0.0742793556696755,0.0800000000000001,0.0859467092058277,0.0921215971661087,0.0985268868585072,0.105164914827678,0.112038135978377,0.1191491287186,0.126500600480481,0.134095393649504,0.141936491935757,0.150027027224452,0.158370286946912,0.166969722017664,0.175828955388229,0.184951791273852,0.194342225115734,0.204004454348508,0.213942890050825,0.224162169566151,0.234667170191411,0.245463024043095,0.256555134224199,0.267949192431123,0.279651198157769,0.29166747967499,0.304004716987692,0.316669966999935,0.329670691150993,0.343014785823362,0.356710615868282,0.370767051646696,0.385193510045244,0.4,0.415197173147398,0.430796380325358,0.446809734771686,0.463250183016116,0.480131584642934,0.497468802320564,0.515277803762603,0.533575777614131,0.552381265664194,0.57171431429143,0.59159664868334,0.612051874168202,0.633105710012658,0.654786262335981,0.677124344467705,0.700153855258246,0.723912228724058,0.748440972227039,0.773786315522454,0.8,0.82714024708834,0.855272958299665,0.884473218609253,0.914827202699957,0.946434624714726,0.979411934226154,1.01389655715032,1.05005263303697,1.08807895078576,1.12822021129187,1.17078350233489,1.21616328230938,1.26488096202044,1.31765111563072,1.37550020016016,1.44,1.51379016875427,1.60200502515735,1.71786528040668,2]).reshape(-1, 1)

location_shape = np.concatenate((location, shape), axis=1)

# Define ranges for inputs
input_ranges = [(60, 60), (1.5, 4.5), (-1.15, -1), (3e-8, 2.2e-7)]

# Genetic Algorithm Parameters
population_size = 200
generations = 5
mutation_rate = 0.1

# Initialize population with random values
population = []
for _ in range(population_size):
    individual = []
    for low, high in input_ranges:
        value = np.random.uniform(low, high)
        individual.append(np.full(101, value))
    population.append(individual)

# ============================================================================
# Evaluate an individual
def evaluate_individual(individual):
    input_data = np.expand_dims(np.concatenate((np.array(individual).T, location_shape), axis=1), axis=1)
    
    output_1 = model_1.predict(input_data)
    input_data = np.concatenate((input_data, output_1), axis=2)

    output_2 = model_2.predict(input_data)
    input_data = np.concatenate((input_data, output_2), axis=2)

    output_3 = model_3.predict(input_data)
    input_data = np.concatenate((input_data, output_3), axis=2)

    wear_output = model_wear.predict(input_data)
    input_data = np.concatenate((input_data, wear_output), axis=2)

    corr_output = model_corr.predict(input_data)

    x_values = location.flatten()
    y_values = corr_output.flatten()
    area_total = np.trapz(np.abs(y_values), x_values)

    wear_values = wear_output.flatten()
    wear_area = np.trapz(np.abs(wear_values), x_values)

    return area_total, wear_area

# ============================================================================
# Evaluate initial population
scores = [evaluate_individual(ind) for ind in population]

# ============================================================================
# Run genetic algorithm
results = []
for gen in range(generations):
    print(f"Generation: {gen + 1}")
    sorted_population = [ind for _, ind in sorted(zip(scores, population))]
    top_individuals = sorted_population[:population_size // 5]

    # Generate offspring through crossover
    offspring = []
    for _ in range(population_size - len(top_individuals)):
        parent1, parent2 = np.random.choice(top_individuals, 2, replace=False)
        offspring.append(np.mean([parent1, parent2], axis=0))

    # Apply mutations
    for ind in offspring:
        for i in range(len(ind)):
            mutation = np.random.normal(0, mutation_rate) * ind[i][0]
            ind[i] = np.full(101, ind[i][0] + mutation)

    # Update population and scores
    population = top_individuals + offspring
    scores = [evaluate_individual(ind) for ind in population]

    # Log results
    for ind, score in zip(population, scores):
        results.append(list(ind[0]) + [score])

# ============================================================================
# Save results to Excel
path = os.getcwd()
df = pd.DataFrame(results, columns=["param1", "param2", "param3", "param4", "score"])
df.to_excel(os.path.join(path, "results.xlsx"), index=False)

final_inputs = np.array(population[0])
pd.DataFrame(final_inputs.T).to_excel(os.path.join(path, "final_inputs.xlsx"), index=False)