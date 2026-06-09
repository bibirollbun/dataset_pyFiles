import pandas as pd


init_data = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
init_data = init_data.dropna()


init_data.shape





testt = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")


data = pd.concat([init_data.drop(columns=['Price']),testt])


data.shape


id = testt['id']



data.head()


data = data.drop(columns=['id'])



df = data.copy()

# Define features (X) and target (y)
X = df



y = init_data['Price'].values  # Target variable


init_data.shape


id = testt['id']


data.head()


# decision to make
"""


before drop_________________
rows : 3,00,000
column : 11

option 1) drop all rows containg them?
option 2

after drop :-

rows : 2,46,686 
column : 11


"""


from sklearn.preprocessing import LabelEncoder


encoder = LabelEncoder()


X['Brand'] = encoder.fit_transform(X['Brand'])
X['Material'] = encoder.fit_transform(X['Material'])
X['Size'] = encoder.fit_transform(X['Size'])
X['Laptop Compartment'] = encoder.fit_transform(X['Laptop Compartment'])
X['Waterproof'] = encoder.fit_transform(X['Waterproof'])
X['Style'] = encoder.fit_transform(X['Style'])
X['Color'] = encoder.fit_transform(X['Color'])


data['Brand'].unique()


test_rows = 246686
X[test_rows:]


from sklearn.model_selection import train_test_split


submit_rows = 246686  # Number of rows for submission
X_train, X_test, y_train, y_test = train_test_split(X[:submit_rows] , y, test_size=0.2, random_state=42)





submit_X =X[submit_rows:]


submit_X.shape


X.shape


y.shape





import numpy as np
from deap import base, creator, tools, algorithms
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor



# Train multiple models
models = [XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=i) for i in range(5)]
for model in models:
    model.fit(X_train, y_train)

# Get predictions from all models
predictions = np.array([model.predict(X_test) for model in models])  # Shape: (num_models, num_samples)

# GASEN: Genetic Algorithm for Selecting Best Models
num_models = len(models)
POP_SIZE = 20  # Population size
N_GEN = 50  # Number of generations
THRESHOLD = 0.2  # Minimum weight for model selection

# Define optimization objective (minimize RMSE)
def eval_fitness(individual):
    weighted_preds = np.average(predictions, axis=0, weights=individual)
    return np.sqrt(mean_squared_error(y_test, weighted_preds)),

# Genetic Algorithm Setup
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # Minimize RMSE
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_float", np.random.rand)  # Random weights
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, num_models)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Genetic Operators
toolbox.register("evaluate", eval_fitness)
toolbox.register("mate", tools.cxBlend, alpha=0.5)  # Crossover
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)  # Mutation
toolbox.register("select", tools.selTournament, tournsize=3)  # Selection

# Run Genetic Algorithm
pop = toolbox.population(n=POP_SIZE)
algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=N_GEN, verbose=True)

# Get best individual (optimal weights)
best_weights = tools.selBest(pop, k=1)[0]

# Select models with weight > THRESHOLD
selected_models = [models[i] for i, w in enumerate(best_weights) if w > THRESHOLD]
selected_weights = [w for w in best_weights if w > THRESHOLD]

# Final Prediction using Selected Models
final_predictions = np.average([model.predict(X_test) for model in selected_models], axis=0, weights=selected_weights)

# Evaluate Final Model
final_rmse = np.sqrt(mean_squared_error(y_test, final_predictions))
print("Final RMSE after GASEN:", final_rmse)
print("Selected Models:", len(selected_models))



_predictions = np.average([model.predict(submit_X) for model in selected_models], axis=0, weights=selected_weights)
final = pd.DataFrame(columns=['id','Price'])
final['id'] = id
final['Price'] = _predictions
final.to_csv("submission.csv", index=False)





# import tensorflow as tf
# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from xgboost import XGBRegressor


# from sklearn.metrics import mean_absolute_error, mean_squared_error



# from sklearn.preprocessing import StandardScaler

# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)  # Apply only on features, NOT target


# test_rows = 246686  # Specify the exact number of rows for the test set

# X_train, X_test = X[:test_rows], X[test_rows:]  # First 500 for testing, rest for training
# y_train= y
# import xgboost as xgb

# xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
# xgb_model.fit(X_train, y_train)
# y_pred_xgb = xgb_model.predict(X_test)


# #########################

# y_test = y_test.ravel()

# import numpy as np
# from deap import base, creator, tools, algorithms
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error
# from xgboost import XGBRegressor


# # Train multiple models
# models = [XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=i) for i in range(5)]
# for model in models:
#     model.fit(X_train, y_train)

# # Get predictions from all models
# predictions = np.array([model.predict(X_test) for model in models])  # Shape: (num_models, num_samples)

# # GASEN: Genetic Algorithm for Selecting Best Models
# num_models = len(models)
# POP_SIZE = 20  # Population size
# N_GEN = 50  # Number of generations
# THRESHOLD = 0.2  # Minimum weight for model selection

# # Define optimization objective (minimize RMSE)
# def eval_fitness(individual):
#     weighted_preds = np.average(predictions, axis=0, weights=individual)
#     return np.sqrt(mean_squared_error(y_test, weighted_preds)),

# # Genetic Algorithm Setup
# creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # Minimize RMSE
# creator.create("Individual", list, fitness=creator.FitnessMin)

# toolbox = base.Toolbox()
# toolbox.register("attr_float", np.random.rand)  # Random weights
# toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, num_models)
# toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# # Genetic Operators
# toolbox.register("evaluate", eval_fitness)
# toolbox.register("mate", tools.cxBlend, alpha=0.5)  # Crossover
# toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)  # Mutation
# toolbox.register("select", tools.selTournament, tournsize=3)  # Selection

# # Run Genetic Algorithm
# pop = toolbox.population(n=POP_SIZE)
# algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=N_GEN, verbose=True)

# # Get best individual (optimal weights)
# best_weights = tools.selBest(pop, k=1)[0]

# # Select models with weight > THRESHOLD
# selected_models = [models[i] for i, w in enumerate(best_weights) if w > THRESHOLD]
# selected_weights = [w for w in best_weights if w > THRESHOLD]

# # Final Prediction using Selected Models
# final_predictions = np.average([model.predict(submit_X) for model in selected_models], axis=0, weights=selected_weights)


# final = pd.DataFrame(columns=['id','Price'])
# final['id'] = id
# final['Price'] = final_predictions
# final.to_csv("submission.csv", index=False)


# print("y_test shape:", y_test.shape)  # Should match y_pred
# print("y_pred shape:", y_pred.shape)



## ensemble technique


import pandas as pd


data = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
data = data.dropna()


data.head()


data = data.drop(columns=['id'])


X = data.drop(columns=['Price'])


X.head()


y = data['Price']





# from sklearn.preprocessing import LabelEncoder
# encoder = LabelEncoder()


# X['Brand'] = encoder.fit_transform(X['Brand'])
# X['Material'] = encoder.fit_transform(X['Material'])
# X['Size'] = encoder.fit_transform(X['Size'])
# X['Laptop Compartment'] = encoder.fit_transform(X['Laptop Compartment'])
# X['Waterproof'] = encoder.fit_transform(X['Waterproof'])
# X['Style'] = encoder.fit_transform(X['Style'])
# X['Color'] = encoder.fit_transform(X['Color'])











# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error
# import numpy as np



# # Split into training and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Train a Random Forest Regressor
# rf = RandomForestRegressor(n_estimators=100, random_state=42)
# rf.fit(X_train, y_train)

# # Predict
# y_pred_random_forest = rf.predict(X_test)

# # Evaluate
# print("MSE:", np.sqrt(mean_squared_error(y_test, y_pred)))



# from xgboost import XGBRegressor


# # Train an XGBoost Regressor
# xgb = XGBRegressor(n_estimators=200, learning_rate=0.01, random_state=42,objective= 'reg:squarederror', max_depth=7,reg_lambda=1)
# xgb.fit(X_train, y_train)

# # Predict
# y_pred_xgb = xgb.predict(X_test)

# # Evaluate
# print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))



### deep learning


# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense


# # Function to create a simple deep learning model
# def build_model():
#     model = Sequential([
#         Dense(128, activation='relu',),
#         Dense(64, activation='relu'),
#         Dense(32, activation='relu'),
#         Dense(1)  # Output layer for regression
#     ])
#     model.compile(optimizer='adam', loss='mse')
#     return model

# # Train multiple models
# model1 = build_model()
# model2 = build_model()
# model3 = build_model()

# model1.fit(X_train, y_train, epochs=10, verbose=0)
# model2.fit(X_train, y_train, epochs=10, verbose=0)
# model3.fit(X_train, y_train, epochs=10, verbose=0)

# # Get predictions from each model
# y_pred1 = model1.predict(X_test)
# y_pred2 = model2.predict(X_test)
# y_pred3 = model3.predict(X_test)

# # Averaging the predictions
# y_final_deep = (y_pred1 + y_pred2 + y_pred3) / 3

# # Evaluate RMSE
# rmse = np.sqrt(np.mean((y_test - y_final) ** 2))
# print("Final Ensemble RMSE:", rmse)



# # Evaluate RMSE
# print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred1)))
# print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred2)))
# print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred3)))





# import numpy as np
# from deap import base, creator, tools, algorithms
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error
# from xgboost import XGBRegressor


# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Train multiple models
# models = [XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=i) for i in range(5)]
# for model in models:
#     model.fit(X_train, y_train)

# # Get predictions from all models
# predictions = np.array([model.predict(X_test) for model in models])  # Shape: (num_models, num_samples)

# # GASEN: Genetic Algorithm for Selecting Best Models
# num_models = len(models)
# POP_SIZE = 20  # Population size
# N_GEN = 50  # Number of generations
# THRESHOLD = 0.2  # Minimum weight for model selection

# # Define optimization objective (minimize RMSE)
# def eval_fitness(individual):
#     weighted_preds = np.average(predictions, axis=0, weights=individual)
#     return np.sqrt(mean_squared_error(y_test, weighted_preds)),

# # Genetic Algorithm Setup
# creator.create("FitnessMin", base.Fitness, weights=(-1.0,))  # Minimize RMSE
# creator.create("Individual", list, fitness=creator.FitnessMin)

# toolbox = base.Toolbox()
# toolbox.register("attr_float", np.random.rand)  # Random weights
# toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, num_models)
# toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# # Genetic Operators
# toolbox.register("evaluate", eval_fitness)
# toolbox.register("mate", tools.cxBlend, alpha=0.5)  # Crossover
# toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)  # Mutation
# toolbox.register("select", tools.selTournament, tournsize=3)  # Selection

# # Run Genetic Algorithm
# pop = toolbox.population(n=POP_SIZE)
# algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=N_GEN, verbose=True)

# # Get best individual (optimal weights)
# best_weights = tools.selBest(pop, k=1)[0]

# # Select models with weight > THRESHOLD
# selected_models = [models[i] for i, w in enumerate(best_weights) if w > THRESHOLD]
# selected_weights = [w for w in best_weights if w > THRESHOLD]

# # Final Prediction using Selected Models
# final_predictions = np.average([model.predict(X_test) for model in selected_models], axis=0, weights=selected_weights)

# # Evaluate Final Model
# final_rmse = np.sqrt(mean_squared_error(y_test, final_predictions))
# print("Final RMSE after GASEN:", final_rmse)
# print("Selected Models:", len(selected_models))



X_test





## gasen : 1 :  38.84981003922961
## gasen : 2 :  38.84981003922961








# X_train, X_test = X[:test_rows], X[test_rows:]  # First 500 for testing, rest for training
# y_train= y








# X_train.shape








# mean ensemble

# y_final = (y_pred_random_forest+ y_pred_xgb)/  2

# print("RMSE:", np.sqrt(mean_squared_error(y_test, y_final)))


# min max ensemble

# y_final_min = np.min(np.array([y_pred_random_forest , y_pred_xgb]), axis=0)  # Conservative estimate
# y_final_max = np.max(np.array([y_pred_random_forest ,  y_pred_xgb ]), axis=0)  # Aggressive estimate


# print("min ensemble RMSE:", np.sqrt(mean_squared_error(y_test, y_final_min)))
# print("max ensemble RMSE:", np.sqrt(mean_squared_error(y_test, y_final_max)))


# # weighted average

# y_final_weighted = (0.50 * y_pred_random_forest) + (0.50 * y_pred_xgb)  # More weight to model 1

#from xgboost import XGBRegressor





