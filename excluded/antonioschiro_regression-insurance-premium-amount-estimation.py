# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import mutual_info_regression
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_log_error
from tensorflow import keras
from tensorflow.keras import layers
from keras import losses
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

pathlist = []
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        pathlist.append(os.path.join(dirname, filename))

print(pathlist)

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


df_train = pd.read_csv(pathlist[1], index_col = 'id')

df_train.head(10)


print(df_train.shape)
print(df_train.dtypes)


print(f'NaN values: \n {df_train.isna().sum()}')


df_train['Month Start Date'] = df_train['Policy Start Date'].apply(lambda date: str(date).split('-')[1])
df_train['Year Start Date'] = df_train['Policy Start Date'].apply(lambda date: str(date).split('-')[0])

df_train.drop('Policy Start Date', axis = 1, inplace = True)

print(df_train[['Month Start Date', 'Year Start Date']].head(10))


cat_features = [col for col in df_train.columns if df_train[col].dtype != 'float'] # Categorical features
print(cat_features)


# Convert dtype for categorical features
for col in cat_features:
    df_train[col], _ = df_train[col].factorize()
# Check if all categorical features were converted in dtype integer    
print(all(df_train[cat_features].dtypes == int))


# Check which categorical features have more than 5 unique values.
for col in cat_features:
    if df_train[col].nunique() > 5:
        print(col)


# Numerical features
num_features = [col for col in df_train.columns if df_train[col].dtype == 'float']  
num_features.remove('Premium Amount') # Remove target values
print(num_features)


X = df_train[[col for col in list(cat_features + num_features)]].copy()
y = df_train['Premium Amount']

x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.3, random_state = 1)


imputer_median = SimpleImputer(strategy = 'median')
scaler = preprocessing.StandardScaler()

pipeline = Pipeline([
    ('imputer', imputer_median),
    ('std_scaler', scaler) 
    ])

x_train_proc = pipeline.fit_transform(x_train)
x_train_proc = pd.DataFrame(x_train_proc, columns = x_train.columns)

x_valid_proc = pipeline.transform(x_valid)
x_valid_proc = pd.DataFrame(x_valid_proc, columns = x_valid.columns)


x_valid_proc.head()





# Calculate correlation values among each pair of numerical features
correlation_num_features = x_train_proc[num_features].corr(method = 'spearman') # Use Spearman any kind of monotone relationship.

# Calculate correlation between each numerical features and target.
correlation_num_target = x_train_proc[num_features].corrwith(y_train, method = 'spearman')

print('Correlation among numerical features \n')
print(correlation_num_features)
print('_'*30, '\n')
print('Correlation between numerical features and target \n')
print(correlation_num_target.sort_values(ascending= False))


def calc_mireg(X, y, discrete_features):
    
    mi_scores = mutual_info_regression(X, y, discrete_features = discrete_features)
    mi_scores = pd.Series(mi_scores, name = 'MI Scores', index = X.columns)
    mi_scores = mi_scores.sort_values(ascending = False)
    
    return mi_scores

def plot_mi_scores(scores):
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")

discrete_features = x_train_proc.dtypes == 'int64' # boolean mask to identify discrete features

mireg_scores = calc_mireg(x_train_proc, y_train, discrete_features)

plt.figure(dpi=100, figsize=(8, 5))
plot_mi_scores(mireg_scores)


# Relevant Mutual info regression features.

selected_col = [col for col in mireg_scores.index if mireg_scores[col] > 0.]
selected_col
x_train_selected = x_train_proc[selected_col]
x_valid_selected = x_valid_proc[selected_col]

print('Selected features:', len(selected_col),'\n','All features:', len(x_train_proc.columns))


def calc_score(x, y, model = HistGradientBoostingRegressor()):
    for col in x.columns:
        if x[col].dtype in ('object', 'category'):
            x[col], _ = x[col].factorize()
    score = cross_val_score(model, x, y, scoring = 'neg_mean_squared_log_error')

    score = -1*score
    score = np.sqrt(score)
    score = score.mean()
    
    return f'RMSLE: {score:.2f}'



rmsle_all_features = calc_score(x_train_proc, y_train)
rmsle_mi_features = calc_score(x_train_selected, y_train)

print(f'RMSLE calculated on all features: {rmsle_all_features}')
print(f'RMSLE calculated on mutual info regression selected features: {rmsle_mi_features}')

# Since the performances between the reduced dataset and the full one are comparable, the former will be used.


selected_col_cat = [True if col in cat_features else False for col in selected_col]


hist_gbr = HistGradientBoostingRegressor()
    
hist_gbr.fit(x_train_selected, y_train)

y_valid_predict_gbr = hist_gbr.predict(x_valid_selected)

gbr_score = np.sqrt(mean_squared_log_error(y_valid, y_valid_predict_gbr)) # Root mean squared log error

print(f'{gbr_score = :.2f}')
    



rfr = RandomForestRegressor(n_estimators=100)

rfr.fit(x_train_selected, y_train)

y_valid_predict_rfr = rfr.predict(x_valid_selected)

rfr_score = np.sqrt(mean_squared_log_error(y_valid, y_valid_predict_rfr)) # Root mean squared log error

print(f'{rfr_score = :.2f}')



input_shape = [x_train_selected.shape[1]]
msle_metric = keras.metrics.MeanSquaredLogarithmicError()
msle_loss = losses.MeanSquaredLogarithmicError(
    reduction="sum_over_batch_size", name="mean_squared_logarithmic_error", dtype=None
)


model = keras.Sequential([
                layers.Input(shape = input_shape),
                layers.Dense(32, activation = 'relu'),
                layers.Dense(1, activation = 'relu')
                ])

model.compile(
        optimizer = 'adam',
        loss = msle_loss,
        metrics = [msle_metric]
)

early_stopping = keras.callbacks.EarlyStopping(
                patience = 10,
                min_delta = 0.001,
                restore_best_weights = True
)


history = model.fit(
    x_train_selected, y_train,
    validation_data = (x_valid_selected, y_valid),
    batch_size = 512,
    epochs = 100,
    callbacks = [early_stopping],
    verbose = 0
)


history_df = pd.DataFrame(history.history)

y_bottom, y_up = 1.15, 1.3 # Define y-axis range
# Loss plot
loss_plot = history_df.loc[:, ['loss', 'val_loss']].plot()
loss_plot.set_title('Loss plots for train and validation sets')
loss_plot.set_ylim(y_bottom, y_up)
loss_plot.legend(['Train dataset', 'Validation dataset'], title = 'Loss:')

#MSLE
msle_plot = history_df.loc[:, ['mean_squared_logarithmic_error', 'val_mean_squared_logarithmic_error']].plot()
msle_plot.set_title('Mean squared logarithmic error plots for train and validation sets')
msle_plot.set_ylim(y_bottom, y_up)
msle_plot.legend(['Train dataset', 'Validation dataset'], title = 'MSLE:')


rmsle_val = np.sqrt(history_df['val_mean_squared_logarithmic_error'].iloc[-1])
rmsle = np.sqrt(history_df['mean_squared_logarithmic_error'].iloc[-1])

print(f'RMSLE for train dataset: {rmsle:.2f}')

print(f'RMSLE for validation dataset: {rmsle_val:.2f}')


df_test = pd.read_csv(pathlist[2], index_col = 'id')

df_test.head()

df_test.dtypes


df_test.head()


# Process test dataset before predicting
def processing_dataset(x, pipeline = pipeline, sorted_col = x_train_proc.columns, selected_col = selected_col):

    # Create features 
    x['Month Start Date'] = x['Policy Start Date'].apply(lambda date: str(date).split('-')[1])
    x['Year Start Date'] = x['Policy Start Date'].apply(lambda date: str(date).split('-')[0])    

    # Drop feature
    x.drop('Policy Start Date', axis = 1, inplace = True)

    columns = x.columns
    
    # Convert object dtypes in 'int'
    for col in columns:
        if x[col].dtype == 'object':
            x[col], _ = x[col].factorize()

    # Sort columns
    x = x.loc[:, sorted_col]
    
    # Processing data
    x = pipeline.transform(x)
    
    x = pd.DataFrame(x, columns = sorted_col) # Reconvert x in Dataframe since imputer converted it in ndarray

    x = x[selected_col] # Select column with mutual information regression value > 0.

    return x
    


processed_test = processing_dataset(df_test)


y_test_predict = model.predict(processed_test, batch_size=512)


y_test_predict_ravel = y_test_predict.ravel()
print(f'ravel: {y_test_predict_ravel.shape}')


submission = pd.DataFrame({'id': df_test.index, 'Premium Amount': y_test_predict_ravel})
submission.reset_index(drop = True)

print(submission.head())


submission.to_csv('./submission.csv', index = False)

