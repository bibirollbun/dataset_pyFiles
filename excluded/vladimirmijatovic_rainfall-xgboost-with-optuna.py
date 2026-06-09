import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler # for scaling
import math

# plotting
import matplotlib.pyplot as plt
import seaborn as sns


#Ignore warnings
import warnings
warnings.filterwarnings('ignore')


from sklearn.model_selection import train_test_split, RandomizedSearchCV,cross_val_score,StratifiedKFold
from sklearn.impute import SimpleImputer

from xgboost import XGBClassifier

import optuna


train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")


train.shape


train.head()


train.info()


train.isnull().sum()


test.isnull().sum()


# impute missing value

imputer_model = SimpleImputer(strategy='most_frequent')
test['winddirection'] = imputer_model.fit_transform(test[['winddirection']])


train.describe().T


plt.figure(figsize = (14, 8))
sns.heatmap(train.corr(), annot = True)


# histograms

sns.histplot(data = train, x = "maxtemp",kde = True, color = "darkorchid")


sns.histplot(data = train, x = "mintemp",kde = True, color = "hotpink")


sns.histplot(data = train, x = "dewpoint",kde = True, color = 'steelblue')


# note that temperature is wrongly spelled as 'temparature'

sns.histplot(data = train, x = "temparature",kde = True, color = 'lavender')


# define all numerical columns

columns_numerical = [col for col in train.columns if col not in ['id', 'rainfall']]


columns_numerical


# Dynamically calculate number of rows & columns @adrien97

num_vars = len(columns_numerical)

num_cols = 2  # Keep 2 columns for readability
num_rows = math.ceil(num_vars / num_cols)  # Calculate rows dynamically






# Create subplots
fig, axes = plt.subplots(num_rows, num_cols, figsize=(14, num_rows * 4.5))
axes = axes.flatten()

# Color palette (color-blind friendly)
palette = sns.color_palette("Set3", n_colors=num_vars)

# Plotting each variable
for i, var in enumerate(columns_numerical):
    sns.histplot(
        data=train,
        x=var,
        kde=True,
        color=palette[i],
        bins=50,
        edgecolor="white",
        linewidth=1.3,
        ax=axes[i]
    )
    axes[i].set_title(f"{var}", fontsize=14, weight="bold")
    axes[i].set_xlabel("")
    #axes[i].set_ylabel("")
    # axes[i].tick_params(axis='x', labelrotation=15)

# Remove unused axes
#for j in range(i + 1, len(axes)):
#    fig.delaxes(axes[j])

# Adjust layout
plt.tight_layout(h_pad=2.5)
plt.show()





X_train = train.copy()
X_test = test.copy()

# remove ID column from the dataset

X_train = X_train.drop("id", axis = 1)
X_test = X_test.drop('id', axis = 1)


columns = X_train.columns


# prepare X_train and Y_train


# drop last column (target variable) in X_train to leave only predictors

X_train = X_train[columns[:-1]]


X_train.head()


Y_train = train[columns[-1]]


Y_train.head()





# add more features (feature engineering)

def feature_engineering(df):
    df = df.copy()

        
    # add time series features
    df['sin_day'] = np.sin(2 * np.pi * df['day'] / 365)
    df['cos_day'] = np.cos(2 * np.pi * df['day'] / 365)

    # add interactions 
    df['humidity_cloud'] = df['humidity'] * df['cloud']
    df['humidity_sunshine'] = df['humidity'] * df['sunshine']
    df['cloud_sunshine_ratio'] = df['cloud'] / (df['sunshine'] + 1e-5)
    df['dryness'] = 100 - df['humidity']
    df['percent_of_sunshine'] = df['sunshine'] / (df['sunshine'] + df['cloud'] + 1e-5)
    df['wi'] = (0.4 * df['humidity']) + (0.3 * df['cloud']) - (0.3 * df['sunshine'])

    # add more from @swandipsingha's idea
    df['temp_range'] = df['maxtemp'] - df['mintemp']
    df['temp_dew_diff'] = df['temparature'] - df['dewpoint']
    df['humidity_cloud_ratio'] = df['humidity'] / (df['cloud'] + 1e-3)
    df['sunshine_cloud_ratio'] = df['sunshine'] / (df['cloud'] + 1e-3)
    df['pressure_wind_interaction'] = df['pressure'] * df['winddirection']
    df['temp_pressure_ratio'] = df['temparature'] / (df['pressure'] + 1e-3)
    df['wind_pressure_ratio'] = df['windspeed'] / (df['pressure'] + 1e-3)
    

    
    return df

X_train = feature_engineering(X_train)
X_test = feature_engineering(X_test)


# dropping 'day' column

X_train = X_train.drop(columns=['day'])
X_test = X_test.drop(columns=['day'])

# scaling the input

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)




cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

X_train, X_val, Y_train, y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=42, stratify=Y_train)


# define XGBoost Optuna 


def objective_xgb(trial):
        # Define hyperparameters to tune

    params = {
    # hyperparameters for XGBClassifier
    'n_estimators' : trial.suggest_int("n_estimators", 50, 500),
    'learning_rate' : trial.suggest_float("learning_rate", 0.001, 0.3, log=True),
    'max_depth' : trial.suggest_int("max_depth", 3, 10),
    'subsample' : trial.suggest_float("subsample", 0.5, 1.0),
    'colsample_bytree' : trial.suggest_float("colsample_bytree", 0.5, 1.0),
    'use_label_encoder' : False, 
    'eval_metric': 'logloss', 
    'random_state' : 20
    }

    
    clf = XGBClassifier(**params)
    
    score = cross_val_score(clf, X_train, Y_train, cv=cv, scoring='accuracy').mean()
    return score








study_xgb = optuna.create_study(direction="maximize")


# optimize with 100 trials 

study_xgb.optimize(objective_xgb, n_trials=200)





best_params_xgb = study_xgb.best_trial.params
print("Best params for XGBoost:", best_params_xgb)


xgb_final = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, **best_params_xgb)


model = xgb_final.fit(X_train, Y_train)


predictions = xgb_final.predict_proba(X_test)





import optuna.visualization as vis
from IPython.display import IFrame, display

# Plot the optimization history (objective value over trials)
fig_history = vis.plot_optimization_history(study_xgb)
fig_history.write_html("optimization_history.html")

# Plot the parameter importances
fig_importances = vis.plot_param_importances(study_xgb)
fig_importances.write_html("parameter_importances.html")

# Display the saved HTML files inline using IFrame
display(IFrame(src="optimization_history.html", width="100%", height=500))
display(IFrame(src="parameter_importances.html", width="100%", height=500))





X_train.shape





submission = pd.DataFrame()

submission['id'] = test['id']
submission["rainfall"] = predictions[:, 1]


submission.head()


# write to csv
submission.to_csv("submission.csv", index = False)

