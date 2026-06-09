# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import optuna
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor, XGBClassifier
from catboost import CatBoostRegressor, CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import KBinsDiscretizer
import warnings
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
warnings.filterwarnings("ignore", category=FutureWarning)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


df_train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')


df_train.describe()


df_train.head()


df_test.head()


df_train['Fertilizer Name'].value_counts()


df_train.columns


numerical_features = [ 'Temparature', 'Humidity', 'Moisture',
       'Nitrogen', 'Potassium', 'Phosphorous']

for feature in numerical_features:
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.histplot(df_train[feature], kde=True, bins=30)
    plt.title(f"Histogram of {feature}")
    plt.xlabel(feature)
    plt.ylabel("Frequency")

    plt.subplot(1, 2, 2)
    sns.boxplot(x=df_train[feature])
    plt.title(f"Box Plot of {feature}")

    plt.tight_layout()
    plt.show()

    print(f"\nStatistics for {feature}:")
    print(f"Skewness: {df_train[feature].skew():.2f}")
    print(f"Number of Missing Values: {df_train[feature].isnull().sum()}")





categorical_features = [
    'Soil Type',
    'Crop Type',
    'Fertilizer Name'
]
for feature in categorical_features:
    counts = df_train[feature].value_counts()
    
    plt.figure(figsize=(6, 6))
    plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90)
    plt.title(f"Distribution of {feature}")
    plt.axis("equal")
    plt.show()
    
    print(f"Number of Unique {feature}: {df_train[feature].nunique()}")
    print(f"Missing Values in {feature}: {df_train[feature].isnull().sum()}")


colors = sns.color_palette('husl', len(numerical_features))

rows = -(-len(numerical_features) // 4)
plt.figure(figsize=(20, 5 * rows))

for i, (col, color) in enumerate(zip(numerical_features, colors), 1):
    plt.subplot(rows, 4, i)
    sns.kdeplot(data=df_train, x=col, fill=True, color=color)
    plt.title(f'KDE Plot of {col}', fontsize=14, color=color)
    plt.xlabel(col)
    plt.ylabel('Density')

plt.tight_layout()
plt.show()


numeric_df = df_train.select_dtypes(include='number')

sns.pairplot(numeric_df, corner=True, plot_kws={'alpha': 0.5})
plt.suptitle('Pairwise Scatter Plots', y=1.02)
plt.show()


for feature in numerical_features[:-1]:  
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x=df_train[feature], y=df_train["Fertilizer Name"], alpha=0.5
    )
    plt.title(f"{feature} vs. Fertilizer Name")
    plt.xlabel(feature)
    plt.ylabel("Fertilizer Name")
    plt.show()

correlation_matrix = df_train[numerical_features].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix of Numerical Features")
plt.show()



colors = sns.color_palette('husl', len(numerical_features))

rows = -(-len(numerical_features) // 4)
plt.figure(figsize=(20, 5 * rows))

for i, (col, color) in enumerate(zip(numerical_features, colors), 1):
    plt.subplot(rows, 4, i)
    sns.lineplot(data=df_train[col], color=color)
    plt.title(f'Trend Plot of {col}', fontsize=14, color=color)
    plt.xlabel('Index')
    plt.ylabel(col)

plt.tight_layout()
plt.show()


colors = sns.color_palette('husl', len(numerical_features))
rows = -(-len(numerical_features) // 4)
plt.figure(figsize=(20, 5 * rows))

for i, (col, color) in enumerate(zip(numerical_features, colors), 1):
    plt.subplot(rows, 4, i)
    sns.kdeplot(data=df_train, x=col, fill=True, color=color)
    sns.lineplot(data=df_train[col].sort_values().reset_index(drop=True), color='black', linewidth=1)
    plt.title(f'KDE + Trend of {col}', fontsize=14, color=color)
    plt.xlabel(col)
    plt.ylabel('Density')

plt.tight_layout()
plt.show()


colors = sns.color_palette('husl', len(numerical_features))
rows = -(-len(numerical_features) // 4)
plt.figure(figsize=(20, 5 * rows))

for i, (col, color) in enumerate(zip(numerical_features, colors), 1):
    plt.subplot(rows, 4, i)
    sns.violinplot(data=df_train, y=col, color=color)
    plt.title(f'Violin Plot of {col}', fontsize=14, color=color)
    plt.xlabel('')
    plt.ylabel(col)

plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 6))
sns.countplot(data=df_train, x='Soil Type', hue='Fertilizer Name')
plt.title('Fertilizer Distribution per Soil Type')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



plt.figure(figsize=(10, 6))
sns.countplot(data=df_train, x='Crop Type', hue='Fertilizer Name')
plt.title('Fertilizer Distribution per Crop Type')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



cross_tab_soil = pd.crosstab(df_train['Soil Type'], df_train['Fertilizer Name'], normalize='index')
plt.figure(figsize=(10, 6))
sns.heatmap(cross_tab_soil, annot=True, cmap='viridis')
plt.title('Normalized Fertilizer Usage by Soil Type')
plt.ylabel('Soil Type')
plt.xlabel('Fertilizer Name')
plt.tight_layout()
plt.show()



cross_tab_crop = pd.crosstab(df_train['Crop Type'], df_train['Fertilizer Name'], normalize='index')
plt.figure(figsize=(10, 6))
sns.heatmap(cross_tab_crop, annot=True, cmap='plasma')
plt.title('Normalized Fertilizer Usage by Crop Type')
plt.ylabel('Crop Type')
plt.xlabel('Fertilizer Name')
plt.tight_layout()
plt.show()



df_train['Soil_Crop'] = df_train['Soil Type'] + '_' + df_train['Crop Type']

plt.figure(figsize=(12, 6))
sns.countplot(data=df_train, x='Soil_Crop', hue='Fertilizer Name')
plt.title('Fertilizer Distribution per Soil-Crop Combination')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()



nutrients = ['Nitrogen', 'Potassium', 'Phosphorous', 'Temparature', 'Humidity', 'Moisture']

for nutrient in nutrients:
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df_train, x='Soil Type', y=nutrient)
    plt.title(f'{nutrient} levels across Soil Types')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()



for nutrient in nutrients:
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df_train, x='Crop Type', y=nutrient)
    plt.title(f'{nutrient} levels across Crop Types')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()



for nutrient in nutrients:
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df_train, x='Fertilizer Name', y=nutrient)
    plt.title(f'{nutrient} levels across Fertilizer Names')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


from itertools import combinations

def generate_interaction_features(df, feature_list):
    for f1, f2 in combinations(feature_list, 2):
        df[f'{f1}_x_{f2}'] = df[f1] * df[f2]
    return df


df_train = generate_interaction_features(df_train, numerical_features)
df_test = generate_interaction_features(df_test, numerical_features)
df_train.head()


df_test.head()


print(df_train.columns)
print(df_test.columns)


# One-hot encode
df_train_encoded = pd.get_dummies(df_train, columns=['Soil Type', 'Crop Type'])
df_test_encoded = pd.get_dummies(df_test, columns=['Soil Type', 'Crop Type'])

# Standardize all numerical columns (including interaction terms)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
num_cols = df_train.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Exclude target
if 'Fertilizer Name' in num_cols:
    num_cols.remove('Fertilizer Name')

df_train_encoded[num_cols] = scaler.fit_transform(df_train_encoded[num_cols])
df_test_encoded[num_cols] = scaler.transform(df_test_encoded[num_cols])




df_train_encoded


df_test_encoded


print(df_train_encoded.columns)
print(df_test_encoded.columns)


df_train.loc[0]


df_train_encoded.loc[0]


df_train = df_train_encoded
df_test = df_test_encoded
df_train, df_test = df_train.align(df_test, join='left', axis=1, fill_value=0)


print(df_train.columns)
print(df_test.columns)


drop_cols = ['id', 'Fertilizer Name', 'Soil_Crop']
x_train = df_train.drop(columns = drop_cols)
y_train = df_train['Fertilizer Name']
x_test = df_test.drop(columns = drop_cols)



one_hot_cols = x_train.select_dtypes(include=['bool']).columns.tolist()
for col in one_hot_cols:
    x_train[col]= x_train[col].astype('int')


for col in one_hot_cols:
    x_test[col]= x_test[col].astype('int')


y_train.value_counts()


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y_train = le.fit_transform(y_train)


np.unique(y_train)


import numpy as np

def apk(actual, predicted, k=3):
    """
    Computes the average precision at k.
    actual: true label (single value)
    predicted: list of predicted labels (top-k)
    """
    if actual in predicted:
        return 1.0 / (np.where(predicted == actual)[0][0] + 1)
    return 0.0

def mapk(actuals, predicteds, k=3):
    """
    Computes the mean average precision at k.
    actuals: list of true labels
    predicteds: list of lists of predicted labels (each list is top-k)
    """
    return np.mean([apk(a, p, k) for a, p in zip(actuals, predicteds)])



from sklearn.metrics import make_scorer

def map3_scorer(y, y_pred):
    top3_preds = np.argsort(y_pred, axis=1)[:, -3:][:, ::-1]  # top 3 labels
    return mapk(y, top3_preds, k=3)

custom_map3_scorer = make_scorer(map3_scorer, needs_proba=True)



from sklearn.model_selection import train_test_split

x_train_tune, _, y_train_tune, _ = train_test_split(x_train, y_train, train_size=0.1, stratify=y_train, random_state=42)



def get_model_with_params(trial, model_name):
    if model_name == "RandomForest":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 100),
            "max_depth": trial.suggest_int("max_depth", 2, 32, log=True),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        }
        return RandomForestClassifier(**params, random_state=42)

    elif model_name == "XGBoost":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        }
        return XGBClassifier(**params, use_label_encoder=False, eval_metric='mlogloss', random_state=42)

    elif model_name == "LightGBM":
        if LGBMClassifier is None:
            raise ImportError("LightGBM is not installed.")
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 700),
            "max_depth": trial.suggest_int("max_depth", -1, 20),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "num_leaves": trial.suggest_int("num_leaves", 20, 300),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        }
        return LGBMClassifier(**params,verbose=-1, random_state=42)

    elif model_name == "CatBoost":
        if CatBoostClassifier is None:
            raise ImportError("CatBoost is not installed.")
        params = {
            "iterations": trial.suggest_int("iterations", 50, 100),
            "depth": trial.suggest_int("depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        }
        return CatBoostClassifier(**params, verbose=0, random_state=42)

    elif model_name == "LogisticRegression":
        params = {
            "C": trial.suggest_float("C", 0.01, 10.0, log=True),
            "solver": trial.suggest_categorical("solver", ["lbfgs", "saga", "newton-cg"]),
        }
        return LogisticRegression(**params, multi_class='multinomial', max_iter=1000, random_state=42)

    elif model_name == "MLP":
        params = {
            "hidden_layer_sizes": trial.suggest_categorical("hidden_layer_sizes", [(64,), (128,), (128, 64), (256, 128)]),
            "activation": trial.suggest_categorical("activation", ["relu", "tanh"]),
            "learning_rate_init": trial.suggest_float("learning_rate_init", 0.0001, 0.1, log=True),
            "alpha": trial.suggest_float("alpha", 0.0001, 0.01),
        }
        return MLPClassifier(**params, max_iter=500, random_state=42)

    elif model_name == "SVC":
        params = {
            "C": trial.suggest_float("C", 0.1, 10.0, log=True),
            "kernel": trial.suggest_categorical("kernel", ["linear", "rbf", "poly"]),
            "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
        }
        return SVC(**params, probability=True, random_state=42)

    else:
        raise ValueError(f"Unsupported model: {model_name}")



def objective(trial,model_name):
    model = get_model_with_params(trial, model_name)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, x_train_tune, y_train_tune, cv=cv, scoring=custom_map3_scorer)

    return scores.mean()



allow_tuning = True
models={}


if allow_tuning:
    study={}
    model_names=["XGBoost", "LightGBM", "LogisticRegression","RandomForest","CatBoost"]
    
    for model_name in model_names:
        num_trials = 50
        # if(model_name in ["RandomForest", "CatBoost"]):
        #     num_trials = 20
        study[model_name] = optuna.create_study(direction="maximize")
        study[model_name].optimize(lambda trial: objective(trial, model_name), n_trials=num_trials)

        print("-------------------------",model_name,"tuning completed--------------------------------------/n/n")
        print(f"Best {model_name} MAP@3 and params:", study[model_name].best_value, study[model_name].best_params)
        if model_name == "XGBoost":
            models[model_name]= XGBClassifier(**study[model_name].best_params, use_label_encoder=False, eval_metric='mlogloss', random_state=42)
        elif model_name == "LightGBM":
            models[model_name]= LGBMClassifier(**study[model_name].best_params,verbose=-1, random_state=42)
        elif model_name == "LogisticRegression":
            models[model_name]= LogisticRegression(**study[model_name].best_params, multi_class='multinomial', max_iter=1000, random_state=42)
        elif model_name == "RandomForest":
            models[model_name]= RandomForestClassifier(**study[model_name].best_params, random_state=42)
        elif model_name == "CatBoost":
            models[model_name]= CatBoostClassifier(**study[model_name].best_params, verbose=0, random_state=42)
else:
    models["XGBoost"]= XGBClassifier(**{'n_estimators': 426, 'max_depth': 3, 'learning_rate': 0.20758390773057933, 'subsample': 0.8738195216573006, 'colsample_bytree': 0.5027190507906605}, use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    models["LightGBM"] = LGBMClassifier(**{'n_estimators': 261, 'max_depth': 2, 'learning_rate': 0.22876751846126406, 'num_leaves': 215, 'subsample': 0.9820654475202691},verbose=-1, random_state=42)
    models["LogisticRegression"]= LogisticRegression(**{'C': 0.012521264644369402, 'solver': 'newton-cg'}, multi_class='multinomial', max_iter=1000, random_state=42)
    models["RandomForest"]= RandomForestClassifier(**{'n_estimators': 62, 'max_depth': 8, 'min_samples_split': 15, 'min_samples_leaf': 20}, random_state=42)
    models["CatBoost"]= CatBoostClassifier(**{'iterations': 97, 'depth': 3, 'learning_rate': 0.23406524785717, 'l2_leaf_reg': 3.2420338085234546}, verbose=0, random_state=42)


import joblib
if allow_tuning:
    for model_name in model_names:
        joblib.dump(study[model_name], f"{model_name}_study.pkl")



import optuna.visualization as vis

if allow_tuning:
    model_names=["XGBoost", "LightGBM", "LogisticRegression","RandomForest","CatBoost"]
    for model_name in model_names:
        print(f"Model: {model_name}")
        vis.plot_optimization_history(study[model_name]).show()
        vis.plot_param_importances(study[model_name]).show()
        vis.plot_parallel_coordinate(study[model_name]).show()


np.argsort(np.array([[3,7,4,1,5]]), axis=1)[:, -3:][:, ::-1]


print('helo')


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Logistic Regression
# lr_model = LogisticRegression(max_iter=1000, random_state=42, multi_class='multinomial')
# lr_model.fit(x_train, y_train)

# Random Forest
# x_sample = x_train.sample(n=500, random_state=42)
# y_sample = y_train[x_sample.index] 
# rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
# rf_model.fit(x_sample, y_sample)

# xgb_model= XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
# xgb_model.fit(x_train, y_train)

# lgb_model = LGBMClassifier()
# lgb_model.fit(x_train, y_train)

# cat_model = CatBoostClassifier(verbose=0)
# cat_model.fit(x_train, y_train)

# mlp_model = MLPClassifier()
# mlp_model.fit(x_train, y_train)

# svc_model = SVC()
# svc_model.fit(x_train, y_train)


# Predict class probabilities
# probs = rf_model.predict_proba(x_test)  # or rf_model.predict_proba(X_test)

# # Get top-3 predicted class indices
# top3_preds = np.argsort(probs, axis=1)[:, -3:][:, ::-1]  # shape: (n_samples, 3)

# # Convert back to fertilizer names
# top3_labels = le.inverse_transform(top3_preds.flatten()).reshape(top3_preds.shape)
# top3_labels


# # Join top-3 predictions with space
# top3_joined = [' '.join(preds) for preds in top3_labels]



# submission = pd.read_csv('/kaggle/input/playground-series-s5e6/sample_submission.csv')
# submission


# submission['Fertilizer Name'] = top3_joined
# submission.to_csv("/kaggle/working/submission.csv", index=False)
# print('submission saved')
# submission.head()

