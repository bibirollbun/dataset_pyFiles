import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as sp
import warnings

from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

import xgboost as xgb
import catboost as cb
from imblearn.combine import SMOTETomek

warnings.filterwarnings("ignore")


train=pd.read_csv(r'/kaggle/input/playground-series-s5e3/train.csv')
test=pd.read_csv(r'/kaggle/input/playground-series-s5e3/test.csv')
print(train.shape, test.shape)


train


train.duplicated().sum()


train.isnull().sum()


test.duplicated().sum()


test.isnull().sum()


test.fillna(test.median(), inplace=True)


test.isnull().sum()


train.info()


train.describe().T


numerical_features = ['pressure', 'maxtemp', 'temparature', 'mintemp',
       'dewpoint', 'humidity', 'cloud', 'sunshine',
       'windspeed']
categorical_feature = ['winddirection']
target = ['rainfall']


sns.set(style='whitegrid')


for feature in numerical_features:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4)) 

    combined_data = pd.concat([train[feature], test[feature]], axis=1, keys=["Train", "Test"])
    sns.boxplot(data=combined_data, ax=axes[0])
    axes[0].set_xlabel(feature)
    axes[0].set_title(f"Box Plot for {feature}")

    sns.histplot(data=train, x=feature, kde=True, bins=30, label="Train", ax=axes[1])
    sns.histplot(data=test, x=feature, kde=True, bins=30, label="Test", ax=axes[1])
    axes[1].set_xlabel(feature)
    axes[1].set_ylabel("Frequency")
    axes[1].set_title(f"Histogram for {feature} [TRAIN, TEST]")
    axes[1].legend()

    plt.tight_layout() 
    plt.show()



def create_wind_rose(ax, data, dataset_name, color):
    wind_direction_radians = np.radians(data['winddirection'].dropna())

    bins = np.linspace(0, 2*np.pi, 37)  
    counts, bin_edges = np.histogram(wind_direction_radians, bins=bins)

    bars = ax.bar(bin_edges[:-1], counts, width=np.radians(10), edgecolor='black',color=color, alpha=0.8)

    ax.set_theta_zero_location("N") 
    ax.set_theta_direction(-1) 
    ax.set_xticks(np.radians(np.arange(0, 360, 45)))  
    ax.set_xticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'], fontsize=10, fontweight='bold')

    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_yticklabels([]) 
    ax.set_title(f"Wind Direction ({dataset_name})", fontsize=12, fontweight='bold', pad=10)

fig, axes = plt.subplots(1, 2, figsize=(18, 6), subplot_kw={'projection': 'polar'})

create_wind_rose(axes[0], train, "Train Data", 'blue')  
create_wind_rose(axes[1], test, "Test Data", 'red')   

plt.tight_layout()
plt.show()


fig, axes = plt.subplots(1, 2, figsize=(18, 6), subplot_kw={'projection': 'polar'})

create_wind_rose(axes[0], train[train.rainfall == 1], "Train Data with rainfall", 'green')  
create_wind_rose(axes[1], train[train.rainfall == 0], "Train Data without rainfall", 'yellow')   

plt.tight_layout()
plt.show()


sns.countplot(x=train.rainfall)
plt.title('Countplot for target variable (rainfall)')
plt.show()


fig, axes = plt.subplots(1, 2, figsize=(20, 8))
sns.heatmap(train.corr(method='spearman'), annot=True, cmap="coolwarm", fmt=".2f", ax=axes[0])
axes[0].set_title("Spearman Feature Correlation Heatmap")

sns.heatmap(train.corr(method='pearson'), annot=True, cmap="coolwarm", fmt=".2f", ax=axes[1])
axes[1].set_title("Pearson Feature Correlation Heatmap")

plt.tight_layout()
plt.show()


X = train.drop(columns=['rainfall', 'id'])  
y = train['rainfall']

mi_scores = mutual_info_classif(X, y, discrete_features=False)
feature_importance = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)

sns.heatmap(pd.DataFrame(feature_importance), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Mutual Information Score Heatmap")
plt.show()


train_sample_1 = train[train.rainfall == 1].sample(train[train.rainfall == 0].shape[0], random_state=42)
train_for_kde = pd.concat([train_sample_1,train[train.rainfall == 0]])
for feature in numerical_features:
    fig, axes = plt.subplots(1, 2, figsize=(15, 4))

    sns.lineplot(x='id', y=feature, data=train, label='Train Data', ax=axes[0], alpha=0.7)
    sns.lineplot(x='id', y=feature, data=test, label='Test Data', ax=axes[0], alpha=0.7)
    axes[0].set_title(f"Trend Plot for {feature}")

    u_crit = sp.mannwhitneyu(train_for_kde[feature][train_for_kde.rainfall == 0], train_for_kde[feature][train_for_kde.rainfall == 1])
    t_coef = sp.ttest_ind(train_for_kde[feature][train_for_kde.rainfall == 0], train_for_kde[feature][train_for_kde.rainfall == 1])
    ks_coef = sp.kstest(train_for_kde[feature][train_for_kde.rainfall == 0], train_for_kde[feature][train_for_kde.rainfall == 1])
    result = f"KS-test: {ks_coef.statistic}, p-value: {ks_coef.pvalue}\n\nU-criteria: {u_crit.statistic}, p-value: {u_crit.pvalue}\nT-test: {t_coef.statistic}, p-value: {t_coef.pvalue}"
    
    sns.kdeplot(data=train_for_kde, x=feature, hue='rainfall', ax=axes[1], fill=True, common_norm=False, alpha=0.6)
    axes[1].set_title(f"KDE of {feature} for rainfall")
    axes[1].text(axes[1].get_xlim()[1] + 1, (axes[1].get_ylim()[1] + axes[1].get_ylim()[0])/2,result, fontsize=10)
    plt.tight_layout()  
    plt.show()


def preprocessing(df):
    # log-scaling to highly skewed features
    df['log_cloud'] = np.log1p(df.cloud)
    df['log_sunshine'] = np.log1p(df.sunshine)
    
    df['sin_day'] = np.sin(2 * np.pi * df.day / 365)
    df['cos_day'] = np.cos(2 * np.pi * df.day / 365)

    # combination of strong features
    df['cloud_mply_humidity'] = (df.cloud * df.humidity).fillna(0)
    df['humidity_dvd_sunshine'] = (df.humidity / (df.sunshine + 1e-5)).fillna(0)
    df['cloud_dvd_sunshine'] = (df.cloud / (df.sunshine + 1e-5)).fillna(0)

    # reduction of the number of colinear features (and making a sense of these values)
    df['temp_range'] = (df.maxtemp - df.mintemp).fillna(0)

    def windpolus(direction): # for dividing winddirection feature into 4 sectors
        if pd.isna(direction):
            return np.nan  
        direction = float(direction)
        if direction >= 315 or direction < 45:
            return 0
        elif direction >= 45 and direction < 135:
            return 1
        elif direction >= 135 and direction < 225:
            return 2
        else:
            return 3
    
    df['wind_polus'] = df['winddirection'].apply(windpolus)

    # features below show clearer sinusoidal pattern in trend plot, 
    # which is why they was combined with sin_day and cos_day
    df['pressure_mply_sin_day'] = df.pressure * df.sin_day
    df['pressure_mply_cos_day'] = df.pressure * df.cos_day
    df['temperature_mply_sin_day'] = df.temparature * df.sin_day
    df['temperature_mply_cos_day'] = df.temparature * df.cos_day
    df['dewpoint_mply_sin_day'] = df.dewpoint * df.sin_day
    df['dewpoint_mply_cos_day'] = df.dewpoint * df.cos_day

    if 'rainfall' in df.columns:
        df[[f for f in df if f not in ['rainfall']] + ['rainfall']] # to place target at the end of the table

new_features = ['log_cloud', 'log_sunshine', 'sin_day', 'cos_day', 'cloud_mply_humidity', 'humidity_dvd_sunshine', 'cloud_dvd_sunshine', 'temp_range', 'wind_polus', 
                'pressure_mply_sin_day', 'pressure_mply_cos_day', 'temperature_mply_sin_day', 'temperature_mply_cos_day', 'dewpoint_mply_sin_day', 'dewpoint_mply_cos_day']
preprocessing(train)



fig, axes = plt.subplots(2, 1, figsize=(12, 12))
sns.heatmap(train[new_features + ['rainfall']].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=axes[0])
axes[0].set_title("New Features' Correlation Heatmap")

sns.heatmap(train[numerical_features + categorical_feature + ['rainfall']].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=axes[1])
axes[1].set_title("Old Features' Correlation Heatmap")

plt.tight_layout()
plt.show()


X_with_new = train.drop(columns=['rainfall', 'id'])[new_features]
X_with_old = train.drop(columns=['rainfall', 'id'])[numerical_features + categorical_feature]
y = train['rainfall']

mi_scores_new = mutual_info_classif(X_with_new, y, discrete_features=False)
feature_importance_new = pd.Series(mi_scores_new, index=X_with_new.columns).sort_values(ascending=False)
mi_scores_old = mutual_info_classif(X_with_old, y, discrete_features=False)
feature_importance_old = pd.Series(mi_scores_old, index=X_with_old.columns).sort_values(ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(15, 4))

sns.heatmap(pd.DataFrame(feature_importance_new), annot=True, cmap="coolwarm", fmt=".2f", ax=axes[0])
sns.heatmap(pd.DataFrame(feature_importance_old), annot=True, cmap="coolwarm", fmt=".2f", ax=axes[1])
axes[0].set_title("Mutual Information Score Heatmap of old features")
axes[1].set_title("Mutual Information Score Heatmap of new features")
plt.show()


train_filtered = train[['log_sunshine', 'cloud_mply_humidity', 'temp_range', 'dewpoint_mply_cos_day', 'windspeed', 'humidity_dvd_sunshine', 'sin_day', 'rainfall']]


sns.heatmap(train_filtered.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Final Features' Correlation Heatmap")
plt.show()


sum_num_deleted = 0

for feature in train_filtered.drop(columns=['rainfall']).columns:
    Q1 = train[feature].quantile(0.1)
    Q3 = train[feature].quantile(0.9)
    lower_bound = Q1 - 1.5 * (Q3 - Q1)
    upper_bound = Q3 + 1.5 * (Q3 - Q1)
    
    
    filtered_data = train[(train_filtered[feature] >= lower_bound) & (train_filtered[feature] <= upper_bound)]
    
    num_deleted = len(train_filtered) - len(filtered_data)
    
    sum_num_deleted += num_deleted
    print(f"{feature}: {num_deleted}")
print(sum_num_deleted)



sns.countplot(x=train.rainfall)
plt.title('Countplot for target variable (rainfall)')
plt.show()


resampler = SMOTETomek(random_state=42)
X_resampled, y_resampled = resampler.fit_resample(train_filtered, train.rainfall)


sns.countplot(x=y_resampled)
plt.title('Countplot for target variable (rainfall)')
plt.show()


scaler = StandardScaler()
train_filtered_scaled = scaler.fit_transform(X_resampled.drop(columns='rainfall'))


models_params = {
    "LogisticRegression": (LogisticRegression(), {
        "C": [0.01, 0.1, 1, 2],  
        "solver": ["lbfgs", "liblinear"]
    }),
    "SVC": (SVC(probability=True), {
        "C": [0.1, 1, 2],
        "kernel": ["linear", "rbf"]
    }),
    "CatBoost": (cb.CatBoostClassifier(verbose=0), {
        "iterations": [100, 200],
        "learning_rate": [0.01, 0.1, 0.2],
        "depth": [4, 6, 8]
    })
}

X_train, y_train = train_filtered_scaled, y_resampled


best_models = {}

for name, (model, params) in models_params.items():
    print(f"Running GridSearchCV for {name}...")
    grid_search = GridSearchCV(model, param_grid=params, cv=5, scoring='roc_auc', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_models[name] = grid_search.best_estimator_
    print(f"Best params for {name}: {grid_search.best_params_}")
    print(f"Best ROC AUC: {grid_search.best_score_}\n")



preprocessing(test)
test_final_features = test[['log_sunshine', 'cloud_mply_humidity', 'temp_range', 'dewpoint_mply_cos_day', 'windspeed', 'humidity_dvd_sunshine', 'sin_day']]
test_scaled = scaler.transform(test_final_features)


for i in best_models:
    model = best_models[i]
    test_proba = model.predict_proba(test_scaled)[:, 1]
    submission_df = pd.DataFrame({
    'id': test['id'],
    'rainfall': test_proba 
    })
    submission_df.to_csv(f"submission_{i}.csv", index=False)


scaler = StandardScaler()
train_filtered_scaled = scaler.fit_transform(train_filtered.drop(columns='rainfall'))


test_scaled = scaler.transform(test_final_features)


X_train, y_train = train_filtered_scaled, train.rainfall


best_models_without_oversampling = {}

for name, (model, params) in models_params.items():
    print(f"Running GridSearchCV for {name}...")
    grid_search = GridSearchCV(model, param_grid=params, cv=5, scoring='roc_auc', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_models_without_oversampling[name] = grid_search.best_estimator_
    print(f"Best params for {name}: {grid_search.best_params_}")
    print(f"Best ROC AUC: {grid_search.best_score_}\n")



for i in best_models:
    model = best_models_without_oversampling[i]
    test_proba = model.predict_proba(test_scaled)[:, 1]
    submission_df = pd.DataFrame({
    'id': test['id'],
    'rainfall': test_proba 
    })
    submission_df.to_csv(f"submission_{i}_without_oversampling.csv", index=False)

