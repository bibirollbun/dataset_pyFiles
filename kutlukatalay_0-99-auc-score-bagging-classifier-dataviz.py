import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
sns.set_style("whitegrid")
sns.color_palette("husl", 10)
%matplotlib inline

from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
from sklearn.metrics import make_scorer
from scipy.stats import gaussian_kde

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

import warnings
from warnings import filterwarnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)


train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")

print(train.shape, test.shape)


train.head()


test.head()


train = train.rename(columns={'temparature': 'temperature'})
test = test.rename(columns={'temparature': 'temperature'})


train.info()


test.info()


def fill_wind_direction(df):
    wind_direction = df['winddirection']
    
    if wind_direction.isna().all():
        df['winddirection'] = wind_direction.fillna(0)
        return df

    rad = np.radians(wind_direction)
    
    mean_cos = np.nanmean(np.cos(rad))
    mean_sin = np.nanmean(np.sin(rad))
    
    mean_direction = np.degrees(np.arctan2(mean_sin, mean_cos))
    
    mean_direction = (mean_direction + 360) % 360
    
    df['winddirection'] = wind_direction.fillna(mean_direction)
    
    return df

test = fill_wind_direction(test)


test["winddirection"].isna().sum()


features = ['maxtemp', 'temperature', 'pressure', 'dewpoint', 
            'humidity', 'cloud', 'sunshine', 'winddirection', 'windspeed']

fig, axes = plt.subplots(3, 3, figsize=(18, 15))
axes = axes.flatten()

train_color = "blue"
test_color = "red"

for i, feature in enumerate(features):
    ax = axes[i]
    
    train_data = train[feature].dropna()
    test_data = test[feature].dropna()
    
    if feature == 'winddirection':

        pass

    try:
        min_val = min(train_data.min(), test_data.min())
        max_val = max(train_data.max(), test_data.max())
        x_grid = np.linspace(min_val, max_val, 1000)
    except:
        continue
    

    try:
        if len(train_data) > 1:
            train_kde = gaussian_kde(train_data)
            train_density = train_kde(x_grid)
        else:
            train_density = np.zeros_like(x_grid)
            
        if len(test_data) > 1:
            test_kde = gaussian_kde(test_data)
            test_density = test_kde(x_grid)
        else:
            test_density = np.zeros_like(x_grid)
    except:
        continue

    ax.fill_between(x_grid, 0, train_density, color=train_color, alpha=0.7, label='Train')
    ax.fill_between(x_grid, train_density, train_density + test_density, color=test_color, alpha=0.7, label='Test')
    
    ax.set_title(f'Distribution of {feature}', fontsize=14)
    ax.set_xlabel(feature, fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.legend()

    train_mean = train_data.mean()
    test_mean = test_data.mean()
    ax.axvline(train_mean, color='darkblue', linestyle='dashed', linewidth=1, label='_Train mean')
    ax.axvline(test_mean, color='darkred', linestyle='dashed', linewidth=1, label='_Test mean')
    
    ax.text(0.05, 0.95, f'Train mean: {train_mean:.2f}\nTest mean: {test_mean:.2f}', 
            transform=ax.transAxes, verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    ax.grid(True, linestyle='dotted', alpha=0.7, color='darkgray')

plt.tight_layout()
plt.suptitle('Comparing Train vs Test Distributions (Stacked)', fontsize=16, y=1.02)
plt.show()


features = ['maxtemp', 'temperature', 'pressure', 'humidity', 'windspeed']

date_column = 'day'

fig, axes = plt.subplots(len(features), 2, figsize=(18, 4*len(features)))

train_color = 'blue'
test_color = 'red'

for i, feature in enumerate(features):

    ax_train = axes[i, 0]
    if date_column in train.columns and feature in train.columns:
        train_sorted = train.sort_values(by=date_column)
        ax_train.plot(train_sorted[date_column], train_sorted[feature], 
                color=train_color, linewidth=0.8)

    ax_train.set_title(f'Train: {feature}', fontsize=14)
    ax_train.set_xlabel('Date', fontsize=12)
    ax_train.set_ylabel(feature, fontsize=12)
    ax_train.grid(True, linestyle='dotted', alpha=0.7, color='darkgray')

    ax_test = axes[i, 1]
    if date_column in test.columns and feature in test.columns:
        test_sorted = test.sort_values(by=date_column)
        ax_test.plot(test_sorted[date_column], test_sorted[feature], 
                color=test_color, linewidth=0.8)

    ax_test.set_title(f'Test: {feature}', fontsize=14)
    ax_test.set_xlabel('Date', fontsize=12)
    ax_test.set_ylabel(feature, fontsize=12)
    ax_test.grid(True, linestyle='dotted', alpha=0.7, color='darkgray')

    for ax in [ax_train, ax_test]:
        try:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        except:
            pass

plt.suptitle('Timeline Plots of Weather Features', fontsize=16, y=1.02)
plt.tight_layout()
plt.show()


features = ['winddirection', 'pressure', 'maxtemp', 'temperature', 'mintemp', 
            'dewpoint', 'humidity', 'cloud', 'sunshine', 'windspeed']

fig, axes = plt.subplots(5, 2, figsize=(16, 20))
axes = axes.flatten()

colors = ["blue", "red"]

for i, feature in enumerate(features):
    ax = axes[i]

    try:
        sns.kdeplot(
            data=train, x=feature, hue='rainfall',
            fill=True, common_norm=False, palette=colors,
            alpha=0.5, linewidth=1, ax=ax)
        
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, [f"rainfall : {label}" for label in labels])

        ax.set_title(f'Distribution of {feature} by Rainfall', fontsize=12)
        ax.set_xlabel(feature, fontsize=10)
        ax.set_ylabel('Density', fontsize=10)

        ax.grid(True, linestyle='dotted', alpha=0.7, color='darkgray')

        if feature == 'winddirection':
            ax.set_xlim(0, 360)
    
    except Exception as e:
        ax.text(0.5, 0.5, f"Error plotting {feature}:\n{str(e)}", 
                ha='center', va='center', transform=ax.transAxes)

plt.tight_layout()
plt.suptitle('Feature Distributions by Rainfall', fontsize=16, y=1.02)
plt.show()


fig, ax = plt.subplots(1, 2, figsize=(16, 8), subplot_kw={'projection': 'polar'})

rain_color = 'navy'
no_rain_color = 'orange'

rain_data = train[train['rainfall'] == 1]
no_rain_data = train[train['rainfall'] == 0]

rain_wind_rad = np.deg2rad(rain_data['winddirection'])
no_rain_wind_rad = np.deg2rad(no_rain_data['winddirection'])

ax[0].scatter(no_rain_wind_rad, no_rain_data['windspeed'], 
             color=no_rain_color, alpha=0.5, s=10)
ax[0].set_title('Wind Speed vs Direction - No Rain (rainfall = 0)', fontsize=14)

ax[1].scatter(rain_wind_rad, rain_data['windspeed'], 
             color=rain_color, alpha=0.5, s=10)
ax[1].set_title('Wind Speed vs Direction - Rain (rainfall = 1)', fontsize=14)

for a in ax:
    a.set_theta_zero_location('N')
    a.set_theta_direction(-1)
    a.grid(True, linestyle='dotted', alpha=0.7, color='darkgray')
    a.set_xticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])
    
    a.set_ylabel('Wind Speed')
    if a == ax[0]:
        a.text(0.5, -0.1, "Each point represents a day with no rainfall", 
               ha='center', transform=a.transAxes)
    else:
        a.text(0.5, -0.1, "Each point represents a day with rainfall", 
               ha='center', transform=a.transAxes)

plt.tight_layout()
plt.show()


train.head()


test.head()


X_train = train.drop(columns=['day', 'rainfall', 'id'],axis = 1)
y_train = train['rainfall']
X_test = test.drop(columns=['day', 'id'],axis = 1)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


models = {
    'DecisionTree': {
        'model': DecisionTreeClassifier(random_state=42),
        'params': {'max_depth': [3, 5, 7, 10],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]}
    },
    'RandomForest': {
        'model': RandomForestClassifier(random_state=42),
        'params': {'n_estimators': [100, 200],
            'max_depth': [5, 10],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]}
    },
    'GradientBoosting': {
        'model': GradientBoostingClassifier(random_state=42),
        'params': {'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5]}
    },
    'AdaBoost': {
        'model': AdaBoostClassifier(random_state=42),
        'params': {'n_estimators': [50, 100],
            'learning_rate': [0.01, 0.1, 1.0]}
    },
    'ExtraTrees': {
        'model': ExtraTreesClassifier(random_state=42),
        'params': {'n_estimators': [100, 200],
            'max_depth': [5, 10],
            'min_samples_split': [2, 5]}
    },
    'Bagging': {
        'model': BaggingClassifier(random_state=42),
        'params': {'n_estimators': [10, 50],
            'max_samples': [0.5, 0.7, 1.0]}
    },
    'SVC': {
        'model': SVC(probability=True, random_state=42),
        'params': {'C': [0.1, 1, 10],
            'kernel': ['rbf', 'linear'],
            'gamma': ['scale', 'auto']}
    }
}

results = {}

for name, model_info in models.items():
    grid_search = GridSearchCV(estimator=model_info['model'],
        param_grid=model_info['params'],cv=5,scoring='roc_auc',
        n_jobs=-1,verbose=1)
    
    grid_search.fit(X_train_scaled, y_train)
    
    results[name] = {'best_score': grid_search.best_score_,
        'best_params': grid_search.best_params_}


sorted_results = sorted(results.items(), key=lambda x: x[1]['best_score'])
model_names = [x[0] for x in sorted_results]
roc_auc_scores = [x[1]['best_score'] for x in sorted_results]

plt.figure(figsize=(12, 6))
sns.set_style("whitegrid")

bars = plt.bar(model_names, roc_auc_scores)

plt.title('ROC-AUC Scores Comparison Across Models', fontsize=14, pad=20)
plt.xlabel('Models', fontsize=12)
plt.ylabel('ROC-AUC Score', fontsize=12)

plt.xticks(rotation=45, ha='right')

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.4f}', ha='center', va='bottom')

mean_score = np.mean(roc_auc_scores)
plt.axhline(y=mean_score, color='r', linestyle='--', alpha=0.5)
plt.text(plt.xlim()[1], mean_score, f'Mean: {mean_score:.4f}', 
         va='bottom', ha='right', color='r')

plt.tight_layout()

plt.show()


predictions = {}

for name, result in results.items():
    if name == 'DecisionTree':
        model = DecisionTreeClassifier(random_state=42, **result['best_params'])
    elif name == 'RandomForest':
        model = RandomForestClassifier(random_state=42, **result['best_params'])
    elif name == 'GradientBoosting':
        model = GradientBoostingClassifier(random_state=42, **result['best_params'])
    elif name == 'AdaBoost':
        model = AdaBoostClassifier(random_state=42, **result['best_params'])
    elif name == 'ExtraTrees':
        model = ExtraTreesClassifier(random_state=42, **result['best_params'])
    elif name == 'Bagging':
        model = BaggingClassifier(random_state=42, **result['best_params'])
    elif name == 'SVC':
        model = SVC(probability=True, random_state=42, **result['best_params'])

    model.fit(X_train_scaled, y_train)
    predictions[name] = model.predict_proba(X_train_scaled)[:, 1]

plt.figure(figsize=(20, 15))
for i, (name, y_pred) in enumerate(predictions.items(), 1):
    plt.subplot(3, 3, i)
    fpr, tpr, _ = roc_curve(y_train, y_pred)
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {name}')
    plt.legend(loc="lower right")

plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 8))

colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black']

for (name, y_pred), color in zip(predictions.items(), colors):
    fpr, tpr, _ = roc_curve(y_train, y_pred)
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, color=color, lw=2,
             label=f'{name} (AUC = {roc_auc:.4f})')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves - All Models')
plt.legend(loc="lower right")
plt.grid(True)

plt.tight_layout()
plt.show()


best_bagging_params = results['Bagging']['best_params']

best_bagging = BaggingClassifier(random_state=42, **best_bagging_params)
best_bagging.fit(X_train_scaled, y_train)

y_pred_proba = best_bagging.predict_proba(X_test_scaled)[:, 1]

predictions_df = pd.DataFrame({'id': test['id'], 
                               'rainfall_probability': y_pred_proba})

predictions_df


feature_names = X_train.columns

importances = np.mean([tree.feature_importances_ for tree in best_bagging.estimators_], axis=0)

feature_importance_df = pd.DataFrame({'Feature': feature_names,
                                      'Importance': importances})

feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(data=feature_importance_df, x='Importance', y='Feature')

plt.title('Feature Importance in Bagging Classifier', fontsize=14, pad=20)
plt.xlabel('Average Feature Importance', fontsize=12)
plt.ylabel('Features', fontsize=12)

plt.grid(axis='x', linestyle='--', alpha=0.6)

plt.tight_layout()

plt.show()


plt.figure(figsize=(10, 6))
plt.hist(y_pred_proba, bins=50, edgecolor='black')
plt.title('Distribution of Rainfall Probability Predictions')
plt.xlabel('Predicted Probability')
plt.ylabel('Count')
plt.grid(True, alpha=0.3)
plt.show()


predictions_df.columns = ["id","rainfall"]
predictions_df


predictions_df.to_csv('predictions.csv', index=False)

