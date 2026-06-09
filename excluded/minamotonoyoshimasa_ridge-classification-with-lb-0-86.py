!pip install optuna


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.linear_model import RidgeClassifier
import warnings
warnings.filterwarnings('ignore')


train_df = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')


train_df.head(5)


test_df.head(5)


train_df.info()


test_df.info()


def feature_distribution(df, col):
  sns.histplot(data=df[col], kde=True)
  plt.show()


for col in train_df.columns:
    if col != 'id':
        feature_distribution(train_df, col)


train_df.isna().sum()


test_df.isna().sum()


test_df['winddirection'].fillna(test_df['winddirection'].mean(), inplace=True)


train_df_copy = train_df.copy()


def feature_target_correlation_plot(df, col):
  fig,axes = plt.subplots(1,2, figsize=(10, 5))
  sns.kdeplot(df[df['rainfall'] == 0][col], label='No Rain', shade=True, ax=axes[1])
  sns.kdeplot(df[df['rainfall'] == 1][col], label='Rain', shade=True, ax=axes[1])
  axes[1].set_title('KDE Plot:' + col + ' Distribution by Rainfall')
  axes[1].legend()

  sns.boxplot(x='rainfall', y=col, data=df, ax=axes[0])
  axes[0].set_title('Box Plot:' + col + ' Distribution by Rainfall')
  plt.tight_layout()
  plt.show()


feature_target_correlation_plot(train_df_copy, 'pressure')


train_df_copy['tempdiff'] = train_df_copy['maxtemp'] - train_df_copy['mintemp']


train_df_copy['winddirection_sin'] = np.sin(2 * np.pi * train_df_copy['winddirection'] / 360)
train_df_copy['winddirection_cos'] = np.cos(2 * np.pi * train_df_copy['winddirection'] / 360)


train_df_copy['day_sin'] = np.sin(2 * np.pi * train_df_copy['day'] / 365)
train_df_copy['day_cos'] = np.cos(2 * np.pi * train_df_copy['day'] / 365)


train_df_copy['sunshine_cloud'] = np.log1p(train_df_copy['sunshine']) + np.log1p(train_df_copy['cloud'])
train_df_copy['humidity_cloud'] = np.log1p(train_df_copy['humidity']) + np.log1p(train_df_copy['cloud'])
train_df_copy['humidity_sunshine'] = np.log1p(train_df_copy['humidity']) - np.log1p(train_df_copy['sunshine'])
train_df_copy['dewpoint_pressure'] = np.log1p(train_df_copy['dewpoint']) + np.log1p(train_df_copy['pressure'])


train_df_copy.head()


train_df_for_model = train_df_copy.copy()


train_df_x = train_df_for_model.drop(['id', 'rainfall', 'maxtemp', 'winddirection_sin', 'dewpoint', 'windspeed', 'day_cos', 'winddirection', 'day', 'mintemp', 'winddirection_cos', 'temparature', 'pressure','day_sin'], axis=1)
train_df_y = train_df_for_model['rainfall']


import seaborn as sns
import matplotlib.pyplot as plt

# Compute correlation matrix
corr_matrix = train_df_copy[train_df_x.columns].corr()

# Plot heatmap
plt.figure(figsize=(18,16))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Matrix")
plt.show()



# Split the dataset into 70% training and 30% validation while maintaining order
split_index = int(len(train_df) * 0.7)  # Calculate 70% index

# Split data in order (not randomly)
train_df_x, vali_df_x = train_df_x.iloc[:split_index], train_df_x.iloc[split_index:]
train_df_y, vali_df_y = train_df_y.iloc[:split_index], train_df_y.iloc[split_index:]


# Merge features with the target for correlation analysis
df_corr = pd.concat([train_df_x, train_df_y], axis=1)

df_corr_encoded = df_corr.copy()

# Compute correlation matrix
corr_matrix = df_corr_encoded.corr()
corr_with_target = corr_matrix.iloc[-1].sort_values(ascending=False)
print(corr_with_target)


test_df_copy = test_df.copy()


test_df_copy['tempdiff'] = test_df_copy['maxtemp'] - test_df_copy['mintemp']
test_df_copy['winddirection_sin'] = np.sin(2 * np.pi * test_df_copy['winddirection'] / 360)
test_df_copy['winddirection_cos'] = np.cos(2 * np.pi * test_df_copy['winddirection'] / 360)
test_df_copy['day_sin'] = np.sin(2 * np.pi * test_df_copy['day'] / 365)
test_df_copy['day_cos'] = np.cos(2 * np.pi * test_df_copy['day'] / 365)
test_df_copy['sunshine_cloud'] = np.log1p(test_df_copy['sunshine']) + np.log1p(test_df_copy['cloud'])
test_df_copy['humidity_cloud'] = np.log1p(test_df_copy['humidity']) + np.log1p(test_df_copy['cloud'])
test_df_copy['humidity_sunshine'] = np.log1p(test_df_copy['humidity']) - np.log1p(test_df_copy['sunshine'])
test_df_copy['dewpoint_pressure'] = np.log1p(test_df_copy['dewpoint']) + np.log1p(test_df_copy['pressure'])


test_df_x = test_df_copy.drop(['id', 'maxtemp', 'winddirection_sin', 'dewpoint', 'day_sin', 'windspeed', 'day_cos', 'winddirection', 'day', 'mintemp', 'winddirection_cos', 'temparature', 'pressure'], axis=1)


from sklearn.preprocessing import StandardScaler

# Initialize the scaler
scaler = StandardScaler()

# Fit and transform the training data
train_df_x_scaled = scaler.fit_transform(train_df_x)

vali_df_x_scaled = scaler.transform(vali_df_x)

# If you have test data, transform it using the same scaler
test_df_x_scaled = scaler.transform(test_df_x)


import optuna
from sklearn.model_selection import cross_val_score, StratifiedKFold
# Define the objective function for Optuna
def objective(trial):
    alpha = trial.suggest_loguniform('alpha', 1e-5, 100)  # Log-uniform distribution
    class_weight = trial.suggest_categorical('class_weight', [None, 'balanced'])  # Handling class imbalance

    model = RidgeClassifier(alpha=alpha, class_weight=class_weight)

    score = cross_val_score(model, train_df_x_scaled, train_df_y, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), scoring='roc_auc').mean()
    return score

# Run Optuna study
study = optuna.create_study(direction='maximize')  # Maximize ROC AUC
study.optimize(objective, n_trials=100)

# Get the best hyperparameters
best_params = study.best_params
best_score = study.best_value
print(f'Best Hyperparameters: {best_params}')
print(f"Best ROC-AUC: {best_score:.4f}")

# Train final model
best_model = RidgeClassifier(**best_params).fit(train_df_x_scaled, train_df_y)


y_pred_proba = best_model.decision_function(vali_df_x_scaled)

# Compute ROC curve and ROC AUC score
fpr, tpr, thresholds = roc_curve(vali_df_y, y_pred_proba)
roc_auc = roc_auc_score(vali_df_y, y_pred_proba)

print(f'ROC AUC Score: {roc_auc}')

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.5f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()


test_df_probs = best_model.decision_function(test_df_x_scaled)


test_df_result = pd.DataFrame({'id': range(2190, 2190 + len(test_df_probs)), 'rainfall': test_df_probs})


test_df_result.head()


test_df_result.to_csv('submission.csv', index=False)

