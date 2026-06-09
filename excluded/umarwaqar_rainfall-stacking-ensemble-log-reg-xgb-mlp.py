import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings('ignore')
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


df_train= pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv").set_index('id')
df_test= pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv").set_index('id')
df_subm= pd.read_csv("/kaggle/input/playground-series-s5e3/sample_submission.csv")


df_train.tail()


feature_columns = [col for col in df_train.columns if col != 'rainfall']

for col in feature_columns:
    plt.figure(figsize=(15, 6))
    
    # Boxplot: compare feature distribution by rainfall status
    plt.subplot(1, 2, 1)
    sns.boxplot(x=df_train['rainfall'], y=df_train[col], palette='Set2')
    plt.title(f'Boxplot of {col} by Rainfall')
    plt.xlabel('Rainfall')
    plt.ylabel(col)
    
    # Histogram with KDE: overlay distributions split by rainfall
    plt.subplot(1, 2, 2)
    sns.histplot(data=df_train, x=col, hue='rainfall', kde=True, palette='Set1')
    plt.title(f'Histogram of {col} by Rainfall')
    plt.xlabel(col)
    
    plt.tight_layout()
    plt.show()


plt.figure(figsize=(12, 10))
sns.heatmap(data=df_train.corr(), annot=True, linewidths=0.2);


missing_counts = df_test.isnull().sum()
print(missing_counts)


df_test['winddirection'].fillna(df_test['winddirection'].median(), inplace=True)


X_train = df_train.drop(columns=['day', 'rainfall'])
y_train = df_train['rainfall']
# For testing, drop 'day'
X_test = df_test.drop(columns=['day'])

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define base models for the ensemble
base_estimators = [
    ('lr', LogisticRegression(max_iter=200)),
    ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
    ('mlp', MLPClassifier(hidden_layer_sizes=(64,), max_iter=300))
]

# Define the meta-model
meta_estimator = LogisticRegression(max_iter=200)

# Create the stacking classifier using 5-fold cross-validation for the meta-model
stack_model = StackingClassifier(
    estimators=base_estimators,
    final_estimator=meta_estimator,
    cv=5
)

# Train the stacking ensemble
stack_model.fit(X_train_scaled, y_train)

# Predict on the training set to evaluate performance
y_train_pred = stack_model.predict(X_train_scaled)
train_accuracy = accuracy_score(y_train, y_train_pred)
print("Training Accuracy: {:.4f}".format(train_accuracy))

# If you want to generate predictions on the test set:
y_test_pred = stack_model.predict(X_test_scaled)


df_subm['rainfall'] = y_test_pred
df_subm.to_csv('submission.csv', index=False)
df_subm.head()




