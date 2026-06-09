import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.metrics import accuracy_score, accuracy_score, f1_score, roc_curve, auc
from sklearn import metrics
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit,cross_val_score, StratifiedKFold
import lightgbm
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, VotingClassifier, HistGradientBoostingClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense, Dropout, MaxPooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')


df_train= pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
df_test= pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')
df_subm= pd.read_csv('/kaggle/input/playground-series-s5e3/sample_submission.csv')
print("\n Train Data:({},{})".format(df_train.shape[0],df_train.shape[1]))
print("\n Test Data:({},{}) ".format(df_test.shape[0],df_test.shape[1]))
print("missing values in the Train Data:\n {}".format(df_train.isnull().sum()))
print("missing values in the Test Data:\n {}".format(df_test.isnull().sum()))


df_train.head(5)


for col in df_test.columns:

  # Checking if the column contains
  # any null values
  if df_test[col].isnull().sum() > 0:
    val = df_test[col].mean()
    df_test[col] = df_test[col].fillna(val)


correlation_matrix = df_train.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix[['rainfall']].drop('rainfall').sort_values(by='rainfall', ascending=False), annot=True, cmap='coolwarm')
plt.show()


def create_features(df):
    # 创建数据副本，以避免直接修改原始数据
    df_new = df.copy()

    df_new['humidity_cloud_interaction'] = df_new['humidity'] * df_new['cloud']
    df_new['humidity_sunshine_interaction'] = df_new['humidity'] * df_new['sunshine']
    df_new['cloud_sunshine_ratio'] = df_new['cloud'] / (df_new['sunshine'] + 1e-5)
    df_new['relative_dryness'] = 100 - df_new['humidity']
    df_new['sunshine_percentage'] = df_new['sunshine'] / (df_new['sunshine'] + df_new['cloud'] + 1e-5)
    df_new['weather_index'] = (0.4 * df_new['humidity']) + (0.3 * df_new['cloud']) - (0.3 * df_new['sunshine'])
    
    # df_new['relative_dryness_1'] = (0.4 * df_new['humidity']) + (0.3 * df_new['cloud']) - (0.3 * df_new['sunshine'])
    # df_new['cloud_sunshine_ratio_1'] = df_new['cloud'] / (df_new['sunshine'] + 1e-5)
    # df_new['weather_index_1'] = (0.4 * df_new['humidity']) + (0.3 * df_new['cloud']) - (0.3 * df_new['sunshine'])

    df_new.fillna(0, inplace=True)

    return df_new


train_processed = create_features(df_train)
test_processed = create_features(df_test)

train_processed = train_processed.fillna(train_processed.mean())
test_processed = test_processed.fillna(test_processed.mean())

correlation_matrix = train_processed.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix[['rainfall']].drop('rainfall').sort_values(by='rainfall', ascending=False), annot=True, cmap='coolwarm')
plt.show()
train_processed.head(5)
# low_correlation_columns = correlation_matrix[correlation_matrix['rainfall'] < 0].index
# print(low_correlation_columns)


cols_no = ['id','rainfall','maxtemp']
X_train = train_processed.drop(cols_no, axis=1)
y_train = train_processed['rainfall']
X_test = test_processed.drop(columns=['id','maxtemp'])

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n Train Data:({},{})".format(X_train_scaled.shape[0],X_train_scaled.shape[1]))
print("\n Test Data:({},{}) ".format(X_test_scaled.shape[0],X_test_scaled.shape[1]))
# Define a function for Purged Cross-Validation


# 拆分训练集 & 验证集
X_train_svm, X_val_svm, y_train_svm, y_val_svm = train_test_split(
    X_train_scaled, y_train, test_size=0.2, random_state=42)


# Define models to train
models = {
    "SVM": SVC(kernel='rbf', C=1.0, gamma='auto', probability=True),  # 选择 RBF 核,
    "Logistic Regression": LogisticRegression(),
    "XGBoost": xgb.XGBClassifier(learning_rate=0.05, max_depth=6, n_estimators=200, objective='binary:logistic'),
    "Extra Trees Classifier": ExtraTreesClassifier(),
    "LightGBM": lightgbm.LGBMClassifier(learning_rate=0.05, max_depth=6, n_estimators=200),
}


best_model = None
best_auc = 0
best_model_name = ""
# 记录各模型的 AUC
model_auc_scores = {}

for model_name, model in models.items():
    print(f"Training {model_name}...")

    model.fit(X_train_svm, y_train_svm)
    # Validation predictions
    val_preds = model.predict(X_val_svm)
    val_proba = model.predict_proba(X_val_svm)[:, 1]

    # Calculate metrics
    accuracy = accuracy_score(y_val_svm, val_preds)
    f1 = f1_score(y_val_svm, val_preds)
    fpr, tpr, _ = roc_curve(y_val_svm, val_proba)
    roc_auc = auc(fpr, tpr)

    print(f"{model_name} - Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}, AUC: {roc_auc:.4f}\n")

    # 存储模型及其 AUC 分数
    model_auc_scores[model_name] = roc_auc

    # 选取 AUC 最高的模型
    if roc_auc > best_auc:
        best_auc = roc_auc
        best_model = model
        best_model_name = model_name

print(f"\nBest Model: {best_model_name} with AUC: {best_auc:.4f}")
# 可选：用完整训练集重新训练最佳模型
best_model_final = models[best_model_name]



#%%
X_train = X_train_svm.reshape((X_train_svm.shape[0], X_train_svm.shape[1], 1))
X_val = X_val_svm.reshape((X_val_svm.shape[0], X_val_svm.shape[1], 1))
X_test_scaled = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))
model = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    Conv1D(filters=32, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=1e-5, verbose=1)

history = model.fit(
    X_train, y_train_svm,
    epochs=200, batch_size=32, validation_data=(X_val, y_val_svm),
    callbacks=[early_stopping, reduce_lr], verbose=1
)

test_preds = model.predict(X_test_scaled).flatten()
df_subm['rainfall'] = test_preds
df_subm.head(10)


df_subm.to_csv('submission.csv', index=False)

