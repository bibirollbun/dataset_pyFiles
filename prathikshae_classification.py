# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


df=pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
data_test=pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')


df.columns


display(df.head())
display(df.info())


display(df.describe())


display(df.isnull().sum())


categorical_cols = df.select_dtypes(include='object').columns
for col in categorical_cols:
    print(f"Column '{col}': {df[col].nunique()} unique values")
    if df[col].nunique() < 20:
        print(df[col].value_counts())


import matplotlib.pyplot as plt
import seaborn as sns


corr = df.corr(numeric_only=True)
# Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()


#Figure size for the plot
plt.figure(figsize=(6,6))

#Create a pie chart
counts = df['y'].value_counts().sort_index()  # ensures 0 comes before 1
labels = ['Class 0', 'Class 1']

plt.pie(counts, labels=labels, autopct='%1.1f%%', colors = ['#a9a9a9', '#e74c3c'])
plt.title('Target y ')


df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
display(df_encoded.head())


from sklearn.preprocessing import StandardScaler

numerical_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
scaler = StandardScaler()
df_encoded[numerical_cols] = scaler.fit_transform(df_encoded[numerical_cols])
display(df_encoded.head())


from sklearn.model_selection import train_test_split

# Drop rows with missing values
df_cleaned = df_encoded.dropna()

X = df_cleaned.drop(['y', 'id'], axis=1)
y = df_cleaned['y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

display(X_train.head())
display(X_test.head())
display(y_train.head())
display(y_test.head())


from sklearn.linear_model import LogisticRegression

model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")


from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

y_pred_rf = rf_model.predict(X_test)
y_pred_proba_rf = rf_model.predict_proba(X_test)[:, 1]

accuracy_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf)
recall_rf = recall_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)
roc_auc_rf = roc_auc_score(y_test, y_pred_proba_rf)

print(f"Random Forest Accuracy: {accuracy_rf:.4f}")
print(f"Random Forest Precision: {precision_rf:.4f}")
print(f"Random Forest Recall: {recall_rf:.4f}")
print(f"Random Forest F1-score: {f1_rf:.4f}")
print(f"Random Forest ROC AUC: {roc_auc_rf:.4f}")


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Define the model
model_nn = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')  # Sigmoid for binary classification
])

# Compile the model
model_nn.compile(optimizer=Adam(learning_rate=0.001),
                 loss='binary_crossentropy',  # Binary crossentropy for binary classification
                 metrics=['accuracy'])

# Train the model
history = model_nn.fit(X_train, y_train, epochs=20, batch_size=40, validation_split=0.2)


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Evaluate the model on the test data
loss, accuracy_nn = model_nn.evaluate(X_test, y_test, verbose=0)
print(f"Neural Network Accuracy: {accuracy_nn:.4f}")

# Get predictions for other metrics
y_pred_nn = (model_nn.predict(X_test) > 0.5).astype("int32")
y_pred_proba_nn = model_nn.predict(X_test)

precision_nn = precision_score(y_test, y_pred_nn)
recall_nn = recall_score(y_test, y_pred_nn)
f1_nn = f1_score(y_test, y_pred_nn)
roc_auc_nn = roc_auc_score(y_test, y_pred_proba_nn)


print(f"Neural Network Precision: {precision_nn:.4f}")
print(f"Neural Network Recall: {recall_nn:.4f}")
print(f"Neural Network F1-score: {f1_nn:.4f}")
print(f"Neural Network ROC AUC: {roc_auc_nn:.4f}")


from lightgbm import LGBMClassifier
import numpy as np

# Calculate scale_pos_weight manually
neg_count = np.sum(y_train == 0)
pos_count = np.sum(y_train == 1)
scale_pos_weight_value = neg_count / pos_count

lgbm_model = LGBMClassifier(random_state=42, scale_pos_weight=scale_pos_weight_value)
lgbm_model.fit(X_train, y_train)



from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

y_pred_lgbm = lgbm_model.predict(X_test)
y_pred_proba_lgbm = lgbm_model.predict_proba(X_test)[:, 1]

accuracy_lgbm = accuracy_score(y_test, y_pred_lgbm)
precision_lgbm = precision_score(y_test, y_pred_lgbm)
recall_lgbm = recall_score(y_test, y_pred_lgbm)
f1_lgbm = f1_score(y_test, y_pred_lgbm)
roc_auc_lgbm = roc_auc_score(y_test, y_pred_proba_lgbm)

print(f"LightGBM Accuracy: {accuracy_lgbm:.4f}")
print(f"LightGBM Precision: {precision_lgbm:.4f}")
print(f"LightGBM Recall: {recall_lgbm:.4f}")
print(f"LightGBM F1-score: {f1_lgbm:.4f}")
print(f"LightGBM ROC AUC: {roc_auc_lgbm:.4f}")


from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []
roc_auc_scores = []

for fold, (train_index, val_index) in enumerate(skf.split(X, y)):
    print(f"Fold {fold+1}/5")
    X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
    y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]

    lgbm_model_fold = LGBMClassifier(random_state=42, scale_pos_weight=scale_pos_weight_value)
    lgbm_model_fold.fit(X_train_fold, y_train_fold)

    y_pred_fold = lgbm_model_fold.predict(X_val_fold)
    y_pred_proba_fold = lgbm_model_fold.predict_proba(X_val_fold)[:, 1]

    accuracy_scores.append(accuracy_score(y_val_fold, y_pred_fold))
    precision_scores.append(precision_score(y_val_fold, y_pred_fold))
    recall_scores.append(recall_score(y_val_fold, y_pred_fold))
    f1_scores.append(f1_score(y_val_fold, y_pred_fold))
    roc_auc_scores.append(roc_auc_score(y_val_fold, y_pred_proba_fold))


print("\nCross-validation results:")
print(f"Average Accuracy: {np.mean(accuracy_scores):.4f} (+/- {np.std(accuracy_scores):.4f})")
print(f"Average Precision: {np.mean(precision_scores):.4f} (+/- {np.std(precision_scores):.4f})")
print(f"Average Recall: {np.mean(recall_scores):.4f} (+/- {np.std(recall_scores):.4f})")
print(f"Average F1-score: {np.mean(f1_scores):.4f} (+/- {np.std(f1_scores):.4f})")
print(f"Average ROC AUC: {np.mean(roc_auc_scores):.4f} (+/- {np.std(roc_auc_scores):.4f})")


import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# LightGBM
lgb_model = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, random_state=42)
lgb_model.fit(X_train, y_train)

# Random Forest
rf_model = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)

# XGBoost
xgb_model = XGBClassifier(n_estimators=500, learning_rate=0.05, random_state=42, use_label_encoder=False, eval_metric="logloss")
xgb_model.fit(X_train, y_train)



from sklearn.ensemble import VotingClassifier

ensemble_model = VotingClassifier(
    estimators=[('lgb', lgb_model), ('rf', rf_model), ('xgb', xgb_model)],
    voting='soft'   # 'soft' = average predicted probabilities
)

ensemble_model.fit(X_train, y_train)



# Predictions on training set
y_train_pred = ensemble_model.predict(X_train)

# Print the predictions
print("Training Predictions:")
print(y_train_pred[:50])   # print first 50 predictions for a quick look

# Optionally, check accuracy or other metrics
from sklearn.metrics import accuracy_score, classification_report

acc = accuracy_score(y_train, y_train_pred)
print(f"Training Accuracy: {acc:.4f}")
print("\nClassification Report:\n", classification_report(y_train, y_train_pred))



test = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")
categorical_cols = test.select_dtypes(include='object').columns
for col in categorical_cols:
    print(f"Column '{col}': {test[col].nunique()} unique values")
    if test[col].nunique() < 20:
        print(test[col].value_counts())


test_encoded = pd.get_dummies(test, columns=categorical_cols, drop_first=True)
display(test_encoded.head())


from sklearn.preprocessing import StandardScaler

numerical_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
scale = StandardScaler()
test_encoded[numerical_cols] = scale.fit_transform(test_encoded[numerical_cols])
display(test_encoded.head())


test_cleaned = test_encoded.dropna()
xy = test_cleaned.drop(['id'], axis=1)


import pandas as pd





# Generate predictions
y_pred = ensemble_model.predict(xy)

submission = pd.DataFrame({
    "id": test["id"],       
    "target": y_pred        
})

# Save to CSV
submission.to_csv("submission.csv", index=False)

print("submission.csv file created successfully!")





