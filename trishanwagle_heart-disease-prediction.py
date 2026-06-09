import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("darkgrid")


df = pd.read_csv("/kaggle/input/heart-disease-prediction-dataquest/heart_train.csv")
df.head()


df.describe()


df.isna().sum()


df.duplicated().sum()


df.info()


df.shape


X = df.iloc[:, :-1]
y = df.iloc[:, -1]

categorical_features = X.select_dtypes(include=['object']).columns
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns


sns.pairplot(df, hue="HeartDisease")
plt.show()


class_counts = df['HeartDisease'].value_counts()
labels = ['Normal', 'Heart Disease']

plt.pie(class_counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=["#01D4C5", "#FF6685"])
plt.title('Target Variable Distribution - HeartDisease')
plt.axis('equal')
plt.show()


fig, axis = plt.subplots(nrows=2, ncols=3, figsize=(16, 10))

for ax, x_value in zip(axis.flat, numerical_features):
    sns.kdeplot(data=df, x=x_value, hue='HeartDisease', fill=True, common_norm=False, alpha=0.5, ax=ax)
    ax.set_title(x_value)
fig.suptitle("Data Distribution", fontsize=16)
plt.tight_layout()
plt.show()


fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 8))
axes = axes.flatten()

categorical_cols = ['Sex', 'ChestPainType', 'FastingBS', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
for index, feature_name in enumerate(categorical_cols):
    sns.countplot(x=feature_name, hue='HeartDisease', data=df, ax=axes[index], alpha=0.8)
    axes[index].set_title(f"{feature_name} v/s Heart Disease")
    axes[index].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.suptitle("Categorical Features v/s Heart Disease", fontsize=16, y=1.03)
plt.show()


from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore', drop='first')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough'
)


from sklearn.pipeline import Pipeline
import xgboost as xgb

model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('xgb', xgb.XGBClassifier()) 
])


from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV

params = {
    'xgb__n_estimators': [100, 200],
    'xgb__max_depth': [3, 5, 7],
    'xgb__learning_rate': [0.01, 0.1, 0.2],
    'xgb__subsample': [0.8, 1],
    'xgb__colsample_bytree': [0.8, 1]
}

cv = StratifiedKFold(n_splits=5, shuffle=True)
grid = GridSearchCV(model_pipeline, param_grid=params, cv=cv, scoring='accuracy', n_jobs=-1, verbose=1)
grid.fit(X, y)

print("Best Parameters:", grid.best_params_)
print("Best CV Accuracy:", grid.best_score_)


from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y)

best_model = grid.best_estimator_
best_model


y_pred_val = best_model.predict(X_val)
y_pred_proba_val = best_model.predict_proba(X_val)[:, 1]


from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report, 
    roc_auc_score, precision_score, recall_score, f1_score, roc_curve
)

accuracy = accuracy_score(y_val, y_pred_val)
precision = precision_score(y_val, y_pred_val)
recall = recall_score(y_val, y_pred_val)
f1 = f1_score(y_val, y_pred_val)
roc_auc = roc_auc_score(y_val, y_pred_proba_val) 
fpr, tpr, thresholds = roc_curve(y_val, y_pred_proba_val)
conf_matrix = confusion_matrix(y_val, y_pred_val)
class_report = classification_report(y_val, y_pred_val)


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

# Confusion Matrix
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", ax=axes[0])
axes[0].set_title('Confusion Matrix')

# ROC Curve
axes[1].plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.4f})', color='blue')
axes[1].plot([0, 1], [0, 1], 'k--', label='Random guess')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('ROC Curve')
axes[1].legend(loc='lower right')
axes[1].grid(True)

plt.tight_layout()
plt.show()

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC AUC Score: {roc_auc:.4f}")
print("\nClassification Report:\n", class_report)


test_df = pd.read_csv('/kaggle/input/heart-disease-prediction-dataquest/heart_test.csv')
test_features = test_df[X.columns] 
test_predictions = best_model.predict(test_features)
test_predictions


submission_df = pd.DataFrame({'id': test_df.index.values, 'HeartDisease': test_predictions})
submission_df.head()


submission_df.to_csv("submission.csv", index=False)

