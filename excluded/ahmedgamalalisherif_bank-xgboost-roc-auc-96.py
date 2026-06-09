import pandas as pd 
import numpy as  np 
import matplotlib.pyplot as plt
import seaborn as sns


bank_df = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
bank_df.head()


bank_df_test = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")
bank_df_test.head()


# Show the information at dataset
bank_df.info()


# Check for missimg values 
bank_df.isnull().sum()


categorical_cols = ['job', 'marital', 'education', 'default', 
                    'housing', 'loan', 'contact', 'month', 'poutcome']


# Show the distrubution of job status
plt.figure(figsize=(10,8)) 
sns.countplot(x='job' , data=bank_df  ,palette="Set2") 
plt.title("Distribution of job")
plt.xlabel("job")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()


# Show the distrubution of marital status
plt.figure(figsize=(10,8)) 
sns.countplot(x='marital' , data=bank_df  ,palette="Set2") 
plt.title("Distribution of Marital Status")
plt.xlabel("Marital Status")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()


# Show the distrubution of Education
plt.figure(figsize=(10,8)) 
sns.countplot(x='education' , data=bank_df  ,palette="Set2") 
plt.title("Distribution of The Education")
plt.xlabel("Education")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()


# Show the distrubution of Housing
plt.figure(figsize=(10,8)) 
sns.countplot(x='housing' , data=bank_df  ,palette="Set1") 
plt.title("Distribution of The Housing")
plt.xlabel("Housing")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()


# Show the distrubution of Loan
plt.figure(figsize=(10,8)) 
sns.countplot(x='loan' , data=bank_df  ,palette="Set1") 
plt.title("Distribution of Loan")
plt.xlabel("Loan")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()


# Show the distrubution of poutcome
plt.figure(figsize=(10,8)) 
sns.countplot(x='poutcome' , data=bank_df  ,palette="Set1") 
plt.title("Distribution of Poutcome")
plt.xlabel("poutcome")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()


from category_encoders import TargetEncoder

target_enc = TargetEncoder(cols=['job', 'marital', 'education', 'default', 'housing', 
                                 'loan', 'contact', 'month', 'poutcome'])

bank_df[categorical_cols] = target_enc.fit_transform(bank_df[categorical_cols], bank_df['y'])
bank_df_test[categorical_cols] = target_enc.transform(bank_df_test[categorical_cols])



num_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
for col in num_cols:
    sns.boxplot(x=bank_df[col])
    plt.title(col)
    plt.show()


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
bank_df[num_cols] = scaler.fit_transform(bank_df[num_cols])
bank_df_test[num_cols] = scaler.transform(bank_df_test[num_cols])


# Show the correlation for numerical data 
plt.figure(figsize=(12,8)) 
corr = bank_df[num_cols + ['y']].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm")


bank_df.info()


from sklearn.model_selection import train_test_split 


X = bank_df[categorical_cols + num_cols]
y = bank_df["y"]
X.shape , y.shape


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


X_train.shape  , X_val.shape


X_test = bank_df_test[categorical_cols + num_cols]


from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, confusion_matrix

# Train the model
xgb = XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    use_label_encoder=False,
    random_state=42,
    learning_rate=0.001,
    n_estimators=5000,
    max_depth=6,
    subsample=0.7,
    colsample_bytree=0.8,
)

xgb.fit(X_train, y_train)

y_val_pred_class = xgb.predict(X_val)
y_val_pred_prob = xgb.predict_proba(X_val)[:, 1]

# Evaluation
print("Accuracy:", accuracy_score(y_val, y_val_pred_class))
print("ROC AUC Score:", roc_auc_score(y_val, y_val_pred_prob))
print("Classification Report:\n", classification_report(y_val, y_val_pred_class))
print("Confusion Matrix:\n", confusion_matrix(y_val, y_val_pred_class))

#Final prediction 
test_preds = xgb.predict_proba(X_test)[:, 1]

# Save submission
submission = pd.DataFrame({
    "id": bank_df_test["id"],   
    "y": test_preds
})
submission.to_csv("sample_submission_v1.csv", index=False)
print("sample_submission_v1.csv saved!")



from sklearn.metrics import roc_curve, auc

fpr, tpr, thresholds = roc_curve(y_val, y_val_pred_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.4f})")
plt.plot([0,1],[0,1],'--', color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("XGBoost ROC Curve")
plt.legend()
plt.show()


cm = confusion_matrix(y_val, y_val_pred_class)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No","Yes"], yticklabels=["No","Yes"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()



importances = xgb.feature_importances_
features = X_train.columns

plt.figure(figsize=(10,6))
plt.barh(features, importances)
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.title("XGBoost Feature Importance")
plt.show()


