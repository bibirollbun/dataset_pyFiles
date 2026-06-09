import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e8/sample_submission.csv")


train.head()


train.shape


for col in train.columns:
    print(f"{col}: {train[col].unique()}")


train.drop(columns=["id"], inplace=True)
test.drop(columns=["id"], inplace=True)


train.info()


train.describe()


num_cols = train.select_dtypes(include=['int','float']).columns
cat_cols = train.select_dtypes(exclude=['int','float']).columns


for col in num_cols:
    plt.figure(figsize=(12,4))

    # Histogram
    plt.subplot(1,2,1)
    sns.histplot(train[col], kde=True, bins=30)
    plt.title(f'Distribution of {col}')

    # Boxplot
    plt.subplot(1,2,2)
    sns.boxplot(x=train[col])
    plt.title(f'Boxplot of {col}')

    plt.tight_layout()
    plt.show()
    


for col in cat_cols:
    plt.figure(figsize=(8,4))
    sns.countplot(x=train[col], order=train[col].value_counts().index)
    plt.title(f'Countplot of {col}')
    plt.xticks(rotation=45)
    plt.show()


num_cols = [col for col in num_cols if col != "y"]


from scipy.stats import skew

# Check skewness for numeric columns
skewed_feats = train[num_cols].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("Skewness of numeric features:\n", skewed_feats)


# Check skewness for numeric columns
skewed_feats_test = test[num_cols].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("Skewness of numeric features:\n", skewed_feats)


skewed_feats = train[num_cols].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
skewed_cols = skewed_feats[abs(skewed_feats) > 0.75].index



def log_transform(df, cols):
    for col in cols:
        if (df[col] > 0).any():
            df[col] = np.log1p(df[col].clip(lower=0))
        else:
            # skip if no positive values
            pass
    return df


train = log_transform(train, skewed_cols)
test  = log_transform(test, skewed_cols)


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
train[num_cols] = scaler.fit_transform(train[num_cols])   # fit + transform train
test[num_cols]  = scaler.transform(test[num_cols])        # transform only test


train.head()


from sklearn.preprocessing import LabelEncoder
for col in cat_cols:
    le = LabelEncoder()
    # fit on combined train+test to capture all categories
    le.fit(pd.concat([train[col], test[col]], axis=0))
    
    train[col] = le.transform(train[col])
    test[col] = le.transform(test[col])


train.head()


from sklearn.model_selection import train_test_split

X= train.drop(columns=["y"])
y = train["y"]


X
y


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


X_train


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

LR = LogisticRegression()
LR.fit(X_train, y_train)


y_pred_proba = LR.predict_proba(X_test)[:, 1]  
roc_auc = roc_auc_score(y_test, y_pred_proba)
roc_auc


LR.fit(X,y)


y_pred_proba = LR.predict_proba(X)[:, 1]  
roc_auc = roc_auc_score(y, y_pred_proba)
roc_auc


y_pred = LR.predict(test)
y_pred_proba = LR.predict_proba(test)[:, 1] 


submission = pd.read_csv('/kaggle/input/playground-series-s5e8/sample_submission.csv')
submission["y"] = y_pred_proba
submission.to_csv("submission.csv", index=False)
print("submission.csv saved.")
submission.head()


y_pred = LR.predict(X_test)
y_pred_proba = LR.predict_proba(X_test)[:, 1]


from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Confusion Matrix 
cm = confusion_matrix(y_test, y_pred)

# Heatmap  plot
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix Heatmap")
plt.show()



from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

# Classification Report
print(classification_report(y_test, y_pred))

# ROC-AUC Score
roc_auc = roc_auc_score(y_test, y_pred_proba)
print("ROC-AUC Score:", roc_auc)


