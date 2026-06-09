import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.impute import SimpleImputer
test=pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')
train=pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
train.head()
train.info()
train.isnull().sum()

train.describe()
train.dropna()





x=train.drop(columns=['rainfall','id'])
y=train['rainfall']
x_train,x_val,y_train,y_val=train_test_split(x,y,test_size=0.2,random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)


test_features = test.drop(columns=['id'])

# Align features
test_features = test_features[x.columns]

# Fill missing values
test_features = test_features.fillna(test_features.mean())

# Convert types if needed
test_features = test_features.astype(x.dtypes)

# Predict on test set
test['rainfall'] = model.predict_proba(test_features)[:, 1]



y_val_pred = model.predict_proba(x_val)[:, 1]
test_features = test_features.dropna()
test_features = test_features.fillna(test_features.mean())
test_features = test_features.fillna(test_features.median())
test_features = test_features.fillna(0)


imputer = SimpleImputer(strategy='mean')  # Or 'median', 'most_frequent'
test_features = pd.DataFrame(imputer.fit_transform(test_features), columns=test_features.columns)

# Evaluate using AUC-ROC
auc = roc_auc_score(y_val, y_val_pred)
print(f"Validation AUC-ROC Score: {auc:.4f}")

# Plot ROC Curve
fpr, tpr, thresholds = roc_curve(y_val, y_val_pred)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
plt.plot([0, 1], [0, 1], linestyle='--', color='red')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# Predict on test set
test_features = test.drop(columns=['id'])


# Create submission file
submission = test[['id', 'rainfall']]
submission.to_csv('submission.csv', index=False)

print("✅ Submission file created successfully!")




