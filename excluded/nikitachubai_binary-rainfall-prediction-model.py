import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns


train_data = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
test_features = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e3/sample_submission.csv")



test_ids = test_features['id']


sample_submission.info()


train_data.info()


corr_matrix = train_data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation matrix of features")
plt.show()


import warnings
warnings.filterwarnings("ignore")

features = ['pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint', 'humidity', 'cloud', 'sunshine', 'windspeed']

plt.figure(figsize=(16, 12))
for i, feature in enumerate(features):
    plt.subplot(3, 3, i+1)
    sns.histplot(train_data[feature], kde=True)
    plt.title(f"Distribution of {feature}")
plt.tight_layout()
plt.show()


corr_matrix = train_data.corr()
print("Correlation with target (rainfall):")
print(corr_matrix["rainfall"].sort_values(ascending=False))


# Remove the id column from the training and test sets since it is not a feature
train_data = train_data.drop('id', axis=1)
test_features = test_features.drop('id', axis=1)


X = train_data.drop('rainfall', axis=1)
y = train_data['rainfall']


# Correlation analysis: output which features are most strongly associated with rain
corr_matrix = train_data.corr()
print("Correlation with target (rainfall):")
print(corr_matrix["rainfall"].sort_values(ascending=False))


# Split the training set into training and validation samples (80/20)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)




imputer = SimpleImputer(strategy='mean')
X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
X_val_imp = pd.DataFrame(imputer.transform(X_val), columns=X_val.columns)
test_imp = pd.DataFrame(imputer.transform(test_features), columns=test_features.columns) 


# training model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_imp, y_train)


# Model evaluation on the validation set using the AUC ROC metric
y_val_pred_proba = model.predict_proba(X_val_imp)[:, 1]
auc = roc_auc_score(y_val, y_val_pred_proba)
print("AUC ROC on validation:", auc)


from sklearn.metrics import roc_auc_score
y_pred_proba = model.predict_proba(X_val)[:, 1]  # Probabilities for class 1
auc = roc_auc_score(y_val, y_pred_proba)
print(f"AUC ROC on validation: {auc}")


# Get probability predictions for the test set
test_pred_proba = model.predict_proba(test_imp)[:, 1]


# Generate the final sending file
submission = pd.DataFrame({'id': sample_submission['id'], 'rainfall': test_pred_proba})
submission.to_csv('submission_v3.csv', index=False)


submission_v3 = pd.read_csv("submission_v3.csv")


submission_v3.info()


submission_v3['rainfall'].describe()




