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


RANDOM_STATE = 442


# Import train and test datasets
train_data = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')
print(f'train_data has: \n\t{train_data.shape[0]} rows\n\t{train_data.shape[1]} columns')
print(f'test_data has: \n\t{test_data.shape[0]} rows\n\t{test_data.shape[1]} columns')


train_data.head()


train_data['rainfall'].unique()


train_data.isna().sum()


test_data.isna().sum()


# Find one row where winddirection is NaN
test_data[test_data['winddirection'].isna()]


# Replace it with mean value
test_data.loc[test_data['winddirection'].isna(), 'winddirection'] = test_data['winddirection'].mean().round()


test_data.isna().sum()


X_train = train_data.drop(columns=['rainfall', 'id'])
y_train = train_data['rainfall']

cv_scores = dict()


train_data = train_data.drop(columns=['id'])


import seaborn as sns
import matplotlib.pyplot as plt


corr = train_data.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


from sklearn import linear_model
from sklearn.model_selection import GridSearchCV


param_grid = {'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1, 10, 100, 1000]}

clf_l1 = GridSearchCV(linear_model.Lasso(), param_grid, cv=5)
clf_l1.fit(X_train, y_train)

l1 = clf_l1.best_estimator_
print(clf_l1.best_params_)


features = X_train.columns
coefs = l1.coef_

df_coefs = pd.DataFrame({'feature': features, 'coefs': coefs}).sort_values(by='coefs', ascending=False)
df_coefs['coefs'] = df_coefs['coefs'].map('{:.6f}'.format)
df_coefs


print(l1.intercept_)


X_train = X_train.drop(columns=['maxtemp', 'mintemp', 'temparature', 'winddirection', 'pressure'])


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)


from sklearn.linear_model import LogisticRegression

param_grid = [
    {'solver': ['liblinear'], 'penalty': ['l1', 'l2'], 'C': [0.1, 1, 10]},
    {'solver': ['lbfgs', 'newton-cg'], 'penalty': ['l2'], 'C': [0.1, 1, 10]},
    {'solver': ['saga'], 'penalty': ['l1', 'l2'], 'C': [0.1, 1, 10]},
    {'solver': ['saga'], 'penalty': ['elasticnet'], 'C': [0.1, 1, 10], 'l1_ratio': [0.1, 0.5, 0.9]}
]

clf_log = GridSearchCV(LogisticRegression(max_iter=2000, random_state=RANDOM_STATE), param_grid, cv=5)
clf_log.fit(X_train_scaled, y_train)


print(clf_log.best_params_)
print(clf_log.best_score_)
cv_scores['LogReg'] = clf_log.best_score_


from sklearn.svm import SVC

param_grid = [
    {'C': [0.1, 1, 10], 'kernel': ['linear']},
    {'C': [0.1, 1, 10], 'kernel': ['rbf', 'poly', 'sigmoid'], 'gamma': ['scale', 'auto']}
]

clf_svm = GridSearchCV(SVC(probability=True), param_grid, cv=5)
clf_svm.fit(X_train_scaled, y_train)


print(clf_svm.best_params_)
print(clf_svm.best_score_)
cv_scores['SVM'] = clf_svm.best_score_


from sklearn.ensemble import RandomForestClassifier

param_grid = {'criterion': ['gini', 'entropy', 'log_loss'], 'max_features': ['sqrt', 'log2', None]}

clf_rf = GridSearchCV(RandomForestClassifier(random_state=RANDOM_STATE), param_grid, cv=5)
clf_rf.fit(X_train, y_train)


print(clf_rf.best_params_)
print(clf_rf.best_score_)
cv_scores['RandomForest'] = clf_rf.best_score_


from sklearn.ensemble import GradientBoostingClassifier

param_grid = {'loss': ['log_loss', 'exponential'], 'criterion': ['friedman_mse', 'squared_error'], 'learning_rate': [0.05, 0.1, 0.3, 0.5]}

clf_gb = GridSearchCV(GradientBoostingClassifier(random_state=RANDOM_STATE), param_grid, cv=5)
clf_gb.fit(X_train, y_train)


print(clf_gb.best_params_)
print(clf_gb.best_score_)
cv_scores['GradBoost'] = clf_gb.best_score_


# No hyperparameters tuning for GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB

clf_nb = GaussianNB()
clf_nb.fit(X_train, y_train)


nb_best_score = cross_val_score(clf_nb, X_train, y_train, cv=5).mean()

print(nb_best_score)
cv_scores['NaiveBayes'] = nb_best_score


df_plot = pd.DataFrame(list(cv_scores.items()), columns=['Model', 'Accuracy'])

plt.figure(figsize=(8, 5))
sns.set_style("whitegrid")
ax = sns.barplot(data=df_plot, x='Model', y='Accuracy', palette='coolwarm', edgecolor='black')

plt.ylim(0.75, 0.9)
plt.ylabel("Cross-Validation Accuracy", fontsize=12)
plt.xlabel("Model", fontsize=12)
plt.title("Model Performance (5-Fold CV)", fontsize=14, fontweight="bold")

for p in ax.patches:
    ax.annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2, p.get_height()),
                ha='center', va='bottom', fontsize=11, fontweight='bold', color='black')


import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


X_train


X_ = X_train.values
y_ = y_train.values

# Float tensor
y_ = y_.astype(np.float32)

# Normalize train data
X_ = scaler.fit_transform(X_)


X_tensor = torch.tensor(X_, dtype=torch.float32)
y_tensor = torch.tensor(y_, dtype=torch.float32).view(-1, 1)


X_tensor_train, X_tensor_test, y_tensor_train, y_tensor_test = train_test_split(\
    X_tensor, y_tensor, test_size=0.25, random_state=RANDOM_STATE)


class FeedForwardNN(nn.Module):
    def __init__(self, input_size):
        super(FeedForwardNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 8)
        self.fco = nn.Linear(8, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fco(x))
        return x


# Criterion and Optimizer selection
model = FeedForwardNN(6)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)


# Training Loop
epochs = 100

for epoch in range(epochs+1):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_tensor_train)
    loss = criterion(outputs, y_tensor_train)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f'[Epoch: {epoch}]: Loss = {loss.item():.5f}')


# Evaluation
model.eval()
with torch.no_grad():
    y_pred = model(X_tensor_test)
    y_pred = (y_pred >= 0.5).float()
    acc = accuracy_score(y_tensor_test.numpy(), y_pred.numpy())
    print(f"Validation Accuracy: {acc:.4f}")


ids = test_data['id']
X_test = test_data.drop(columns=['id', 'maxtemp', 'temparature', 'mintemp', 'winddirection', 'pressure'])
X_test_ = X_test.values
X_test_scaled_ = scaler.fit_transform(X_test_)

X_test_tensor_ = torch.tensor(X_test_scaled_, dtype=torch.float32)


model.eval()
with torch.no_grad():
    y_pred = model(X_test_tensor_)


result = pd.DataFrame({'id': ids, 'rainfall': y_pred[:, 0]})
result.to_csv('/kaggle/working/FF.csv', index=False)


X_test_scaled = scaler.fit_transform(X_test)


y_pred.shape


svm = clf_svm.best_estimator_
y_pred = svm.predict_proba(X_test_scaled)


result = pd.DataFrame({'id': ids, 'rainfall': y_pred[:, 1]})
result.to_csv('/kaggle/working/SVM.csv', index=False)


gradBM = clf_gb.best_estimator_
y_pred = gradBM.predict_proba(X_test)


result = pd.DataFrame({'id': ids, 'rainfall': y_pred[:, 1]})
result.to_csv('/kaggle/working/GradBM.csv', index=False)

