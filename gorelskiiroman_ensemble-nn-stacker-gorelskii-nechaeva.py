!pip install torch scikit-learn numpy pandas xgboost


%%time
%%capture
%reset -f
from IPython.core.interactiveshell import InteractiveShell as IS; IS.ast_node_interactivity = "all"
import numpy as np, pandas as pd, time, matplotlib.pyplot as plt, os, random

from torch.utils.data import random_split
import torch, torchvision
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from torch import nn
import torch.nn.functional as F
from torch.nn import Sequential, Flatten, Linear, LazyLinear, Dropout, AdaptiveAvgPool2d, MaxPool2d, Conv2d, AvgPool2d

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import StackingClassifier, StackingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression, SGDClassifier, RidgeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.model_selection import cross_val_predict
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from itertools import product
from collections import defaultdict
from tqdm import tqdm

torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.autograd.set_detect_anomaly(False)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
generator = torch.Generator().manual_seed(42)

np.random.seed(42)
random.seed(42)


data = pd.read_csv('/kaggle/input/bank-churn-competition-by-ipii-hs-ex-mts/train.csv')
X = data.drop(['id', 'CustomerId', 'Surname', 'Exited'], axis=1)
X['AgeSquared'] = X['Age'] * X['Age']
X['FinBehaviour'] = X['Balance'] / X['EstimatedSalary']
X['AgeActivity'] = X['Age'] * X['IsActiveMember']
conditions = [
    (X['CreditScore'] < 580),
    (X['CreditScore'] >= 580) & (X['CreditScore'] <= 669),
    (X['CreditScore'] >= 670) & (X['CreditScore'] <= 739),
    (X['CreditScore'] >= 740) & (X['CreditScore'] <= 799),
    (X['CreditScore'] >= 800)
]
choices = [0, 1, 2, 3, 4]
X['CreditScore'] = np.select(conditions, choices)
X['Loyalty'] = X['Tenure']/X['Age']
y = data['Exited']


print("Percentage of churners: ", len(data[data['Exited'] == 1]) / len(data['Exited']) * 100)
scale_pos_weight = (len(data[data['Exited'] == 0])) / (len(data[data['Exited'] == 1]))


categorical_cols = ['Geography', 'Gender']
numerical_cols = X.select_dtypes(include=['float64']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(drop='first'), categorical_cols)
    ])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)


X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size = 32, shuffle = True, generator = generator)
test_loader = DataLoader(test_dataset, batch_size = 32, shuffle = False, generator = generator)


class NeuralNetwork(nn.Module):
    def __init__(self, input_dim):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        self.leaky_relu = nn.LeakyReLU(negative_slope = 0.015)
        self.dropout = nn.Dropout(p = 0.4)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.leaky_relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.leaky_relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.leaky_relu(self.fc3(x))
        x = self.sigmoid(self.fc4(x))
        return x


estimators = [
    ('xgb', XGBClassifier(random_state = 42)),\
    ('rfc', RandomForestClassifier(random_state = 42)),
    ('gbc', GradientBoostingClassifier(random_state = 42)),
    ('sgd', SGDClassifier(random_state = 42)),
    ('lda', LinearDiscriminantAnalysis()),
    ('qda', QuadraticDiscriminantAnalysis()),
    ('ridge', RidgeClassifier(random_state = 42))
]

stacker = StackingClassifier(
    estimators = estimators,
    final_estimator = LogisticRegression(class_weight = 'balanced', random_state = 42),
    cv = 5
)
stacker.fit(X_train, y_train)


y_pred_stack = stacker.predict_proba(X_test)[:, 1]
y_pred_stack = y_pred_stack.reshape(3000, 1)


model = NeuralNetwork(X_train.shape[1])
criterion = nn.BCELoss()
optimizer = optim.NAdam(model.parameters(), lr = 0.001, weight_decay = 1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 50)

num_epochs = 15
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    scheduler.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

model.eval()
y_true = []
y_pred = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch)
        y_true.extend(y_batch.cpu().numpy())
        y_pred.extend(outputs.cpu().numpy())

y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_pred = y_pred * 0.65 + y_pred_stack * 0.35

fpr, tpr, thresholds = roc_curve(y_true, y_pred)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')
plt.show()

print(f'AUC-ROC Score: {roc_auc:.4f}')


test_data = pd.read_csv('/kaggle/input/bank-churn-competition-by-ipii-hs-ex-mts/test.csv')
X_test_sub = test_data.drop(['id', 'CustomerId', 'Surname'], axis=1)
X_test_sub['AgeSquared'] = X_test_sub['Age'] * X_test_sub['Age']
X_test_sub['FinBehaviour'] = X_test_sub['Balance'] / X_test_sub['EstimatedSalary']
X_test_sub['AgeActivity'] = X_test_sub['Age'] * X_test_sub['IsActiveMember']
conditions = [
    (X_test_sub['CreditScore'] < 580),
    (X_test_sub['CreditScore'] >= 580) & (X_test_sub['CreditScore'] <= 669),
    (X_test_sub['CreditScore'] >= 670) & (X_test_sub['CreditScore'] <= 739),
    (X_test_sub['CreditScore'] >= 740) & (X_test_sub['CreditScore'] <= 799),
    (X_test_sub['CreditScore'] >= 800)
]
choices = [0, 1, 2, 3, 4]
X_test_sub['CreditScore'] = np.select(conditions, choices)
X_test_sub['Loyalty'] = X_test_sub['Tenure']/X_test_sub['Age']

X_test_sub_processed = preprocessor.transform(X_test_sub)
X_test_tensor_sub = torch.tensor(X_test_sub_processed, dtype=torch.float32)

y_pred_test_stack = stacker.predict_proba(X_test_sub_processed)[:, 1]
y_pred_test_stack = y_pred_test_stack.reshape(10000, )

test_dataset_sub = TensorDataset(X_test_tensor_sub)
test_loader_sub = DataLoader(test_dataset_sub, batch_size = 32, shuffle = False)

predictions_prob = []
with torch.no_grad():
    for X_batch in test_loader_sub:
        outputs = model(X_batch[0])
        predictions_prob.extend(outputs.squeeze().tolist())

predictions_prob = np.array(predictions_prob)
predictions_prob = predictions_prob * 0.6 + y_pred_test_stack * 0.4


submission = pd.DataFrame({
    'id': test_data['id'],
    'Exited': predictions_prob
})

submission['Exited'] = submission['Exited'].fillna(0)

submission.to_csv('submission.csv', index=False)

