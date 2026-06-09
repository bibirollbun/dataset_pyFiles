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


train_data_path = "/kaggle/input/playground-series-s5e8/train.csv"
submission_test_data_path = "/kaggle/input/playground-series-s5e8/test.csv"
submission_sample_data_path = "/kaggle/input/playground-series-s5e8/sample_submission.csv"


train_data  = pd.read_csv(train_data_path)
submission_test_data = pd.read_csv(submission_test_data_path)
sample_submission_data = pd.read_csv(submission_sample_data_path)


train_data.head()


submission_test_data.head()


sample_submission_data.head()


print("Train data shape:", train_data.shape)
print("Test data shape: ", submission_test_data.shape)


train_data.info()


train_data.describe()


print("Null values in dataset :")
print(train_data.isnull().sum())


train_data.duplicated().sum()


import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")


fig, axes = plt.subplots(1, 2, figsize = (12, 5))
sns.histplot(train_data["age"], color="skyblue", edgecolor="black", ax = axes[0])
axes[0].set_title("Distrbution of age")
sns.boxplot(train_data["age"] , ax = axes[1])
axes[1].set_title("Age Spread (Box Splot)")
plt.tight_layout()
plt.show()


plt.figure(figsize=(10,6))  # Bigger figure

sns.histplot(
    data=train_data,
    x="age",
    hue="y",               # color by target (yes/no)
    bins=30,
    multiple="dodge",      # show side-by-side, not stacked
    kde=True,              # smooth line
    palette="Set2",        # good contrasting colors
    alpha=0.7              # little transparent
)

# Add labels and title
plt.xlabel("Age of Customers", fontsize=12)
plt.ylabel("Number of Customers", fontsize=12)
plt.title("Age Distribution by Target (Subscribed = Yes/No)", fontsize=14, weight="bold")

plt.legend(title="Subscribed (y)", labels=["No", "Yes"])
plt.grid(axis="y", linestyle="--", alpha=0.7)

plt.show()


plt.figure(figsize=(10,6))

sns.histplot(
    data=train_data,
    x="balance",
    hue="y",
    bins=40,
    multiple="stack",
    kde=False,              
    palette="Set2"     
)

# Formatting
plt.title("Balance Distribution by Subscription Status (y)", fontsize=16, fontweight="bold")
plt.xlabel("Account Balance", fontsize=14)
plt.ylabel("Number of Clients", fontsize=14)
plt.xticks(rotation=30)
plt.legend(title="Subscribed (y)", labels=["No", "Yes"])
plt.grid(axis="y", linestyle="--", alpha=0.6)

plt.show()



plt.figure(figsize=(8,6))
sns.boxplot(x="y", y="duration", data=train_data, palette="Set2")
plt.title("Call Duration vs Subscription (y)")
plt.xlabel("Subscribed (y)")
plt.ylabel("Call Duration (seconds)")
plt.yscale("log")  # optional if durations are skewed
plt.show()





# Set the style for better aesthetics
sns.set_style("whitegrid")

# Create a larger figure for clarity
plt.figure(figsize=(14, 6))

# Plot count of each day
sns.countplot(
    x="day", 
    data=train_data, 
    color="skyblue",   # consistent color
    edgecolor="black"  # adds clear borders for better visibility
)

# Add title and labels
plt.title("Distribution of Contact Days", fontsize=16)
plt.xlabel("Day of the Month", fontsize=12)
plt.ylabel("Number of Contacts", fontsize=12)

# Show counts above each bar for clarity
for p in plt.gca().patches:
    height = p.get_height()
    plt.text(p.get_x() + p.get_width()/2., height + 2, int(height), 
             ha="center", fontsize=8)

plt.show()



# Set Seaborn style
sns.set_style("whitegrid")

# Create the figure
plt.figure(figsize=(12,6))

# Plot histogram with hue for subscription, more bins, and KDE for smoothness
sns.histplot(
    data=train_data, 
    x="duration", 
    hue="y", 
    bins=50, 
    kde=True,        # adds smooth curve
    alpha=0.6        # slightly transparent for overlapping bars
)

# Add title and labels
plt.title("Call Duration Distribution by Subscription Status", fontsize=16)
plt.xlabel("Call Duration (seconds)", fontsize=12)
plt.ylabel("Number of Calls", fontsize=12)

# Show the plot
plt.show()


# Set style for better aesthetics
sns.set_style("whitegrid")

# Create figure
plt.figure(figsize=(12,6))

# Histogram of 'campaign' with log-scaled y-axis
sns.histplot(
    train_data["campaign"], 
    bins=30, 
    color="skyblue", 
    edgecolor="black"
)

# Apply log scale to y-axis
plt.yscale("log")

# Add title and labels
plt.title("Distribution of Campaign Contacts (Log Scale)", fontsize=16)
plt.xlabel("Number of Contacts in Campaign", fontsize=12)
plt.ylabel("Count (log scale)", fontsize=12)

# Optional: show values on top of bars for clarity
for p in plt.gca().patches:
    height = p.get_height() 
    if height > 0:  # avoid log(0) issues
        plt.text(p.get_x() + p.get_width()/2., height, int(height), 
                 ha="center", fontsize=8, rotation=0)

plt.show()


# Set a clean style
sns.set_style("whitegrid")

# Create figure
plt.figure(figsize=(12,6))

# Histogram of 'pdays'
sns.histplot(
    train_data["pdays"], 
    bins=50, 
    color="lightcoral", 
    edgecolor="black"
)

# Add title and labels
plt.title("Distribution of Days Since Last Contact (pdays)", fontsize=16)
plt.xlabel("Days Since Last Contact", fontsize=12)
plt.ylabel("Number of Clients", fontsize=12)



plt.show()


sns.set_style("whitegrid")

plt.figure(figsize=(8,5))

# Count plot
ax = sns.countplot(
    x="y", 
    data=train_data, 
    palette="pastel", 
    edgecolor="black"
)

plt.title("Distribution of Target Variable (y)", fontsize=16)
plt.xlabel("Subscription (y)", fontsize=12)
plt.ylabel("Number of Clients", fontsize=12)

# Annotate counts inside the bars
for p in ax.patches:
    height = p.get_height()
    ax.text(
        p.get_x() + p.get_width()/2.,  # x position (center)
        height / 2,                    # y position (middle of the bar)
        int(height),                    # text
        ha="center", 
        va="center", 
        fontsize=11,
        color="black"                  # text color
    )

plt.show()


print(train_data["job"].unique())
print("-" * 100)
print(train_data["marital"].unique())
print("-" * 100)
print(train_data["education"].unique())
print("-" * 100)
print(train_data["default"].unique())
print("-" * 100)
print(train_data["housing"].unique())
print("-" * 100)
print(train_data["loan"].unique())
print("-" * 100)
print(train_data["contact"].unique())
print("-" * 100)
print(train_data["month"].unique())
print("-" * 100)
print(train_data["poutcome"].unique())
print("-" * 100)
print(train_data["y"].unique())




education_mapping = {
    'primary' : 0,
    'secondary' : 1,
    'tertiary' : 2,
    'unknown' :-1
}


train_data["education"] = train_data["education"].map(education_mapping)
submission_test_data["education"] = submission_test_data["education"].map(education_mapping)


train_data["education"].unique()


train_data = pd.get_dummies(train_data, columns = ["job", "marital", "contact"], drop_first = True)
test_data = pd.get_dummies(submission_test_data, columns = ["job", "marital", "contact"], drop_first = True)


from sklearn.preprocessing import LabelEncoder

label_encoders = {}
for col in [ "month"]:
    le = LabelEncoder()
    train_data[col] = le.fit_transform(train_data[col])
    test_data[col] = le.fit_transform(test_data[col])
    label_encoders[col] = le  # store encoder for later use



yes_no_mapping = {
    "yes" : 1,
"no" : 0
}


for col in ["default", "housing", "loan"]:
    train_data[col] = train_data[col].map(yes_no_mapping)
    test_data[col] = test_data[col].map(yes_no_mapping)


x = train_data.drop(columns = ["y", "poutcome"])
y = train_data["y"]
test = test_data.drop(columns = [ "poutcome"])


complete = train_data.drop(columns=["poutcome"])


complete.head()


import seaborn as sns
plt.figure(figsize=(15, 15))
sns.heatmap(complete.corr(), annot=True)



correlation = complete.corr()['y'].sort_values(ascending=False)
top_corr = correlation.drop('y').abs().sort_values(ascending=False)
print(top_corr)



plt.scatter(complete['duration'], complete['balance'], c=complete['y'], cmap='coolwarm', alpha=0.6)
plt.xlabel('Duration')
plt.ylabel('Balance')
plt.title('Duration vs Balance colored by target')
plt.show()



import numpy as np
corr_matrix = complete.corr()

# Select upper triangle of the correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# Find feature pairs with correlation > 0.8
high_corr = [(column, row, corr_val) for column in upper.columns for row, corr_val in upper[column].items() if abs(corr_val) > 0.5]
print(high_corr)



from sklearn.model_selection import train_test_split


X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)


X_train.shape, X_test.shape, Y_train.shape, Y_test.shape


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix


DTC_params = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5],
    'max_features': [None, 'sqrt', 'log2'],
    'class_weight' : ['balanced']
}
DTC = DecisionTreeClassifier()

DTC_gridcv = GridSearchCV(
    estimator=DTC,
    param_grid=DTC_params,
    cv=4,
    n_jobs=-1,  # use all CPUs if possible
    scoring=make_scorer(f1_score),  # Correct way to use f1_score
    verbose=0
)
DTC_gridcv.fit(X_train, Y_train)
print(DTC_gridcv.best_params_)
print(DTC_gridcv.best_score_)


from sklearn.model_selection import GridSearchCV
import lightgbm as lgb

# Base model
lgb_estimator = lgb.LGBMClassifier(
    objective="binary",
    boosting_type="gbdt",
    random_state=42,
    n_jobs=-1
)



import torch


num_pos = (Y_train==1).sum().item()
num_neg = (Y_train==0).sum().item()
pos_weight = torch.tensor([num_neg/num_pos])


param_grid = {
    "learning_rate": [0.01, 0.05, 0.1],
    "num_leaves": [31, 100, 127],
    "max_depth": [-1, 8, 30],
    "min_child_samples": [5, 10, 20],
    "n_estimators": [200, 500, 1000],
    "subsample": [0.8, 1.0],              # same as bagging_fraction
    "colsample_bytree": [0.8, 1.0],       # same as feature_fraction
    "reg_alpha": [0, 1.0],                # L1 regularization
    "reg_lambda": [1.0],                  # L2 regularization
    "scale_pos_weight": [pos_weight]  # imbalance handling
}




# GridSearchCV
grid = GridSearchCV(
    estimator=lgb_estimator,
    param_grid=param_grid,
    scoring="f1",   # since data is imbalanced, optimize F1 instead of accuracy/auc
    cv=3,
    verbose=2,
    n_jobs=-1
)


import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from lightgbm import early_stopping, log_evaluation
import numpy as np



params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'learning_rate': 0.01,
    'num_leaves': 127,
    'max_depth': 8,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'reg_alpha': 1.0,
    'reg_lambda': 1.0,
    'seed': 42,
    'verbose': -1
}



folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(x.shape[0])
test_preds = np.zeros(X_test.shape[0])


categorical_features = ["job_self-employed", "job_services", "job_technician", "job_unemployed", "job_unknown", "marital_married", "marital_single", "contact_telephone", "contact_unknown"]


for fold, (train_idx, val_idx) in enumerate(folds.split(x, y), 1):
    X_tr, X_val = x.iloc[train_idx], x.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    train_data = lgb.Dataset(X_tr, label=y_tr, categorical_feature=categorical_features)
    val_data = lgb.Dataset(X_val, label=y_val, categorical_feature=categorical_features)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=5000,
        valid_sets=[val_data],
        callbacks=[early_stopping(stopping_rounds=100), log_evaluation(100)]
    )

    # Validation predictions
    val_pred = model.predict(X_val, num_iteration=model.best_iteration)
    oof_preds[val_idx] = val_pred
    fold_auc = roc_auc_score(y_val, val_pred)
    print(f"Fold {fold} AUC: {fold_auc:.5f}")

    # Test predictions
    test_pred = model.predict(X_test, num_iteration=model.best_iteration)
    test_preds += test_pred / folds.n_splits


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler_x_train = scaler.fit_transform(X_train)
scaler_x_test = scaler.transform(X_test)


from sklearn.neighbors import KNeighborsClassifier




knn = KNeighborsClassifier(n_neighbors=7, weights="distance", algorithm="auto", leaf_size=30, p=2)

knn.fit(X_train, Y_train)

knn_preds  = knn.predict(X_test)


from sklearn.metrics import classification_report,  confusion_matrix


print(classification_report(Y_test, knn_preds))


print(confusion_matrix(Y_test, knn_preds))


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


logreg = LogisticRegression(class_weight="balanced")
score = cross_val_score(logreg, scaler_x_train, Y_train, cv=5, scoring='f1')
print(score.mean())


import time
import torch

import torch.nn as nn
from torch.nn import BCELoss
import torch.optim as optim
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


x_train = torch.tensor(scaler_x_train, dtype = torch.float32)
x_test = torch.tensor(scaler_x_test, dtype = torch.float32)
y_train = torch.tensor(np.array(Y_train), dtype=torch.long)
y_test = torch.tensor(np.array(Y_test), dtype=torch.long)



from torch.utils.data import WeightedRandomSampler

class_counts = torch.bincount(y_train)
class_weights = 1. / class_counts.float()
sample_weights = class_weights[y_train]

train_dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=128)
test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=128)


from sklearn.metrics import f1_score, roc_auc_score
from sklearn.metrics import precision_recall_curve



class MLP(nn.Module):
    def __init__(self, input_size, epochs=100, learning_rate=0.005, pos_weight=None):
        super(MLP, self).__init__()
        # Reduced hidden sizes

        self.hidden2 = nn.Linear(input_size, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.hidden3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.hidden4 = nn.Linear(32, 16)
        self.bn4 = nn.BatchNorm1d(16)

        self.output = nn.Linear(16, 1)

        # Dropout only in last two hidden layers
        self.dropout = nn.Dropout(0.1)

        # Loss
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # Optimizer & scheduler
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3
        )
        self.epochs = epochs

    def forward(self, x):
        x = F.relu(self.bn2(self.hidden2(x)))
        x = F.relu(self.bn3(self.hidden3(x)))
        x = self.dropout(F.relu(self.bn4(self.hidden4(x))))
        out = self.output(x)
        return out

    def fit(self, train_loader, test_loader, device="cpu", early_stop=True):
        self.to(device)

        for epoch in range(1, self.epochs + 1):
            epoch_start = time.time()
            epoch_loss = 0
            self.train()

            # -------- Training --------
            for batchx, batchy in train_loader:
                batchx, batchy = batchx.to(device), batchy.to(device).float().unsqueeze(1)
                self.optimizer.zero_grad()
                outputs = self(batchx)
                loss = self.loss_fn(outputs, batchy)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(train_loader)

            # -------- Validation --------
            self.eval()
            all_preds, all_probs, all_labels = [], [], []

            with torch.no_grad():
                for batchx, batchy in test_loader:
                    batchx, batchy = batchx.to(device), batchy.to(device).float().unsqueeze(1)
                    outputs = self(batchx)
                    probs = torch.sigmoid(outputs)
                    preds = (probs > 0.6).float()

                    all_preds.append(preds.cpu())
                    all_probs.append(probs.cpu())
                    all_labels.append(batchy.cpu())

            all_preds = torch.cat(all_preds).numpy()
            all_probs = torch.cat(all_probs).numpy()
            all_labels = torch.cat(all_labels).numpy()

            # Metrics
            accuracy = (all_preds == all_labels).mean()
            f1 = f1_score(all_labels, all_preds)
            try:
                auc = roc_auc_score(all_labels, all_probs)
            except ValueError:
                auc = float('nan')

            # Threshold tuning
            precision, recall, thresholds = precision_recall_curve(all_labels, all_probs)
            f1_scores = 2 * precision * recall / np.maximum(precision + recall, 1e-8)


            optimal_idx = f1_scores.argmax()
            optimal_threshold = thresholds[optimal_idx] if len(thresholds) > 0 else 0.5

            # Scheduler
            self.scheduler.step(avg_loss)

            epoch_end = time.time()
            print(f"Epoch [{epoch}/{self.epochs}], Loss: {avg_loss:.4f}, "
                  f"Accuracy: {accuracy*100:.2f}%, F1: {f1:.4f}, AUC: {auc:.4f}, "
                  f"Time: {epoch_end - epoch_start:.2f}s, Best threshold: {optimal_threshold:.4f}")




num_pos = (y_train==1).sum().item()
num_neg = (y_train==0).sum().item()
pos_weight = torch.tensor([num_neg/num_pos])


model = MLP(input_size=28, epochs=30, learning_rate=0.001, pos_weight=pos_weight)



model.fit(train_loader, test_loader)  





