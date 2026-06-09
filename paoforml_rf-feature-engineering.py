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


import pandas as pd
import numpy as np


from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler, RobustScaler,PowerTransformer,QuantileTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import RocCurveDisplay, roc_auc_score
#from sklearn.svm import SVC   

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
#from cuml.svm import SVC              # GPU SVC
#from cuml.ensemble import RandomForestClassifier  # GPU RF



#graphics
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


train = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv')
orig = pd.read_csv('/kaggle/input/bank-marketing-dataset-full/bank-full.csv', sep =';')


train.head(2)


bin_features = ['default', 'housing', 'loan']

for col in bin_features:
    train[col] = train[col].map({'no':0, 'yes':1})
    #test[col] = test[col].map({'no':0, 'yes':1})



'''
orig['y'] = orig['y'].map({'no': 0, 'yes': 1})
train=train.drop(columns='id')
orig = orig[orig.y==1]
combined = pd.concat([train, orig], axis=0, ignore_index=True)
train= combined.copy()
train['id'] = range(len(train))
train = combined.copy()

train.shape
'''



# 1. Map month abbreviations to month numbers (1–12)
month_map = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
             'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}

train['month_num'] = train['month'].str.lower().map(month_map)

# 2. Convert to radians
month_angle = 2 * np.pi * (train['month_num'] - 1) / 12   # 0 for Jan, 2π/12 for Feb, …

# 3. Cyclic features
train['month_sin'] = np.sin(month_angle)
train['month_cos'] = np.cos(month_angle)

# Optionally drop the helper columns
train = train.drop(columns=['month_num'])


#  Convert to radians
day_angle = 2 * np.pi * (train['day'] - 1) / 31  

# 3. Cyclic features
train['day_sin'] = np.sin(day_angle)
train['day_cos'] = np.cos(day_angle)

# Optionally drop the helper columns
#train = train.drop(columns=['day'])


df = train.copy()
target_col = 'y'
X = df.drop(columns=[target_col,'id'])
y = df[target_col].astype('float32')

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.1, stratify=y, random_state=42
)


X.head()


X.shape


''''
# ------------------------------------------------------------------
# 1) same preprocessing block
# ------------------------------------------------------------------
cat_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns

preprocess = ColumnTransformer(
    [
        ('std', StandardScaler(), ['age']),
        ('qt',  PowerTransformer(method='yeo-johnson', standardize=True),
               ['day', 'pdays', 'previous', 'duration',
                'campaign', 'balance']),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False),
               cat_cols), 
    ], remainder='passthrough'
)

# ------------------------------------------------------------------
# 2) pipeline with a Random-Forest ensemble
# ------------------------------------------------------------------
model = Pipeline(
    steps=[
        ('prep', preprocess),
        ('clf',KNeighborsClassifier(n_neighbors=340,
                                  n_jobs=-1) )
    ]
)

# ------------------------------------------------------------------
# 3) fit & evaluate
# ------------------------------------------------------------------
model.fit(X_train, y_train)

y_prob = model.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_val, y_prob)
print(f"AUC-ROC (positive class, y == 1): {auc:.4f}")
'''


'''
e_prob = (y_prob_rf*0.6 + y_prob*0.4)/2.0
auc = roc_auc_score(y_val, e_prob)
print(f"AUC-ROC (positive class, y == 1): {auc:.4f}")
'''


#PowerTransformer(method='yeo-johnson', standardize=True)
#QuantileTransformer(output_distribution='uniform', random_state=42)

num_cols = X.select_dtypes(include=['int64','float64']).columns
cat_cols = X.select_dtypes(include=['object','category','bool']).columns


preprocess = ColumnTransformer(
        [('std', StandardScaler(), ['age']),
        ('yj', PowerTransformer(method='yeo-johnson', standardize=True), ['day','pdays', 'previous','duration', 'campaign','balance']),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
        ], remainder='passthrough')
'''
pipe = Pipeline(
    steps=[
        ('prep', preprocess),               # numeric scaling + one-hot encoding
       # ('pca',  PCA(n_components=0.95,    # keep 95 % variance
        #             random_state=42))
    ]
)
'''

# Fit + transform
X_train_prep = preprocess.fit_transform(X_train).astype('float32')
X_val_prep = preprocess.transform(X_val).astype('float32')


y_train_np = y_train.values.reshape(-1, 1)
y_val_np = y_val.values.reshape(-1, 1)



rf = RandomForestClassifier(
                    n_estimators = 300,
                    max_depth = 30,
                    random_state = 42,
                    bootstrap=True,
                    n_jobs = -1
                )
rf.fit(X_train_prep, y_train)

y_prob_rf = rf.predict_proba(X_val_prep)[:, 1]
auc = roc_auc_score(y_val, y_prob_rf)
print(f"AUC-ROC (positive class, y == 1): {auc:.6f}")


train_ds  = TensorDataset(torch.tensor(X_train_prep), torch.tensor(y_train_np))
val_ds = TensorDataset(torch.tensor(X_val_prep), torch.tensor(y_val_np))


BATCH = 256
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
val_loader = DataLoader(val_ds,  batch_size=BATCH, shuffle=False)



class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE = nn.functional.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none')
        p  = torch.sigmoid(inputs)
        pt = torch.where(targets == 1, p, 1 - p)
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        loss = alpha_t * (1 - pt)**self.gamma * BCE
        if self.reduction == 'mean':
            return loss.mean()
        return loss.sum()



class Net(nn.Module):
    def __init__(self, in_features): #in_features
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(1)   # raw logit


X_train_prep.shape[1]


device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = Net(X_train_prep.shape[1]).to(device)

criterion = BinaryFocalLoss(alpha = 0.25, gamma= 2)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4) #, lr=1e-4


train_losses = []
val_losses   = []
best_val_loss = float('inf')
EPOCHS = 50
probs=[]
for epoch in range(EPOCHS):
    # --- Training phase ---
    model.train()
    epoch_train_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.squeeze().to(device) #removes any singleton dimensions from yb, 
                                                        # so its shape becomes (batch_size,) 
                                                        #instead of (batch_size, 1)
        
        optimizer.zero_grad() #Clears any accumulated gradients from the previous mini-batch. 
        logits = model(xb) # Performs the forward pass: 
                            # the network takes the mini-batch xb and produces raw, 
                            # unnormalized predictions (logits). 
                            # Shape is usually (batch_size, num_classes).
        loss = criterion(logits, yb)
        loss.backward() # Back-propagates the loss through the network, 
                        # computing gradients for every parameter that has requires_grad=True
        
        optimizer.step() # Updates all learnable parameters using the gradients computed in the previous step, 
                        # according to the chosen optimization algorithm, in this case, Adam
        
        epoch_train_loss += loss.item() * xb.size(0)   # accumulate multiplied by batch size
        
    epoch_train_loss /= len(train_loader.dataset)        # average over the whole dataset
    train_losses.append(epoch_train_loss)

    # --- Validation phase ---
    model.eval()
    epoch_val_loss = 0.0
    with torch.no_grad():
        for xb, yb in val_loader: # for each mini-batch
            xb, yb = xb.to(device), yb.squeeze().to(device)
            logits = model(xb)
            probs.append(torch.sigmoid(logits).cpu())
            loss = criterion(logits, yb)
            epoch_val_loss += loss.item() * xb.size(0)
    
    epoch_val_loss /= len(val_loader.dataset)
    val_losses.append(epoch_val_loss)
    
    # Optional: save best model
    #if epoch_val_loss < best_val_loss:
     #   best_val_loss = epoch_val_loss
      #  torch.save(model.state_dict(), 'best_model.pt')

    # Logging
    if epoch % 2 == 0 or epoch == EPOCHS - 1:
        print(f"Epoch {epoch:3d}  "
              f"train_loss={epoch_train_loss:.4f}  val_loss={epoch_val_loss:.4f}")


# --- Plot ---
import seaborn as sns
import matplotlib.pyplot as plt
with pd.option_context('mode.use_inf_as_na', True):
    plt.figure(figsize=(8, 5))
    sns.lineplot(x=range(EPOCHS), y=train_losses, label='Training loss')
    sns.lineplot(x=range(EPOCHS), y=val_losses,   label='Validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()


model.eval()
probs = []
with torch.no_grad():
    for xb, _ in val_loader:
        xb = xb.to(device)
        logits = model(xb)
        probs.append(torch.sigmoid(logits).cpu())

val_probs = torch.cat(probs).numpy()   # shape (n_test,)

print("Focal Loss Neural Network ROC-AUC:", roc_auc_score(y_val_np, val_probs))


fin_prob = (val_probs + y_prob_rf)/2.0
auc = roc_auc_score(y_val, fin_prob)
print(f"AUC-ROC (positive class, y == 1): {auc:.5f}")


train.head(1)


test = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv')

bin_features = ['default', 'housing', 'loan']

for col in bin_features:
    test[col] = test[col].map({'no':0, 'yes':1})

# 1. Map month abbreviations to month numbers (1–12)
month_map = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
             'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}

test['month_num'] = test['month'].str.lower().map(month_map)

# 2. Convert to radians
month_angle = 2 * np.pi * (test['month_num'] - 1) / 12   # 0 for Jan, 2π/12 for Feb, …

# 3. Cyclic features
test['month_sin'] = np.sin(month_angle)
test['month_cos'] = np.cos(month_angle)

# Optionally drop the helper columns
test = test.drop(columns=['month_num'])

#  Convert to radians
day_angle = 2 * np.pi * (test['day'] - 1) / 31  

# 3. Cyclic features
test['day_sin'] = np.sin(day_angle)
test['day_cos'] = np.cos(day_angle)


X_test = test.drop(columns=['id'])


X_test_prep  = preprocess.transform(X_test) #model.transform(X_val).astype('float32')
X_test_torch = torch.tensor(X_test_prep.astype(np.float32)).to(device)
test_loader  = DataLoader(TensorDataset(X_test_torch),  batch_size = BATCH, shuffle =False)


# ------------------------------------------------------------------
#  Generate predictions on the test set
# ------------------------------------------------------------------
model.eval()
nn_probs = []
with torch.no_grad():
    for xb,in test_loader:
        xb = xb.to(device)
        logits = model(xb)
        nn_probs.append(torch.sigmoid(logits).cpu())


y_prob_rf = rf.predict_proba(X_test_prep)[:, 1]
nn_probs = torch.cat(nn_probs).numpy() 

# --------------------
# Ensenble
# ---------------------

fin_prob = (nn_probs + y_prob_rf)/2.0


subvc = pd.read_csv("/kaggle/input/playground-series-s5e8/sample_submission.csv")


submission = pd.DataFrame({
    'id': subvc.id, 
    'y': fin_prob})


submission.to_csv("submission.csv", index=False)


submission.head()

