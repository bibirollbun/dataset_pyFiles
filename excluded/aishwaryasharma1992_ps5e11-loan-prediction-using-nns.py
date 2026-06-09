# importing libraries to be used 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
from torch.utils.data import TensorDataset, DataLoader


# loading data 
test_data = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv") # 254K
train_data = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv") # 593K rows

train_data.head() # having a look at the train data


# viewing categorical data columns values for more context
for col in train_data.select_dtypes(exclude=['number']).columns :
    print(f"Unique row values for {col} are : {train_data[col].unique()}")


import numpy as np
import torch
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

class Data_Transform:
    def __init__(self, input_data, y_col, poly_degree=2, interactions_only=False, include_bias=False):
        # store inputs
        self.input_data = input_data
        self.y_col = y_col

        # polynomial hyperparams
        self.poly_degree = poly_degree
        self.interactions_only = interactions_only
        self.include_bias = include_bias

        # initializing columns by type
        # drop id and target from numeric features (assumes 'id' exists; if not, ensure it's present or modify)
        self.num_cols = input_data.select_dtypes(include=['number']).columns.drop(['id', y_col])
        self.ohe_cols = ['gender', 'marital_status', 'employment_status', 'loan_purpose', 'grade_subgrade']
        self.oe_cols = ['education_level']

        # intializing ordinal encoding categorical columns
        self.oe_cat = [['Other', 'High School', "Master's", "Bachelor's", 'PhD']]

        # initializing one hot encoding (use sparse=False for broad compatibility)
        self.ohe = OneHotEncoder(sparse=False, handle_unknown="ignore")

        # initializing ordinal encoding (use the attribute)
        self.oe = OrdinalEncoder(categories=self.oe_cat)

        # initializing the scaler (use the attribute)
        self.scaler = StandardScaler()

        # initializing polynomial feature transformer (will be used inside numeric pipeline)
        self.poly = PolynomialFeatures(
            degree=self.poly_degree,
            interaction_only=self.interactions_only,
            include_bias=self.include_bias
        )

    def data_split(self, test_size=0.2, random_state=42, stratify=True):
        """Split self.input_data into train/val and store/return them."""
        X = self.input_data.drop(columns=self.y_col)
        y = self.input_data[self.y_col]
        strat = y if stratify else None

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=strat
        )

        # store splits as attributes and return them
        self.X_train = X_train
        self.X_val = X_val
        self.y_train = y_train
        self.y_val = y_val
        return X_train, X_val, y_train, y_val

    def build_pipeline(self):
        # numeric pipeline: scale -> polynomial features
        num_pipeline = Pipeline(steps=[
            ("scaler", self.scaler),
            ("poly", self.poly),
        ])

        # step 1 is to have a preprocessor (use self.* everywhere)
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", num_pipeline, list(self.num_cols)),
                ("ohe", self.ohe, self.ohe_cols),
                ("ord", self.oe, self.oe_cols),
            ],
            remainder="drop",
            sparse_threshold=0
        )

        # step 2 is to have a pipeline to process above steps
        self.pipeline = Pipeline(steps=[("preprocessor", self.preprocessor)])

        return self.pipeline

    def tensor_convert(self, input_data_proc):
        # convert numpy-like array to torch.FloatTensor
        out = torch.from_numpy(input_data_proc.astype(np.float32))
        return out


# Assuming you already have your full DataFrame 'train_data'
dt = Data_Transform(train_data, y_col='loan_paid_back')

# 1ï¸�âƒ£ Split the data
X_train, X_val, y_train, y_val = dt.data_split()

# 2ï¸�âƒ£ Build the pipeline (preprocessor)
pipe = dt.build_pipeline()

# 3ï¸�âƒ£ Fit + transform training data, and transform validation data
X_train_scaled = pipe.fit_transform(X_train)
X_val_scaled = pipe.transform(X_val)

# 4ï¸�âƒ£ Convert to tensors
X_train_scaled_t = dt.tensor_convert(X_train_scaled)
X_val_scaled_t   = dt.tensor_convert(X_val_scaled)
y_train_t        = dt.tensor_convert(y_train.to_numpy().reshape(-1, 1))
y_val_t          = dt.tensor_convert(y_val.to_numpy().reshape(-1, 1))


# turning on our cuda device for faster processing
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# update settings for training our model
train_loader = DataLoader(TensorDataset(X_train_scaled_t, y_train_t), batch_size=1024, shuffle=True) # increased from 1024 -> 2048 -> 1024
val_loader   = DataLoader(TensorDataset(X_val_scaled_t, y_val_t), batch_size=1024, shuffle=False) # increased from 1024 -> 2048 -> 1024

# the model has been tweaked with an addtional layer, neurons per layer being increased and dropout decreased from 0.3 to 0.2
model = nn.Sequential(
    nn.Linear(X_train_scaled_t.shape[1], 512),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(64, 1),  # single output neuron for binary
).to(DEVICE)

criterion = nn.BCEWithLogitsLoss()   # combines sigmoid + BCE in one step
optimizer = optim.Adam(model.parameters(), lr=0.01) # need to read more about this from karpathy's lecture

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters: {total_params:,}")


# --- Training ---
for epoch in range(0, 50): # increased the number of iterations from 25 -> 100 -> 10 -> 25
    model.train()
    total_loss = 0.0
    for xb, yb in train_loader:
        xb = xb.to(DEVICE, non_blocking=True).float()
        # ensure target shape is (batch, 1) and float
        yb = yb.to(DEVICE, non_blocking=True).float().view(-1, 1)

        optimizer.zero_grad()
        out = model(xb)                   # shape: (batch, 1)
        loss = criterion(out, yb)         # both shapes match now
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # --- Validation ---
    model.eval()
    val_probs, val_preds, val_true = [], [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(DEVICE).float()
            logits = model(xb)                          # (batch,1)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()  # shape (batch,)
            
            # discrete preds for acc/f1 (use business threshold if you must)
            preds = (probs >= 0.5).astype(int)

            val_probs.extend(probs)
            val_preds.extend(preds)
            # bring true labels to CPU and flatten
            val_true.extend(yb.cpu().numpy().flatten())

    acc = accuracy_score(val_true, val_preds)
    f1  = f1_score(val_true, val_preds)
    # use probabilities for AUC
    try:
        auc = roc_auc_score(val_true, val_probs)
    except ValueError:
        auc = float("nan")

    print(
        f"Epoch {epoch:02d} | Train loss {total_loss/len(train_loader):.4f} | "
        f"Val Acc {acc:.4f} | F1 {f1:.4f} | AUC {auc:.4f}"
    )


# calculate FPR, TPR, thresholds
fpr, tpr, thresholds = roc_curve(val_true, val_preds)

# calculate AUC
auc = roc_auc_score(val_true, val_preds)

# plot
plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
plt.plot([0, 1], [0, 1], 'k--')  # diagonal line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()


# --- Transform the test data using the pipeline inside the class ---
test_data_proc = dt.pipeline.transform(test_data)

# --- Convert transformed test data to tensor using the class method ---
test_data_proc_t = dt.tensor_convert(test_data_proc)

# create loader (no shuffle, since we need predictions in order)
test_loader = DataLoader(TensorDataset(test_data_proc_t), batch_size=1024, shuffle=False)


model.eval()

all_preds = []

with torch.no_grad():
    for xb in test_loader:
        xb = xb[0].to(DEVICE, non_blocking=True)  # since TensorDataset returns tuple
        logits = model(xb)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()
        all_preds.extend(probs)


all_preds_flat = np.array(all_preds).flatten()

sub_data = pd.concat([test_data['id'],pd.Series(all_preds_flat, name ='loan_paid_back')], axis = 1)

sub_data.to_csv('submission.csv',index=False)

