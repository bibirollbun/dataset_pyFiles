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


main_train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
final = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
orignal_train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
external = pd.read_csv("/kaggle/input/fertilizer-prediction/Fertilizer Prediction.csv")


# data overlook
main_train


# checking for null values
main_train.info()


# categorical columns unique values

print(f"The columns:    {main_train.columns}")
print("-"*60)
print(f"Soil Type: {main_train['Soil Type'].unique()}")
print("-"*60)
print(f"Crop type: {main_train['Crop Type'].unique()}")
print("-"*60)
print(f"Fertilizers: {main_train['Fertilizer Name'].unique()}")
print("-"*60)


# Described data for proportion
main_train.describe()


def generate_features(df):
    # Total nutrient content
    df['Total_NPK'] = df['Nitrogen'] + df['Phosphorous'] + df['Potassium']
    
    # Nutrient ratios
    df['N_to_P_ratio'] = round(df['Nitrogen'] / (df['Phosphorous'] + 1),1)
    df['K_to_N_ratio'] = round(df['Potassium'] / (df['Nitrogen'] + 1),1)
    df['PK_to_N_ratio'] = round((df['Phosphorous']+df['Potassium'])/ (df['Nitrogen'] + 1),1)

    # Soil Moisture Category
    df['Soil_Moisture_Level'] = pd.cut(
        df['Moisture'],
        bins=[-np.inf, 35, 55, np.inf],
        labels=['Low', 'Medium', 'High']
    )

    # soil retention
    retention_map = {
        'Clayey': 'High',
        'Loamy': 'Medium-High',
        'Black': 'Medium-High',
        'Red': 'Medium-Low',
        'Sandy': 'Low'
    }
    df['Soil_Retention'] = df['Soil Type'].map(retention_map)

    
    # Temperature category (optional, for some models)
    df['Temp_Level'] = pd.cut(
        df['Temparature'],
        bins=[-np.inf, 28, 33, np.inf],
        labels=['Cool', 'Normal', 'Hot']
    )

    # Humidity category
    df['Humidity_Level'] = pd.cut(
        df['Humidity'],
        bins=[-np.inf, 55, 65, np.inf],
        labels=['Low', 'Medium', 'High']
    )

    df["NxPxP_Binned"] = df.progress_apply(
        lambda x: f"{x['Nitrogen']}_{x['Potassium']}_{x['Phosphorous']}",
        axis=1
    )
    df["SoilxCrop_Binned"] = df.progress_apply(
        lambda x: f"{x['Soil Type']}_{x['Crop Type']}",
        axis=1
    )
    
    # Binary Nutrient Deficiency flags
    df['Low_Nitrogen'] = df['Nitrogen'] < 20
    df['Low_Phosphorous'] = df['Phosphorous'] < 15
    df['Low_Potassium'] = df['Potassium'] < 6

    # Combine nutrient deficiency count
    df['Deficiency_Count'] = df[['Low_Nitrogen', 'Low_Phosphorous', 'Low_Potassium']].sum(axis=1)

    return df



main_train.drop(['id'], axis=1, inplace=True)


# --- Data Augmentation Configuration ---
# Set the number of times to add the external data
main_train = pd.concat([main_train,main_train], axis=0) # orignal copy x1
for i in range(1,6):   #here  external copy x6
    
    main_train = pd.concat([main_train,external], axis=0)



from tqdm import tqdm
tqdm.pandas()
main_train = generate_features(main_train)
final = generate_features(final)


import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, OneHotEncoder, LabelEncoder

def process(train=None, val=None, test=None, onehot=False, label=False, scale=False,
            onehot_catg=['object', 'category', 'bool'], binning=False, scaler=StandardScaler):

    def scaling_func(scaler, train=None, val=None, test=None, cols=None):
        scal = scaler()

        # auto-detect non-binary columns for scaling
        if cols is None:
            one_hot_cols = [col for col in train.columns if set(train[col].unique()) <= {0, 1}]
            cols = [col for col in train.columns if col not in one_hot_cols]

        # Apply scaling
        scaled_train, scaled_test = train.copy(), test.copy()
        scaled_train[cols] = scal.fit_transform(train[cols])
        scaled_test[cols] = scal.transform(test[cols])

        if val is not None:
            scaled_val = val.copy()
            scaled_val[cols] = scal.transform(val[cols])
            return scaled_train, scaled_val, scaled_test
        return scaled_train, scaled_test

    # Drop 'id' column if present
    for df in [train, val, test]:
        if df is not None and 'id' in df.columns:
            df.drop(columns='id', inplace=True)

    # Label Encoding
    if label:
        cat_cols = train.select_dtypes(include=onehot_catg).columns.tolist()
        encoders = {}
        for col in cat_cols:
            le = LabelEncoder()
            train[col] = le.fit_transform(train[col].astype(str))
            if val is not None:
                val[col] = le.transform(val[col].astype(str))
            test[col] = le.transform(test[col].astype(str))
            encoders[col] = le

        train_final, test_final, val_final = train, test, val

    # One-Hot Encoding
    elif onehot:
        cat_cols = train.select_dtypes(include=onehot_catg).columns.tolist()
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoder.fit(train[cat_cols])

        def encode(df):
            encoded = pd.DataFrame(
                encoder.transform(df[cat_cols]),
                columns=encoder.get_feature_names_out(cat_cols),
                index=df.index
            )
            df = df.drop(columns=cat_cols)
            return pd.concat([df, encoded], axis=1)

        train_final = encode(train)
        test_final = encode(test)
        val_final = encode(val) if val is not None else None
    else:
        train_final, test_final, val_final = train, test, val

    # binning
    if binning:
        for colm in train_final.columns:
            train_final[f"{colm}_bins"] = pd.cut(train_final[colm], bins=3, include_lowest=True, labels=False)
            test_final[f"{colm}_bins"] = pd.cut(test_final[colm], bins=3, include_lowest=True, labels=False)
            if val is not None:   
                val_final[f"{colm}_bins"] = pd.cut(val_final[colm], bins=3, include_lowest=True, labels=False)                             

    # Scaling
    if scale:
        return scaling_func(scaler, train_final, val_final, test_final)

    return (train_final, val_final, test_final) if val is not None else (train_final, test_final)



labels = main_train['Fertilizer Name']
le = LabelEncoder()
ytrain_encoded = le.fit_transform(labels.values.ravel())

label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
inverse_mapping = {v: k for k, v in label_mapping.items()}

ytrain_df = pd.DataFrame({'Fertilizer_Name': ytrain_encoded})


main_train.drop(columns=['Fertilizer Name'], inplace=True, axis=1)


main_train, final = process(train=main_train, test=final, onehot=False, label=True,scale=False, binning=True, scaler=MinMaxScaler)


main_train


# import statements

from sklearn.model_selection import train_test_split,KFold, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder

import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, log_loss, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import copy


import matplotlib.pyplot as plt
import seaborn as sns

import torch.nn.init as init
import torch.nn as nn
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class FNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(44, 128),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.ELU(),

            # nn.Linear(128, 512),
            # nn.BatchNorm1d(512),
            # nn.ELU(),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.SiLU(),

            
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Mish(),

            # nn.Linear(256, 256),
            # nn.BatchNorm1d(256),
            # nn.Dropout(0.4),
            # nn.SiLU(),
            
            nn.Linear(256,7),
            # nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.net(x)

def init_weights(m):
    if isinstance(m, nn.Linear):
        init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')
        if m.bias is not None:
                init.zeros_(m.bias)
            
    elif isinstance(m, nn.BatchNorm1d):
        init.constant_(m.weight, 1)  # gamma = 1
        init.constant_(m.bias, 0) 


model = FNN().to(device)
model = torch.nn.DataParallel(model)  # Wrap it
model.apply(init_weights)




import gc

# MAP@k scorer
def mapk(preds, targets, k=3):
    topk = preds.topk(k, dim=1).indices  # [N, k]
    targets = targets.view(-1, 1).expand_as(topk)
    correct = (topk == targets).float()
    scores = correct / torch.arange(1, k + 1, device=preds.device).float()
    return scores.sum(1).mean().item()

# Training function
def train_best_fullbatch(main_train, ytrain_encoded, FNN,
                         num_epochs=10,
                         n_splits=10,
                         learning_rate=1e-4,
                         weight_decay=1e-3):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Convert dataset to tensors on GPU
    X_tensor = torch.tensor(main_train.values, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(ytrain_encoded, dtype=torch.long).to(device)

    # Compute class weights from original labels
    class_weights_np = compute_class_weight(class_weight='balanced',
                                            classes=np.unique(ytrain_encoded),
                                            y=ytrain_encoded)
    class_weights = torch.tensor(class_weights_np, dtype=torch.float32).to(device)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_accuracies, fold_losses, fold_map3s, fold_reports = [], [], [], []
    best_accuracy = 0
    best_model_state = None

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_tensor.cpu(), y_tensor.cpu())):
        print(f"\nFold {fold + 1}/{n_splits}")

        # Initialize new model for each fold
        model = FNN().to(device)
        model.apply(init_weights)

        train_idx_tensor = torch.tensor(train_idx, dtype=torch.long).to(device)
        val_idx_tensor = torch.tensor(val_idx, dtype=torch.long).to(device)

        X_train = X_tensor[train_idx_tensor]
        y_train = y_tensor[train_idx_tensor]
        X_val = X_tensor[val_idx_tensor]
        y_val = y_tensor[val_idx_tensor]

        optimizer = optim.AdamW(model.parameters(), lr=learning_rate,
                                weight_decay=weight_decay, eps=1e-8, amsgrad=True)
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)

        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()
            output = model(X_train)
            loss = loss_fn(output, y_train)
            loss.backward()
            optimizer.step()

            train_loss = loss.item()

            # Evaluation
            model.eval()
            with torch.no_grad():
                val_output = model(X_val)
                val_loss = loss_fn(val_output, y_val).item()
                preds = val_output.argmax(dim=1)
                accuracy = (preds == y_val).float().mean().item()
                map3 = mapk(val_output, y_val, k=3)

            print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, "
                  f"Val Loss = {val_loss:.4f}, Accuracy = {accuracy:.4f}, MAP@3 = {map3:.4f}")

        # Fold summary
        fold_losses.append(val_loss)
        fold_accuracies.append(accuracy)
        fold_map3s.append(map3)
        report = classification_report(y_val.cpu().numpy(), preds.cpu().numpy(), digits=4)
        fold_reports.append(report)

        print(f"\nClassification Report for Fold {fold+1}:\n{report}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_state = copy.deepcopy(model.state_dict())

        # Free memory each fold
        del model, X_train, y_train, X_val, y_val, optimizer, loss_fn, output, val_output, preds
        gc.collect()
        torch.cuda.empty_cache()

    print("\n=== Cross-validation Summary ===")
    print(f"Average Accuracy: {np.mean(fold_accuracies):.4f}")
    print(f"Average Validation Loss: {np.mean(fold_losses):.4f}")
    print(f"Average MAP@3: {np.mean(fold_map3s):.4f}")
    print(f"Best Fold Accuracy: {best_accuracy:.4f}")

    # Load best model
    best_model = FNN().to(device)
    best_model.load_state_dict(best_model_state)
    print("Best model loaded and ready.")

    return best_model

# Example weight initialization function
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)



# best_model = train_best_fullbatch(main_train, ytrain_encoded, FNN, num_epochs = 50,
#                     n_splits = 20,
#                     learning_rate = 1e-4,
#                     weight_decay = 1e-3)


import numpy as np
from xgboost import XGBClassifier

def mapk(preds, actuals, k=3):
    """
    Computes the mean average precision at k.
    preds: array-like of shape [n_samples, n_classes], probabilities per class.
    actuals: array-like of true labels.
    """
    preds_topk = np.argsort(-preds, axis=1)[:, :k]
    score = 0.0
    for i in range(len(actuals)):
        if actuals[i] in preds_topk[i]:
            score += 1 / (np.where(preds_topk[i] == actuals[i])[0][0] + 1)
    return score / len(actuals)


from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
import cupy as cp

xgbp = {
    'objective': 'multi:softprob',
    'num_class': 7,
    'max_depth': 16,                       # reduce depth to prevent overfitting
    'learning_rate': 0.1,               # a bit faster to converge, but balanced
    'subsample': 0.6,
    'colsample_bytree': 0.3,             # allow more features per tree
    'colsample_bylevel':0.8,
    'colsample_bynode': 0.9,
    'max_bin': 128,
    'reg_alpha': 2,                    # L1 regularization
    'reg_lambda': 8,                   # L2 regularization
    'tree_method': 'hist',               # or 'gpu_hist' if using GPU
    'device': 'cuda',
    'random_state': 15,
    'n_jobs': 4,
    'eval_metric': 'mlogloss',
    'n_estimators': 500,
    'early_stopping_rounds': 200,
}

def tree_model_cv(X, y, n_splits=10, random_state=42):

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    fold_accuracies = []
    fold_reports = []
    fold_losses = []
    
    best_accuracy = 0
    best_model = None
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\nFold {fold + 1}")

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx].values.ravel(), y.iloc[val_idx].values.ravel()
        

        model = XGBClassifier(**xgbp)

        model.fit(X_train, y_train,
                 eval_set=[(X_train, y_train),(X_val,y_val)],
                  verbose=100
                 )
        preds = model.predict(X_val)
        probas = model.predict_proba(X_val)

        acc = (preds == y_val).mean()   
        loss = log_loss(y_val, probas)
        
        print(f"Validation Accuracy: {acc:.4f} | Log Loss: {loss:.4f}")

        fold_accuracies.append(acc)
        fold_losses.append(loss)

        cm = confusion_matrix(y_val, preds)
        ConfusionMatrixDisplay(cm).plot()
        plt.title(f"Confusion Matrix - Fold {fold+1}")
        plt.show()

        report = classification_report(y_val, preds, digits=4)
        print(f"Classification Report:\n{report}")
        fold_reports.append(report)

        if acc > best_accuracy:
            best_accuracy = acc
            best_model = copy.deepcopy(model)

    print("\n=== Cross-validation Summary ===")
    print(f"Average Accuracy: {np.mean(fold_accuracies):.4f}")
    print(f"Average Log Loss: {np.mean(fold_losses):.4f}")
    print(f"Best Accuracy: {best_accuracy:.4f}")
    print("Best model loaded")

    return best_model, fold_accuracies, fold_losses, fold_reports


best_model, fold_accuracies, fold_losses, fold_reports= tree_model_cv(main_train, ytrain_df, 3)


# Using Bayesian Optimization
from skopt import BayesSearchCV
from skopt.space import Real, Integer,Categorical
param_spaces = {
    'xgb':{
        'n_estimators': Integer(100,10000),
        'max_depth': Integer(0,10),
        'max_leaves':Integer(0,5),
        'max_bin': Integer(10,300),
        'subsample': Real(0.1, 1.0),
        'grow_policy':Categorical(['depthwise', 'lossguide']),
        'learning_rate':Real(0.01,0.3,prior='log-uniform'),
        'subsample': Real(0.5, 1.0),
        'colsample_bytree': Real(0.1, 1.0),
        'colsample_bynode': Integer(0,10),
        'reg_alpha': Integer(0,10),
        'reg_lambda': Integer(0,10)
    },
    'lgbm': {
        'learning_rate': Real(0.01, 0.2, prior='log-uniform'),
        'num_leaves': Integer(16, 256),
        'max_depth': Integer(3, 12),
        'min_child_samples': Integer(10, 100),
        'min_child_weight': Real(1e-3, 10, prior='log-uniform'),
        'subsample': Real(0.5, 1.0),
        'colsample_bytree': Real(0.5, 1.0),
        'reg_alpha': Real(1e-3, 10, prior='log-uniform'),
        'reg_lambda': Real(1e-3, 10, prior='log-uniform'),
        'bagging_fraction': Real(0.5, 1.0),
        'bagging_freq': Integer(1, 10),
        'boosting_type': Categorical(['gbdt', 'dart', 'goss']),
    },
    
    'cat': {
        'learning_rate': Real(0.01, 0.2, prior='log-uniform'),
        'depth': Integer(4, 10),
        'l2_leaf_reg': Real(1, 10, prior='log-uniform'),
        'bagging_temperature': Real(0.0, 1.0),
        'border_count': Integer(32, 255),
        'random_strength': Real(1, 20),
        'od_type': Categorical(['Iter', 'IncToDec']),
        'od_wait': Integer(10, 50),
        'grow_policy': Categorical(['SymmetricTree', 'Depthwise', 'Lossguide']),
        'leaf_estimation_iterations': Integer(1, 20),
        'bootstrap_type': Categorical(['Bayesian', 'Bernoulli', 'MVS']),
    }
}


# # for neural_network
# import torch
# import pandas as pd

# def submission(model, final, filename="submission.csv", device='cuda'):
#     model.eval()
#     test_tensor = torch.tensor(final.values, dtype=torch.float32).to(device)

#     with torch.no_grad():
#         outputs = model(test_tensor)  # (N, num_classes)
#         top3 = outputs.topk(3, dim=1).indices.cpu().numpy()

#     df = pd.DataFrame({
#         "id": pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')['id'],
#         "Fertilizer Name": [' '.join([str(uniques[i]) for i in row]) for row in top3]
#     })

#     df.to_csv(filename, index=False)
#     print(f"Saved submission file as: {filename}")

#     return df

# # dff = submission(best_model, final, filename="submission.csv", device='cuda')


# for trees based model : nongpu
import pandas as pd
import numpy as np

test_csv_path = '/kaggle/input/playground-series-s5e6/test.csv'

def submission_map3(model, test_features, class_map, test_csv_path, filename="submission.csv"):
    # Predict class probabilities
    probs = model.predict_proba(test_features)

    # Get top 3 predicted class indices
    top3 = np.argsort(probs, axis=1)[:, -3:][:, ::-1]

    # Map indices to class labels
    top3_labels = [' '.join(class_map[i] for i in row) for row in top3]

    # Load IDs from test.csv
    ids = pd.read_csv(test_csv_path)['id']

    # Create submission
    submission = pd.DataFrame({'id': ids, 'Fertilizer Name': top3_labels})
    submission.to_csv(filename, index=False)
    print(f"âœ… Saved submission file: {filename}")
    return submission

import pandas as pd
import numpy as np


dkf= submission_map3(best_model, final, inverse_mapping, test_csv_path, filename="submission.csv")




