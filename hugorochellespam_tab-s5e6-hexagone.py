# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import MinMaxScaler, RobustScaler, OrdinalEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.impute import SimpleImputer


from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, AdaBoostClassifier

from lightgbm.sklearn import LGBMRegressor, LGBMClassifier
from catboost import CatBoostRegressor, CatBoostClassifier, Pool
import xgboost as xgb

import category_encoders as ce

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


!pip install --upgrade git+https://github.com/scikit-learn-contrib/category_encoders


df_train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
sub = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")


# # Dans X, on va prendre toutes les colonnes sauf "id" qui est unique
# FEATURES = [c for c in df_test.columns if c != "id"] 

# CAT_COLS = ["Soil Type", "Crop Type"] # DonnÃ©es catÃ©gorielles
# NUM_COLS = [c for c in FEATURES if c not in CAT_COLS] # DonnÃ©es numÃ©riques
# LABEL = "Fertilizer Name" # Le label ou le Y

# # Nombre de classes
# CLASSES = 7


# enc = OrdinalEncoder()
# enc.fit(df_train[CAT_COLS])
# df_train[CAT_COLS] = enc.transform(df_train[CAT_COLS])
# df_test[CAT_COLS] = enc.transform(df_test[CAT_COLS]) 


# X = df_train[FEATURES]
# y = df_train[LABEL]
# X_test = df_test[FEATURES]


# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.05, random_state=42)


# model = AdaBoostClassifier()
# model.fit(X_train, y_train)


# pred = model.predict(X_test)


# Calculer le score Ã  partir de la mÃ©trique, en comparant pred et y_test. 
# N'oubliez pas d'importer la mÃ©trique ! 


# sub[LABEL] = model.predict(X_test)
# sub.to_csv("submission.csv", index=False)


# # ğŸ“¦ Imports
# import numpy as np
# import pandas as pd

# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, classification_report
# from sklearn.ensemble import AdaBoostClassifier
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import RobustScaler
# import category_encoders as ce

# # ğŸ“� Chargement des donnÃ©es
# df_train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
# df_test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
# sub = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")

# # ğŸ�·ï¸� Variables
# CAT_COLS = ["Soil Type", "Crop Type"]
# LABEL = "Fertilizer Name"
# FEATURES = [col for col in df_test.columns if col != "id"]
# NUM_COLS = [col for col in FEATURES if col not in CAT_COLS]
# CLASSES = df_train[LABEL].nunique()

# # ğŸ§¼ PrÃ©traitement
# ## Imputation des valeurs manquantes (numÃ©riques)
# num_imputer = SimpleImputer(strategy='median')
# df_train[NUM_COLS] = num_imputer.fit_transform(df_train[NUM_COLS])
# df_test[NUM_COLS] = num_imputer.transform(df_test[NUM_COLS])

# # ğŸ”§ Encodage uniquement des colonnes catÃ©gorielles
# encoder = ce.OrdinalEncoder(
#     cols=CAT_COLS,
#     handle_unknown='impute',   # ou 'value' si tu veux spÃ©cifier une valeur
#     handle_missing='impute'    # ou 'value' aussi
# )
# df_train[CAT_COLS] = encoder.fit_transform(df_train[CAT_COLS])
# df_test[CAT_COLS] = encoder.transform(df_test[CAT_COLS])


# ## Standardisation des features numÃ©riques
# scaler = RobustScaler()
# df_train[NUM_COLS] = scaler.fit_transform(df_train[NUM_COLS])
# df_test[NUM_COLS] = scaler.transform(df_test[NUM_COLS])

# # ğŸ�¯ Features et cible
# X = df_train[FEATURES]
# y = df_train[LABEL]
# X_test = df_test[FEATURES]

# # ğŸ§ª Split train / val
# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# # ğŸ¤– ModÃ©lisation
# model = AdaBoostClassifier(n_estimators=1000, learning_rate=0.5, random_state=42)
# model.fit(X_train, y_train)

# # ğŸ“Š Ã‰valuation
# val_preds = model.predict(X_val)
# print("Accuracy sur validation:", accuracy_score(y_val, val_preds))
# print(classification_report(y_val, val_preds))

# # ğŸ“¤ PrÃ©diction test
# sub[LABEL] = model.predict(X_test)
# sub.to_csv("submission.csv", index=False)

# # ğŸ“Š Ã‰valuation validation
# val_preds = model.predict(X_val)
# score = accuracy_score(y_val, val_preds)
# print(f"Validation Accuracy: {score:.5f}")

# # ğŸ’¾ Sauvegarde avec nom dynamique
# filename = f"submission_acc_{score:.5f}.csv"
# sub.to_csv(filename, index=False)
# print(f"Saved submission to: {filename}")

# # ğŸ“� Log du score
# with open("log.txt", "a") as f:
#     f.write(f"{filename} - Accuracy: {score:.5f}\n")




# # ğŸ“¦ Imports
# import pandas as pd
# import numpy as np
# import torch
# import torch.nn as nn
# from torch.utils.data import TensorDataset, DataLoader
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# import category_encoders as ce
# import torch.optim as optim
# import os

# # ğŸ“‚ Chargement des donnÃ©es
# df_train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
# df_test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
# sub = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")

# # ğŸ§¹ PrÃ©paration des donnÃ©es
# CAT_COLS = ["Soil Type", "Crop Type"]
# LABEL = "Fertilizer Name"
# FEATURES = [c for c in df_test.columns if c != "id"]
# NUM_COLS = [c for c in FEATURES if c not in CAT_COLS]

# # ğŸ§ª Encode le label
# label2idx = {name: idx for idx, name in enumerate(sorted(df_train[LABEL].unique()))}
# idx2label = {v: k for k, v in label2idx.items()}
# df_train["label"] = df_train[LABEL].map(label2idx)

# # ğŸ§· SÃ©paration avant encodage
# X = df_train[FEATURES].copy()
# y = df_train["label"].copy()
# X_test = df_test[FEATURES].copy()

# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

# # ğŸ�¯ Target Encoding
# target_enc = ce.TargetEncoder(cols=CAT_COLS)
# X_train[CAT_COLS] = target_enc.fit_transform(X_train[CAT_COLS], y_train)
# X_val[CAT_COLS] = target_enc.transform(X_val[CAT_COLS])
# X_test[CAT_COLS] = target_enc.transform(X_test[CAT_COLS])

# # ğŸ”„ Standardisation
# scaler = StandardScaler()
# X_train[NUM_COLS] = scaler.fit_transform(X_train[NUM_COLS])
# X_val[NUM_COLS] = scaler.transform(X_val[NUM_COLS])
# X_test[NUM_COLS] = scaler.transform(X_test[NUM_COLS])

# # ğŸ“¦ Dataset PyTorch
# X_train_tensor = torch.tensor(X_train.values.astype(np.float32))
# y_train_tensor = torch.tensor(y_train.values)
# X_val_tensor = torch.tensor(X_val.values.astype(np.float32))
# y_val_tensor = torch.tensor(y_val.values)
# X_test_tensor = torch.tensor(X_test.values.astype(np.float32))

# train_ds = TensorDataset(X_train_tensor, y_train_tensor)
# val_ds = TensorDataset(X_val_tensor, y_val_tensor)
# test_ds = TensorDataset(X_test_tensor)

# train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)
# val_loader = DataLoader(val_ds, batch_size=512)
# test_loader = DataLoader(test_ds, batch_size=512)

# # ğŸ§  ModÃ¨le
# class SimpleNN(nn.Module):
#     def __init__(self, in_features, num_classes):
#         super().__init__()
#         self.model = nn.Sequential(
#             nn.Linear(in_features, 256),
#             nn.ReLU(),
#             nn.Dropout(0.3),
#             nn.Linear(256, 64),
#             nn.LeakyReLU(0.1),
#             nn.Dropout(0.3),
#             nn.Linear(64, num_classes)
#         )
#     def forward(self, x):
#         return self.model(x)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = SimpleNN(X_train.shape[1], len(label2idx)).to(device)

# # âš™ï¸� EntraÃ®nement
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
# epochs = 50

# def map3(preds, targets):
#     total = 0
#     for i in range(len(targets)):
#         target = targets[i]
#         pred = preds[i]
#         for j in range(3):
#             if pred[j] == target:
#                 total += 1 / (j + 1)
#                 break
#     return total / len(targets)

# for epoch in range(epochs):
#     model.train()
#     for xb, yb in train_loader:
#         xb, yb = xb.to(device), yb.to(device)
#         optimizer.zero_grad()
#         out = model(xb)
#         loss = criterion(out, yb)
#         loss.backward()
#         optimizer.step()

#     model.eval()
#     val_preds = []
#     val_targets = []
#     with torch.no_grad():
#         for xb, yb in val_loader:
#             xb = xb.to(device)
#             logits = model(xb)
#             top3 = torch.topk(logits, k=3, dim=1).indices.cpu().numpy()
#             val_preds.append(top3)
#             val_targets.append(yb.numpy())
#     val_preds = np.vstack(val_preds)
#     val_targets = np.concatenate(val_targets)
#     score = map3(val_preds, val_targets)
#     print(f"Epoch {epoch+1}/{epochs} - Val MAP@3: {score:.5f}")

# # ğŸ“¤ PrÃ©diction test
# model.eval()
# all_preds = []
# with torch.no_grad():
#     for xb in test_loader:
#         xb = xb[0].to(device)
#         logits = model(xb)
#         top3 = torch.topk(logits, k=3, dim=1).indices.cpu().numpy()
#         all_preds.append(top3)

# all_preds = np.vstack(all_preds)

# # ğŸ“„ Soumission
# submission_preds = [
#     " ".join([idx2label[idx] for idx in row]) for row in all_preds
# ]
# sub[LABEL] = submission_preds
# sub.to_csv("submission_nn_targetenc.csv", index=False)
# print("âœ… Fichier de soumission sauvegardÃ© : submission_nn_targetenc.csv")



# df_train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
# df_test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
# sub = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")

# FEATURES = [c for c in df_test.columns if c != "id"] 

# CAT_COLS = ["Soil Type", "Crop Type"] # DonnÃ©es catÃ©gorielles
# NUM_COLS = [c for c in FEATURES if c not in CAT_COLS] # DonnÃ©es numÃ©riques
# LABEL = "Fertilizer Name" # Le label ou le Y

# ENCODER = "OE"

# # Nombre de classes
# CLASSES = 7

# if ENCODER == "CAT":
#     df_train[CAT_COLS] = df_train[CAT_COLS].fillna("None").astype("category")
#     df_test[CAT_COLS] = df_test[CAT_COLS].fillna("None").astype("category")
# elif ENCODER == "TE":
#     enc = ce.TargetEncoder(cols=CAT_COLS)
#     y_train = pd.Series(df_train[LABEL].tolist(), index=df_train[LABEL].index)
#     enc.fit(df_train[FEATURES], y_train)
#     df_train[FEATURES] = enc.transform(df_train[FEATURES], y_train)
#     df_test[FEATURES] = enc.transform(df_test[FEATURES])
# elif ENCODER == "OE":
#     enc = OrdinalEncoder()
#     enc.fit(df_train[CAT_COLS])
#     df_train[CAT_COLS] = enc.transform(df_train[CAT_COLS])
#     df_test[CAT_COLS] = enc.transform(df_test[CAT_COLS]) 

# X = df_train[FEATURES]
# y = df_train[LABEL]
# X_test = df_test[FEATURES]

# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.05, random_state=42)

# model = AdaBoostClassifier()
# model.fit(X_train, y_train)

# pred_val = model.predict(X_val)
# pred = model.predict(X_test)


# # Predict probabilities for each class
# probs = model.predict_proba(X_test)

# # Get class names (after encoding)
# class_labels = model.classes_  # should match encoded class labels

# # Get top 3 predictions per row
# top_3 = np.argsort(probs, axis=1)[:, -3:][:, ::-1]  # Get top 3, ordered highest first

# # Map indices to label names
# top_3_labels = np.vectorize(lambda x: class_labels[x])(top_3)

# # Join top 3 predictions into a space-separated string
# sub[LABEL] = [' '.join(row.astype(str)) for row in top_3_labels]

# # Save the submission file
# sub.to_csv("submission.csv", index=False)



# sub.head()
# print(probs)


# def apk(actual, predicted, k=10):
#     """
#     Computes the average precision at k.
#     This function computes the average prescision at k between two lists of
#     items.
#     Parameters
#     ----------
#     actual : list
#              A list of elements that are to be predicted (order doesn't matter)
#     predicted : list
#                 A list of predicted elements (order does matter)
#     k : int, optional
#         The maximum number of predicted elements
#     Returns
#     -------
#     score : double
#             The average precision at k over the input lists
#     """
#     if len(predicted)>k:
#         predicted = predicted[:k]

#     score = 0.0
#     num_hits = 0.0

#     for i,p in enumerate(predicted):
#         if p in actual and p not in predicted[:i]:
#             num_hits += 1.0
#             score += num_hits / (i+1.0)

#     if not actual:
#         return 0.0

#     return score

# def mapk(actual, predicted, k=10):
#     """
#     Computes the mean average precision at k.
#     This function computes the mean average prescision at k between two lists
#     of lists of items.
#     Parameters
#     ----------
#     actual : list
#              A list of lists of elements that are to be predicted 
#              (order doesn't matter in the lists)
#     predicted : list
#                 A list of lists of predicted elements
#                 (order matters in the lists)
#     k : int, optional
#         The maximum number of predicted elements
#     Returns
#     -------
#     score : double
#             The mean average precision at k over the input lists
#     """
#     return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])



# enc = LabelEncoder()
# y = enc.fit_transform(y)
    
# actual = [[label] for label in y]
# map3_score = mapk(actual, top_3)

# print(map3_score)


# print(y)


# import numpy as np
# import pandas as pd

# from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
# from sklearn.model_selection import StratifiedKFold
# from sklearn.impute import SimpleImputer

# from lightgbm import LGBMClassifier

# # Chargement des donnÃ©es
# df_train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
# df_test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
# sub = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")

# FEATURES = [c for c in df_test.columns if c != "id"]
# CAT_COLS = ["Soil Type", "Crop Type"]
# NUM_COLS = [c for c in FEATURES if c not in CAT_COLS]
# LABEL = "Fertilizer Name"

# # Imputation des valeurs manquantes numÃ©riques
# num_imputer = SimpleImputer(strategy="median")
# df_train[NUM_COLS] = num_imputer.fit_transform(df_train[NUM_COLS])
# df_test[NUM_COLS] = num_imputer.transform(df_test[NUM_COLS])

# # Encodage des colonnes catÃ©gorielles
# enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
# df_train[CAT_COLS] = enc.fit_transform(df_train[CAT_COLS])
# df_test[CAT_COLS] = enc.transform(df_test[CAT_COLS])

# # Encodage de la cible
# label_enc = LabelEncoder()
# y = label_enc.fit_transform(df_train[LABEL])
# X = df_train[FEATURES]
# X_test = df_test[FEATURES]

# # Fonction MAP@3
# def apk(actual, predicted, k=3):
#     if len(predicted) > k:
#         predicted = predicted[:k]
#     return int(actual in predicted) / (predicted.index(actual)+1) if actual in predicted else 0

# def map3(actuals, preds, k=3):
#     return np.mean([apk(a, list(p), k) for a, p in zip(actuals, preds)])

# # Cross-validation Stratified K-Fold
# skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# oof_preds = np.zeros((X.shape[0], len(label_enc.classes_)))
# test_preds = np.zeros((X_test.shape[0], len(label_enc.classes_)))

# for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
#     print(f"\nğŸ”� Fold {fold+1}")

#     X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
#     y_train, y_val = y[train_idx], y[val_idx]

#     clf = LGBMClassifier(
#         random_state=42,
#         device='gpu',
#         n_estimators=3000,
#         learning_rate=0.03,
#         num_leaves=31,
#         colsample_bytree=0.8,
#         subsample=0.8,
#         subsample_freq=1,
#         reg_alpha=1,
#         reg_lambda=1,
#         class_weight='balanced',
#         importance_type='gain',
#         n_jobs=-1,
#     )

#     clf.fit(
#         X_train, y_train,
#         eval_set=[(X_val, y_val)],
#         eval_metric='multi_logloss'
#     )

#     oof_preds[val_idx] = clf.predict_proba(X_val)
#     test_preds += clf.predict_proba(X_test) / skf.n_splits

# # Score OOF
# top3_oof = np.argsort(oof_preds, axis=1)[:, -3:][:, ::-1]
# map3_score = map3(y, top3_oof)
# print(f"\nğŸ“Š OOF MAP@3 score (Stratified K-Fold CV): {map3_score:.5f}")

# # PrÃ©dictions finales
# top3_test = np.argsort(test_preds, axis=1)[:, -3:][:, ::-1]
# top3_test_labels = label_enc.inverse_transform(top3_test.ravel()).reshape(top3_test.shape)
# sub[LABEL] = [' '.join(row) for row in top3_test_labels]
# sub.to_csv("submission.csv", index=False)

# print("âœ… Submission file saved as submission.csv")



# import numpy as np
# import pandas as pd

# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import StratifiedKFold
# from sklearn.impute import SimpleImputer

# from lightgbm import LGBMClassifier
# import category_encoders as ce  # <-- ici TargetEncoder

# # Chargement des donnÃ©es
# df_train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
# df_test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
# sub = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")

# FEATURES = [c for c in df_test.columns if c != "id"]
# CAT_COLS = ["Soil Type", "Crop Type"]
# NUM_COLS = [c for c in FEATURES if c not in CAT_COLS]
# LABEL = "Fertilizer Name"

# # Imputation des valeurs manquantes numÃ©riques
# num_imputer = SimpleImputer(strategy="median")
# df_train[NUM_COLS] = num_imputer.fit_transform(df_train[NUM_COLS])
# df_test[NUM_COLS] = num_imputer.transform(df_test[NUM_COLS])

# # Encodage de la cible
# label_enc = LabelEncoder()
# y = label_enc.fit_transform(df_train[LABEL])

# X = df_train[NUM_COLS + CAT_COLS]
# X_test = df_test[NUM_COLS + CAT_COLS]

# # Fonction MAP@3
# def apk(actual, predicted, k=3):
#     if len(predicted) > k:
#         predicted = predicted[:k]
#     return int(actual in predicted) / (predicted.index(actual)+1) if actual in predicted else 0

# def map3(actuals, preds, k=3):
#     return np.mean([apk(a, list(p), k) for a, p in zip(actuals, preds)])

# # Cross-validation Stratified K-Fold avec Target Encoding
# skf = StratifiedKFold(n_splits=15, shuffle=True, random_state=42)
# oof_preds = np.zeros((X.shape[0], len(label_enc.classes_)))
# test_preds = np.zeros((X_test.shape[0], len(label_enc.classes_)))

# for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
#     print(f"\nğŸ”� Fold {fold+1}")

#     X_train, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
#     y_train, y_val = y[train_idx], y[val_idx]

#     # â�• Target Encoding des variables catÃ©gorielles
#     te = ce.TargetEncoder(cols=CAT_COLS, smoothing=0.3)
#     X_train[CAT_COLS] = te.fit_transform(X_train[CAT_COLS], y_train)
#     X_val[CAT_COLS] = te.transform(X_val[CAT_COLS])
#     X_test_enc = X_test.copy()
#     X_test_enc[CAT_COLS] = te.transform(X_test[CAT_COLS])

#     clf = LGBMClassifier(
#         random_state=42,
#         device='gpu',
#         n_estimators=3000,
#         learning_rate=0.05,
#         num_leaves=31,
#         colsample_bytree=0.8,
#         subsample=0.8,
#         subsample_freq=1,
#         reg_alpha=1,
#         reg_lambda=1,
#         class_weight='balanced',
#         importance_type='gain',
#         n_jobs=-1,
#     )

#     clf.fit(
#         X_train, y_train,
#         eval_set=[(X_val, y_val)],
#         eval_metric='multi_logloss'
#     )

#     oof_preds[val_idx] = clf.predict_proba(X_val)
#     test_preds += clf.predict_proba(X_test_enc) / skf.n_splits

# # Score OOF
# top3_oof = np.argsort(oof_preds, axis=1)[:, -3:][:, ::-1]
# map3_score = map3(y, top3_oof)
# print(f"\nğŸ“Š OOF MAP@3 score (Target Encoding): {map3_score:.5f}")

# # PrÃ©dictions finales
# top3_test = np.argsort(test_preds, axis=1)[:, -3:][:, ::-1]
# top3_test_labels = label_enc.inverse_transform(top3_test.ravel()).reshape(top3_test.shape)
# sub[LABEL] = [' '.join(row) for row in top3_test_labels]
# sub.to_csv("submission.csv", index=False)

# print("âœ… Submission file saved as submission.csv")



# import numpy as np
# import pandas as pd

# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import StratifiedKFold
# from sklearn.impute import SimpleImputer
# from sklearn.linear_model import LogisticRegression

# from lightgbm import LGBMClassifier
# from xgboost import XGBClassifier
# import category_encoders as ce

# # Chargement des donnÃ©es
# df_train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
# df_test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
# sub = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")

# FEATURES = [c for c in df_test.columns if c != "id"]
# CAT_COLS = ["Soil Type", "Crop Type"]
# NUM_COLS = [c for c in FEATURES if c not in CAT_COLS]
# LABEL = "Fertilizer Name"

# # Imputation des valeurs manquantes
# num_imputer = SimpleImputer(strategy="median")
# df_train[NUM_COLS] = num_imputer.fit_transform(df_train[NUM_COLS])
# df_test[NUM_COLS] = num_imputer.transform(df_test[NUM_COLS])

# # Encodage de la cible
# label_enc = LabelEncoder()
# y = label_enc.fit_transform(df_train[LABEL])

# X = df_train[NUM_COLS + CAT_COLS]
# X_test = df_test[NUM_COLS + CAT_COLS]

# # MAP@3
# def apk(actual, predicted, k=3):
#     if len(predicted) > k:
#         predicted = predicted[:k]
#     return int(actual in predicted) / (predicted.index(actual)+1) if actual in predicted else 0

# def map3(actuals, preds, k=3):
#     return np.mean([apk(a, list(p), k) for a, p in zip(actuals, preds)])

# # Cross-validation + Target Encoding
# skf = StratifiedKFold(n_splits=25, shuffle=True, random_state=42)
# oof_preds = np.zeros((X.shape[0], len(label_enc.classes_)))
# test_preds = np.zeros((X_test.shape[0], len(label_enc.classes_)))

# for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
#     print(f"\nğŸ”� Fold {fold+1}")

#     X_train, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
#     y_train, y_val = y[train_idx], y[val_idx]

#     # â�• Target Encoding
#     te = ce.TargetEncoder(cols=CAT_COLS, smoothing=0.3)
#     X_train[CAT_COLS] = te.fit_transform(X_train[CAT_COLS], y_train)
#     X_val[CAT_COLS] = te.transform(X_val[CAT_COLS])
#     X_test_enc = X_test.copy()
#     X_test_enc[CAT_COLS] = te.transform(X_test[CAT_COLS])

#     # ModÃ¨le 1 : LightGBM
#     lgbm = LGBMClassifier(
#         random_state=42,
#         device='gpu',
#         n_estimators=3000,
#         learning_rate=0.05,
#         num_leaves=31,
#         colsample_bytree=0.8,
#         subsample=0.8,
#         subsample_freq=1,
#         reg_alpha=1,
#         reg_lambda=1,
#         class_weight='balanced',
#         importance_type='gain',
#         n_jobs=-1,
#     )
#     lgbm.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='multi_logloss')
#     pred_val_lgb = lgbm.predict_proba(X_val)
#     pred_test_lgb = lgbm.predict_proba(X_test_enc)

#     # ModÃ¨le 2 : XGBoost
#     xgb = XGBClassifier(
#         n_estimators=3000,
#         learning_rate=0.05,
#         max_depth=6,
#         colsample_bytree=0.8,
#         subsample=0.8,
#         reg_alpha=1,
#         reg_lambda=1,
#         use_label_encoder=False,
#         eval_metric='mlogloss',
#         tree_method='gpu_hist',
#         random_state=42,
#         n_jobs=-1,
#     )
#     xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)])
#     pred_val_xgb = xgb.predict_proba(X_val)
#     pred_test_xgb = xgb.predict_proba(X_test_enc)

#     # ModÃ¨le 3 : Logistic Regression
#     logreg = LogisticRegression(
#         C=1.0,
#         max_iter=2000,
#         multi_class='multinomial',
#         solver='saga',
#         n_jobs=-1
#     )
#     logreg.fit(X_train, y_train)
#     pred_val_logreg = logreg.predict_proba(X_val)
#     pred_test_logreg = logreg.predict_proba(X_test_enc)

#     pred_val_lgb_csv = pd.DataFrame(data=pred_val_lgb)
#     pred_val_xgb_csv = pd.DataFrame(data=pred_val_xgb)
#     pred_val_logreg_csv = pd.DataFrame(data=pred_val_logreg)

#     pred_val_lgb_csv.to_csv("pred_val_lgb_csv.csv",index=False)
#     pred_val_xgb_csv.to_csv("pred_val_xgb_csv.csv",index=False)
#     pred_val_logreg_csv.to_csv("pred_val_logreg_csv.csv",index=False)
    
    
#     # ğŸ�¯ Blending (poids ajustables)
#     oof_preds[val_idx] = (
#         0.2787 * pred_val_lgb +
#         0.0160 * pred_val_xgb +
#         0.7053 * pred_val_logreg
#     )
#     test_preds += (
#         0.2787 * pred_test_lgb +
#         0.0160 * pred_test_xgb +
#         0.7053 * pred_test_logreg
#     ) / skf.n_splits

# # Score OOF
# top3_oof = np.argsort(oof_preds, axis=1)[:, -3:][:, ::-1]
# map3_score = map3(y, top3_oof)
# print(f"\nğŸ“Š OOF MAP@3 score (Blending): {map3_score:.5f}")

# # Submission
# top3_test = np.argsort(test_preds, axis=1)[:, -3:][:, ::-1]
# top3_test_labels = label_enc.inverse_transform(top3_test.ravel()).reshape(top3_test.shape)
# sub[LABEL] = [' '.join(row) for row in top3_test_labels]
# sub.to_csv("submission.csv", index=False)

# print("âœ… Submission file saved as submission.csv")



# import numpy as np
# import pandas as pd

# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import StratifiedKFold
# from sklearn.impute import SimpleImputer
# from sklearn.linear_model import LogisticRegression

# from lightgbm import LGBMClassifier
# from xgboost import XGBClassifier
# import category_encoders as ce  # <-- TargetEncoder

# # Chargement des donnÃ©es
# df_train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
# df_test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
# sub = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")

# FEATURES = [c for c in df_test.columns if c != "id"]
# CAT_COLS = ["Soil Type", "Crop Type"]
# NUM_COLS = [c for c in FEATURES if c not in CAT_COLS]
# LABEL = "Fertilizer Name"

# # Imputation des valeurs manquantes numÃ©riques
# num_imputer = SimpleImputer(strategy="median")
# df_train[NUM_COLS] = num_imputer.fit_transform(df_train[NUM_COLS])
# df_test[NUM_COLS] = num_imputer.transform(df_test[NUM_COLS])

# # Encodage de la cible
# label_enc = LabelEncoder()
# y = label_enc.fit_transform(df_train[LABEL])

# X = df_train[NUM_COLS + CAT_COLS]
# X_test = df_test[NUM_COLS + CAT_COLS]

# # Fonction MAP@3
# def apk(actual, predicted, k=3):
#     if len(predicted) > k:
#         predicted = predicted[:k]
#     return int(actual in predicted) / (predicted.index(actual)+1) if actual in predicted else 0

# def map3(actuals, preds, k=3):
#     return np.mean([apk(a, list(p), k) for a, p in zip(actuals, preds)])

# # PrÃ©-encodage global pour Ã©viter de rÃ©pÃ©ter
# te_global = ce.TargetEncoder(cols=CAT_COLS, smoothing=0.3)
# te_global.fit(X[CAT_COLS], y)
# X[CAT_COLS] = te_global.transform(X[CAT_COLS])
# X_test_enc = X_test.copy()
# X_test_enc[CAT_COLS] = te_global.transform(X_test[CAT_COLS])

# # Cross-validation Stratified K-Fold avec Blending + pondÃ©ration
# skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
# oof_preds = np.zeros((X.shape[0], len(label_enc.classes_)))
# test_preds = np.zeros((X_test.shape[0], len(label_enc.classes_)))

# # ğŸ”� GÃ©nÃ©ration alÃ©atoire de poids (Dirichlet)
# np.random.seed(42)
# best_score = 0
# best_weights = None

# for _ in range(1000):
#     w = np.random.dirichlet(np.ones(3))
#     temp_oof = np.zeros_like(oof_preds)

#     for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
#         X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
#         y_train, y_val = y[train_idx], y[val_idx]

#         clf_lgb = LGBMClassifier(
#             random_state=42,
#             device='gpu',
#             n_estimators=3000,
#             learning_rate=0.05,
#             num_leaves=31,
#             colsample_bytree=0.8,
#             subsample=0.8,
#             subsample_freq=1,
#             reg_alpha=1,
#             reg_lambda=1,
#             class_weight='balanced',
#             importance_type='gain',
#             n_jobs=-1,
#         )

#         clf_xgb = XGBClassifier(
#             tree_method='hist',
#             predictor='gpu_predictor',
#             eval_metric='mlogloss',
#             learning_rate=0.05,
#             n_estimators=3000,
#             max_depth=10,
#             subsample=0.8,
#             colsample_bytree=0.8,
#             reg_alpha=1,
#             reg_lambda=1,
#             use_label_encoder=False,
#             n_jobs=-1,
#             random_state=42
#         )

#         clf_lr = LogisticRegression(max_iter=2000, class_weight='balanced', solver='saga')

#         clf_lgb.fit(X_train, y_train)
#         clf_xgb.fit(X_train, y_train)
#         clf_lr.fit(X_train, y_train)

#         pred_val_lgb = clf_lgb.predict_proba(X_val)
#         pred_val_xgb = clf_xgb.predict_proba(X_val)
#         pred_val_lr = clf_lr.predict_proba(X_val)

#         temp_oof[val_idx] = (
#             w[0] * pred_val_lgb +
#             w[1] * pred_val_xgb +
#             w[2] * pred_val_lr
#         )

#     top3 = np.argsort(temp_oof, axis=1)[:, -3:][:, ::-1]
#     score = map3(y, top3)
#     if score > best_score:
#         best_score = score
#         best_weights = w

# print(f"\nâœ… Best Weights Found: LGB={best_weights[0]:.4f}, XGB={best_weights[1]:.4f}, LR={best_weights[2]:.4f} | MAP@3: {best_score:.5f}")

# # RÃ©exÃ©cution avec meilleurs poids
# oof_preds = np.zeros_like(oof_preds)
# test_preds = np.zeros_like(test_preds)

# for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
#     X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
#     y_train, y_val = y[train_idx], y[val_idx]

#     clf_lgb = LGBMClassifier(
#         random_state=42,
#         device='gpu',
#         n_estimators=3000,
#         learning_rate=0.05,
#         num_leaves=31,
#         colsample_bytree=0.8,
#         subsample=0.8,
#         subsample_freq=1,
#         reg_alpha=1,
#         reg_lambda=1,
#         class_weight='balanced',
#         importance_type='gain',
#         n_jobs=-1,
#     )

#     clf_xgb = XGBClassifier(
#         tree_method='hist',
#         predictor='gpu_predictor',
#         eval_metric='mlogloss',
#         learning_rate=0.05,
#         n_estimators=3000,
#         max_depth=10,
#         subsample=0.8,
#         colsample_bytree=0.8,
#         reg_alpha=1,
#         reg_lambda=1,
#         use_label_encoder=False,
#         n_jobs=-1,
#         random_state=42
#     )

#     clf_lr = LogisticRegression(max_iter=1000, class_weight='balanced', solver='saga')

#     clf_lgb.fit(X_train, y_train)
#     clf_xgb.fit(X_train, y_train)
#     clf_lr.fit(X_train, y_train)

#     pred_val_lgb = clf_lgb.predict_proba(X_val)
#     pred_val_xgb = clf_xgb.predict_proba(X_val)
#     pred_val_lr = clf_lr.predict_proba(X_val)

#     oof_preds[val_idx] = (
#         best_weights[0] * pred_val_lgb +
#         best_weights[1] * pred_val_xgb +
#         best_weights[2] * pred_val_lr
#     )

#     pred_test_lgb = clf_lgb.predict_proba(X_test_enc)
#     pred_test_xgb = clf_xgb.predict_proba(X_test_enc)
#     pred_test_lr = clf_lr.predict_proba(X_test_enc)

#     test_preds += (
#         best_weights[0] * pred_test_lgb +
#         best_weights[1] * pred_test_xgb +
#         best_weights[2] * pred_test_lr
#     ) / skf.n_splits

# pred_val_lgb_csv = pd.DataFrame(data=pred_val_lgb)
# pred_val_xgb_csv = pd.DataFrame(data=pred_val_xgb)
# pred_val_logreg_csv = pd.DataFrame(data=pred_val_logreg)

# pred_val_lgb_csv.to_csv("pred_val_lgb_csv.csv",index=False)
# pred_val_xgb_csv.to_csv("pred_val_xgb_csv.csv",index=False)
# pred_val_logreg_csv.to_csv("pred_val_logreg_csv.csv",index=False)

# # Score OOF
# top3_oof = np.argsort(oof_preds, axis=1)[:, -3:][:, ::-1]
# map3_score = map3(y, top3_oof)
# print(f"\nğŸ“Š Final OOF MAP@3 score (Optimized Blending): {map3_score:.5f}")

# # PrÃ©dictions finales
# top3_test = np.argsort(test_preds, axis=1)[:, -3:][:, ::-1]
# top3_test_labels = label_enc.inverse_transform(top3_test.ravel()).reshape(top3_test.shape)
# sub[LABEL] = [' '.join(row) for row in top3_test_labels]
# sub.to_csv("submission.csv", index=False)

# print("âœ… Submission file saved as submission.csv")



# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import StratifiedKFold
# from sklearn.impute import SimpleImputer
# from xgboost import XGBClassifier
# import category_encoders as ce


# # Chargement des donnÃ©es principales
# df_train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
# df_test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
# sub = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")

# # Chargement des donnÃ©es supplÃ©mentaires (assurez-vous que ce fichier existe bien)
# df_extra = pd.read_csv("/kaggle/input/hex-s5e6/FertilizerPrediction1.csv")
# df_extra2 = pd.read_csv("/kaggle/input/hex-s5e6/FertilizerPrediction2.csv")

# df_train.drop(columns=["id"], inplace=True)

# # Fusion des datasets
# df_train = pd.concat([df_train, df_extra, df_extra2], axis=0, ignore_index=True)

# # Colonnes
# FEATURES = [c for c in df_test.columns if c != "id"]
# CAT_COLS = ["Soil Type", "Crop Type"]
# NUM_COLS = [c for c in FEATURES if c not in CAT_COLS]
# LABEL = "Fertilizer Name"

# # Imputation des valeurs manquantes
# num_imputer = SimpleImputer(strategy="median")
# df_train[NUM_COLS] = num_imputer.fit_transform(df_train[NUM_COLS])
# df_test[NUM_COLS] = num_imputer.transform(df_test[NUM_COLS])

# # Encodage de la cible
# label_enc = LabelEncoder()
# y = label_enc.fit_transform(df_train[LABEL])
# X = df_train[NUM_COLS + CAT_COLS]
# X_test = df_test[NUM_COLS + CAT_COLS]

# # MAP@3
# def apk(actual, predicted, k=3):
#     predicted = predicted[:k]
#     return int(actual in predicted) / (predicted.index(actual)+1) if actual in predicted else 0

# def map3(actuals, preds, k=3):
#     return np.mean([apk(a, list(p), k) for a, p in zip(actuals, preds)])

# # Cross-validation
# skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
# oof_preds = np.zeros((X.shape[0], len(label_enc.classes_)))
# test_preds = np.zeros((X_test.shape[0], len(label_enc.classes_)))

# for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
#     print(f"\nğŸ”� Fold {fold+1}")

#     X_train, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
#     y_train, y_val = y[train_idx], y[val_idx]

#     # Target Encoding
#     te = ce.TargetEncoder(cols=CAT_COLS, smoothing=0.3)
#     X_train[CAT_COLS] = te.fit_transform(X_train[CAT_COLS], y_train)
#     X_val[CAT_COLS] = te.transform(X_val[CAT_COLS])
#     X_test_enc = X_test.copy()
#     X_test_enc[CAT_COLS] = te.transform(X_test[CAT_COLS])

#     # ModÃ¨le XGBoost
#     xgb = XGBClassifier(
#         n_estimators=6000,
#         learning_rate=0.01,
#         max_depth=10,
#         colsample_bytree=0.8,
#         subsample=0.8,
#         reg_alpha=1,
#         reg_lambda=1,
#         use_label_encoder=False,
#         eval_metric='mlogloss',
#         tree_method='gpu_hist',
#         random_state=42,
#         n_jobs=-1,
#     )

#     xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)

#     pred_val = xgb.predict_proba(X_val)
#     pred_test = xgb.predict_proba(X_test_enc)

#     oof_preds[val_idx] = pred_val
#     test_preds += pred_test / skf.n_splits

# # Score
# top3_oof = np.argsort(oof_preds, axis=1)[:, -3:][:, ::-1]
# map3_score = map3(y, top3_oof)
# print(f"\nğŸ“Š OOF MAP@3 score (XGBoost): {map3_score:.5f}")

# # Soumission
# top3_test = np.argsort(test_preds, axis=1)[:, -3:][:, ::-1]
# top3_test_labels = label_enc.inverse_transform(top3_test.ravel()).reshape(top3_test.shape)
# sub[LABEL] = [' '.join(row) for row in top3_test_labels]
# sub.to_csv("submission.csv", index=False)

# print("âœ… Submission file saved as submission.csv")



# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import StratifiedKFold
# from sklearn.impute import SimpleImputer
# import category_encoders as ce

# # Installer pytorch_tabnet depuis le fichier fourni
# !pip -q install /kaggle/input/pytorchtabnet/pytorch_tabnet-4.1.0-py3-none-any.whl

# from pytorch_tabnet.tab_model import TabNetClassifier
# import torch

# # Chargement des donnÃ©es
# df_train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
# df_test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
# sub = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")

# df_extra = pd.read_csv("/kaggle/input/hex-s5e6/FertilizerPrediction1.csv")
# df_extra2 = pd.read_csv("/kaggle/input/hex-s5e6/FertilizerPrediction2.csv")

# df_train.drop(columns=["id"], inplace=True)
# df_train = pd.concat([df_train, df_extra, df_extra2], axis=0, ignore_index=True)

# # DÃ©finition des colonnes
# FEATURES = [c for c in df_test.columns if c != "id"]
# CAT_COLS = ["Soil Type", "Crop Type"]
# NUM_COLS = [c for c in FEATURES if c not in CAT_COLS]
# LABEL = "Fertilizer Name"

# # Imputation des valeurs manquantes
# num_imputer = SimpleImputer(strategy="median")
# df_train[NUM_COLS] = num_imputer.fit_transform(df_train[NUM_COLS])
# df_test[NUM_COLS] = num_imputer.transform(df_test[NUM_COLS])

# # Encodage de la cible
# label_enc = LabelEncoder()
# y = label_enc.fit_transform(df_train[LABEL])
# X = df_train[NUM_COLS + CAT_COLS]
# X_test = df_test[NUM_COLS + CAT_COLS]

# # MAP@3
# def apk(actual, predicted, k=3):
#     predicted = predicted[:k]
#     return int(actual in predicted) / (predicted.index(actual)+1) if actual in predicted else 0

# def map3(actuals, preds, k=3):
#     return np.mean([apk(a, list(p), k) for a, p in zip(actuals, preds)])

# # Cross-validation TabNet
# skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
# oof_preds = np.zeros((X.shape[0], len(label_enc.classes_)))
# test_preds = np.zeros((X_test.shape[0], len(label_enc.classes_)))

# for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
#     print(f"\nğŸ”� Fold {fold+1}")

#     X_train, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
#     y_train, y_val = y[train_idx], y[val_idx]

#     # Target Encoding
#     te = ce.TargetEncoder(cols=CAT_COLS, smoothing=0.3)
#     X_train[CAT_COLS] = te.fit_transform(X_train[CAT_COLS], y_train)
#     X_val[CAT_COLS] = te.transform(X_val[CAT_COLS])
#     X_test_enc = X_test.copy()
#     X_test_enc[CAT_COLS] = te.transform(X_test[CAT_COLS])

#     # TabNetClassifier
#     clf = TabNetClassifier(
#         n_d=32, n_a=32,
#         n_steps=5,
#         gamma=1.5,
#         lambda_sparse=1e-4,
#         optimizer_fn=torch.optim.Adam,
#         optimizer_params=dict(lr=2e-2),
#         mask_type='entmax',
#         scheduler_params={"step_size":10, "gamma":0.95},
#         scheduler_fn=torch.optim.lr_scheduler.StepLR,
#         verbose=0,
#         seed=42,
#         device_name='cuda' if torch.cuda.is_available() else 'cpu'
#     )

#     clf.fit(
#         X_train=X_train.values,
#         y_train=y_train,
#         eval_set=[(X_val.values, y_val)],
#         eval_name=["val"],
#         eval_metric=["accuracy"],
#         max_epochs=20,
#         patience=10,
#         batch_size=1024,
#         virtual_batch_size=128,
#         num_workers=0,
#         drop_last=False
#     )

#     pred_val = clf.predict_proba(X_val.values)
#     pred_test = clf.predict_proba(X_test_enc.values)

#     oof_preds[val_idx] = pred_val
#     test_preds += pred_test / skf.n_splits

# # Score MAP@3
# top3_oof = np.argsort(oof_preds, axis=1)[:, -3:][:, ::-1]
# map3_score = map3(y, top3_oof)
# print(f"\nğŸ“Š OOF MAP@3 score (TabNet): {map3_score:.5f}")

# # Soumission
# top3_test = np.argsort(test_preds, axis=1)[:, -3:][:, ::-1]
# top3_test_labels = label_enc.inverse_transform(top3_test.ravel()).reshape(top3_test.shape)
# sub[LABEL] = [' '.join(row) for row in top3_test_labels]
# sub.to_csv("submission_tabnet.csv", index=False)

# print("âœ… Submission file saved as submission_tabnet.csv")



from lightgbm import LGBMClassifier

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
import category_encoders as ce


# Chargement des donnÃ©es principales
df_train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
sub = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")

# Chargement des donnÃ©es supplÃ©mentaires (assurez-vous que ce fichier existe bien)
df_extra = pd.read_csv("/kaggle/input/hex-s5e6/FertilizerPrediction1.csv")
df_extra2 = pd.read_csv("/kaggle/input/hex-s5e6/FertilizerPrediction2.csv")

df_train.drop(columns=["id"], inplace=True)

# Fusion des datasets
df_train = pd.concat([df_train, df_extra, df_extra2], axis=0, ignore_index=True)

# Colonnes
FEATURES = [c for c in df_test.columns if c != "id"]
CAT_COLS = ["Soil Type", "Crop Type"]
NUM_COLS = [c for c in FEATURES if c not in CAT_COLS]
LABEL = "Fertilizer Name"

# Imputation des valeurs manquantes
num_imputer = SimpleImputer(strategy="median")
df_train[NUM_COLS] = num_imputer.fit_transform(df_train[NUM_COLS])
df_test[NUM_COLS] = num_imputer.transform(df_test[NUM_COLS])

# Encodage de la cible
label_enc = LabelEncoder()
y = label_enc.fit_transform(df_train[LABEL])
X = df_train[NUM_COLS + CAT_COLS]
X_test = df_test[NUM_COLS + CAT_COLS]

# Cross-validation
skf = StratifiedKFold(n_splits=30, shuffle=True, random_state=42)
oof_preds = np.zeros((X.shape[0], len(label_enc.classes_)))
test_preds = np.zeros((X_test.shape[0], len(label_enc.classes_)))

# MAP@3
def apk(actual, predicted, k=3):
    predicted = predicted[:k]
    return int(actual in predicted) / (predicted.index(actual)+1) if actual in predicted else 0

def map3(actuals, preds, k=3):
    return np.mean([apk(a, list(p), k) for a, p in zip(actuals, preds)])


for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\nğŸ”� Fold {fold+1}")
    print("ğŸ“¦ Using LightGBM model...")

    X_train, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
    y_train, y_val = y[train_idx], y[val_idx]

    # Target Encoding
    te = ce.TargetEncoder(cols=CAT_COLS, smoothing=0.3)
    X_train[CAT_COLS] = te.fit_transform(X_train[CAT_COLS], y_train)
    X_val[CAT_COLS] = te.transform(X_val[CAT_COLS])
    X_test_enc = X_test.copy()
    X_test_enc[CAT_COLS] = te.transform(X_test[CAT_COLS])

    # ModÃ¨le LightGBM avec paramÃ¨tres optimaux
    lgbm = LGBMClassifier(
        boosting_type='gbdt',
        learning_rate=0.04238464760043869,
        n_estimators=140,
        max_depth=28,
        num_leaves=998,
        min_child_samples=11,
        subsample=0.6308977348487621,
        colsample_bytree=0.546323401022693,
        reg_alpha=0.06926229885170429,
        reg_lambda=0.5241361697903782,
        random_state=42,
        n_jobs=-1
    )

    lgbm.fit(X_train, y_train, eval_set=[(X_val, y_val)])

    pred_val = lgbm.predict_proba(X_val)
    pred_test = lgbm.predict_proba(X_test_enc)

    oof_preds[val_idx] = pred_val
    test_preds += pred_test / skf.n_splits

# Score
top3_oof = np.argsort(oof_preds, axis=1)[:, -3:][:, ::-1]
map3_score = map3(y, top3_oof)
print(f"\nğŸ“Š OOF MAP@3 score (LightGBM): {map3_score:.5f}")

# Soumission
top3_test = np.argsort(test_preds, axis=1)[:, -3:][:, ::-1]
top3_test_labels = label_enc.inverse_transform(top3_test.ravel()).reshape(top3_test.shape)
sub[LABEL] = [' '.join(row) for row in top3_test_labels]
sub.to_csv("submission.csv", index=False)

print("âœ… Submission file saved as submission.csv")


