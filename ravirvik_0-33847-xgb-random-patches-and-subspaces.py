import numpy as np
import pandas as pd


df = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
df.drop("id", axis = 1, inplace = True)


catcol = []
for x in df.columns:
    if (type(df[x][0]) == type(" ")):
        print(df[x].value_counts())
        print("="*70)
        catcol.append(x)
catcol = catcol[:-1]
catcol


df2 = df.copy()

for x in catcol:
    df2 = pd.get_dummies(df2, columns=[x], prefix=x.split()[0], drop_first=True)

df2.info()


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

numeric_cols = df2.select_dtypes(include='number').columns
df2[numeric_cols] = scaler.fit_transform(df2[numeric_cols])

df2


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df2["Fertilizer Name"] = le.fit_transform(df2["Fertilizer Name"])



df3 = df2


from sklearn.model_selection import train_test_split

X = df3.drop("Fertilizer Name", axis=1)
y = df3["Fertilizer Name"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



from xgboost import XGBClassifier

def mapk(y_true, y_pred, k=3):
    score = 0.0
    for true, preds in zip(y_true, y_pred):
        if true in preds:
            score += 1.0 / (list(preds).index(true) + 1)
    return score / len(y_true)

def train_xgb_random_patches_subspaces_ensemble(X_train, y_train, X_test, y_test, 
                                                n_estimators=20, sample_frac=0.3, feature_frac=0.7, random_state=42):
    np.random.seed(random_state)
    models = []
    feature_indices_list = []
    
    X_train_np = X_train.values if hasattr(X_train, "values") else X_train
    X_test_np = X_test.values if hasattr(X_test, "values") else X_test
    y_train_np = y_train.values if hasattr(y_train, "values") else y_train
    y_test_np = y_test.values if hasattr(y_test, "values") else y_test
    
    n_samples, n_features = X_train_np.shape
    
    # Store predictions for ensemble averaging
    train_preds_sum = np.zeros((n_samples, len(np.unique(y_train_np))))
    test_preds_sum = np.zeros((X_test_np.shape[0], len(np.unique(y_train_np))))
    
    for i in range(n_estimators):
        idx_samples = np.random.choice(n_samples, size=int(sample_frac * n_samples), replace=False)
        idx_features = np.random.choice(n_features, size=int(feature_frac * n_features), replace=False)
        
        X_sub = X_train_np[idx_samples[:, None], idx_features]
        y_sub = y_train_np[idx_samples]
        
        model = XGBClassifier(n_estimators = 500, max_depth=5, use_label_encoder=False, eval_metric='mlogloss', random_state=random_state+i)
        model.fit(X_sub, y_sub)
        
        models.append(model)
        feature_indices_list.append(idx_features)
        
        train_probs = model.predict_proba(X_train_np[:, idx_features])
        top3_train_preds = np.argsort(-train_probs, axis=1)[:, :3]
        
        test_probs = model.predict_proba(X_test_np[:, idx_features])
        top3_test_preds = np.argsort(-test_probs, axis=1)[:, :3]
        
        train_map3 = mapk(y_train_np, top3_train_preds)
        test_map3 = mapk(y_test_np, top3_test_preds)
        
        print(f"Model {i+1} trained on patch+subspace:")
        print(f"  Train MAP@3: {train_map3:.5f}")
        print(f"  Test MAP@3: {test_map3:.5f}\n")
        
        # Accumulate predictions for ensemble (sum of probabilities)
        train_preds_sum += train_probs
        test_preds_sum += test_probs
    
    # Ensemble prediction: average probabilities over all models
    train_preds_avg = train_preds_sum / n_estimators
    test_preds_avg = test_preds_sum / n_estimators
    
    top3_train_ensemble = np.argsort(-train_preds_avg, axis=1)[:, :3]
    top3_test_ensemble = np.argsort(-test_preds_avg, axis=1)[:, :3]
    
    train_map3_ensemble = mapk(y_train_np, top3_train_ensemble)
    test_map3_ensemble = mapk(y_test_np, top3_test_ensemble)
    
    print(f"Ensemble Train MAP@3: {train_map3_ensemble:.5f}")
    print(f"Ensemble Test MAP@3: {test_map3_ensemble:.5f}")
    
    return models, feature_indices_list



models, feat_idxs = train_xgb_random_patches_subspaces_ensemble(X_train, y_train, X_test, y_test)
"""

##################################################### 
        Output was:
#####################################################


Model 1 trained on patch+subspace:
  Train MAP@3: 0.36411
  Test MAP@3: 0.30961

Model 2 trained on patch+subspace:
  Train MAP@3: 0.35854
  Test MAP@3: 0.30292

Model 3 trained on patch+subspace:
  Train MAP@3: 0.34127
  Test MAP@3: 0.29714

Model 4 trained on patch+subspace:
  Train MAP@3: 0.37366
  Test MAP@3: 0.31089

Model 5 trained on patch+subspace:
  Train MAP@3: 0.37040
  Test MAP@3: 0.30879

Model 6 trained on patch+subspace:
  Train MAP@3: 0.36820
  Test MAP@3: 0.30763

Model 7 trained on patch+subspace:
  Train MAP@3: 0.37393
  Test MAP@3: 0.31062

Model 8 trained on patch+subspace:
  Train MAP@3: 0.37338
  Test MAP@3: 0.31071

Model 9 trained on patch+subspace:
  Train MAP@3: 0.35603
  Test MAP@3: 0.30211

Model 10 trained on patch+subspace:
  Train MAP@3: 0.36408
  Test MAP@3: 0.30827

Model 11 trained on patch+subspace:
  Train MAP@3: 0.36769
  Test MAP@3: 0.31042

Model 12 trained on patch+subspace:
  Train MAP@3: 0.36935
  Test MAP@3: 0.30690

Model 13 trained on patch+subspace:
  Train MAP@3: 0.35963
  Test MAP@3: 0.30455

Model 14 trained on patch+subspace:
  Train MAP@3: 0.38380
  Test MAP@3: 0.31543

Model 15 trained on patch+subspace:
  Train MAP@3: 0.36506
  Test MAP@3: 0.30989

Model 16 trained on patch+subspace:
  Train MAP@3: 0.36797
  Test MAP@3: 0.30669

Model 17 trained on patch+subspace:
  Train MAP@3: 0.36460
  Test MAP@3: 0.30848

Model 18 trained on patch+subspace:
  Train MAP@3: 0.34531
  Test MAP@3: 0.29954

Model 19 trained on patch+subspace:
  Train MAP@3: 0.33282
  Test MAP@3: 0.29422

Model 20 trained on patch+subspace:
  Train MAP@3: 0.36554
  Test MAP@3: 0.30968

Ensemble Train MAP@3: 0.45420
Ensemble Test MAP@3: 0.34101


######### ######################################
final submission result
0.33847
trained on 80% of full train data
"""


df_test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
for x in catcol:
    df_test = pd.get_dummies(df_test, columns=[x], prefix=x.split()[0], drop_first=True)
df_test = df_test.reindex(columns=X_train.columns, fill_value=0)
df_test[numeric_cols] = scaler.transform(df_test[numeric_cols])



import numpy as np
import pandas as pd

# --- Ensemble prediction function for top 3 predictions ---
def predict_ensemble_top3(models, feat_idxs, X):
    preds_proba = []

    for model, feat_idx in zip(models, feat_idxs):
        X_sub = X[:, feat_idx]
        proba = model.predict_proba(X_sub)
        preds_proba.append(proba)

    # Average predicted probabilities over all models
    avg_proba = np.mean(preds_proba, axis=0)

    # Get top 3 class indices per row
    top3_preds = np.argsort(avg_proba, axis=1)[:, -3:][:, ::-1]  # descending order
    return top3_preds

# --- Step 1: Predict top 3 encoded labels ---
top3_pred_encoded = predict_ensemble_top3(models, feat_idxs, df_test.values)

# --- Step 2: Decode to original fertilizer names ---
top3_pred_labels = np.array([le.inverse_transform(row) for row in top3_pred_encoded])

# --- Step 3: Concatenate top 3 predictions into single string per row ---
fertilizer_name_preds = [' '.join(row) for row in top3_pred_labels]

# --- Step 4: Create submission DataFrame ---
sample = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")
submission = pd.DataFrame({
    "id": sample["id"],
    "Fertilizer Name": fertilizer_name_preds
})

# --- Step 5: Save submission ---
submission.to_csv("/kaggle/working/submission.csv", index=False)
print("Submission file saved as submission.csv")

