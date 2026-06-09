import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt


#mlp
#catboost
# node


df_train = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv', index_col='id')
df_test =  pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv', index_col='id')
df_train.head()


# The higher the annual income, the higher the credit score and also loan_paid_back
df_train.dtypes


df_train.describe()


# Are we predicting a rare event or a roughly even split?
plt.pie(
    df_train["loan_paid_back"].value_counts(),
    labels=["Paid Back", "Not Paid Back"],
    autopct="%.1f%%",
    startangle=90,
    counterclock=False
)
plt.title("Loan Payback Distribution")
plt.show()


cat_features = [
    "employment_status",
    "gender",
    "marital_status",
    "education_level",
    "loan_purpose"
]

sns.set(style="whitegrid")
for col in cat_features:
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True, gridspec_kw={'height_ratios':[2,1]})

    sns.countplot(
        data=df_train, x=col, hue="loan_paid_back",
        palette="dark", dodge=True, ax=axes[0]
    )
    axes[0].set_title(f"Counts by {col}", fontsize=13, weight="bold")
    axes[0].set_ylabel("Number of Loans")

    sns.histplot(
        data=df_train, x=col, hue="loan_paid_back",
        multiple="fill", palette="dark", shrink=0.8, ax=axes[1]
    )
    axes[1].set_ylabel("Proportion within Category")
    axes[1].set_xlabel("")
    axes[1].legend([],[], frameon=False)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.show()



num_features = [
    "annual_income",
    "loan_amount",
    "debt_to_income_ratio",
    "credit_score",
    "interest_rate"
]

sns.set(style="whitegrid")

for col in num_features:
    plt.figure(figsize=(7, 4))
    sns.kdeplot(
        data=df_train,
        x=col,
        hue="loan_paid_back",
        common_norm=False,
        fill=True,
        alpha=0.5,
        palette="dark"
    )
    plt.title(f"Distribution of {col} by Loan Repayment", fontsize=13, weight="bold")
    plt.xlabel(col)
    plt.ylabel("Density")
    plt.legend(title="Loan Paid Back", labels=["No", "Yes"])
    plt.tight_layout()
    plt.show()



from sklearn.preprocessing import StandardScaler

# Numeric features you want to compare
numeric_features = ["credit_score", "interest_rate", "debt_to_income_ratio"]

# Categorical features to analyze
categorical_features = [
    "employment_status",
    "education_level",
    "loan_purpose",
    "marital_status",
    "gender"
]

def plot_effect_size_by_category(df, cat_col, num_cols):
    """Computes and visualizes normalized mean differences
    (paid - not paid) for numeric features within a category."""
    
    # Drop missing values
    df_temp = df.dropna(subset=[cat_col, "loan_paid_back"] + num_cols).copy()
    
    # Normalize numeric features
    scaler = StandardScaler()
    df_temp[num_cols] = scaler.fit_transform(df_temp[num_cols])
    
    # Compute mean per category Ã— repayment group
    group_stats = (
        df_temp.groupby([cat_col, "loan_paid_back"])[num_cols]
        .mean()
        .unstack()  # MultiIndex columns: (feature, 0/1)
    )

    # Effect size = mean(paid) - mean(not paid)
    effect_size = group_stats.xs(1, level=1, axis=1) - group_stats.xs(0, level=1, axis=1)
    effect_size = effect_size.round(2)

    # Plot
    plt.figure(figsize=(8, max(3, len(effect_size) * 0.5)))
    sns.heatmap(
        effect_size,
        annot=True,
        fmt=".2f",
        cmap="vlag",
        center=0,
        linewidths=0.5,
        cbar_kws={"label": "Effect size (std. devs, paid âˆ’ not paid)"}
    )
    plt.title(f"Normalized Effect Size by {cat_col} and Feature", fontsize=13, weight="bold")
    plt.xlabel("Numeric Feature")
    plt.ylabel(cat_col)
    plt.tight_layout()
    plt.show()

    return effect_size


all_effects = {}
for cat in categorical_features:
    print(f"\nğŸ“Š Analyzing {cat} ...")
    all_effects[cat] = plot_effect_size_by_category(df_train, cat, numeric_features)



from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.compose import ColumnTransformer 
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score 
from catboost import CatBoostClassifier


num_features = [
    "annual_income",
    "loan_amount",
    "debt_to_income_ratio",
    "credit_score",
    "interest_rate"
]

cat_features = [
    "gender",
    "marital_status",
    "education_level",
    "employment_status",
    "loan_purpose",
    "grade_subgrade"
]

target = "loan_paid_back"


X = df_train[num_features + cat_features]
y = df_train[target]

# Build preprocessing steps

# Numeric pipeline
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical pipeline
cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Combine
preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_features),
    ("cat", cat_pipeline, cat_features)
])

# ---------------------------------
# Full model pipeline (very basic CatBoost baseline, combined with preprocessing although CatBoost would handle raw data)
# ---------------------------------
model = CatBoostClassifier(
    iterations=1500,
    learning_rate=0.03,
    depth=6,
    loss_function="Logloss",
    thread_count =-1,
    auto_class_weights ="Balanced", 
    eval_metric="AUC",
    verbose=False,
    random_seed=42
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = cross_val_score(
    pipeline, X, y,
    cv=cv,
    scoring="roc_auc",
    n_jobs=-1
)

print(f"Mean CV ROC-AUC: {cv_scores.mean():.4f} Â± {cv_scores.std():.4f}")



print(f"Cross - Validation Catboost AUC Score metrics: {cv_scores}")
pipeline.fit(X,y)


cat_clf = pipeline.named_steps["model"]
cat_clf.get_feature_importance(prettified=True)


# Get all transformed feature names from the pipeline
feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()

# Combine with importances from CatBoost
importances = pd.DataFrame({
    "Feature": feature_names,
    "Importance": pipeline.named_steps["model"].get_feature_importance()
})

# Sort descending
importances = importances.sort_values(by="Importance", ascending=False).reset_index(drop=True)
importances.head(15)



from catboost import Pool
import shap

X_preprocessed = pipeline.named_steps["preprocessor"].transform(X)
feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out().tolist()
train_pool = Pool(
    data=X_preprocessed,
    label=y,
    feature_names=feature_names  
)
explainer = shap.TreeExplainer(cat_clf)
shap_values = explainer.shap_values(train_pool)

shap.summary_plot(shap_values, features=X_preprocessed, feature_names=feature_names)


X_dense = pd.DataFrame(
    pipeline.named_steps["preprocessor"].transform(X).toarray(),
    columns=pipeline.named_steps["preprocessor"].get_feature_names_out()
)

X_dense = X_dense.astype(float)

top_features = [
    "num__credit_score",
    "num__debt_to_income_ratio",
    "num__interest_rate",
    "cat__employment_status_Unemployed"
]

for feat in top_features:
    shap.dependence_plot(
        feat,
        shap_values,
        X_dense.values,
        feature_names=X_dense.columns,
        interaction_index=None,
        alpha=0.6,
        dot_size=10,
        show=False
    )
    plt.title(f"SHAP Dependence Plot â€” {feat}", fontsize=12, weight="bold")
    plt.xlabel(f"{feat} (value)")
    plt.ylabel("SHAP value (impact on repayment)")
    plt.tight_layout()
    plt.show()



# make predictions on the test set
test_features = num_features + cat_features
X_test = df_test[test_features]

# Apply pipeline (preprocessing + model)
y_test_pred = pipeline.predict_proba(df_test)[:, 1] 

# Create submission DataFrame
preds_catboost = pd.DataFrame({
    "id": df_test.index,
    "loan_paid_back": y_test_pred
})

# Save to CSV
preds_catboost.to_csv("submission.csv", index=False)

print("âœ… submission.csv saved successfully!")
preds_catboost.head()


# Here data preprocessing and scaling is really necessary 
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.compose import ColumnTransformer 
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score 


num_features = [
    "annual_income",
    "loan_amount",
    "debt_to_income_ratio",
    "credit_score",
    "interest_rate"
]

cat_features = [
    "gender",
    "marital_status",
    "education_level",
    "employment_status",
    "loan_purpose",
    "grade_subgrade"
]

target = "loan_paid_back"


X_nn = df_train[num_features + cat_features]
y = df_train[target]


# Numeric pipeline
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Use num_pipeline from catboost for numerical and a new one for categorial 
X_num_scaled = num_pipeline.fit_transform(X_nn[num_features])
cat_imputer = SimpleImputer(strategy="most_frequent")
X_cat_imputed = pd.DataFrame(cat_imputer.fit_transform(X_nn[cat_features]), columns=cat_features)

# Apply LabelEncoder per column
label_encoders = {}
for col in cat_features:
    le = LabelEncoder()
    X_cat_imputed[col] = le.fit_transform(X_cat_imputed[col].astype(str))
    label_encoders[col] = le

X_cat_encoded = X_cat_imputed.values

# --- Combine numeric + categorical (for later use in NN) ---
X_prepared = [X_num_scaled, X_cat_encoded]
X_prepared


# Create a DataLoader class that would look after data points and provide mini-batches
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F

class TabularDataset(Dataset):
    def __init__(self, X_num, X_cat, y):
        self.X_num = torch.tensor(X_num, dtype=torch.float32)
        self.X_cat = torch.tensor(X_cat, dtype=torch.long)
        self.y = torch.tensor(y.values, dtype=torch.float32).view(-1,1)
    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X_num[idx], self.X_cat[idx], self.y[idx]

class TabularMLP(nn.Module):
    def __init__(self, num_features, cat_cardinalities, emb_dims, hidden_dims=[128,64], dropout=0.3):
        super(TabularMLP, self).__init__()
        self.emb_layers = nn.ModuleList([
            nn.Embedding(num_embeddings=card, embedding_dim=emb)
            for card, emb in zip(cat_cardinalities, emb_dims)
        ])
        total_emb_dim = sum(emb_dims)
        input_dims = num_features + total_emb_dim
        layers = []
        in_dim = input_dims
        for h in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.model = nn.Sequential(*layers)
        
    def forward(self, x_num, x_cat):
        cat_embeds = [emb(x_cat[:, i]) for i, emb in enumerate(self.emb_layers)]
        x_cat = torch.cat(cat_embeds, dim=1)
        x = torch.cat([x_num, x_cat], dim=1)
        
        return self.model(x)


from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score
)
train_dataset = TabularDataset(X_num_scaled, X_cat_encoded, y)
train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define embedding sizes
cat_cardinalities = [int(X_cat_encoded[:, i].max() + 1) for i in range(X_cat_encoded.shape[1])]
emb_dims = [min(50, round(card ** 0.25)) for card in cat_cardinalities]

# Instantiate the model
model = TabularMLP(
    num_features=X_num_scaled.shape[1],
    cat_cardinalities=cat_cardinalities,
    emb_dims=emb_dims,
    hidden_dims=[128, 64],
    dropout=0.2
).to(device)

criterion = nn.BCEWithLogitsLoss() 
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for x_num, x_cat, y in loader:
        x_num, x_cat, y = x_num.to(device), x_cat.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x_num, x_cat)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
    return total_loss / len(loader.dataset)

def evaluate(model, loader, device, plot_pr=False):
    """
    Evaluate the model on a DataLoader:
      - Computes AUC, Average Precision
      - Finds best F1 threshold
      - Reports precision, recall, F1 at best threshold
    """
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x_num, x_cat, y in loader:
            x_num, x_cat, y = x_num.to(device), x_cat.to(device), y.to(device)
            logits = model(x_num, x_cat)
            preds.append(torch.sigmoid(logits).cpu().numpy())
            targets.append(y.cpu().numpy())

    preds = np.vstack(preds)
    targets = np.vstack(targets)

    # --- Core metrics ---
    auc = roc_auc_score(targets, preds)
    ap = average_precision_score(targets, preds)

    # --- Threshold tuning (best F1) ---
    precision, recall, thresholds = precision_recall_curve(targets, preds)
    f1s = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_idx = np.argmax(f1s)
    best_thresh = thresholds[best_idx]
    best_f1 = f1s[best_idx]
    best_prec = precision[best_idx]
    best_rec = recall[best_idx]

    if plot_pr:
        plt.figure(figsize=(6,4))
        plt.plot(recall, precision, label=f"PR curve (AP={ap:.3f})")
        plt.scatter(best_rec, best_prec, color="red", label=f"Best F1={best_f1:.3f}")
        plt.xlabel("Recall (positive class retrieved)")
        plt.ylabel("Precision")
        plt.title("Precisionâ€“Recall Curve")
        plt.legend()
        plt.show()

    print(f"AUC: {auc:.4f} | AP: {ap:.4f}")
    print(f"Best threshold: {best_thresh:.3f}")
    print(f"Precision: {best_prec:.3f} | Recall: {best_rec:.3f} | F1: {best_f1:.3f}")

    return {
        "auc": auc,
        "ap": ap,
        "best_thresh": best_thresh,
        "precision": best_prec,
        "recall": best_rec,
        "f1": best_f1
    }


import optuna
from sklearn.model_selection import StratifiedKFold
import numpy as np

def objective(trial):
    # --- Hyperparameters to tune ---
    hidden1 = trial.suggest_int("hidden1", 64, 512, step=64)
    hidden2 = trial.suggest_int("hidden2", 32, 256, step=32)
    dropout = trial.suggest_float("dropout", 0.1, 0.4)
    lr = trial.suggest_loguniform("lr", 1e-4, 1e-2)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-3)

    # --- Cross-validation setup ---
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_scores = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X_num_scaled, y)):
        # Prepare fold data
        X_num_train, X_num_val = X_num_scaled[train_idx], X_num_scaled[val_idx]
        X_cat_train, X_cat_val = X_cat_encoded[train_idx], X_cat_encoded[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        train_ds = TabularDataset(X_num_train, X_cat_train, y_train)
        val_ds = TabularDataset(X_num_val, X_cat_val, y_val)
        train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=1024, shuffle=False)

        # Build model with current hyperparameters
        model = TabularMLP(
            num_features=X_num_scaled.shape[1],
            cat_cardinalities=cat_cardinalities,
            emb_dims=emb_dims,
            hidden_dims=[hidden1, hidden2],
            dropout=dropout
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.BCEWithLogitsLoss()

        # --- Simple training loop per fold (short to save time) ---
        best_auc, no_improve, patience = 0, 0, 5
        for epoch in range(30):
            loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            metrics = evaluate(model, val_loader, device)
            val_auc = metrics["auc"]

            # Early stopping
            if val_auc > best_auc:
                best_auc = val_auc
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

        auc_scores.append(best_auc)

    # Return mean AUC across folds
    mean_auc = np.mean(auc_scores)
    print(f"Trial {trial.number}: mean AUC={mean_auc:.4f}")
    return mean_auc


# --- Run the Optuna study ---
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)  # e.g. 20â€“30 trials for a first run

print("Best trial:")
print(study.best_trial.params)



print("Best trial:")
print(study.best_trial.params)


full_ds = TabularDataset(X_num_scaled, X_cat_encoded, y)
full_loader = DataLoader(full_ds, batch_size=512, shuffle=True)

# --- Use best Optuna parameters ---
best_params = {
    "hidden1": 256,
    "hidden2": 192,
    "dropout": 0.3410734862244389,
    "lr": 0.0013071318979336183,
    "weight_decay": 1.6026311608302762e-05
}

model_final = TabularMLP(
    num_features=X_num_scaled.shape[1],
    cat_cardinalities=cat_cardinalities,
    emb_dims=emb_dims,
    hidden_dims=[best_params["hidden1"], best_params["hidden2"]],
    dropout=best_params["dropout"]
).to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(
    model_final.parameters(),
    lr=best_params["lr"],
    weight_decay=best_params["weight_decay"]
)

# --- Train on all data ---
epochs = 80
for epoch in range(epochs):
    loss = train_one_epoch(model_final, full_loader, optimizer, criterion, device)
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs} | Loss: {loss:.4f}")



# --- Preprocess test data ---
X_test_num = num_pipeline.transform(df_test[num_features])

X_test_cat = df_test[cat_features].copy()
for col, le in label_encoders.items():
    X_test_cat[col] = X_test_cat[col].map(lambda x: le.transform([x])[0] if x in le.classes_ else 0)
X_test_cat = X_test_cat.values

# Clamp for safety
for i, card in enumerate(cat_cardinalities):
    X_test_cat[:, i] = np.clip(X_test_cat[:, i], 0, card - 1)

# --- Predict probabilities ---
X_num_t = torch.tensor(X_test_num, dtype=torch.float32).to(device)
X_cat_t = torch.tensor(X_test_cat, dtype=torch.long).to(device)

model_final.eval()
with torch.no_grad():
    preds_test = torch.sigmoid(model_final(X_num_t, X_cat_t)).cpu().numpy().ravel()



submission = pd.DataFrame({
    "id": df_test.index,
    "loan_paid_back": preds_test
})
submission.to_csv("submission_nn_optuna.csv", index=False)
print("âœ… Saved submission_nn_optuna.csv")



blended = 0.6 * preds_catboost['loan_paid_back'] + 0.4 * submission['loan_paid_back']  # weights can be tuned

submission = pd.DataFrame({
    "id": df_test.index,
    "loan_paid_back": blended
})
submission.to_csv("submission_blend_2.csv", index=False)




