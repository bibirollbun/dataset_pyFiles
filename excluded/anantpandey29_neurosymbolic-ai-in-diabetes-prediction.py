!pip3 install pytorch-tabnet


import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import scipy.stats as stats
from scipy.stats import rankdata
from sklearn.model_selection import RepeatedStratifiedKFold
import optuna
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, TensorDataset
from pytorch_tabnet.tab_model import TabNetClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Kaggle/GitHub Compatibility: Ensure Plotly figures are embedded in the notebook output
pio.renderers.default = "notebook" 

warnings.filterwarnings('ignore')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"âœ… System Initialized. Compute Engine: {DEVICE}")


train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")
sample_sub = pd.read_csv("/kaggle/input/playground-series-s5e12/sample_submission.csv")

print(f"ğŸ“‚ Clinical Database Loaded: {len(train):,} Training Samples, {len(test):,} Test Samples")


target_counts = train['diagnosed_diabetes'].value_counts().reset_index()
target_counts.columns = ['Status', 'Count']
target_counts['Label'] = target_counts['Status'].map({0: 'Non-Diabetic', 1: 'Diabetic'})

fig = px.bar(target_counts, 
             x='Label', 
             y='Count', 
             color='Label', 
             title="<b>Figure 1:</b> Distribution of Diagnosed Diabetes",
             template='plotly_white',
             color_discrete_sequence=['#3498db', '#e74c3c'])
fig.show()


fig = px.violin(train, 
                y='bmi', 
                x='diagnosed_diabetes', 
                color='diagnosed_diabetes', 
                box=True, 
                points=None,
                title="<b>Figure 2:</b> Clinical BMI Distribution vs Outcome",
                template='plotly_white',
                labels={'diagnosed_diabetes': 'Diabetes Status', 'bmi': 'Body Mass Index'},
                color_discrete_map={0: '#3498db', 1: '#e74c3c'})
fig.update_layout(xaxis=dict(tickmode='array', tickvals=[0, 1], ticktext=['Negative', 'Positive']))
fig.show()


num_cols = train.select_dtypes(include=[np.number]).columns
corr = train[num_cols].corr()

fig = px.imshow(corr, 
                text_auto=True, 
                aspect="auto", 
                color_continuous_scale='RdBu_r',
                title="<b>Figure 3:</b> Feature Interaction Correlation Matrix",
                template='plotly_white')
fig.show()


def advanced_engineering(df):
    df_p = df.copy()
    
    df_p['metabolic_intensity'] = (df_p['bmi'] * df_p['systolic_bp']) / 1000
    df_p['lipid_stress_ratio'] = df_p['cholesterol_total'] / (df_p['hdl_cholesterol'] + 1e-9)
    df_p['age_bmi_interaction'] = df_p['age'] * df_p['bmi']
    if 'glucose' in df_p.columns: df_p['glucose_bmi_load'] = (df_p['glucose'] * df_p['bmi']) / 100
    df_p['bp_strain_index'] = (df_p['systolic_bp'] * df_p['age']) / 1000
    
    if 'id' in df_p.columns: df_p.drop('id', axis=1, inplace=True)
    
    cat_cols = df_p.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df_p[col] = LabelEncoder().fit_transform(df_p[col].astype(str))
        
    return df_p

train_processed = advanced_engineering(train)
test_processed = advanced_engineering(test)

X_full = train_processed.drop('diagnosed_diabetes', axis=1)
y = train_processed['diagnosed_diabetes']

print("ğŸ”� Running Selective Pruning to eliminate noise...")
selector_model = LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
selector_model.fit(X_full, y)

importances = pd.Series(selector_model.feature_importances_, index=X_full.columns)
threshold = importances.max() * 0.01 
selected_features = importances[importances >= threshold].index.tolist()

X = X_full[selected_features]
test_selected = test_processed[selected_features]

print(f"âœ… Feature Engineering Complete. Original: {X_full.shape[1]} | Pruned: {X.shape[1]} features.")


class ModelFactory:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        self.results = {}

    def benchmark(self, models_dict):
        # Using TQDM for real-time progress tracking
        pbar = tqdm(models_dict.items(), desc="ğŸ�¥ Training Model Pantheon")
        for name, model in pbar:
            pbar.set_description(f"ğŸ”¬ Optimizing {name}")
            scores = cross_val_score(model, self.X, self.y, cv=self.skf, scoring='roc_auc')
            self.results[name] = {'auc': np.mean(scores), 'model': model}
        
factory = ModelFactory(X, y)
factory.benchmark({
    "LogReg": LogisticRegression(max_iter=2000, C=0.01, solver='lbfgs'),
    "RandomForest": RandomForestClassifier(n_estimators=150, max_depth=10, min_samples_leaf=20),
    "HistGradBoost": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=8),
    "LightGBM": LGBMClassifier(n_estimators=500, learning_rate=0.03, num_leaves=63, verbose=-1),
    "CatBoost": CatBoostClassifier(iterations=500, learning_rate=0.03, depth=6, verbose=0)
})
print("âœ… Factory Benchmarking Complete. Top performing models identified.")


perf_df = pd.DataFrame([{'Model': k, 'ROC-AUC': v['auc']} for k, v in factory.results.items()])
perf_df = perf_df.sort_values(by='ROC-AUC', ascending=False).head(5)

fig = px.bar(perf_df, 
             x='ROC-AUC', 
             y='Model', 
             orientation='h', 
             color='ROC-AUC', 
             title="<b>Figure 4:</b> Top 5 Performance Ranking (ROC-AUC)",
             template='plotly_white', 
             color_continuous_scale='Viridis',
             text_auto='.4f')
fig.update_layout(yaxis={'categoryorder':'total ascending'})
fig.show()


class ClinicalDNN(nn.Module):
    def __init__(self, input_dim):
        super(ClinicalDNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)

def run_training():
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)
    X_ten = torch.FloatTensor(X_sc)
    y_ten = torch.FloatTensor(y.values).view(-1, 1)
    
    loader = DataLoader(TensorDataset(X_ten, y_ten), batch_size=512, shuffle=True)
    model = ClinicalDNN(X.shape[1]).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=0.001)
    crit = nn.BCELoss()
    
    for epoch in tqdm(range(30), desc="Training PyTorch DNN"): 
        model.train()
        for bx, by in loader:
            opt.zero_grad()
            loss = crit(model(bx.to(DEVICE)), by.to(DEVICE))
            loss.backward(); opt.step()
            
    model.eval()
    with torch.no_grad():
        p = model(X_ten.to(DEVICE)).cpu().numpy()
        auc = roc_auc_score(y, p)
    return model, scaler, auc

torch_model, torch_scaler, torch_auc = run_training()
factory.results['PyTorch_DNN'] = {'auc': torch_auc, 'model': torch_model}
print(f"ğŸ§  PyTorch DNN Status: Active | ROC-AUC: {torch_auc:.6f}")


print("ğŸ�—ï¸� Initializing TabNet Architecture...")
tabnet_model = TabNetClassifier(
    n_a=32, n_d=32, n_steps=3,
    gamma=1.3, n_independent=2, n_shared=2,
    momentum=0.02, clip_value=2.,
    lambda_sparse=1e-3, optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=2e-2),
    scheduler_params=dict(mode="min", patience=5, min_lr=1e-5, factor=0.9),
    scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
    verbose=0
)

# Training TabNet with Early Stopping
tabnet_model.fit(
    X_train=X.values, y_train=y.values,
    eval_set=[(X.values, y.values)],
    eval_name=['train'],
    eval_metric=['auc'],
    max_epochs=50, patience=10,
    batch_size=1024, virtual_batch_size=128,
    num_workers=0, drop_last=False
)

factory.results['TabNet'] = {'auc': tabnet_model.best_cost, 'model': tabnet_model}
print(f"âœ… TabNet Status: Active | Best ROC-AUC: {tabnet_model.best_cost:.6f}")


# Identify the champion from the Model Factory
winning_name = sorted(factory.results.items(), key=lambda x: x[1]['auc'], reverse=True)[0][0]
print(f"ğŸ�† Current Best Model: {winning_name}")

if 'PyTorch' in winning_name or 'TabNet' in winning_name:
    print(f"â�© Finalist {winning_name} is a Deep Architecture. Using optimal weights from training...")
else:
    def objective(trial):
        if "CatBoost" in winning_name:
            p = {
                'depth': trial.suggest_int('depth', 4, 10),
                'learning_rate': trial.suggest_float('lr', 0.01, 0.1),
                'l2_leaf_reg': trial.suggest_float('l2', 1, 10),
                'iterations': 300
            }
            m = CatBoostClassifier(**p, verbose=0)
        elif "LightGBM" in winning_name:
            p = {
                'num_leaves': trial.suggest_int('nl', 31, 127),
                'learning_rate': trial.suggest_float('lr', 0.01, 0.1),
                'min_child_samples': trial.suggest_int('mcs', 20, 100),
                'feature_fraction': trial.suggest_float('ff', 0.5, 1.0),
                'n_estimators': 300
            }
            m = LGBMClassifier(**p, verbose=-1)
        else:
            m = HistGradientBoostingClassifier(
                learning_rate=trial.suggest_float('lr', 0.01, 0.2),
                max_iter=300,
                max_depth=trial.suggest_int('md', 5, 15)
            )
            
        rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)
        return cross_val_score(m, X, y, cv=rskf, scoring='roc_auc').mean()

    # Advanced 50-trial optimization with Multivariate TPE
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(multivariate=True))
    study.optimize(objective, n_trials=50)
    print(f"ğŸ”® Optimization Finished. Final Oracle Best Score: {study.best_value:.6f}")


def clinical_logic_refiner(probs, df):
    refined = probs.copy()
    
    # Rule A: Metabolic Syndrome Hard Constraint
    # If extreme BMI AND extreme BP are present, biologically, risk increases significantly
    mask = (df['bmi'] > 38) & (df['systolic_bp'] > 145)
    refined[mask] = np.maximum(refined[mask], 0.92)
    
    # Rule B: Health Shield for Young Adults
    # Spurious correlations in youth are suppressed if metabolic markers are stable
    youth_mask = (df['age'] < 22) & (df['metabolic_intensity'] < 1.0)
    refined[youth_mask] *= 0.80
    
    return np.clip(refined, 0, 1)

print("âœ¨ Neurosymbolic Logic Layer Initialized.")


from scipy.stats import rankdata

print("ğŸš€ Starting Consensus Strategic Inference...")
model_preds = []

# Ensembling the top 3 performers for maximum stability
top_performers = sorted(factory.results.items(), key=lambda x: x[1]['auc'], reverse=True)[:3]

for i, (name, data) in enumerate(top_performers):
    print(f"ğŸ“¡ Generating probabilities from rank {i+1}: {name}")
    model = data['model']
    
    if hasattr(model, 'predict_proba'):
        # Ensure model is fitted on full data before inference
        model.fit(X, y)
        probs = model.predict_proba(test_selected)[:, 1]
    else:
        # Handling Deep Learning models (PyTorch)
        model.eval()
        with torch.no_grad():
            full_test_ten = torch.FloatTensor(torch_scaler.transform(test_selected))
            probs = model(full_test_ten.to(DEVICE)).cpu().numpy().flatten()
            
    model_preds.append(probs)

# Perform Rank-Based Ensembling (Rank Blending)
ranks = [rankdata(p) for p in model_preds]
avg_rank = np.mean(ranks, axis=0)

# Normalize ranks back to [0, 1] range
final_predictions = (avg_rank - avg_rank.min()) / (avg_rank.max() - avg_rank.min())

submission = pd.DataFrame({
    'id': sample_sub['id'],
    'diagnosed_diabetes': final_predictions
})

submission.to_csv('/kaggle/working/submission.csv', index=False)
print(f"âœ¨ Rank Blending Complete. {len(model_preds)} models contributed to the oracle consensus.")
print("ğŸš€ Candidate has been saved.")

