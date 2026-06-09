import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt  
import plotly.express as px
import itertools
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import math
import plotly.subplots as sp
import itertools
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.metrics import precision_recall_curve, f1_score, make_scorer
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix
import xgboost as xgb
import time
import plotly.figure_factory as ff  
import optuna
from sklearn.preprocessing import LabelEncoder
import warnings
from xgboost import XGBClassifier
warnings.filterwarnings("ignore", category=RuntimeWarning)
pio.renderers.default = 'iframe_connected'
warnings.filterwarnings("ignore", category=FutureWarning)


train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')


train.head()


train.info()


train.describe().T


train.isnull().sum()


print("Duplicated Rows:",train.duplicated().sum())
print("-"*30)
print("Number of Rows:",train.shape[0])
print("-"*30)
print("Number of Columns:",train.shape[1])
print("-"*30)


print("Numeric Col Names",train.select_dtypes(include=['number']).columns)
print("-"*30)
print("Categorical Col Names",train.select_dtypes(include=['object']).columns)


num_col =  ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside',
       'Friends_circle_size', 'Post_frequency']
cat_col = ['Stage_fear', 'Drained_after_socializing']
target_col = 'Personality'


stage_fear_counts = train['Stage_fear'].value_counts().reset_index()
stage_fear_counts.columns = ['Stage_fear', 'Count']
print(stage_fear_counts)


fig = px.pie(
    stage_fear_counts,
    names='Stage_fear',
    values='Count',
    color_discrete_sequence=px.colors.qualitative.Pastel 
)

fig.update_traces(textposition='inside', textinfo='percent+label')

fig.update_layout(
    title_text='Stage Fear Distribution',
    width=500,
    height=500
)

fig.show()


social_counts = train['Drained_after_socializing'].value_counts().reset_index()
social_counts.columns = ['Drained_after_socializing', 'Count']
print(social_counts)


fig = px.pie(
    social_counts,
    names='Drained_after_socializing',
    values='Count',
    color_discrete_sequence=px.colors.qualitative.Pastel 
)

fig.update_traces(textposition='inside', textinfo='percent+label')

fig.update_layout(
    title_text='Drained_after_socializing Distribution',
    width=500,
    height=500
)

fig.show()


Personality_counts = train['Personality'].value_counts().reset_index()
Personality_counts.columns = ['Personality', 'Count']
print(Personality_counts)


fig = px.pie(
    Personality_counts,
    names='Personality',
    values='Count',
    color_discrete_sequence=px.colors.qualitative.Pastel 
)

fig.update_traces(textposition='inside', textinfo='percent+label')

fig.update_layout(
    title_text='Personality Distribution',
    width=500,
    height=500
)

fig.show()


for col in num_col:
    stats = train[col].describe().round(2)  
    print(f"--- {col} ---")
    print(stats.T) 
    print() 


fig = sp.make_subplots(rows=3, cols=2, subplot_titles=num_col)

colors = px.colors.qualitative.Pastel

for i, col in enumerate(num_col):
    row = i // 2 + 1
    col_pos = i % 2 + 1
    fig.add_trace(
        go.Histogram(
            x=train[col],
            name=col,
            marker_color=colors[i % len(colors)]
        ),
        row=row,
        col=col_pos
    )

fig.update_layout(
    height=800,
    width=750,
    title_text='Distribution of Numerical Features',
    showlegend=False,
    template='simple_white'
)

fig.show()


fig = sp.make_subplots(rows=3, cols=2, subplot_titles=num_col)
colors = px.colors.qualitative.Pastel

for i, col in enumerate(num_col):
    row = i // 2 + 1
    col_pos = i % 2 + 1

    fig.add_trace(
        go.Box(
            y=train[num_col].dropna(),
            name=col,
            marker_color=colors[i % len(colors)],
            boxpoints='outliers'  
        ),
        row=row,
        col=col_pos
    )

fig.update_layout(
    height=800,
    width=750,
    title_text='Distribution of Numerical Features (Box Plots)',
    showlegend=False,
    template='simple_white'
)

fig.show()


sns.set_theme(style="whitegrid")

for col in num_col:
    plt.figure(figsize=(7, 6))
    
    sns.violinplot(
        x=train[target_col],
        y=train[col],
        palette='Set2',
        inner='quartile',
        linewidth=1.2
    )
    
    sns.stripplot(
        x=train[target_col],
        y=train[col],
        color='black',
        size=2,
        jitter=0.2,
        alpha=0.6
    )
    
    plt.title(f"{col} by {target_col}", pad=12)
    plt.xlabel(target_col)
    plt.ylabel(col)
    plt.tight_layout()
    plt.show()


fig = px.histogram(
    train,
    x='Stage_fear',
    color='Personality',
    barmode='group',
    color_discrete_sequence=px.colors.qualitative.Pastel,
    title='Stage Fear vs Personality',
    labels={'Stage_fear': 'Stage Fear', 'count': 'Count'},
    category_orders={'Stage_fear': ['Yes', 'No']},
    height=400,      
    width=600       
)

fig.update_layout(
    bargap=0.2,
    xaxis_title='Stage Fear',
    yaxis_title='Count'
)

fig.show()


corr = train[num_col].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Numeric Feature Correlations")
plt.show()


cat_cols = [col for col in train.select_dtypes(include=['object', 'category']).columns if col != "Personality"]
for col in cat_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col])
    test[col] = le.transform(test[col])


le = LabelEncoder()
train['Personality'] = le.fit_transform(train['Personality'])


X = train.drop(['Personality', 'id'], axis=1)
y = train['Personality']
X_test_final = test.drop(columns=["id"])


X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.1, random_state=42)


test_ids = test['id']
X_final_test = test.drop(['id'], axis=1)


def objective(trial):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1500),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.1, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-5, 1.0, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-5, 1.0, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'objective': 'binary:logistic',
        'use_label_encoder': False,
        'eval_metric': 'logloss',
        'random_state': 42,
        'device': 'cuda',  # change to 'cpu' if not using GPU
        'predictor': 'gpu_predictor'
    }

    model = xgb.XGBClassifier(**param)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    return accuracy


#study = optuna.create_study(direction="maximize")
#study.optimize(objective, n_trials=100, timeout=3600) 


#print("Best trial:")
#trial = study.best_trial


#print("  Accuracy: {}".format(trial.value))
#print("  Best hyperparameters: ")
#for key, value in trial.params.items():
#    print(f"    {key}: {value}")


best_params = {
    'n_estimators': 1013,
    'max_depth': 3,
    'learning_rate': 0.04473761810915283,
    'subsample': 0.7472021066686094,
    'colsample_bytree': 0.6526442450606929,
    'gamma': 4.987525774261538,
    'reg_lambda': 0.1016293050091594,
    'reg_alpha': 0.8381641826774137,
    'min_child_weight': 10,
    'objective': 'binary:logistic',
    'use_label_encoder': False,
    'eval_metric': 'logloss',
    'device': 'cuda',
    'predictor': 'gpu_predictor',
    'random_state': 42
}


skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

oof_preds   = np.zeros(len(X))
oof_proba   = np.zeros(len(X))
test_preds  = np.zeros(len(X_test_final))

fold_accuracies = []
fold_roc_aucs   = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    print(f"--- Fold {fold} ---")
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = xgb.XGBClassifier(**best_params)
    model.fit(X_train, y_train)

    val_preds = model.predict(X_val)
    val_proba = model.predict_proba(X_val)[:, 1]

    oof_preds[val_idx] = val_preds
    oof_proba[val_idx] = val_proba

    acc   = accuracy_score(y_val, val_preds)
    auc   = roc_auc_score(y_val, val_proba)
    fold_accuracies.append(acc)
    fold_roc_aucs.append(auc)

    print(f"Accuracy: {acc:.4f} | ROC AUC: {auc:.4f}")

    test_preds += model.predict_proba(X_test_final)[:, 1]

test_preds /= skf.n_splits
final_preds = (test_preds > 0.5).astype(int)

oof_acc = accuracy_score(y, oof_preds)
oof_auc = roc_auc_score(y, oof_proba)

mean_acc = np.mean(fold_accuracies)
std_acc  = np.std(fold_accuracies)
mean_auc = np.mean(fold_roc_aucs)
std_auc  = np.std(fold_roc_aucs)

print("\n=== CV Summary ===")
print(f"Fold Accuracies: {fold_accuracies}")
print(f" â†’ Mean Accuracy: {mean_acc:.4f} Â± {std_acc:.4f}")
print(f"Fold ROC AUCs:   {fold_roc_aucs}")
print(f" â†’ Mean ROC AUC: {mean_auc:.4f} Â± {std_auc:.4f}")

print("\n=== OOF (all folds) ===")
print(f"Overall OOF Accuracy: {oof_acc:.4f}")
print(f"Overall OOF ROC AUC:  {oof_auc:.4f}")


conf_matrix = confusion_matrix(y, oof_preds)
plt.figure(figsize=(5, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix (OOF)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()


submission["Personality"] = le.inverse_transform(final_preds)
submission.to_csv("submission.csv", index=False)
print("âœ… Submission file saved as 'submission.csv'")


submission.head()

