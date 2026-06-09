import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, GridSearchCV,StratifiedKFold,cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import xgboost as xgb
from lightgbm import LGBMClassifier 
from sklearn.ensemble import GradientBoostingClassifier
import optuna


train_df = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")
sub_df = pd.read_csv("/kaggle/input/playground-series-s5e8/sample_submission.csv")


train_df.head()


from colorama import Fore, Style

# Print the shape of the dataframe (number of rows and columns)
print(Fore.CYAN + "train_df shape: " + Style.RESET_ALL)
print(f"{train_df.shape}\n")

# Print basic information about the dataframe (column names, data types, non-null values)
print(Fore.GREEN + "train_df info: " + Style.RESET_ALL)
print(f"{train_df.info()}\n") 

# Print the count of missing (NaN) values in each column
print(Fore.YELLOW + "train_df isnull sum: " + Style.RESET_ALL)
print(f"{train_df.isnull().sum()}\n")

# Print summary statistics for numerical columns (count, mean, std, min, max, etc.)
print(Fore.MAGENTA + "train_df describe: " + Style.RESET_ALL)
print(f"{train_df.describe()}\n")


# define the numerical and categorical columns
numerical = train_df.select_dtypes(include=['int64', 'float64']).columns
categorical = train_df.select_dtypes(include=['object']).columns

print(f" We have features: {len(numerical)} numerical features {numerical}")
print("\n")
print(f" We have features: {len(categorical)} categorical features {categorical}")


for feature in numerical:
    plt.figure(figsize=(12, 5))
    plt.subplot(1,2,1)

    sns.histplot(data = train_df, x = feature , kde= True, bins = 30, palette="inferno")
    plt.title(f"Histgrom of {feature}")
    plt.xlabel(feature)
    plt.ylabel("Frquency")

    plt.subplot(1,2,2)
    sns.boxplot(train_df[feature])
    plt.title(f"Boxplt of {feature}")
    plt.tight_layout()
    plt.show()


for col in ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact','month', 'poutcome']:
    counts = train_df[col].value_counts()
    plt.figure(figsize = (20,6))
    plt.subplot(1,2,1)
    sns.countplot(data = train_df, x = col, palette = "Set2")
    plt.title(f"Count of {col} values")
    plt.xticks(rotation = 90)
    plt.ylabel("Count")
    # plt.show()

    plt.subplot(1,2,2)
    plt.pie(counts,labels = counts.index,autopct = "%1.1f%%",startangle=90)
    plt.title(f"Percentage of {col}")
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


# Count of y 
train_df['y'].value_counts().plot(kind="pie",autopct="%1.1f%%")
plt.title("Percentage of y")
plt.show()


correlation_matrix = train_df[numerical].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix of Numerical Features")
plt.show()


colors = sns.color_palette('husl', len(numerical))

rows = -(-len(numerical) // 4)
plt.figure(figsize=(20, 5 * rows))

for i, (col, color) in enumerate(zip(numerical, colors), 1):
    plt.subplot(rows, 4, i)
    sns.kdeplot(data=train_df, x=col, fill=True, color=color)
    plt.title(f'KDE Plot of {col}', fontsize=14, color=color)
    plt.xlabel(col)
    plt.ylabel('Density')

plt.tight_layout()
plt.show()


# Binary encoding
for col in ['default', 'housing', 'loan']:
    train_df[col] = train_df[col].map({'no': 0, 'yes': 1})
    test_df[col] = test_df[col].map({'no': 0, 'yes': 1})

# One-hot encoding
train_df = pd.get_dummies(train_df, columns=['job', 'marital', 'education', 'contact', 'month', 'poutcome'])
test_df = pd.get_dummies(test_df, columns=['job', 'marital', 'education', 'contact', 'month', 'poutcome'])

# New features
train_df['balance_bin'] = pd.cut(train_df['balance'], bins=[-np.inf, 0, 1000, 5000, np.inf], labels=['negative', 'low', 'medium', 'high'])
train_df['duration_bin'] = pd.cut(train_df['duration'], bins=[-np.inf, 100, 300, np.inf], labels=['short', 'medium', 'long'])
train_df['contacted_before'] = (train_df['pdays'] != -1).astype(int)
train_df['total_contacts'] = train_df['campaign'] + train_df['previous']

test_df['balance_bin'] = pd.cut(test_df['balance'], bins=[-np.inf, 0, 1000, 5000, np.inf], labels=['negative', 'low', 'medium', 'high'])
test_df['duration_bin'] = pd.cut(test_df['duration'], bins=[-np.inf, 100, 300, np.inf], labels=['short', 'medium', 'long'])
test_df['contacted_before'] = (test_df['pdays'] != -1).astype(int)
test_df['total_contacts'] = test_df['campaign'] + test_df['previous']

# One-hot encoding
train_df = pd.get_dummies(train_df, columns=['balance_bin', 'duration_bin'])
test_df = pd.get_dummies(test_df, columns=['balance_bin', 'duration_bin'])

# Scale numerical features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
num_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
train_df[num_cols] = scaler.fit_transform(train_df[num_cols])
test_df[num_cols] = scaler.fit_transform(test_df[num_cols])



# Separate features (X) and target (y)
X = train_df.drop(columns=["y"])
y = train_df["y"]

# Split into training and testing sets (80/20 split)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# Define evaluation helper function
def evaluate_model(true, predicted):
    """
    Prints confusion matrix and standard classification metrics.
    Arguments:
        true      : Ground truth labels
        predicted : Predicted labels (0 or 1)
    """
    cm = confusion_matrix(true, predicted)
    print("Confusion Matrix:")
    print(cm)
    print('\n')
    print("ROC AUC Score:", roc_auc_score(true, predicted))
    print("Accuracy:", accuracy_score(true, predicted))
    print("Precision:", precision_score(true, predicted))
    print("Recall:", recall_score(true, predicted))
    print("F1-Score:", f1_score(true, predicted))

# models
models = {
    "XGB" : XGBClassifier,
    "CatBoost" : CatBoostClassifier,
    "LGBMClassifier" : LGBMClassifier,
    "GradientBoosting" : GradientBoostingClassifier
}

model_list = []
for i in range(len(list(models))):
    model = list(models.values())[i]()  # Instantiate the model
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    # y_test_pred = model.predict(X_test)

    print(list(models.keys())[i])
    print("Train Results:")
    evaluate_model(y_train, y_train_pred)
    print('\n')
    print("-------------------------------------------------")
    # print("Test Results:")
    # evaluate_model(y_test, y_test_pred)
    print('\n')

    model_list.append(list(models.keys())[i])



from sklearn.utils import resample
import numpy as np

# assume X, y are numpy arrays or pandas DataFrame/Series
X_minority = X[y == 1]
y_minority = y[y == 1]

X_majority = X[y == 0]
y_majority = y[y == 0]

X_minority_upsampled, y_minority_upsampled = resample(
    X_minority,
    y_minority,
    replace=True,           # sample with replacement
    n_samples=len(y_majority),  # match majority size
    random_state=42
)

X_resampled = np.vstack([X_majority, X_minority_upsampled])
y_resampled = np.hstack([y_majority, y_minority_upsampled])



def objective(trial):
    # Choose algorithm
    classifier_name = trial.suggest_categorical('classifier', ["CatBoostClassifier", "xgb.XGBClassifier"])

    if classifier_name == 'xgb.XGBClassifier':
        # Hyperparameters for XGB
        n_estimators = trial.suggest_int('n_estimators', 200, 1000, step=100)
        learning_rate = trial.suggest_float('learning_rate', 0.01, 0.1, log=True)
        max_depth = trial.suggest_int("max_depth", 3, 8)
        subsample = trial.suggest_float("subsample", 0.7, 1.0)
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.6, 1.0)
        reg_lambda = trial.suggest_float("reg_lambda", 1e-2, 10.0, log=True)
        reg_alpha = trial.suggest_float("reg_alpha", 1e-2, 10.0, log=True)

        model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_lambda=reg_lambda,
            reg_alpha=reg_alpha,
            objective='binary:logistic',
            eval_metric='auc',
            random_state=42,
            verbosity=0,
            tree_method='gpu_hist',
            predictor='gpu_predictor',
            use_label_encoder=False
        )

    else:  # CatBoostClassifier
        iterations = trial.suggest_int('iterations', 200, 1000, step=100)
        learning_rate = trial.suggest_float('learning_rate', 0.01, 0.1, log=True)
        depth = trial.suggest_int("depth", 4, 10)

        model = CatBoostClassifier(
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            random_seed=42,
            verbose=False,
            task_type="GPU",
            devices="0"
        )

    # Cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_resampled,y_resampled, cv=skf, scoring="roc_auc")

    return scores.mean()

# Run optimization
study = optuna.create_study(direction='maximize', sampler=optuna.samplers.RandomSampler())
study.optimize(objective, n_trials=50)

# Best result
print("<==============================================>")
print(f'Best trial ROC-AUC: {study.best_trial.value}')
print(f'Best hyperparameters: {study.best_trial.params}')

# Train final model with best params
if study.best_trial.params['classifier'] == 'xgb.XGBClassifier':
    best_model = xgb.XGBClassifier(**{k:v for k,v in study.best_trial.params.items() if k != "classifier"},
                                   objective='binary:logistic',
                                   eval_metric='auc',
                                   random_state=42,
                                   verbosity=1,
                                   tree_method='gpu_hist',
                                   predictor='gpu_predictor',
                                   use_label_encoder=False)
else:
    best_model = CatBoostClassifier(**{k:v for k,v in study.best_trial.params.items() if k != "classifier"},
                                    random_seed=42,
                                    verbose=100,
                                    task_type="GPU",
                                    devices="0")

best_model.fit(X_resampled,y_resampled)

# Predictions
y_proba = best_model.predict_proba(X_test)[:, 1]

# Final CV check
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(best_model, X, y, cv=skf, scoring="roc_auc")

print(f"\nâœ… Final CV ROC-AUC: {scores.mean():.5f} (+/- {scores.std():.5f})")



y_proba = best_model.predict_proba(test_df)[:, 1]
submission = pd.DataFrame({
    'id': test_df['id'],
    'y': y_proba
})
#   submission file
submission.to_csv("submission.csv", index=False)

 #  Check
print(submission.head())

