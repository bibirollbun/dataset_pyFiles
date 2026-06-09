!pip install -q /kaggle/input/pip-install-lifelines/autograd-1.7.0-py3-none-any.whl
!pip install -q /kaggle/input/pip-install-lifelines/autograd-gamma-0.5.0.tar.gz
!pip install -q /kaggle/input/pip-install-lifelines/interface_meta-1.3.0-py3-none-any.whl
!pip install -q /kaggle/input/pip-install-lifelines/formulaic-1.0.2-py3-none-any.whl
!pip install -q /kaggle/input/pip-install-lifelines/lifelines-0.30.0-py3-none-any.whl
!pip install -q /kaggle/input/download-lightning-and-pytorch-tabular/pytorch_lightning-2.4.0-py3-none-any.whl
!pip install -q /kaggle/input/download-lightning-and-pytorch-tabular/scikit_learn-1.6.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
!pip install -q /kaggle/input/download-lightning-and-pytorch-tabular/torchmetrics-1.5.2-py3-none-any.whl
!pip install -q /kaggle/input/download-lightning-and-pytorch-tabular/pytorch_tabnet-4.1.0-py3-none-any.whl
!pip install -q /kaggle/input/download-lightning-and-pytorch-tabular/einops-0.7.0-py3-none-any.whl
!pip install -q /kaggle/input/download-lightning-and-pytorch-tabular/pytorch_tabular-1.1.1-py2.py3-none-any.whl
!pip install -q /kaggle/input/cibmtr-pip-install-pycox


import warnings
warnings.filterwarnings('ignore')


# Data manipulation
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score, RandomizedSearchCV, cross_validate, KFold
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import LabelEncoder

# Models
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier, GradientBoostingClassifier, StackingClassifier, BaggingClassifier, VotingClassifier
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.neural_network import MLPClassifier

# Model evaluation
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score

# Pipeline
from sklearn.pipeline import Pipeline

# Explainable AI
import shap


# Load datatrain
data = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/train.csv')
data.head()


# Load dataset
test_data = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/test.csv')
test_data.head()


data.info()


test_data.info()


print(data.shape)
print(list(data.columns))
print(list(data.dtypes))


print(test_data.shape)
print(list(test_data.columns))
print(list(test_data.dtypes))


plt.hist(data.loc[data.efs==1,"efs_time"],bins=100,label="efs=1, Yes Event")
plt.hist(data.loc[data.efs==0,"efs_time"],bins=100,label="efs=0, Maybe Event")
plt.xlabel("Time of Observation, efs_time")
plt.ylabel("Density")
plt.title("Times of Observation. Either time to event, or time observed without event.")
plt.legend()
plt.show()


from lifelines import KaplanMeierFitter
def transform_survival_probability(df, time_col='efs_time', event_col='efs'):
    kmf = KaplanMeierFitter()
    kmf.fit(df[time_col], df[event_col])
    y = kmf.survival_function_at_times(df[time_col]).values
    return y
    
data["y"] = transform_survival_probability(data, time_col='efs_time', event_col='efs')

plt.hist(data.loc[data.efs==1,"y"],bins=100,label="efs=1, Yes Event")
plt.hist(data.loc[data.efs==0,"y"],bins=100,label="efs=0, Maybe Event")
plt.xlabel("Transformed Target y")
plt.ylabel("Density")
plt.title("KaplanMeier Transformed Target y using both efs and efs_time.")
plt.legend()
plt.show()


RMV = ["ID","efs","efs_time", 'y']
FEATURES = [c for c in data.columns if not c in RMV]
print(f"There are {len(FEATURES)} FEATURES: {FEATURES}")


LOW_IMPACT_FEATURES = ["sex_match", "gvhd_proph", "cmv_status", 
                       "tce_imm_match", "pulm_severe"]

FEATURES = [feat for feat in FEATURES if feat not in LOW_IMPACT_FEATURES]



CATS = []
for c in FEATURES:
    if data[c].dtype=="object":
        CATS.append(c)


# Drop Duplicate Rows
data = data.drop_duplicates()
data.shape


print(data.dtypes.value_counts())


print(test_data.dtypes.value_counts())


# how many total missing values do we have?
missing_values = data.isnull().sum()
total_cells = np.product(data.shape)
total_missing = missing_values.sum()
test_data
# percent of data that is missing
percent_missing = (total_missing/total_cells) * 100
print(percent_missing)

sns.heatmap(data.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# how many total missing values do we have?
missing_values = test_data.isnull().sum()
total_cells = np.product(test_data.shape)
total_missing = missing_values.sum()

# percent of test_data that is missing
percent_missing = (total_missing/total_cells) * 100
print(percent_missing)

sns.heatmap(test_data.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# Separate integer columns
int_cols = data.select_dtypes(include=['int']).columns.tolist()

# Separate float columns
float_cols = data.select_dtypes(include=['float']).columns.tolist()

# Separate categorical columns
cat_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()

print("Integer columns:", int_cols)
print("Float columns:", float_cols)
print("Categorical columns:", cat_cols)


# Plot density plot and histogram for integer columns
for col in int_cols:
    print("Column:", col)
    print("Summary Statistics:")
    print(data[col].describe())
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    sns.histplot(data[col], kde=True, color='blue')
    plt.title(f'Density Plot of {col}')
    plt.subplot(1, 2, 2)
    sns.boxplot(y=data[col], color='green')
    plt.title(f'Boxplot of {col}')
    plt.show()


for col in float_cols:
    print("Column:", col)
    print("Summary Statistics:")
    print(data[col].describe())
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    sns.histplot(data[col], kde=True, color='blue')
    plt.title(f'Density Plot of {col}')
    plt.subplot(1, 2, 2)
    sns.boxplot(y=data[col], color='green')
    plt.title(f'Boxplot of {col}')
    plt.show()


# Univariate analysis for categorical columns
for col in cat_cols:
    print("Column:", col)
    print("Value Counts:")
    print(data[col].value_counts())
    print("Bar plot:")
    data[col].value_counts().plot(kind='bar')
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.show()



def impute_missing_data(data):
    # -------- Categorical Columns --------
    # 1. Impute with Mode
    mod_col_nam = ["graft_type", "prod_type", "race_group", "tbi_status", "prim_disease_hct"]
    for col in mod_col_nam:
        if col in data.columns:
            mode_val = data[col].mode().iloc[0] if not data[col].mode().empty else "Unknown"
            data[col] = data[col].fillna(mode_val)

    # 2. Impute with "Unknown"
    unknown_col_name = [
        "cmv_status", "conditioning_intensity", "ethnicity", "in_vivo_tcd", "dri_score",
        "gvhd_proph", "sex_match", "donor_related", "melphalan_dose", "mrd_hct",
        "cyto_score", "tce_imm_match", "cyto_score_detail", "tce_match", "tce_div_match"
    ]
    for col in unknown_col_name:
        if col in data.columns:
            data[col] = data[col].fillna("Unknown")

    # 3. Impute with "Not done"
    not_done_col_name = [
        "psych_disturb", "diabetes", "arrhythmia", "vent_hist", "renal_issue",
        "pulm_severe", "rituximab", "obesity", "hepatic_severe", "prior_tumor",
        "peptic_ulcer", "rheum_issue", "hepatic_mild", "cardiac", "pulm_moderate"
    ]
    for col in not_done_col_name:
        if col in data.columns:
            data[col] = data[col].fillna("Not done")

    # -------- Numerical Columns --------
    # 5. Impute with Median
    med_col_nam = [
        'hla_match_c_high', 'hla_high_res_8', 'hla_low_res_6', 'hla_high_res_6',
        'hla_high_res_10', 'hla_match_dqb1_high', 'hla_nmdp_6', 'hla_match_c_low',
        'hla_match_drb1_low', 'hla_match_dqb1_low', 'hla_match_a_high', 'donor_age',
        'hla_match_b_low', 'hla_match_a_low', 'hla_match_b_high', 'comorbidity_score',
        'karnofsky_score', 'hla_low_res_8', 'hla_match_drb1_high', 'hla_low_res_10', 'age_at_hct'
    ]
    for col in med_col_nam:
        if col in data.columns:
            median_val = data[col].median()
            data[col] = data[col].fillna(median_val)

    # 6. Impute Year of HCT with Mode (Discrete Numerical)
    if "year_hct" in data.columns:
        mode_year = data["year_hct"].mode()[0]
        data["year_hct"] = data["year_hct"].fillna(mode_year)

    return data



def preprocess_data(data, is_train=True):
    """
    Preprocesses the dataset by handling missing values, encoding categorical features,
    and dropping unnecessary columns.
    
    Parameters:
    data (pd.DataFrame): Input dataset.
    is_train (bool): Flag indicating whether the dataset is training data.
    
    Returns:
    pd.DataFrame: Preprocessed dataset.
    pd.Series (only for test data): Stored ID column for later use.
    """

    # Drop low-impact features
    data = data.drop(columns=[col for col in LOW_IMPACT_FEATURES if col in data.columns], errors="ignore")
    
    # Handle missing values
    data = impute_missing_data(data)
    
    # Store ID for test data and drop unnecessary columns
    if is_train==False:
        data = data.drop(columns=['efs', 'efs_time'], errors='ignore')

    # Identifying numerical features
    numerical_features = data.select_dtypes(include=['int64', 'float64']).columns
    
    # Scaling numerical features
    scaler = StandardScaler()
    data[numerical_features] = scaler.fit_transform(data[numerical_features])
    
    # Encode categorical values
    label_encoders = {}
    for col in CATS:
        if col in data.columns:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
            label_encoders[col] = le
    
    return data


train = preprocess_data(data)  # For training data
test = preprocess_data(test_data, is_train=False)  # For test data


sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')


sns.heatmap(test.isnull(),yticklabels=False,cbar=False,cmap='viridis')


print(train.columns.tolist())


print(test.columns.tolist())


def features_engineering(df):
    # Change year_hct to relative year from 2000
    df['year_hct'] -= 2000
    df["age_dri_interaction"] = df["dri_score"] * np.log1p(df["age_at_hct"])

    scaler_fe = StandardScaler()
    df["karnofsky_score_scaled"] = scaler_fe.fit_transform(df[["karnofsky_score"]])

    return df


train = features_engineering(train)  # For training data
test = features_engineering(test)  # For test data


combined = pd.concat([train,test],axis=0,ignore_index=True)
train = combined.iloc[:len(train)].copy()
test = combined.iloc[len(train):].reset_index(drop=True).copy()


from sklearn.model_selection import KFold
from xgboost import XGBRegressor, XGBClassifier
# from xgbse import XGBSEDebiasedBCE
import xgboost as xgb
print("Using XGBoost version",xgb.__version__)


%%time
FOLDS = 10
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
    
oof_xgb = np.zeros(len(train))
pred_xgb = np.zeros(len(test))

for i, (train_index, test_index) in enumerate(kf.split(train)):

    print("#"*25)
    print(f"### Fold {i+1}")
    print("#"*25)
    
    x_train = train.loc[train_index,FEATURES].copy()
    y_train = train.loc[train_index,"y"]
    x_valid = train.loc[test_index,FEATURES].copy()
    y_valid = train.loc[test_index,"y"]
    x_test = test[FEATURES].copy()

    model_xgb = XGBRegressor(
        device="cuda", 
        enable_categorical=True
    )
    model_xgb.fit(
        x_train, y_train,
        eval_set=[(x_valid, y_valid)],  
        verbose=500 
    )

    # INFER OOF
    oof_xgb[test_index] = model_xgb.predict(x_valid)
    # INFER TEST
    pred_xgb += model_xgb.predict(x_test)

# COMPUTE AVERAGE TEST PREDS
pred_xgb /= FOLDS


from metric import score

y_true = train[["ID","efs","efs_time","race_group"]].copy()
y_pred = train[["ID"]].copy()
y_pred["prediction"] = oof_xgb
m = score(y_true.copy(), y_pred.copy(), "ID")
print(f"\nOverall CV for XGBoost KaplanMeier =",m)


# import shap

# explainer = shap.Explainer(model_xgb, train[FEATURES])
# shap_values = explainer(train[FEATURES])

# shap.summary_plot(shap_values, train[FEATURES])



from catboost import CatBoostRegressor, CatBoostClassifier
import catboost as cb
print("Using CatBoost version",cb.__version__)


%%time
FOLDS = 10
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
    
oof_cat = np.zeros(len(train))
pred_cat = np.zeros(len(test))

for i, (train_index, test_index) in enumerate(kf.split(train)):

    print("#"*25)
    print(f"### Fold {i+1}")
    print("#"*25)
    
    x_train = train.loc[train_index,FEATURES].copy()
    y_train = train.loc[train_index,"y"]
    x_valid = train.loc[test_index,FEATURES].copy()
    y_valid = train.loc[test_index,"y"]
    x_test = test[FEATURES].copy()

    model_cat = CatBoostRegressor(
        task_type="GPU",  
        learning_rate=0.1,    
        grow_policy='Lossguide',
        #early_stopping_rounds=25,
    )
    model_cat.fit(x_train,y_train,
              eval_set=(x_valid, y_valid),
              cat_features=CATS,
              verbose=250)

    # INFER OOF
    oof_cat[test_index] = model_cat.predict(x_valid)
    # INFER TEST
    pred_cat += model_cat.predict(x_test)

# COMPUTE AVERAGE TEST PREDS
pred_cat /= FOLDS



y_true = train[["ID","efs","efs_time","race_group"]].copy()
y_pred = train[["ID"]].copy()
y_pred["prediction"] = oof_cat
m = score(y_true.copy(), y_pred.copy(), "ID")
print(f"\nOverall CV for CatBoost KaplanMeier =",m)


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Convert data to tensors
def prepare_data(df, feature_cols, target_col):
    X = torch.tensor(df[feature_cols].values, dtype=torch.float32)
    y = torch.tensor(df[target_col].values, dtype=torch.float32).view(-1, 1)
    return X, y

# Define NN Model
class NNModel(nn.Module):
    def __init__(self, input_dim):
        super(NNModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)  # Regression output
        )
    
    def forward(self, x):
        return self.model(x)

# KFold Cross Validation
FOLDS = 10
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)

oof_nn = np.zeros(len(train))
pred_nn = np.zeros(len(test))

TARGET = "y"

X_train, y_train = prepare_data(train, FEATURES, TARGET)
X_test, _ = prepare_data(test, FEATURES, TARGET)

for i, (train_index, val_index) in enumerate(kf.split(train)):
    print(f"\n### Fold {i+1} ###")
    
    # Prepare datasets
    X_tr, y_tr = X_train[train_index], y_train[train_index]
    X_val, y_val = X_train[val_index], y_train[val_index]
    
    train_ds = TensorDataset(X_tr, y_tr)
    val_ds = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)
    
    # Model
    model = NNModel(input_dim=len(FEATURES)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    # Training
    for epoch in range(100):  # Adjust epochs if necessary
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
    
    # Validation
    model.eval()
    val_preds = []
    with torch.no_grad():
        for xb, _ in val_loader:
            xb = xb.to(device)
            preds = model(xb).cpu().numpy()
            val_preds.extend(preds)
    
    oof_nn[val_index] = np.array(val_preds).flatten()
    
    # Predict test data
    test_preds = model(X_test.to(device)).cpu().detach().numpy().flatten()
    pred_nn += test_preds

# Compute average test predictions
pred_nn /= FOLDS

# Evaluate
y_true = train[["ID", "efs", "efs_time", "race_group"]].copy()
y_pred = train[["ID"]].copy()
y_pred["prediction"] = oof_nn
m = score(y_true.copy(), y_pred.copy(), "ID")
print(f"\nOverall CV for Neural Network = {m}")



from scipy.stats import rankdata 

y_true = train[["ID","efs","efs_time","race_group"]].copy()
y_pred = train[["ID"]].copy()
y_pred["prediction"] = rankdata(oof_xgb) + rankdata(oof_cat) + rankdata(oof_nn)
m = score(y_true.copy(), y_pred.copy(), "ID")
print(f"\nOverall CV for Ensemble =",m)


sub = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/sample_submission.csv")
sub.prediction = rankdata(pred_xgb) + rankdata(pred_cat) + rankdata(pred_nn)
sub.to_csv("submission.csv",index=False)
print("Sub shape:",sub.shape)
sub.head()

