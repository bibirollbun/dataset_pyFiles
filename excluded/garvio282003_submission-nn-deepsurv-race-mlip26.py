!pip install /kaggle/input/pip-install-lifelines/autograd-1.7.0-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/autograd-gamma-0.5.0.tar.gz
!pip install /kaggle/input/pip-install-lifelines/interface_meta-1.3.0-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/formulaic-1.0.2-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/lifelines-0.30.0-py3-none-any.whl


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
train=pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/train.csv')
test=pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/test.csv')


hla_cols = [col for col in train.columns if col.startswith("hla")]
for col in hla_cols:
    unique_values = train[col].dropna().unique()
    print(f"{col}: {len(unique_values)} unique values → {sorted(unique_values)[:10]}")




train["year_hct"] -= 2000
test["year_hct"] -= 2000


numerical_cols = [
    'year_hct', 'donor_age', 'age_at_hct', 'comorbidity_score',
    'karnofsky_score'
]

categorical_cols = [
    'dri_score', 'psych_disturb', 'cyto_score', 'diabetes', 'tbi_status',
    'arrhythmia', 'graft_type', 'vent_hist', 'renal_issue', 'pulm_severe',
    'prim_disease_hct', 'cmv_status', 'tce_imm_match', 'rituximab',
    'prod_type', 'cyto_score_detail', 'conditioning_intensity', 'ethnicity',
    'obesity', 'mrd_hct', 'in_vivo_tcd', 'tce_match', 'hepatic_severe',
    'prior_tumor', 'peptic_ulcer', 'gvhd_proph', 'rheum_issue', 'sex_match',
    'race_group', 'hepatic_mild', 'tce_div_match', 'donor_related',
    'melphalan_dose', 'cardiac', 'pulm_moderate'
]

hla_cols = [
    'hla_match_c_high', 'hla_high_res_8', 'hla_low_res_6', 'hla_high_res_6',
    'hla_high_res_10', 'hla_match_dqb1_high', 'hla_nmdp_6', 'hla_match_c_low',
    'hla_match_drb1_low', 'hla_match_dqb1_low', 'hla_match_a_high',
    'hla_match_b_low', 'hla_match_a_low', 'hla_match_b_high',
    'hla_low_res_8', 'hla_match_drb1_high', 'hla_low_res_10'
]


targets = ['efs', 'efs_time']


categorical_cols = categorical_cols + hla_cols
for col in categorical_cols:
    train[col] = train[col].astype("category")
    test[col] = test[col].astype("category")


from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder

train_imputed = train.copy()
test_imputed = test.copy()
categorical_cols = train.select_dtypes(include=['category']).columns.tolist()

label_encoders = {}
for col in categorical_cols:
    if train_imputed[col].dtype.name == 'category':
        le = LabelEncoder()
        train_imputed[col] = le.fit_transform(train_imputed[col].astype(str))
        test_imputed[col] = le.transform(test_imputed[col].astype(str))
        label_encoders[col] = le

knn_imputer = KNNImputer(n_neighbors=5)
all_cols = numerical_cols + categorical_cols
train_imputed[all_cols] = knn_imputer.fit_transform(train_imputed[all_cols])
test_imputed[all_cols] = knn_imputer.transform(test_imputed[all_cols])



train['race_group'].unique()


# train['race_group_encoded'] = train['race_group'].astype('category').cat.codes
# correlation_efs = train['race_group_encoded'].corr(train['efs'])
# correlation_efs_time = train['race_group_encoded'].corr(train['efs_time'])

# print(f"Correlation between race_group and efs: {correlation_efs:.4f}")
# print(f"Correlation between race_group and efs_time: {correlation_efs_time:.4f}")

# ---------------------------
# Correlation between race_group and efs: 0.0393
# Correlation between race_group and efs_time: -0.0123


# import scipy.stats as stats
# import pandas as pd

# anova_efs = stats.f_oneway(*(train[train['race_group'] == group]['efs'] for group in train['race_group'].unique()))
# anova_efs_time = stats.f_oneway(*(train[train['race_group'] == group]['efs_time'] for group in train['race_group'].unique()))

# print(f"ANOVA p-value for race_group and efs: {anova_efs.pvalue:.4f}")
# print(f"ANOVA p-value for race_group and efs_time: {anova_efs_time.pvalue:.4f}")

# ---------------------------
# ANOVA p-value for race_group and efs: 0.0000
# ANOVA p-value for race_group and efs_time: 0.0000



# from collections import Counter

# race_group_counts = Counter(train['race_group'])

# for race, count in race_group_counts.items():
#     print(f"Race Group {race}: {count} data points")

# ---------------------------
# Race Group More than one race: 4845 data points
# Race Group Asian: 4832 data points
# Race Group White: 4831 data points
# Race Group American Indian or Alaska Native: 4790 data points
# Race Group Native Hawaiian or other Pacific Islander: 4707 data points
# Race Group Black or African-American: 4795 data points



import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from lifelines.utils import concordance_index

race_groups = train_imputed['race_group'].unique()
class DeepSurv(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, dropout_rate):
        super(DeepSurv, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1) 
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2) 
        self.fc3 = nn.Linear(hidden_dim2, 1) 
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)  

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x.squeeze()  

def cox_loss(hazards, efs, efs_time):
    """Computes Cox proportional hazards loss."""
    risk_order = torch.argsort(efs_time, descending=True)
    hazards = hazards[risk_order]
    efs = efs[risk_order]

    exp_hazards = torch.exp(hazards)
    cum_sum = torch.cumsum(exp_hazards, dim=0)
    log_risk = torch.log(cum_sum)
    loss = -torch.sum((hazards - log_risk) * efs)

    return loss / torch.sum(efs)

def train_race_group_model(race_group, train_imputed, numerical_cols, categorical_cols, targets, best_params):
    race_group_data = train_imputed[train_imputed['race_group'] == race_group]

    X = race_group_data.drop(columns=targets).values
    efs = race_group_data['efs'].values
    efs_time = race_group_data['efs_time'].values

    X_train, X_val, efs_train, efs_val, efs_time_train, efs_time_val = train_test_split(
        X, efs, efs_time, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    efs_train_tensor = torch.tensor(efs_train, dtype=torch.float32)
    efs_val_tensor = torch.tensor(efs_val, dtype=torch.float32)
    efs_time_train_tensor = torch.tensor(efs_time_train, dtype=torch.float32)
    efs_time_val_tensor = torch.tensor(efs_time_val, dtype=torch.float32)

    input_dim = X_train.shape[1]
    model = DeepSurv(input_dim, best_params['hidden_dim1'], best_params['hidden_dim2'], best_params['dropout_rate'])
    optimizer = optim.Adam(model.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'])

    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        hazards_train = model(X_train_tensor)
        loss = cox_loss(hazards_train, efs_train_tensor, efs_time_train_tensor)

        # Backward pass
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            hazards_val = model(X_val_tensor)
            val_loss = cox_loss(hazards_val, efs_val_tensor, efs_time_val_tensor)
            c_index = concordance_index(efs_time_val, -hazards_val.numpy(), efs_val)

        if epoch % 10 == 0:
            print(f"Race Group: {race_group} - Epoch {epoch}: Train Loss = {loss.item():.4f}, Val Loss = {val_loss.item():.4f}, C-index = {c_index:.4f}")

    print(f"Race Group: {race_group} - Final C-index on validation set: {c_index:.4f}")

    return model, scaler, c_index

# Best parameters for each race group
best_params_dict = {
    3.0: {'dropout_rate': 0.3, 'hidden_dim1': 128, 'hidden_dim2': 128, 'lr': 0.0005, 'weight_decay': 0.0001},
    1.0: {'dropout_rate': 0.3, 'hidden_dim1': 128, 'hidden_dim2': 128, 'lr': 0.0005, 'weight_decay': 0.0001},
    5.0: {'dropout_rate': 0.3, 'hidden_dim1': 128, 'hidden_dim2': 64, 'lr': 0.0005, 'weight_decay': 1e-05},
    0.0: {'dropout_rate': 0.3, 'hidden_dim1': 256, 'hidden_dim2': 128, 'lr': 0.0005, 'weight_decay': 1e-05},
    4.0: {'dropout_rate': 0.2, 'hidden_dim1': 128, 'hidden_dim2': 128, 'lr': 0.0005, 'weight_decay': 1e-05},
    2.0: {'dropout_rate': 0.2, 'hidden_dim1': 128, 'hidden_dim2': 128, 'lr': 0.0005, 'weight_decay': 1e-05}
}

race_group_models = {}
race_group_scalers = {}
for race_group in race_groups:
    if race_group in best_params_dict:
        print(f"Training model for race group: {race_group}")
        model, scaler, _ = train_race_group_model(race_group, train_imputed, numerical_cols, categorical_cols, ['efs', 'efs_time'], best_params_dict[race_group])
        race_group_models[race_group] = model
        race_group_scalers[race_group] = scaler


predictions = []

for idx, row in test_imputed.iterrows():
    race_group = row['race_group'] 

    if race_group in race_group_models and race_group in race_group_scalers:
        model = race_group_models[race_group]  
        scaler = race_group_scalers[race_group] 

        # Extract features and apply scaling
        X_row = row.drop(columns=['ID', 'race_group']) 
        X_row_scaled = scaler.transform([X_row])  

        # Convert to PyTorch tensor and predict
        X_row_tensor = torch.tensor(X_row_scaled, dtype=torch.float32)
        model.eval()

        with torch.no_grad():
            hazard_score = model(X_row_tensor)  
            prediction = hazard_score.item() 
        predictions.append((row['ID'], prediction))

submission_df = pd.DataFrame(predictions, columns=['ID', 'prediction'])
submission_df.to_csv('submission.csv', index=False)
print("Test predictions saved successfully.")


# import torch
# import torch.nn as nn
# import torch.optim as optim
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from lifelines.utils import concordance_index
# from sklearn.model_selection import ParameterGrid

# # Assuming train_imputed is already preprocessed and available
# race_groups = train_imputed['race_group'].unique()

# # DeepSurv Model
# class DeepSurv(nn.Module):
#     def __init__(self, input_dim, hidden_dim1=256, hidden_dim2=128, dropout_rate=0.2):
#         super(DeepSurv, self).__init__()
#         self.fc1 = nn.Linear(input_dim, hidden_dim1)
#         self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
#         self.fc3 = nn.Linear(hidden_dim2, 1)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(dropout_rate)

#     def forward(self, x):
#         x = self.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = self.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x.squeeze()

# # Cox loss function
# def cox_loss(hazards, efs, efs_time):
#     """Computes Cox proportional hazards loss."""
#     risk_order = torch.argsort(efs_time, descending=True)
#     hazards = hazards[risk_order]
#     efs = efs[risk_order]

#     exp_hazards = torch.exp(hazards)
#     cum_sum = torch.cumsum(exp_hazards, dim=0)
#     log_risk = torch.log(cum_sum)
#     loss = -torch.sum((hazards - log_risk) * efs)

#     return loss / torch.sum(efs)

# # Training function for each race group model
# def train_race_group_model(race_group, train_imputed, numerical_cols, categorical_cols, targets, hidden_dim1=256, hidden_dim2=128, dropout_rate=0.2, lr=0.001, weight_decay=1e-4, num_epochs=100):
#     race_group_data = train_imputed[train_imputed['race_group'] == race_group]

#     X = race_group_data.drop(columns=targets).values
#     efs = race_group_data['efs'].values
#     efs_time = race_group_data['efs_time'].values

#     # Train-validation split
#     X_train, X_val, efs_train, efs_val, efs_time_train, efs_time_val = train_test_split(
#         X, efs, efs_time, test_size=0.2, random_state=42)

#     # Standardize the data
#     scaler = StandardScaler()
#     X_train = scaler.fit_transform(X_train)
#     X_val = scaler.transform(X_val)

#     # Convert data to tensors
#     X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
#     X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
#     efs_train_tensor = torch.tensor(efs_train, dtype=torch.float32)
#     efs_val_tensor = torch.tensor(efs_val, dtype=torch.float32)
#     efs_time_train_tensor = torch.tensor(efs_time_train, dtype=torch.float32)
#     efs_time_val_tensor = torch.tensor(efs_time_val, dtype=torch.float32)

#     # Initialize model and optimizer
#     input_dim = X_train.shape[1]
#     model = DeepSurv(input_dim, hidden_dim1, hidden_dim2, dropout_rate)
#     optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

#     # Training loop
#     for epoch in range(num_epochs):
#         model.train()
#         optimizer.zero_grad()

#         # Forward pass
#         hazards_train = model(X_train_tensor)
#         loss = cox_loss(hazards_train, efs_train_tensor, efs_time_train_tensor)

#         # Backward pass
#         loss.backward()
#         optimizer.step()

#     # Validation
#     model.eval()
#     with torch.no_grad():
#         hazards_val = model(X_val_tensor)
#         val_loss = cox_loss(hazards_val, efs_val_tensor, efs_time_val_tensor)
#         c_index = concordance_index(efs_time_val, -hazards_val.numpy(), efs_val)

#     return c_index

# # Grid search parameters for each model
# param_grid = {
#     'hidden_dim1': [128, 256],
#     'hidden_dim2': [64, 128],
#     'dropout_rate': [0.2, 0.3],
#     'lr': [0.001, 0.0005],
#     'weight_decay': [1e-4, 1e-5]
# }

# # Function to perform grid search for all race groups
# def grid_search_for_race_groups(train_imputed, race_groups, numerical_cols, categorical_cols, targets):
#     race_group_models = {}
#     race_group_scalers = {}
#     race_group_c_indexes = {}
#     race_group_best_params = {}

#     for race_group in race_groups:
#         print(f"Training model for race group: {race_group}")

#         best_c_index = -1
#         best_model = None
#         best_scaler = None
#         best_params = {}

#         # Iterate through the grid search space
#         for params in ParameterGrid(param_grid):
#             c_index = train_race_group_model(race_group, train_imputed, numerical_cols, categorical_cols, targets, **params)
#             if c_index > best_c_index:
#                 best_c_index = c_index
#                 best_model = model
#                 best_scaler = scaler
#                 best_params = params

#         # Store the best model, scaler, and parameters
#         race_group_models[race_group] = best_model
#         race_group_scalers[race_group] = best_scaler
#         race_group_c_indexes[race_group] = best_c_index
#         race_group_best_params[race_group] = best_params

#         print(f"Race Group: {race_group} - Final C-index on validation set: {best_c_index:.4f} with params: {best_params}")

#     return race_group_models, race_group_scalers, race_group_c_indexes, race_group_best_params

# # Run the grid search for each race group
# race_group_models, race_group_scalers, race_group_c_indexes, race_group_best_params = grid_search_for_race_groups(train_imputed, race_groups, numerical_cols, categorical_cols, targets)

# # Print C-index and best parameters for each race group
# print("C-index and best parameters for each race group:")
# for race_group in race_group_best_params:
#     print(f"Race group {race_group}:")
#     print(f"  Best C-index: {race_group_c_indexes[race_group]:.4f}")
#     print(f"  Best Params: {race_group_best_params[race_group]}")



# C-index and best parameters for each race group:
# Race group 3.0:
#   Best C-index: 0.6412
#   Best Params: {'dropout_rate': 0.3, 'hidden_dim1': 128, 'hidden_dim2': 128, 'lr': 0.0005, 'weight_decay': 0.0001}
# Race group 1.0:
#   Best C-index: 0.6550
#   Best Params: {'dropout_rate': 0.3, 'hidden_dim1': 128, 'hidden_dim2': 128, 'lr': 0.0005, 'weight_decay': 0.0001}
# Race group 5.0:
#   Best C-index: 0.6268
#   Best Params: {'dropout_rate': 0.3, 'hidden_dim1': 128, 'hidden_dim2': 64, 'lr': 0.0005, 'weight_decay': 1e-05}
# Race group 0.0:
#   Best C-index: 0.6520
#   Best Params: {'dropout_rate': 0.3, 'hidden_dim1': 256, 'hidden_dim2': 128, 'lr': 0.0005, 'weight_decay': 1e-05}
# Race group 4.0:
#   Best C-index: 0.6235
#   Best Params: {'dropout_rate': 0.2, 'hidden_dim1': 128, 'hidden_dim2': 128, 'lr': 0.0005, 'weight_decay': 1e-05}
# Race group 2.0:
#   Best C-index: 0.6145
#   Best Params: {'dropout_rate': 0.2, 'hidden_dim1': 128, 'hidden_dim2': 128, 'lr': 0.0005, 'weight_decay': 1e-05}

