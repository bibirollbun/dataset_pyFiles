import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


data = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')


data.info()


numerical_cols = ['annual_income', 'debt_to_income_ratio', 'credit_score', 
                  'loan_amount', 'interest_rate']


for col in numerical_cols:
    print(f"\n{col.upper()}:")
    print(f"  Mean: {data[col].mean():,.2f}")
    print(f"  Median: {data[col].median():,.2f}")
    print(f"  Mode: {data[col].mode()[0]:,.2f}")
    print(f"  Std Dev: {data[col].std():,.2f}")
    print(f"  Variance: {data[col].var():,.2f}")
    print(f"  Skewness: {data[col].skew():.2f}")
    print(f"  Kurtosis: {data[col].kurtosis():.2f}")
    print(f"  Range: {data[col].min():,.2f} - {data[col].max():,.2f}")


fig, axes = plt.subplots(3, 2, figsize=(15, 12))
axes = axes.ravel()

for idx, col in enumerate(numerical_cols):
    axes[idx].hist(data[col], bins=50, edgecolor='black', alpha=0.7)
    axes[idx].set_title(f'Distribution of {col}', fontsize=12, fontweight='bold')
    axes[idx].set_xlabel(col)
    axes[idx].set_ylabel('Frequency')
    axes[idx].grid(True, alpha=0.4)


plt.tight_layout()
plt.show()



# Box plots for numerical features
fig, axes = plt.subplots(3, 2, figsize=(15, 12))
axes = axes.ravel()

for idx, col in enumerate(numerical_cols):
    axes[idx].boxplot(data[col], vert=True, patch_artist=True)
    axes[idx].set_title(f'Box Plot of {col}', fontsize=12, fontweight='bold')
    axes[idx].set_ylabel(col)
    axes[idx].grid(True, alpha=0.3)


# Outlier detection
print("\n" + "="*80)
print("OUTLIER DETECTION (IQR Method)")
print("="*80)
for col in numerical_cols:
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)][col]
    print(f"\n{col}:")
    print(f"  Outliers count: {len(outliers)} ({len(outliers)/len(data)*100:.2f}%)")
    print(f"  Lower bound: {lower_bound:.2f}, Upper bound: {upper_bound:.2f}")


categorical_cols = ['gender', 'marital_status', 'education_level', 
                    'employment_status', 'loan_purpose', 'grade_subgrade']


for col in categorical_cols:
    print(f"\n{col.upper()}:")
    value_counts = data[col].value_counts()
    value_percentages = data[col].value_counts(normalize=True) * 100
    summary = pd.DataFrame({
        'Count': value_counts,
        'Percentage': value_percentages
    })
    print(summary)
    print(f"Unique values: {data[col].nunique()}")


# Visualize categorical features
fig, axes = plt.subplots(3, 2, figsize=(15, 15))
axes = axes.ravel()

for idx, col in enumerate(categorical_cols):
    value_counts = data[col].value_counts()
    axes[idx].bar(range(len(value_counts)), value_counts.values, alpha=0.7)
    axes[idx].set_title(f'Distribution of {col}', fontsize=12, fontweight='bold')
    axes[idx].set_xlabel(col)
    axes[idx].set_ylabel('Count')
    axes[idx].set_xticks(range(len(value_counts)))
    axes[idx].set_xticklabels(value_counts.index, rotation=45, ha='right')
    axes[idx].grid(True, alpha=0.3, axis='y')


# Correlation matrix for numerical features including target
correlation_cols = numerical_cols + ['loan_paid_back']
correlation_matrix = data[correlation_cols].corr()
target_corr = correlation_matrix['loan_paid_back'].sort_values(ascending=False)
print(target_corr)

# Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt='.3f', cmap='coolwarm', 
            center=0, square=True, linewidths=1)
plt.title('Correlation Heatmap - Numerical Features', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()


mask = pd.Series([True] * len(data))

for col in numerical_cols:
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Update mask: keep rows within bounds
    mask = mask & (data[col] >= lower_bound) & (data[col] <= upper_bound)

# Apply mask to remove outliers
data_cleaned = data[mask].copy()


from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer

# Define numerical columns
normal_distributed = ['credit_score', 'interest_rate']
skewed_columns = ['annual_income', 'loan_amount', 'debt_to_income_ratio']

# Define categorical columns
# Non-ordered categorical features (nominal)
nominal_features = ['gender', 'marital_status', 'employment_status', 'loan_purpose']

# Ordered categorical features (ordinal)
ordinal_features = ['education_level', 'grade_subgrade']

# Define ordinal mappings based on natural ordering
education_order = ['High School', 'Associate', "Bachelor's", "Master's", 'Doctorate']  # Adjust based on your data
grade_order = ['A1', 'A2', 'A3', 'A4', 'A5', 
               'B1', 'B2', 'B3', 'B4', 'B5',
               'C1', 'C2', 'C3', 'C4', 'C5',
               'D1', 'D2', 'D3', 'D4', 'D5',
               'E1', 'E2', 'E3', 'E4', 'E5',
               'F1', 'F2', 'F3', 'F4', 'F5',
               'G1', 'G2', 'G3', 'G4', 'G5']  # Adjust based on your data

# Create the ColumnTransformer with all preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('standard', StandardScaler(), normal_distributed),  # Normal distributed numerical
        ('robust', RobustScaler(), skewed_columns),  # Skewed numerical
        ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), 
         nominal_features),  # Non-ordered categorical
        ('ordinal', OrdinalEncoder(categories=[education_order, grade_order], 
                                   handle_unknown='use_encoded_value', unknown_value=-1), 
         ordinal_features)  # Ordered categorical
    ],
    remainder='drop'  # Drop any remaining columns (like 'id' if present)
)



X = data_cleaned.drop('loan_paid_back',axis=1)
y = data_cleaned['loan_paid_back']
# X = data.drop('loan_paid_back',axis=1)
# y = data['loan_paid_back']


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


processed_train = preprocessor.fit_transform(X_train)
processed_test  =preprocessor.transform(X_test)


from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge



reg = SGDRegressor( max_iter=10000, tol=1e-3)
lreg = LinearRegression()
rreg = Ridge()


# reg.fit(processed_train,y_train)
rreg.fit(processed_train,y_train)


pre =  rreg.predict(processed_test)


from sklearn.metrics import roc_auc_score
roc_auc_score(y_test,pre)


from sklearn.metrics import mean_squared_error
mean_squared_error(y_test,pre)


test = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')


ids = test['id'].to_list()


test_preprocessed = preprocessor.transform(test)


pre_test =  rreg.predict(test_preprocessed)


pre_test.shape,len(ids),test_preprocessed.shape


import torch

X_train_tensor = torch.tensor(processed_train, dtype=torch.float32)
X_val_tensor = torch.tensor(processed_test, dtype=torch.float32)

y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
y_val_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)


import torch.nn as nn

class MyRegressor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1) ,  # no activation
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)



criterion = nn.MSELoss()
# criterion = nn.L1Loss()


model = MyRegressor(input_dim=X_train_tensor.shape[1])



optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(300):
    model.train()
    pred = model(X_train_tensor)
    loss = criterion(pred, y_train_tensor)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model.eval()
    val_pred = model(X_val_tensor).detach()
    val_loss = criterion(val_pred, y_val_tensor)

    print(epoch, loss.item(), val_loss.item())




X_test_predict_tensor = torch.tensor(test_preprocessed, dtype=torch.float32)

model.eval()
with torch.no_grad():
    y_val_pred = model(X_test_predict_tensor).numpy().ravel()


y_val_pred.shape,len(ids) ,X_test_predict_tensor.shape


submit = pd.DataFrame({
    "id" :ids,
    "loan_paid_back":y_val_pred
    
})


submit.to_csv('submission.csv',index=False)

