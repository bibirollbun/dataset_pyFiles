import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", category=FutureWarning)


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder


train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv').set_index('id')
test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv').set_index("id")
train.head()


test.head()


train.info()


test.info()


train.describe()


print(train.duplicated().sum())
print(test.duplicated().sum())
print(train.isnull().sum())


def split_columns_by_type(df):
    numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    return numerical_cols, categorical_cols

# Example usage
numerical_cols, categorical_cols = split_columns_by_type(train)
categorical_cols.remove("Fertilizer Name")
print("Numerical:", numerical_cols)
print("Categorical:", categorical_cols)


# Plot
target_col = "Fertilizer Name"
plt.figure(figsize=(8, 5))
sns.countplot(data=train, y=target_col, order=train[target_col].value_counts().index, palette="viridis")
plt.title("Distribution of Target (Fertilizer Name)")
plt.show()


# Set plot style
sns.set(style="whitegrid")
plt.figure(figsize=(16, 24))

# Plot histograms
for i, col in enumerate(numerical_cols):
    plt.subplot(len(numerical_cols), 2, i * 2 + 1)
    sns.histplot(train[col], kde=True, bins=30, color='black')
    plt.title(f'Histogram of {col}')
    plt.xlabel(col)
    
    plt.subplot(len(numerical_cols), 2, i * 2 + 2)
    sns.boxplot(x=train[col], color='lightgreen')
    plt.title(f'Boxplot of {col}')
    plt.xlabel(col)

plt.tight_layout()
plt.show()


for feature in numerical_cols:
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=train, x=target_col, y=feature, palette='Set2')
    plt.title(f"{feature} by Target (Boxplot)", fontsize=14)
    plt.xlabel("Target", fontsize=12)
    plt.ylabel(feature, fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


for col in categorical_cols:
    plt.figure(figsize=(10,5))
    sns.countplot(data=train, x=col, hue=target_col, palette='Set2')
    plt.title(f'Distribution of {col} by {target_col}', fontsize=16)
    plt.xlabel(col, fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=45)
    plt.legend(title=target_col, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


for feature in ["Soil Type", "Crop Type",'Fertilizer Name']:
    counts = train[feature].value_counts()

    # Plot pie chart
    plt.figure(figsize=(6, 6))
    plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90)
    plt.title(f"Distribution of {feature}")
    plt.axis("equal")
    plt.show()

    # Print unique and missing values
    print(f"Number of Unique {feature}: {train[feature].nunique()}")


from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder

# Separate target
target_col = 'Fertilizer Name'
X = train.drop(columns=[target_col])
y = train[target_col]


# 1. Normalize  numerical features
scaler = StandardScaler() 
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
test[numerical_cols] = scaler.transform(test[numerical_cols])

# 2. Encode categorical features
oe = OrdinalEncoder()
X[categorical_cols] = oe.fit_transform(X[categorical_cols])
test[categorical_cols] = oe.transform(test[categorical_cols])

# 3. Encode target
le = LabelEncoder()
y = le.fit_transform(y)


from xgboost import XGBClassifier

# Train model with tuned parameters
model = XGBClassifier(
    n_estimators=100000,        # number of boosting rounds
    learning_rate=0.1,       # step size shrinkage
    max_depth=8,             # maximum depth of a tree
    subsample=0.8,           # subsample ratio of training instances
    colsample_bytree=0.8,    # subsample ratio of columns when constructing each tree
    gamma=1,                 # minimum loss reduction required to make a further partition
    random_state=42,
    use_label_encoder=False,
    eval_metric='mlogloss'   # avoid warning
)

model.fit(X, y)


y_test_pred_proba = model.predict_proba(test)


top_k = 3
top_k_indices = np.argsort(y_test_pred_proba, axis=1)[:, ::-1][:, :top_k]  
top_k_labels = le.inverse_transform(top_k_indices.ravel()).reshape(top_k_indices.shape)

sample_sub = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")

submission = pd.DataFrame({
    'id': sample_sub['id'],
    'Fertilizer Name': [' '.join(row) for row in top_k_labels]
})

submission.to_csv('submission.csv', index=False)


submission.head()

