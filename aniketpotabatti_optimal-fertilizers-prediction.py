import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import seaborn as sns
warnings.filterwarnings('ignore')

sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)


test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
train.head()


print(f"Number of samples: {len(train)}")
print(f"Number of features: {len(train.columns)}")

print("\nData types:")
print(train.dtypes)

train.describe()


print("Training data missing values:")
print(train.isnull().sum())
print("\nTest data missing values:")
print(test.isnull().sum())


numerical_cols = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
print(train[numerical_cols].describe())


print("\nPlotting distributions...")
plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(2, 3, i)
    sns.histplot(train[col], kde=True, bins=30)
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.show()


categorical_cols = ['Soil Type', 'Crop Type', 'Fertilizer Name']
for col in categorical_cols[:-1]:  # Exclude target for now
    print(f"\n{col} value counts:")
    print(train[col].value_counts())
    print(f"\n{col} distribution:")
    plt.figure(figsize=(10, 4))
    sns.countplot(data=train, y=col, order=train[col].value_counts().index)
    plt.title(f'Distribution of {col}')
    plt.tight_layout()
    plt.show()


# 7. Target variable analysis
print("Number of unique fertilizers:", train['Fertilizer Name'].nunique())
print("\nTop 10 most common fertilizers:")
print(train['Fertilizer Name'].value_counts().head(10))

plt.figure(figsize=(12, 6))
sns.countplot(data=train, 
             y='Fertilizer Name', 
             order=train['Fertilizer Name'].value_counts().index[:10])  # Top 10
plt.title('Top 10 Most Common Fertilizers')
plt.tight_layout()
plt.show()


from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
import joblib, gc, warnings, os 
warnings.filterwarnings('ignore')
sns.set(style="whitegrid")


y = train['Fertilizer Name']
X = train.drop(['id', 'Fertilizer Name'], axis=1)
X_test = test.drop('id', axis=1)

print('Encoding target variable...')
le = LabelEncoder()
y_encoded = le.fit_transform(y)


print('Creating preprocessor...')
num_cols = ['Temparature','Humidity','Moisture',
            'Nitrogen','Potassium','Phosphorous']
cat_cols = ['Soil Type','Crop Type']

prep = ColumnTransformer([
    ('num', StandardScaler(with_mean=False), num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
], sparse_threshold=0.3)


print('Transforming data...')
X_all = prep.fit_transform(X)
X_test_prep = prep.transform(X_test)

# Split into train/validation sets
print('Splitting data...')
X_tr, X_va, y_tr, y_va = train_test_split(
    X_all, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)



print('Training XGBoost...')
model = XGBClassifier(
    tree_method='hist',
    max_depth=6,
    n_estimators=200,
    learning_rate=0.05,
    n_jobs=-1,
    random_state=42,
    objective='multi:softprob',
    num_class=len(le.classes_)  
)

model.fit(X_tr, y_tr)

# Evaluate
y_pred = model.predict(X_va)
val_acc = accuracy_score(y_va, y_pred)
print(f'Validation accuracy: {val_acc:.4f}')

# Make predictions on test set
test_pred_encoded = model.predict(X_test_prep)
test_pred = le.inverse_transform(test_pred_encoded) # Convert back to original labels


# Creates submission file
submission = pd.DataFrame({
    'id': test['id'],
    'Fertilizer Name': test_pred
})
submission.to_csv('submission.csv', index=False)
print('Submission file saved as submission.csv')

# Print class mapping for reference
print("\nClass mapping:")
for i, class_name in enumerate(le.classes_):
    print(f"{i}: {class_name}")

