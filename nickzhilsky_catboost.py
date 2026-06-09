import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from catboost import CatBoostClassifier, Pool
from sklearn.linear_model import LogisticRegression
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')

print("Train shape:", train.shape)
print("\nTrain info:")
print(train.info())
print("Test shape:", test.shape)
print("\nTest info:")
print(test.info())


print("\nMissing values in train:")
print(train.isnull().sum())

print("\nTarget distribution:")
print(train['Fertilizer Name'].value_counts(normalize=True))

categorical_cols = train.select_dtypes(include=['object']).columns.tolist()
print("\nCategorical columns:", categorical_cols)

numeric_cols = train.select_dtypes(include=['int64', 'float64']).columns.tolist()
numeric_cols.remove('id')
print("Numeric columns:", numeric_cols)


print("\nMissing values in train:")
print(test.isnull().sum())

categorical_cols_test = test.select_dtypes(include=['object']).columns.tolist()
print("\nCategorical columns:", categorical_cols_test)

numeric_cols_test = test.select_dtypes(include=['int64', 'float64']).columns.tolist()
numeric_cols_test.remove('id')
print("Numeric columns:", numeric_cols_test)


plt.figure(figsize=(7, 6))
corr = train[numeric_cols].corr(method='pearson')
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Numeric Features')
plt.show()


def add_custom_features(df):
    
    df['NPK_Total'] = df['Nitrogen'] + df['Phosphorous'] + df['Potassium']
    df['N_ratio'] = df['Nitrogen'] / df['NPK_Total']
    df['P_ratio'] = df['Phosphorous'] / df['NPK_Total']
    df['K_ratio'] = df['Potassium'] / df['NPK_Total']
    df['SMI'] = df['Humidity'] / df['Temparature']
    df['EvapoIndex'] = df['Temparature'] * (1 - df['Humidity'] / 100)
    df['Nutrient_var'] = df[['Nitrogen', 'Phosphorous', 'Potassium']].std(axis=1)
    df['Dominant_Nutrient'] = df[['Nitrogen', 'Phosphorous', 'Potassium']].idxmax(axis=1)
    
    return df


train = add_custom_features(train)
test = add_custom_features(test)


numeric_cols = train.select_dtypes(include=['number'])
corr = numeric_cols.corr(method='pearson')
plt.figure(figsize=(10, 10))
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', center=0)
plt.title("Correlation Heatmap of Numerical Features")
plt.show()


corr_pairs = (
    corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    .stack()
    .reset_index()
    .rename(columns={0: 'correlation', 'level_0': 'Feature 1', 'level_1': 'Feature 2'})
)

corr_pairs['abs_corr'] = corr_pairs['correlation'].abs()
corr_pairs_sorted = corr_pairs.sort_values(by='abs_corr', ascending=False)

print("\nTop 10 best correlation:")
print(corr_pairs_sorted.head(10))


selected_features = [
    'Temparature',
    'Humidity',
    'Nitrogen',
    'Phosphorous',
    'Nutrient_var',
    'Potassium',
    'Soil Type',
    'Crop Type'
]


X = train[selected_features].copy()
y = train['Fertilizer Name']

X.loc[:, 'Soil Type'] = X['Soil Type'].astype('category')
X.loc[:, 'Crop Type'] = X['Crop Type'].astype('category')


X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

cat_features = ['Soil Type', 'Crop Type']

train_pool = Pool(X_train, y_train, cat_features=cat_features)
valid_pool = Pool(X_valid, y_valid, cat_features=cat_features)

model = CatBoostClassifier(
    iterations = 100,
    learning_rate = 0.3,
    depth = 7,
    l2_leaf_reg = 2,
    loss_function='MultiClass',
    verbose = 50
)

model.fit(train_pool, eval_set=valid_pool, early_stopping_rounds=50)

y_pred = model.predict(X_valid).ravel()

print("Accuracy:", accuracy_score(y_valid, y_pred))
print("\nClassification report:\n", classification_report(y_valid, y_pred))


cm = confusion_matrix(y_valid, y_pred)
plt.figure(figsize=(10, 6))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=model.classes_, yticklabels=model.classes_, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


feature_importances = model.get_feature_importance(train_pool)
feature_names = X_train.columns

plt.figure(figsize=(10, 6))
plt.barh(feature_names, feature_importances)
plt.xlabel("Importance")
plt.title("Feature Importance")
plt.show()


print(train['Crop Type'].value_counts())
pd.crosstab(train['Crop Type'], train['Fertilizer Name'], normalize='index').round(2)
print(train[['Crop Type', 'Soil Type', 'Nitrogen']].head())


train['Nitrogen_bin'] = pd.cut(train['Nitrogen'], bins=5, labels=False)
train['Crop_N'] = train['Crop Type'].astype(str) + '_' + train['Nitrogen_bin'].astype(str)
train['Crop_Soil'] = train['Crop Type'].astype(str) + '_' + train['Soil Type'].astype(str)
print(train[['Crop Type', 'Soil Type', 'Nitrogen', 'Nitrogen_bin', 'Crop_N', 'Crop_Soil']].head())
test['Crop_N'] = test['Crop Type'].astype(str) + '_' + train['Nitrogen_bin'].astype(str)
test['Crop_Soil'] = test['Crop Type'].astype(str) + '_' + train['Soil Type'].astype(str)


selected_features = [
    'Temparature',
    'Humidity',
    'Nitrogen',
    'Phosphorous',
    'Potassium',
    'Crop Type',
    'Crop_N',
    'Crop_Soil'
]

X = train[selected_features]
y = train['Fertilizer Name']
X_test = test[selected_features]

categorical_features = ['Crop Type', 'Crop_N', 'Crop_Soil']
n_splits = 5

skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

test_preds = np.zeros((len(test), len(y.unique())))
val_preds = np.zeros(len(X), dtype=int)

le = LabelEncoder()
y_encoded = le.fit_transform(y)

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\nFold {fold+1}/{n_splits}")
    
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr = y.iloc[train_idx]
    y_val = y.iloc[val_idx]
    
    model = CatBoostClassifier(
        iterations=500,            
        learning_rate=0.05,  
        depth=6,                
        l2_leaf_reg=3, 
        loss_function='MultiClass',
        eval_metric='MultiClass',
        random_seed=42,
        early_stopping_rounds=50,
        verbose=100
    )
    
    model.fit(
        X_tr, y_tr,
        eval_set=(X_val, y_val),
        cat_features=categorical_features
    )

    val_preds[val_idx] = le.transform(model.predict(X_val).ravel())

    test_preds += model.predict_proba(X_test)


test_preds /= n_splits

class_labels = model.classes_

proba_df = pd.DataFrame(test_preds, columns=class_labels, index=test.index)


def get_top3_classes(row):
    return ' '.join(row.sort_values(ascending=False).index[:3])

test['Fertilizer Name'] = proba_df.apply(get_top3_classes, axis=1)

submission = test[['id', 'Fertilizer Name']]
submission.to_csv('submission.csv', index=False)

