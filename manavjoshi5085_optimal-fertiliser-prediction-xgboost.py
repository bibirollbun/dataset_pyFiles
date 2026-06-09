import numpy as np            #Importing Necessary data 
import pandas as pd
import matplotlib.pyplot as plt   
import seaborn as sns            
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from collections import Counter
from catboost import CatBoostClassifier, Pool
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings("ignore")


train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv") #Data Importing
test  = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
orginal = pd.read_csv("/kaggle/input/fertilizer-prediction/Fertilizer Prediction.csv")


orginal.insert(0, 'id', range(len(test), len(test) + len(orginal)))


# Adding a dataset column to track source
train['dataset'] = 'train'
test['dataset'] = 'test'
orginal['dataset'] = 'train'
# Merging train and test sets for consistent preprocessing across the dataset
fert=pd.concat([train, test, orginal], axis=0).reset_index(drop=True)


train  #Printing the data train 


orginal #Printing original data 


test #Printing test data 


fert #Printing complete dataset fert is the name of my dataset i put during concat
#Your can be anything 


fert.shape  #Checking for the dimensions 


fert.head() #Preview the first 5 rows to verify columns and 'dataset' marker


fert.info() #Info helps to get basic details about the data 


fert.describe()


train.isnull().sum() #Checking for null values 


train_fert = fert[fert['dataset'] == 'train'].copy()
num_feats = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
for col in num_feats:
    plt.figure(figsize=(6, 4))
    sns.histplot(train_fert[col], kde=True, color='skyblue', edgecolor='black')
    plt.title(f'Distribution of {col}', fontsize=14)
    plt.xlabel(col, fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    print(f'\n📊 Stats for {col}:\n')
    print(train_fert[col].describe(), '\n' + '-'*40)


categorical_features = ['Soil Type', 'Crop Type']
for feature in categorical_features:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.countplot(x=feature,data=train_fert,order=train_fert[feature].value_counts().index,palette='Spectral',edgecolor='black',ax=ax)
    ax.set_title(f'Distribution of {feature}', fontsize=15, weight='bold')
    ax.set_xlabel(feature, fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.tick_params(axis='x', rotation=40)
    ax.grid(visible=True, axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()
    print(f"\n🔎 Proportions in '{feature}':")
    print(train_fert[feature]
          .value_counts(normalize=True)
          .round(3)
          .rename_axis('Category')
          .reset_index(name='Proportion'), '\n' + '='*50)


fig, ax = plt.subplots(figsize=(9, 5))
sns.barplot(
    x=train_fert['Fertilizer Name'].value_counts().index,
    y=train_fert['Fertilizer Name'].value_counts().values,
    palette='Spectral',
    edgecolor='gray',
    ax=ax
)
ax.set_title('🌿 Fertilizer Class Distribution', fontsize=15, weight='bold', color='darkgreen')
ax.set_xlabel('Fertilizer Type', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
plt.xticks(rotation=30, ha='right')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()
print("\n🌱 Fertilizer Class Proportions:")
for k, v in train_fert['Fertilizer Name'].value_counts(normalize=True).round(3).items():
    print(f"{k}: {v}")


cat_feats = ['Soil Type', 'Crop Type']
for col in cat_feats:
    plt.figure(figsize=(8, 4))
    sns.countplot(data=train_fert,x=col,hue='Fertilizer Name',palette='Set1',edgecolor='black')
    plt.title(f'{col} by Fertilizer Name', fontsize=14)
    plt.xlabel(col, fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Fertilizer Name', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()
    print(f'\n📊 Proportions of Fertilizer within "{col}":\n')
    prop_table = train_fert.groupby(col)['Fertilizer Name'].value_counts(normalize=True).unstack().round(3)
    print(prop_table, '\n' + '-'*50)


numeric_feats = ['Temparature','Humidity','Moisture','Nitrogen','Potassium','Phosphorous']
for col in numeric_feats:
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=train_fert, x='Fertilizer Name', y=col, palette='Set3')
    plt.title(f'{col} by Fertilizer Name', fontsize=14)
    plt.xlabel('Fertilizer Name', fontsize=12)
    plt.ylabel(col, fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


num_feats = ['Temparature','Humidity','Moisture','Nitrogen','Potassium','Phosphorous']
plt.figure(figsize=(6, 4))
sns.heatmap(
    train_fert[num_feats].corr(),
    annot=True,
    fmt='.2f',
    cmap='coolwarm',
    linewidths=0.5,
    square=True,
    cbar_kws={'shrink': 0.75}
)
plt.title('Correlation Matrix of Numerical Features',fontsize=14)
plt.tight_layout()
plt.show()


train.isnull().sum() # Handle Missing Values


num_cols = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Phosphorous', 'Potassium']
cat_cols = ['Soil Type', 'Crop Type']

le = LabelEncoder()
train_mask = fert['dataset'] == 'train'
fert.loc[train_mask, 'Fertilizer Name'] = le.fit_transform(fert.loc[train_mask, 'Fertilizer Name'])


train_fert = fert[fert['dataset'] == 'train'].drop(columns=['dataset'], errors='ignore')
test_fert = fert[fert['dataset'] == 'test'].drop(columns =['dataset'], errors='ignore')
train_fert= train_fert.drop(columns=['id'], errors='ignore')
test_fert= test_fert.drop(columns=['Fertilizer Name'], errors='ignore')
X = train_fert.drop(['Fertilizer Name'], axis=1)
y = train_fert['Fertilizer Name']
y = y.astype(int)


def mapk(actual, predicted, k=3):
    """Compute mean average precision at k (MAP@k)."""
    def apk(a, p, k):
        score = 0.0
        for i in range(min(k, len(p))):
            if p[i] == a:
                score += 1.0 / (i + 1)
                break  
        return score
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


X_enc = X.copy()
for col in cat_cols:
    X_enc[col] = X_enc[col].astype("category").cat.codes
    test_fert[col] = test_fert[col].astype("category").cat.codes
counter_full = Counter(y)
max_count_full = max(counter_full.values())
class_weights_full = {cls: max_count_full / count for cls, count in counter_full.items()}

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
fold_accuracies = []
oof_preds = np.zeros((X.shape[0], len(np.unique(y))))

for fold, (train_idx, val_idx) in enumerate(kfold.split(X_enc, y), 1):
    print(f"\n Fold {fold} ")

    X_tr, X_va = X_enc.iloc[train_idx], X_enc.iloc[val_idx]
    y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]

    counter_fold = Counter(y_tr)
    max_count_fold = max(counter_fold.values())
    sample_weights = y_tr.map(lambda cls: max_count_fold / counter_fold[cls])
    XGB_model = XGBClassifier(
        max_depth=12,
        colsample_bytree=0.467,
        subsample=0.86,
        n_estimators=4000,
        learning_rate=0.03,
        gamma=0.26,
        max_delta_step=4,
        reg_alpha=2.7,
        reg_lambda=1.4,
        objective='multi:softprob',
        random_state=13,
        enable_categorical=True,
        tree_method='hist',     
        device='cuda'        
    )
    XGB_model.fit(
        X_tr,
        y_tr,
        sample_weight=sample_weights,
        eval_set=[(X_va, y_va)],
        early_stopping_rounds=150,
        verbose=200,
    )

    val_labels = XGB_model.predict(X_va)
    val_probas = XGB_model.predict_proba(X_va)

    oof_preds[val_idx] = val_probas
    acc = accuracy_score(y_va, val_labels)
    fold_accuracies.append(acc)
    print(f" Fold {fold} Accuracy: {acc:.4f}")

print("\n Mean CV Accuracy:", np.mean(fold_accuracies))
print(" Std CV Accuracy:", np.std(fold_accuracies))

top3_preds = np.argsort(oof_preds, axis=1)[:, ::-1][:, :3]
map3_score = mapk(y.values, top3_preds, k=3)
print(f"\n Mean Average Precision @3 (MAP@3): {map3_score:.5f}")


# Prepare test features by dropping the 'id' column if it exists
test_features = test_fert.drop(columns=['id'], errors='ignore')

# Predict class probabilities using XGBoost
xgb_test_preds = XGB_model.predict_proba(test_features)

# Get top 3 predictions per sample (indices of top probabilities)
top_3_preds = np.argsort(xgb_test_preds, axis=1)[:, ::-1][:, :3]

# Decode predicted class indices back to original labels
top_3_labels = le.inverse_transform(top_3_preds.ravel()).reshape(top_3_preds.shape)

# Build submission DataFrame
submission = pd.DataFrame({
    'id': test_fert['id'],  # Ensure 'id' exists in test_df
    'Fertilizer Name': [' '.join(preds) for preds in top_3_labels]
})

# Save the submission to CSV
submission.to_csv('submission.csv', index=False)


submission




