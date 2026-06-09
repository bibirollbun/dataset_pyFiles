import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from sklearn.feature_extraction.text import TfidfVectorizer
import lightgbm as lgb
import warnings

warnings.filterwarnings('ignore')

# Load data
train_df = pd.read_csv('/kaggle/input/ctai-ctd-hackathon/train.csv')
test_df = pd.read_csv('/kaggle/input/ctai-ctd-hackathon/test.csv')


def clean_numeric(df):
    """Cleans numeric columns by removing currency symbols and converting to float."""
    numeric_cols = ['QtyShipped', 'UnitPrice', 'ExtendedPrice', 'invoiceTotal']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(r'[$,]', '', regex=True)
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def create_features(df):
    """Engineers new features from existing data."""
    df = df.copy()
    
    # Date-based features
    for col_name, prefix in [('CONSTRUCTION_START_DATE', 'start'), ('SUBSTANTIAL_COMPLETION_DATE', 'completion')]:
        if col_name in df.columns:
            df[col_name] = pd.to_datetime(df[col_name], errors='coerce')
            df[f'{prefix}_month'] = df[col_name].dt.month
            df[f'{prefix}_quarter'] = df[col_name].dt.quarter

    if 'CONSTRUCTION_START_DATE' in df.columns and 'SUBSTANTIAL_COMPLETION_DATE' in df.columns:
        df['project_duration'] = (df['SUBSTANTIAL_COMPLETION_DATE'] - df['CONSTRUCTION_START_DATE']).dt.days

    # Price-based features
    if 'UnitPrice' in df.columns and 'QtyShipped' in df.columns:
        df['total_price'] = df['UnitPrice'] * df['QtyShipped']
    if 'invoiceTotal' in df.columns and 'SIZE_BUILDINGSIZE' in df.columns:
        df['price_per_sqft'] = df['invoiceTotal'] / (df['SIZE_BUILDINGSIZE'] + 1e-6) # Add epsilon for stability

    # Aggregations by project
    if 'PROJECTNUMBER' in df.columns:
        agg_dict = {'invoiceTotal': ['mean', 'sum', 'count'], 'UnitPrice': ['mean', 'median']}
        if 'MasterItemNo' in df.columns:
            agg_dict['MasterItemNo'] = 'nunique'
        
        project_stats = df.groupby('PROJECTNUMBER').agg(agg_dict).round(2)
        project_stats.columns = ['_'.join(col).strip() for col in project_stats.columns]
        df = df.merge(project_stats, on='PROJECTNUMBER', how='left')

    # Frequency encoding for items
    if 'MasterItemNo' in df.columns:
        df['item_frequency'] = df['MasterItemNo'].map(df['MasterItemNo'].value_counts())

    # Fill missing values
    numeric_cols = df.select_dtypes(include=np.number).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    categorical_cols = df.select_dtypes(include='object').columns
    df[categorical_cols] = df[categorical_cols].fillna('Unknown')
    
    return df



def prepare_text_features(train, test):
    """Creates TF-IDF features from 'ItemDescription'."""
    if 'ItemDescription' not in train.columns:
        return train, test, []

    all_descriptions = pd.concat([train['ItemDescription'], test['ItemDescription']], ignore_index=True).fillna('')
    tfidf = TfidfVectorizer(max_features=100, stop_words='english', ngram_range=(1, 2))
    tfidf.fit(all_descriptions)

    train_tfidf = pd.DataFrame(tfidf.transform(train['ItemDescription'].fillna('')).toarray(), index=train.index)
    test_tfidf = pd.DataFrame(tfidf.transform(test['ItemDescription'].fillna('')).toarray(), index=test.index)
    
    tfidf_cols = [f'tfidf_{i}' for i in range(train_tfidf.shape[1])]
    train_tfidf.columns = tfidf_cols
    test_tfidf.columns = tfidf_cols

    return pd.concat([train, train_tfidf], axis=1), pd.concat([test, test_tfidf], axis=1), tfidf_cols



def train_lgb_model(X_train, y_train, X_val, y_val, params):
    """Trains a LightGBM model with early stopping."""
    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)
    
    model = lgb.train(
        params,
        lgb_train,
        valid_sets=[lgb_val],
        num_boost_round=1000,
        callbacks=[lgb.early_stopping(100, verbose=False)]
    )
    return model


train_df = clean_numeric(train_df)
test_df = clean_numeric(test_df)
train_df = create_features(train_df)
test_df = create_features(test_df)
train_df, test_df, tfidf_cols = prepare_text_features(train_df, test_df)


features_to_drop = ['id', 'MasterItemNo', 'QtyShipped', 'ItemDescription', 'CONSTRUCTION_START_DATE', 'SUBSTANTIAL_COMPLETION_DATE']
feature_cols = [col for col in train_df.columns if col not in features_to_drop]

categorical_cols = train_df[feature_cols].select_dtypes(include=['object', 'category']).columns.tolist()
for col in categorical_cols:
    le = LabelEncoder()
    combined_data = pd.concat([train_df[col], test_df[col]]).astype(str)
    le.fit(combined_data)
    train_df[col] = le.transform(train_df[col].astype(str))
    test_df[col] = le.transform(test_df[col].astype(str))


X = train_df[feature_cols]
y_reg = train_df['QtyShipped']

X_test = test_df.reindex(columns=X.columns, fill_value=0)

class_encoder = LabelEncoder()
y_class_encoded = class_encoder.fit_transform(train_df['MasterItemNo'])
num_total_classes = len(class_encoder.classes_) # Correctly get total number of classes


gkf = GroupKFold(n_splits=5)
groups = train_df['PROJECTNUMBER'] if 'PROJECTNUMBER' in train_df.columns else None

class_params = {
    'objective': 'multiclass', 'num_class': num_total_classes, 'metric': 'multi_logloss',
    'boosting_type': 'gbdt', 'num_leaves': 64, 'learning_rate': 0.05,
    'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 5,
    'verbose': -1, 'random_state': 42
}
reg_params = {
    'objective': 'regression', 'metric': 'mae', 'boosting_type': 'gbdt',
    'num_leaves': 32, 'learning_rate': 0.05, 'feature_fraction': 0.8,
    'bagging_fraction': 0.8, 'bagging_freq': 5, 'verbose': -1, 'random_state': 42
}

oof_class_preds = np.zeros(len(X))
oof_reg_preds = np.zeros(len(X))
test_class_preds = np.zeros((len(X_test), num_total_classes))
test_reg_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y_class_encoded, groups)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_class_train, y_class_val = y_class_encoded[train_idx], y_class_encoded[val_idx]
    y_reg_train, y_reg_val = y_reg.iloc[train_idx], y_reg.iloc[val_idx]
    
    # Train and predict for classification
    clf = train_lgb_model(X_train, y_class_train, X_val, y_class_val, class_params)
    oof_class_preds[val_idx] = np.argmax(clf.predict(X_val, num_iteration=clf.best_iteration), axis=1)
    test_class_preds += clf.predict(X_test, num_iteration=clf.best_iteration) / gkf.n_splits
    
    # Train and predict for regression
    reg = train_lgb_model(X_train, y_reg_train, X_val, y_reg_val, reg_params)
    oof_reg_preds[val_idx] = reg.predict(X_val, num_iteration=reg.best_iteration)
    test_reg_preds += reg.predict(X_test, num_iteration=reg.best_iteration) / gkf.n_splits




accuracy = accuracy_score(y_class_encoded, oof_class_preds)
f1 = f1_score(y_class_encoded, oof_class_preds, average='weighted')
mae = mean_absolute_error(y_reg, oof_reg_preds)
y_reg_range = y_reg.max() - y_reg.min()
reg_score = max(0, 1 - (mae / y_reg_range)) if y_reg_range > 0 else 1
final_score = 0.25 * accuracy + 0.25 * f1 + 0.5 * reg_score

print(f"--- Validation Scores ---\n"
      f"Accuracy:       {accuracy:.4f}\n"
      f"F1 Score:       {f1:.4f}\n"
      f"MAE:            {mae:.4f}\n"
      f"Regression Score: {reg_score:.4f}\n"
      f"Final Score:      {final_score:.4f}\n")




final_class_preds = class_encoder.inverse_transform(np.argmax(test_class_preds, axis=1))
final_reg_preds = np.round(np.maximum(0, test_reg_preds)).astype(int)

submission = pd.DataFrame({
    'id': test_df['id'],
    'MasterItemNo': final_class_preds,
    'QtyShipped': final_reg_preds
})

submission['MasterItemNo'] = submission['MasterItemNo'].astype(int)



assert len(submission) == len(test_df), "Submission length mismatch"
assert list(submission.columns) == ['id', 'MasterItemNo', 'QtyShipped'], "Wrong column names"
assert not submission.isnull().any().any(), "Submission contains null values"

submission.to_csv('submission.csv', index=False)
print("Submission file 'submission.csv' created successfully.")
print("\n--- Sample Predictions ---")
print(submission.head(10))

