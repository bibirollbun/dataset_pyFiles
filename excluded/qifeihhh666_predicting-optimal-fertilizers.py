import pandas as pd 
import numpy as np 
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")


train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
original = pd.read_csv("/kaggle/input/fertilizers-original-dataset/Fertilizer Prediction.csv")


print(train.head())
print(original.head())
print(test.head())


original["Fertilizer Name"].value_counts()


#Basic Replication
def basic_replication(original):
    
 original_copy = original.copy()
    
 for i in range(6):

    original = pd.concat([original, original_copy], axis=0, ignore_index= True)
     
 return original


original = basic_replication(original)

original.head()


    
    


# def create_feature(df):
    
#     df['THI'] = df['Temparature'] - (0.55 - 0.0055 * df['Humidity']) * (df['Temparature'] - 14.5)
    
#     return df


# original = create_feature(original)
# train = create_feature(train)
# test = create_feature(test)

# original.head()


# Binnig feature 

def binning_feature(df):

    #quantile base binning

    numerical_col =  ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Phosphorous', 'Potassium']

    for col in numerical_col :

      # df[f'{col}_cat_bin'] = pd.qcut(df[col], q=5, labels=False, duplicates='drop')

    #simple based binning

      df[f'{col}_cat_bin']= df[col].astype(str)

    return df

    


# train = create_feature(train)
train = binning_feature(train)
# test = create_feature(test)
test = binning_feature(test)
# original = create_feature(original)
original = binning_feature(original)


train.head()
# test.head()


cat_col = [col for col in train.select_dtypes(include=["object",'category']).columns if col != "Fertilizer Name"] + [col for col in train.columns if col.endswith("_cat_bin")]

for col in cat_col:

    encode_col = pd.concat([train[col],test[col],original[col]]).unique()
    lb = LabelEncoder().fit(encode_col)
    train[col] = lb.transform(train[col])
    test[col]= lb.transform(test[col])
    original[col]= lb.transform(original[col])
    


train.head()


#Label enencoding on target coloumn 

label_target = LabelEncoder()

label_target.fit(pd.concat([train["Fertilizer Name"],original["Fertilizer Name"]]))

train["Fertilizer Name"]= label_target.transform(train["Fertilizer Name"])

original["Fertilizer Name"] = label_target.transform(original["Fertilizer Name"])







train.head()


for col in cat_col:
    train[col] = train[col].astype("category")
    test[col] = test[col].astype("category")
    original[col] = original[col].astype("category")



#target coloumn 

x = train.drop(["id","Fertilizer Name"], axis=1)
y = train["Fertilizer Name"]
x_test = test.drop("id",axis=1)
x_original = original.drop(["Fertilizer Name"], axis=1)
y_original = original["Fertilizer Name"]
test_ids = test["id"]


# xgboost best param 

params = {
    'objective': 'multi:softprob',
    'num_class': len(np.unique(y)),
    'max_depth': 7,
    'learning_rate': 0.03,
    'subsample': 0.8,
    'max_bin': 128,
    'colsample_bytree': 0.3,
    'colsample_bylevel': 1,
    'colsample_bynode': 1,
    'tree_method': 'hist',
    'random_state': 42,
    'eval_metric': 'mlogloss',
    'device': 'cuda',
    'enable_categorical': True,
    'n_estimators': 10000,
    'early_stopping_rounds': 50,
}



# map3 mean average precision

def mapk(actual, predicted, k=3):
    def apk(a, p, k):
        p = p[:k]
        score = 0.0
        hits = 0
        seen = set()
        for i, pred in enumerate(p):
            if pred in a and pred not in seen:
                hits += 1
                score += hits / (i + 1.0)
                seen.add(pred)
        return score / min(len(a), k)
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


# ==== K-Fold Cross-Validation ====
FOLDS = 15
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)
xgb_model = XGBClassifier(**params)
oof = np.zeros(shape=(len(train), y.nunique()))
pred_prob = np.zeros(shape=(len(test), y.nunique()))
map3_scores = []




for i, (train_idx, valid_idx) in enumerate(skf.split(x, y)):
    print(f"############### FOLD {i+1} ###############")
    x_train, x_valid = x.iloc[train_idx], x.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

    # Augment training data with original dataset
    x_train = pd.concat([x_train, x_original], axis=0, ignore_index=True)
    y_train = pd.concat([y_train, y_original], axis=0, ignore_index=True)

    # Fit model
    xgb_model.fit(
        x_train,
        y_train,
        eval_set=[(x_train, y_train), (x_valid, y_valid)],
        verbose=100
    )

    # Out-of-fold predictions
    oof[valid_idx] = xgb_model.predict_proba(x_valid)

    # Test predictions (accumulate across folds)
    pred_prob += xgb_model.predict_proba(x_test)

    # Calculate MAP@3 for this fold
    top_3_preds = np.argsort(oof[valid_idx], axis=1)[:, -3:][:, ::-1]
    actual = [[label] for label in y_valid]
    map3_score = mapk(actual, top_3_preds)
    map3_scores.append(map3_score)
    print(f"âœ… FOLD {i+1}: MAP@3 Score: {map3_score:.5f}")



# Average test predictions across folds
pred_prob /= FOLDS

# Average MAP@3 across folds
avg_map3 = np.mean(map3_scores)
print(f"\nðŸŽ¯ Average MAP@3 Score across all folds: {avg_map3:.5f}")

# ==== Generate Top-3 Predictions for Submission ====
top_3_preds = np.argsort(pred_prob, axis=1)[:, -3:][:, ::-1]
top_3_labels = label_target.inverse_transform(top_3_preds.ravel()).reshape(top_3_preds.shape)

# Create submission DataFrame
submission = pd.DataFrame({
    'id': test_ids,
    'Fertilizer Name': [' '.join(row) for row in top_3_labels]
})

# Save submission file
submission.to_csv('submission.csv', index=False)
print("âœ… Submission file saved as 'submission_xgboost_kfold_top3.csv'")

# Save OOF and test predictions
np.save('xgb_repeat_train_oof.npy', oof)
np.save('xgb_repeat_test_oof.npy', pred_prob)
print("âœ… OOF and test predictions saved as 'xgb_repeat_train_oof.npy' and 'xgb_repeat_test_oof.npy'")

# Display first few rows of submission
print("\nðŸ“„ Submission Preview:")
print(submission.head())

