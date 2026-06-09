import pandas as pd 
import numpy as np 
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv',index_col='id')
test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv",index_col = 'id')


target = 'Fertilizer Name'
cat_columns = [i for i in train.columns if train[i].dtype == np.object_][:-1]
num_columns = [i for i in train.columns if i not in cat_columns]
label_enc = LabelEncoder()
ordinal_enc = OrdinalEncoder(handle_unknown='error')
train[cat_columns] = ordinal_enc.fit_transform(train[cat_columns])
test[cat_columns] = ordinal_enc.transform(test[cat_columns])
train[cat_columns] = train[cat_columns].astype('category')
test[cat_columns] = test[cat_columns].astype('category')
train['Fertilizer Name'] = label_enc.fit_transform(train['Fertilizer Name'])
train['const'] = 1
test['const'] =1



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


X = train.drop(target, axis = 1)
y = train[target]


oof_ori_1 = np.load('/kaggle/input/xgb-4-model-from-22-06-2025/oof_1_with_org.npy')
oof_ori_2 = np.load('/kaggle/input/xgb-4-model-from-22-06-2025/oof_2_with_org.npy')
oof_lgbgoss_6 = np.load('/kaggle/input/models-lgb/lgb_goss_oof.npy')
oof_1 = np.load('/kaggle/input/model-xgb-240625/oof_240625_1.npy')
oof_2 = np.load('/kaggle/input/xgb-2-8-re-2406/oof_240625_2.npy')
oof_stack = np.load('/kaggle/input/lr-stack/stack_oof.npy')
oof_14 = np.load('/kaggle/input/oof-public-code/ensemble_oof.npy')
oof_15 = np.load('/kaggle/input/oof-public-code/lgb_oof_predictions.npy')
true = y


pred_prob_ori_1 = np.load('/kaggle/input/xgb-4-model-from-22-06-2025/pred_prob_1_with_org.npy')
pred_prob_ori_2 = np.load('/kaggle/input/xgb-4-model-from-22-06-2025/pred_prob_2_with_org.npy')
pred_prob_lgbgoss_6 = np.load('/kaggle/input/models-lgb/lgb_goss_test.npy')
pred_prob_1 = np.load('/kaggle/input/model-xgb-240625/pred_240625_1.npy')
pred_prob_2 = np.load('/kaggle/input/xgb-2-8-re-2406/pred_240625_2.npy')
pred_prob_stack = np.load('/kaggle/input/lr-stack/pred_prob_stack.npy')
pred_prob_14 = np.load('/kaggle/input/oof-public-code/ensemble_pred.npy')
pred_prob_15 = np.load('/kaggle/input/oof-public-code/lgb_test_predictions.npy')



X_meta = np.column_stack([ oof_ori_1, oof_ori_2, oof_15, oof_lgbgoss_6, oof_1,oof_2,oof_14,oof_stack])
x_test = np.column_stack([pred_prob_ori_1,pred_prob_ori_2,pred_prob_15,pred_prob_lgbgoss_6,pred_prob_1,pred_prob_2,pred_prob_14,pred_prob_stack])



skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_stack = np.zeros(shape = (len(train), y.nunique()))
pred_prob_stack = np.zeros(shape= (len(test), y.nunique()))
print("\nStarting 5-Fold LR training...")
for i, (train_idx, valid_idx) in enumerate(skf.split(X_meta, y)):
    lr_model = LogisticRegression(**{
        'C': 1.436289965798556,
        'tol': 0.05692806752309682,
        'penalty': 'l2',
        'solver': 'newton-cholesky',
        'max_iter': 1001,'fit_intercept':True}
        
    )

    x_train,x_valid = X_meta[train_idx], X_meta[valid_idx]
    y_train,y_valid = y.iloc[train_idx],y.iloc[valid_idx]

    lr_model.fit(x_train, y_train)

    oof_stack[valid_idx] = lr_model.predict_proba(x_valid)
    pred_prob_stack +=lr_model.predict_proba(x_test) / skf.get_n_splits()
    actual = [[label] for label in y_valid]
    top_3_preds = np.argsort(oof_stack[valid_idx], axis=1)[:, -3:][:, ::-1]
    map3_score = mapk(actual, top_3_preds)
    print(f"âœ… FOLD {i+1}: MAP@3  Score: {map3_score:.5f}")


actual = [[label] for label in y]

top_3_preds_1 = np.argsort(oof_stack, axis=1)[:, -3:][:, ::-1]  
map3_score_1 = mapk(actual, top_3_preds_1)
print(f'âœ… Final  MAP@3 Score: {map3_score_1:.5f}')


top_3_preds = np.argsort(pred_prob_stack, axis=1)[:, -3:][:, ::-1]
top_3_labels = label_enc.inverse_transform(top_3_preds.ravel()).reshape(top_3_preds.shape)
df_sub = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")
submission = pd.DataFrame({
    'id': df_sub['id'],
    'Fertilizer Name': [' '.join(row) for row in top_3_labels]
})
submission.to_csv('submission.csv', index=False)
print("âœ… Submission file saved as 'submission.csv'")


submission.head()

