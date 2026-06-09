


import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


# 1. Load and preprocess the data
data0 = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
TEST0 = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')


# Label encoding for target variable
class_names=sorted(data0['Personality'].unique().tolist())
print(class_names)
N=list(range(len(class_names)))
normal_mapping=dict(zip(class_names,N)) 
reverse_mapping=dict(zip(N,class_names))     
data0['Personality'] = data0['Personality'].map(normal_mapping)


from sklearn.preprocessing import LabelEncoder

def labelencoder(df):
    for c in df.columns:
        if df[c].dtype=='object': 
            df[c] = df[c].fillna('N')
            lbl = LabelEncoder()
            lbl.fit(list(df[c].values))
            df[c] = lbl.transform(df[c].values)
    return df
    
data=labelencoder(data0)
TEST=labelencoder(TEST0)


from sklearn.model_selection import train_test_split
train, test = train_test_split(data, test_size=0.2, random_state=42)


train.info()


y = train['Personality']
y_test=test['Personality']


# Create empty array for OOF and test predictions (3 models)
oof_preds = np.zeros((len(data)*4//5, 3))
test_preds = np.zeros((len(TEST), 3))

# LGBM
oof1 = np.load('/kaggle/input/introverts-lgbm-for-oof/oof.npy')[:, 1]
test_pred1 = np.load("/kaggle/input/introverts-lgbm-for-oof/FTEST_pred.npy")[:, 1]
oof_preds[:, 0] = oof1
test_preds[:, 0] = test_pred1

# CatBoost
oof2 = np.load('/kaggle/input/introverts-catboost-for-oof/oof.npy')[:, 1]
test_pred2 = np.load("/kaggle/input/introverts-catboost-for-oof/FTEST_pred.npy")[:, 1]
oof_preds[:, 1] = oof2
test_preds[:, 1] = test_pred2

# XGBoost
oof3 = np.load('/kaggle/input/introverts-xgboost-for-oof/oof.npy')[:, 1]
test_pred3 = np.load("/kaggle/input/introverts-xgboost-for-oof/FTEST_pred.npy")[:, 1]
oof_preds[:, 2] = oof3
test_preds[:, 2] = test_pred3


# Train meta model (stacked model)
meta_model = LogisticRegression()
meta_model.fit(oof_preds, y)

# Evaluate on OOF
oof_score = roc_auc_score(y, meta_model.predict_proba(oof_preds)[:, 1])
print(f"Meta Model OOF AUC: {oof_score:.5f}")
print(f"Model Weights: {meta_model.coef_}")

# Predict on test set
final_preds = meta_model.predict_proba(test_preds)[:, 1]
final_labels = (final_preds > 0.5).astype(int)


x_test = []

test_pred1=np.load('/kaggle/input/introverts-lgbm-for-oof/test_pred.npy')
test_pred2=np.load('/kaggle/input/introverts-catboost-for-oof/test_pred.npy')
test_pred3=np.load('/kaggle/input/introverts-xgboost-for-oof/test_pred.npy')

x_test.append(test_pred1)
x_test.append(test_pred2)
x_test.append(test_pred3)



def ensemble_predict(weights, predictions):
    """
    Perform weighted ensemble prediction.
    Args:
        weights: Model weights (shape: n_models)
        predictions: Predictions from each model (shape: n_models, n_samples, n_classes)
    Returns:
        Ensemble prediction (shape: n_samples, n_classes)
    """
    ensemble_pred = np.zeros(predictions[0].shape)
    for i, weight in enumerate(weights):
        ensemble_pred += weight * predictions[i]
    return ensemble_pred


weights=(meta_model.coef_/(meta_model.coef_).sum())[0]
print(weights)


# Generate test predictions using the best weights
final_test_pred = ensemble_predict(weights, x_test)
final_test_pred_class = np.argmax(final_test_pred, axis=1)

print(f"Shape of ensemble predictions: {final_test_pred.shape}")
print(f"Distribution of predicted classes: {np.bincount(final_test_pred_class)}")


y_true=y_test
y_pred=final_test_pred_class
print(len(y_test))


from sklearn.metrics import classification_report
print(classification_report(y_true, y_pred, target_names=None, digits=4))


# Create submission file
submit = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')
submit['Personality']=final_labels
submit['Personality']=submit['Personality'].map(reverse_mapping)
display(submit)
submit.to_csv('submission.csv',index=False)




