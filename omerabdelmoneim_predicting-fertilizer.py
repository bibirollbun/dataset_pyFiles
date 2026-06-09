import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import os


train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv",index_col = 'id')
test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv",index_col = 'id')
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv",index_col = 'id')


label_encoder = LabelEncoder().fit(train['Fertilizer Name'])
X = train.drop(columns = 'Fertilizer Name')
y = label_encoder.transform(train['Fertilizer Name'])
classes = label_encoder.classes_
encoded_classes = label_encoder.transform(classes)


# Custom scoring function for MAP@3
def map3_score(y_true, y_pred_proba):
    """
    Calculate MAP@3 for classification problems
    y_true: array of true labels
    y_pred_proba: 2D array of predicted probabilities (n_samples × n_classes)
    """
    # Get top 3 predicted classes for each sample
    top3_indices = np.argsort(-y_pred_proba, axis=1)[:, :3]
    top3_preds = np.take(encoded_classes, top3_indices)
    # Calculate AP@3 for each sample
    aps = []
    for true_label, preds in zip(y_true, top3_preds):
        ap = 0.0
        correct_found = 0
        
        for k, pred in enumerate(preds, 1):
            if pred == true_label:
                correct_found += 1
                ap += correct_found / k
        
        aps.append(ap)
    
    return np.mean(aps)


estimator_last_score = {}
def eval_model(estimator, X, Y, n_splits=4, estimator_key=None, n_jobs=1, print_result=True):
    map3_scorer = make_scorer(map3_score, needs_proba=True)
    result = cross_validate(estimator, X, Y, cv=n_splits,
                            scoring=map3_scorer, n_jobs=n_jobs)
    scores = result['test_score']
    mean_score = np.mean(scores)
    median_score = np.median(scores)
    if print_result:
        print(f"\n=== Evaluation Results [{estimator_key}] ===")
        print(f"Mean MAP@3: {mean_score:.4f}")
        print(f"Median MAP@3: {median_score:.4f}")
        print(f"Std: MAP@3: {np.std(scores)}")
        print(f"All scores: {np.round(scores, 4)}")
    if print_result and estimator_key:
        if not estimator_key in estimator_last_score:
            print("first evaluation")
        elif estimator_key in estimator_last_score:
            last_mean, last_median = estimator_last_score[estimator_key]
            mean_diff = mean_score - last_mean
            median_diff = median_score - last_median
            
            print("\n=== Performance Change ===")
            print(f"Mean change: {mean_diff:+.4f} ({'improved' if mean_diff > 0 else 'degraded' if mean_diff < 0 else 'no change'})")
            print(f"Median change: {median_diff:+.4f} ({'improved' if median_diff > 0 else 'degraded' if median_diff < 0 else 'no change'})")
        else:
            print("\n(First evaluation of this estimator)")
    
    if estimator_key:
        estimator_last_score[estimator_key] = (mean_score, median_score)
    return result


def make_submission(model,X=X,y=y,test = test):
    model.fit(X,y)
    proba = model.predict_proba(test)
    top3_indices = np.argsort(-proba, axis=1)[:, :3]
    top3_preds = np.take(classes, top3_indices)
    pred_strings = [' '.join(map(str, preds)) for preds in top3_preds]
    submission = pd.DataFrame({
        'id': test.index,
        'Fertilizer Name': pred_strings
    })
    submission.to_csv("submission.csv", index=False)
    return submission


cat_columns = ['Soil Type', 'Crop Type']
encoder_step =  ('preprocessing', ColumnTransformer([
        ('cat', OneHotEncoder(),cat_columns)
    ], remainder='passthrough'))


from xgboost import XGBClassifier
xgb = Pipeline([
    encoder_step,
    ('clf', XGBClassifier(n_jobs=-1,
                          random_state=0,
                          objective='multi:softprob',
                          eval_metric='mlogloss',
                         ))
])
eval_model(xgb, X, y,n_jobs = -1, estimator_key='xgb')


from sklearn.ensemble import AdaBoostClassifier
ada = Pipeline([
    encoder_step,
    ('clf', AdaBoostClassifier(random_state=0,))])
eval_model(ada, X, y,n_jobs = -1, estimator_key='ada')


from sklearn.ensemble import RandomForestClassifier
forest = Pipeline([
    encoder_step,
    ('clf', RandomForestClassifier(random_state=0,n_jobs=-1))])
eval_model(forest, X, y,n_jobs = -1, estimator_key='forest')


from sklearn.ensemble import StackingClassifier
stack = StackingClassifier([('forest',forest),('xgb',xgb),('ada',ada)])
eval_model(stack, X, y,n_jobs = 1, estimator_key='stack')


make_submission(stack)

