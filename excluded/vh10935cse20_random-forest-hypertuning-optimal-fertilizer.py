import pandas as pd
import numpy as np


train=pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test=pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')


x=train.drop(['Fertilizer Name'],axis=1)
y=train['Fertilizer Name']


from sklearn.model_selection import RandomizedSearchCV,train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


model=LabelEncoder()
y_enc=model.fit_transform(y)


le=LabelEncoder()
cat_cols=['Soil Type','Crop Type']
for i in cat_cols:
    x[i]=le.fit_transform(x[i])
    test[i]=le.transform(test[i])











x_train,x_val,y_train,y_val=train_test_split(x,y_enc,test_size=0.2,random_state=42)


rf=RandomForestClassifier(random_state=42)


from scipy.stats import randint


param_dist = {
    'n_estimators': randint(50, 200),  # Example: 50 to 199 trees
    'max_depth': randint(5, 30),      # Example: Max depth from 5 to 29
    'min_samples_split': randint(2, 10) # Example: Min samples to split from 2 to 9
    # Add more Random Forest parameters as needed
}


rs=RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_jobs=-1,
    random_state=42,
    cv=5,
    n_iter=5)


rs.fit(x_train,y_train)


rs.best_score_


y_probs=rs.predict_proba(x_val)


def get_top_k_predictions(probs, k):
    return np.argsort(probs, axis=1)[:, -k:][:, ::-1]


# Single-label MAP@K
def mapk_single_label(y_true, y_pred, k=3):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)[:, :k]
    matches = (y_true.reshape(-1, 1) == y_pred)
    ranks = np.where(matches.any(axis=1), matches.argmax(axis=1) + 1, np.inf)
    return np.mean(ranks ** -1)


# Multi-label MAP@K (each instance has one label in a list)
def apk(actual, predicted, k=10):
    if not actual:
        return 0.0
    predicted = predicted[:k]
    score = 0.0
    num_hits = 0
    seen = set()
    actual_set = set(actual)
    for i, p in enumerate(predicted):
        if p in actual_set and p not in seen:
            num_hits += 1
            score += num_hits / (i + 1)
            seen.add(p)
    return score / min(len(actual), k)
def mapk(actual, predicted, k=10):
    return np.mean([apk([a], p, k) for a, p in zip(actual, predicted)])


#Evaluate MAP@K for k = 1 to k_max
k_max = 8
k_values = range(1, k_max)
mapk_single_scores = []
mapk_multi_scores = []


test_proba = rs.predict_proba(test)



preds = np.argsort(test_proba, axis=1)[:, ::-1]
preds


test_top_3 = np.argsort(test_proba, axis=1)[:, -3:][:, ::-1]
test_top_3


test_top_3_names = model.inverse_transform(test_top_3.ravel())
test_3_picks = test_top_3_names.reshape(test_top_3.shape)

test_3_picks


preds_df = pd.DataFrame({
    'id': test['id'],
    'Fertilizer Name': [' '.join(preds) for preds in test_3_picks]
})



preds_df.to_csv('/kaggle/working/submission.csv', index=False)

