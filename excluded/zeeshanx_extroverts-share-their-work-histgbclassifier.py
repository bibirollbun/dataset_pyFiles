import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import datasets

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


train_data = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv', index_col='id')
train_data[:2]


test_data = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv', index_col='id')
X = train_data.drop(['Personality'], axis=1)
y = train_data['Personality']
print("Train Data: ", train_data.shape)
print("Test Data: ", test_data.shape)


train_data.info()


train_data.isna().sum()


import missingno as msno

msno.matrix(train_data)
plt.show()
msno.heatmap(train_data)
plt.show()



from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.compose import ColumnTransformer


print("Features Shape: ", X.shape)
print("Target Shape: ", y.shape)

num_cols = X.select_dtypes(include='float').columns.tolist()
cat_cols = X.select_dtypes(include='object').columns.tolist()


num_imputer = IterativeImputer(random_state=0)
cat_imputer = SimpleImputer(strategy='most_frequent')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_imputer, num_cols),
        ('cat', cat_imputer, cat_cols)
    ]
)


from sklearn.pipeline import Pipeline
from catboost import CatBoostClassifier

cat_feature_indices = list(range(len(num_cols), len(num_cols) + len(cat_cols)))

clf = Pipeline(
    steps=[
        ('preprocess', preprocessor),
        ('model', CatBoostClassifier(verbose=0))
    ]
)

clf.fit(X, y, model__cat_features=cat_feature_indices)
# clf.fit(X, y, model__cat_features=cat_cols)
print("CatBoost model trained successfully.")


test_data[cat_cols] = test_data[cat_cols].astype('category')
test_preds = clf.predict(test_data)

submission = pd.DataFrame({'id': test_data.index,'Personality': test_preds})

submission.to_csv('submission.csv', index=False)
print("Predictions saved to submission.csv")


from sklearn.ensemble import HistGradientBoostingClassifier

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_imputer, num_cols),
        # ('cat', cat_imputer, cat_cols)
    ]
)
X[cat_cols] = X[cat_cols].astype('category')

clf = Pipeline(
    steps=[
        ('preprocess', preprocessor),
        ('model', HistGradientBoostingClassifier())
    ]
)

clf.fit(X, y)
print("HistGBClassifier model trained successfully.")


from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, cross_val_score

clf = Pipeline(
    steps=[
        ('preprocess', preprocessor),
        ('model', HistGradientBoostingClassifier())
    ]
)

param_grid = {
    'model__learning_rate': [0.01, 0.05, 0.1],
    'model__max_iter': [100, 200],
    'model__max_leaf_nodes': [15, 31, 63],
    'model__l2_regularization': [0.0, 0.1, 1.0]
}

grid = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
grid.fit(X, y)

print("Best parameters:", grid.best_params_)
print("Best cross-val accuracy:", grid.best_score_)


best_model = grid.best_estimator_

# Predict on test set
test_preds = best_model.predict(test_data)

submission = pd.DataFrame({
    'id': test_data.index,
    'Personality': test_preds
})
submission.to_csv('submission.csv', index=False)
print("Predictions saved to submissions.csv")


test_data[cat_cols] = test_data[cat_cols].astype('category')
test_preds = best_model.predict(test_data) # Change this according to model used

submission = pd.DataFrame({'id': test_data.index,'Personality': test_preds})

submission.to_csv('submission.csv', index=False)
print("Predictions saved to submission.csv")

