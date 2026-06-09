import numpy as np
import pandas as pd
%matplotlib inline
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


df_train = pd.read_csv('/kaggle/input/forest-cover-type-prediction/train.csv')
df_test = pd.read_csv('/kaggle/input/forest-cover-type-prediction/test.csv')


df_train.head().T


df_test.head().T


df_train['Cover_Type'].value_counts()


X_train, X_valid, y_train, y_valid = train_test_split(
    df_train.drop('Cover_Type', axis=1),df_train['Cover_Type'],
    test_size=0.3, random_state=17)


logit = LogisticRegression(C=1, solver='lbfgs', max_iter=500,
                           random_state=17, n_jobs=4,
                          multi_class='multinomial')
logit_pipe = Pipeline([('scaler', StandardScaler()), 
                       ('logit', logit)])


logit_pipe.fit(X_train, y_train)


logit_val_pred = logit_pipe.predict(X_valid)


accuracy_score(y_valid, logit_val_pred)


first_forest = RandomForestClassifier(
    n_estimators=100, random_state=17, n_jobs=4)


first_forest.fit(X_train, y_train)


forest_val_pred = first_forest.predict(X_valid)
accuracy_score(y_valid, forest_val_pred)


pd.DataFrame(first_forest.feature_importances_,
             index=X_train.columns, columns=['Importance']).sort_values(
    by='Importance', ascending=False)[:10]


param_grid = {
    'n_estimators': [100, 300, 500],
    'max_depth': [None, 20],  
}

rf = RandomForestClassifier(
    random_state=42,
    n_jobs=-1,            
)

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,                   
    scoring='accuracy',     
    verbose=0,              
    n_jobs=-1,                            
)

grid_search.fit(X_train, y_train)

print("Лучшие параметры")
print(grid_search.best_params_)
print(f"\nЛучшая точность: {grid_search.best_score_:.4f}")


best_params = grid_search.best_params_

best_model = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    n_jobs=-1,
    random_state=42,
    verbose=0
)

best_model.fit(df_train.drop('Cover_Type', axis=1), df_train['Cover_Type'])

test_predictions = best_model.predict(df_test)

submission = pd.DataFrame({
    "Id": df_test["Id"],
    "Cover_Type": test_predictions.astype(int)
})
submission.to_csv("submission.csv", index=False)




