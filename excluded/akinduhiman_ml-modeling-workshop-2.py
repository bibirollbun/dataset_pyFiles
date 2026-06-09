import pandas as pd

train = pd.read_csv('/kaggle/input/s4e12-preproccesed/s4e12_train_preproccesed.csv')
test = pd.read_csv('/kaggle/input/s4e12-preproccesed/s4e12_test_preproccesed.csv')

train = train.drop(columns=['Policy Start Date'], axis=1)  


train


from sklearn.model_selection import train_test_split

X = train.drop(columns=['Premium Amount'], axis=1)  
y = train['Premium Amount']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#When we have Classification task
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("Evaluation Metrics:")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")


def predict(X_train,y_train,X_test,y_test,model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print("Evaluation Metrics:")
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")


from sklearn.tree import DecisionTreeRegressor

tree_model = DecisionTreeRegressor(max_depth=2) 
predict(X_train, y_train, X_test, y_test, tree_model)


from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(n_estimators=10, random_state=42)
predict(X_train, y_train, X_test, y_test, rf_model)


from sklearn.ensemble import GradientBoostingRegressor

gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)
predict(X_train, y_train, X_test, y_test, gb_model)


import pandas as pd

train = pd.read_csv('/kaggle/input/playground-series-s4e6/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s4e6/test.csv')


train['Target'] = train['Target'].replace('Dropout',0)
train['Target'] = train['Target'].replace('Enrolled',1)
train['Target'] = train['Target'].replace('Graduate',2)


X = train.drop(columns=['Target'], axis=1)  
y = train['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)

y_pred = logistic_model.predict(X_test)
print("\nClassification Report:\n", classification_report(y_test, y_pred))


import seaborn as sns
import matplotlib.pyplot as plt

labels = ['Dropout', 'Enrolled','Graduate']

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


def predict(X_train, y_train, X_test, y_test, model):

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    
    cm = confusion_matrix(y_test, y_pred)

    labels = ['Dropout', 'Enrolled','Graduate']
    
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()


from sklearn.tree import DecisionTreeClassifier

tree_model = DecisionTreeClassifier(max_depth=10, random_state=42)

predict(X_train, y_train, X_test, y_test, tree_model)


from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

predict(X_train, y_train, X_test, y_test, rf_model)


from sklearn.ensemble import GradientBoostingClassifier

gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)

predict(X_train, y_train, X_test, y_test, gb_model)


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

overfit_model = DecisionTreeClassifier(max_depth=40)
overfit_model.fit(X_train, y_train)
train_acc = accuracy_score(y_train, overfit_model.predict(X_train))
test_acc = accuracy_score(y_test, overfit_model.predict(X_test))
print("Overfitting Example - Training Accuracy:", train_acc, "Test Accuracy:", test_acc)

underfit_model = DecisionTreeClassifier(max_depth=1)
underfit_model.fit(X_train, y_train)
train_acc = accuracy_score(y_train, underfit_model.predict(X_train))
test_acc = accuracy_score(y_test, underfit_model.predict(X_test))
print("Underfitting Example - Training Accuracy:", train_acc, "Test Accuracy:", test_acc)

normal_model = DecisionTreeClassifier(max_depth=15)
normal_model.fit(X_train, y_train)
train_acc = accuracy_score(y_train, normal_model.predict(X_train))
test_acc = accuracy_score(y_test, normal_model.predict(X_test))
print("Normal Example - Training Accuracy:", train_acc, "Test Accuracy:", test_acc)



from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

dt_model = DecisionTreeClassifier(random_state=42)

param_grid = {
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy']
}

grid_search = GridSearchCV(estimator=dt_model, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation Accuracy:", grid_search.best_score_)

best_model = grid_search.best_estimator_
test_accuracy = best_model.score(X_test, y_test)
print("Test Accuracy:", test_accuracy)


final_model = DecisionTreeClassifier(max_depth=10, min_samples_leaf = 4, min_samples_split = 10,criterion = 'entropy' )
final_model.fit(X, y)
predictions = final_model.predict(test)
predictions


submission = pd.read_csv('/kaggle/input/playground-series-s4e6/sample_submission.csv')
submission


submission['Target'] = predictions
submission


submission['Target'] = submission['Target'].replace(0,'Dropout')
submission['Target'] = submission['Target'].replace(1,'Enrolled')
submission['Target'] = submission['Target'].replace(2,'Graduate')


submission.to_csv("submission.csv", index=False)


from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

xgb_model = XGBClassifier(random_state=42)
cv = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
val_predictions = pd.DataFrame()
test_predictions = pd.DataFrame()
fold_preds = []

for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
    X_train_fold, y_train_fold = X_train.iloc[train_idx], y_train.iloc[train_idx]
    X_val_fold, y_val_fold = X_train.iloc[val_idx], y_train.iloc[val_idx]
    
    xgb_model.fit(X_train_fold, y_train_fold)
    off = xgb_model.predict(X_val_fold)
    off_acc = accuracy_score(y_val_fold, off)
    print(f"fold {fold+1} acc : {off_acc}")
    test_preds = xgb_model.predict(test)
    test_predictions[f'fold_{fold + 1}'] = test_preds


test_predictions['final_predictions'] = test_predictions.mode(axis=1)[0]


test_predictions


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

log_reg = LogisticRegression(max_iter=200, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

log_reg.fit(X_train, y_train)
test_predictions = log_reg.predict(X_test)

accuracy = accuracy_score(y_test, test_predictions)
print(f"Test Accuracy: {accuracy:.4f}")

preds = log_reg.predict(test)


from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

log_reg = LogisticRegression(max_iter=200, random_state=42)
adaboost_model = AdaBoostClassifier(base_estimator=log_reg, n_estimators=50, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

adaboost_model.fit(X_train, y_train)
test_predictions = adaboost_model.predict(X_test)

accuracy = accuracy_score(y_test, test_predictions)
print(f"Test Accuracy: {accuracy:.4f}")

preds = adaboost_model.predict(test)


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

xgb_model = XGBClassifier(random_state=42)
lgbm_model = lgb.LGBMClassifier(random_state=42)
catboost_model = CatBoostClassifier(silent=True, random_state=42)

xgb_model.fit(X_train, y_train)
lgbm_model.fit(X_train, y_train)
catboost_model.fit(X_train, y_train)

xgb_test_preds = xgb_model.predict(X_test)
lgbm_test_preds = lgbm_model.predict(X_test)
catboost_test_preds = catboost_model.predict(X_test).ravel()
accuracy = accuracy_score(y_test, xgb_test_preds)
print(f"XGB Test Accuracy: {accuracy:.4f}")
accuracy = accuracy_score(y_test, lgbm_test_preds)
print(f"LGBM Test Accuracy: {accuracy:.4f}")
accuracy = accuracy_score(y_test, catboost_test_preds)
print(f"CATB Test Accuracy: {accuracy:.4f}")


train_meta_features = pd.DataFrame({
    'xgb': xgb_test_preds,
    'lgbm': lgbm_test_preds,
    'catboost': catboost_test_preds
})

meta_model = LogisticRegression()

meta_model.fit(train_meta_features, y_test)


xgb_preds = xgb_model.predict(test)
lgbm_preds = lgbm_model.predict(test)
catboost_preds = catboost_model.predict(test).ravel() 

test_meta_features = pd.DataFrame({
    'xgb': xgb_preds,
    'lgbm': lgbm_preds,
    'catboost': catboost_preds
})

final_predictions = meta_model.predict(test_meta_features)
final_predictions

