import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn as sk
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier



import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")

train_data["isMale"] = (train_data["Sex"] == "male").astype(int)
noage_df = train_data[train_data.isna().any(axis=1)]
train_data = train_data.drop(["Cabin", "Embarked", "Sex"], axis = 1)
train_data = train_data.dropna()
train_data.head()

X = train_data[["Pclass" ,"isMale" , "Age" , "SibSp" , "Parch"]].values
Y = train_data["Survived"].values

print(len(test_data))



scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)

Xtr, Xte, ytr, yte = train_test_split(X_scaled, Y, test_size=0.2, stratify=Y, random_state=42)


param_distributions = {
    'max_iter': [50, 100, 150],
    'max_leaf_nodes': [15, 31, 63, 127],
    'max_depth': [None, 5, 10, 15],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'min_samples_leaf': [20, 50, 100],
    'l2_regularization': [0, 0.1, 1.0],
    'max_bins': [128, 255]
}


rand_search = RandomizedSearchCV(
    estimator=HistGradientBoostingClassifier(random_state=42),
    param_distributions=param_distributions,
    n_iter=50,
    scoring='neg_log_loss',
    cv=5,
    random_state=42
)

rand_search.fit(Xtr, ytr)
print("Best Params:", rand_search.best_params_)
print("Best Log Loss:", -rand_search.best_score_)

#Gradient Boosting Classifier

Hist_tree = HistGradientBoostingClassifier(min_samples_leaf = 20, max_leaf_nodes = 63,max_iter = 50, max_depth = 5,max_bins = 128, learning_rate = 0.05,l2_regularization = 1.0,   random_state = 42)

Hist_tree.fit(Xtr,ytr)

model_predy=Hist_tree.predict(Xte)

print(Hist_tree.score(Xte,yte))

Hist_tree_base = HistGradientBoostingClassifier( random_state = 42)

Hist_tree_base.fit(Xtr,ytr)

model_predy_base=Hist_tree.predict(Xte)

print(Hist_tree_base.score(Xte,yte))

#Random Forest


def evaluate_rf(X_train, X_val, y_train, y_val, n_estimators, max_depth, min_samples_leaf):
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            n_jobs=-1,
            random_state=42
        )
        model.fit(X_train, y_train)
        train_acc = model.score(X_train, y_train)
        val_acc = model.score(X_val, y_val)
        overfit_gap = train_acc - val_acc
        return {
            'model': model,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'overfit_gap': overfit_gap,
            'params': (n_estimators, max_depth, min_samples_leaf)
        }

def prune_best_rf(X_train_full, y_train_full):
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full)

        n_estimators_list = [1, 2, 5, 10, 20, 50, 100, 150]
        max_depth_list = [None, 10, 20, 30, 40]
        min_samples_leaf_list = [1, 3, 5, 10]

        results = []
        for n in n_estimators_list:
            for d in max_depth_list:
                for leaf in min_samples_leaf_list:
                    result = evaluate_rf(X_train, X_val, y_train, y_val, n, d, leaf)
                    results.append(result)
                    print(f"Tested: n={n}, depth={d}, leaf={leaf} , Training Accuracy: {result['train_acc']:.4f}, Validation Accuracy: {result['val_acc']:.4f}, Gap: {result['overfit_gap']:.4f}")

        overfit_threshold = np.percentile([r['overfit_gap'] for r in results], 30)
        candidates = [r for r in results if r['overfit_gap'] <= overfit_threshold]

        best_model = max(candidates, key=lambda r: r['val_acc'], default=None)

        if best_model:
            print(" Best Random Forest:")
            print(f"  Parameters used: n_estimators={best_model['params'][0]}, max_depth={best_model['params'][1]}, min_samples_leaf={best_model['params'][2]}")
            print(f"  Training Accuracy: {best_model['train_acc']:.4f}, Validation Accuracy: {best_model['val_acc']:.4f}, Gap: {best_model['overfit_gap']:.4f}")
            return best_model['model'], results
        else:
            print(" No suitable model found after pruning.")
            return None, results

prune_best_rf(Xtr,ytr)



#No age 

Xa = noage_df[["Pclass" ,"isMale"  , "SibSp" , "Parch"]].values
Ya = noage_df["Survived"].values

scaler = StandardScaler().fit(Xa)
Xa_scaled = scaler.transform(Xa)

Xatr, Xate, yatr, yate = train_test_split(Xa_scaled, Ya, test_size=0.2, stratify=Ya, random_state=42)


param_distributions = {
    'max_iter': [50, 100, 150],
    'max_leaf_nodes': [15, 31, 63, 127],
    'max_depth': [None, 5, 10, 15],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'min_samples_leaf': [20, 50, 100],
    'l2_regularization': [0, 0.1, 1.0],
    'max_bins': [128, 255]
}



def evaluate_rf(X_train, X_val, y_train, y_val, n_estimators, max_depth, min_samples_leaf):
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            n_jobs=-1,
            random_state=42
        )
        model.fit(X_train, y_train)
        train_acc = model.score(X_train, y_train)
        val_acc = model.score(X_val, y_val)
        overfit_gap = train_acc - val_acc
        return {
            'model': model,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'overfit_gap': overfit_gap,
            'params': (n_estimators, max_depth, min_samples_leaf)
        }

def prune_best_rf(X_train_full, y_train_full):
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full)

        n_estimators_list = [1, 2, 5, 10, 20, 50, 100, 150]
        max_depth_list = [None, 10, 20, 30, 40]
        min_samples_leaf_list = [1, 3, 5, 10]

        results = []
        for n in n_estimators_list:
            for d in max_depth_list:
                for leaf in min_samples_leaf_list:
                    result = evaluate_rf(X_train, X_val, y_train, y_val, n, d, leaf)
                    results.append(result)
                    print(f"Tested: n={n}, depth={d}, leaf={leaf} , Training Accuracy: {result['train_acc']:.4f}, Validation Accuracy: {result['val_acc']:.4f}, Gap: {result['overfit_gap']:.4f}")

        overfit_threshold = np.percentile([r['overfit_gap'] for r in results], 30)
        candidates = [r for r in results if r['overfit_gap'] <= overfit_threshold]

        best_model = max(candidates, key=lambda r: r['val_acc'], default=None)

        if best_model:
            print(" Best Random Forest:")
            print(f"  Parameters used: n_estimators={best_model['params'][0]}, max_depth={best_model['params'][1]}, min_samples_leaf={best_model['params'][2]}")
            print(f"  Training Accuracy: {best_model['train_acc']:.4f}, Validation Accuracy: {best_model['val_acc']:.4f}, Gap: {best_model['overfit_gap']:.4f}")
            return best_model['model'], results
        else:
            print(" No suitable model found after pruning.")
            return None, results

prune_best_rf(Xatr,yatr)



noage_df = test_data[test_data["Age"].isna()].copy()
age_df   = test_data[test_data["Age"].notna()].copy()

noage_df["isMale"] = (noage_df["Sex"] == "male").astype(int)
age_df["isMale"]   = (age_df["Sex"] == "male").astype(int)

X_noage = noage_df[["Pclass", "isMale", "SibSp", "Parch"]].values
noage_df["Survived"] = best_model_noage.predict(X_noage)

X_age = age_df[["Pclass", "isMale", "Age", "SibSp", "Parch"]].values
age_df["Survived"] = best_model.predict(X_age)

df3 = pd.concat([noage_df, age_df], ignore_index=True)

print(len(test_data))   
print(len(df3))      


print(df3)

submission = df3[["PassengerId", "Survived"]].copy()

submission["PassengerId"] = submission["PassengerId"].astype(int)

print(submission.head())
print(len(submission))  # should be 310

submission.to_csv("submission.csv", index=False)


