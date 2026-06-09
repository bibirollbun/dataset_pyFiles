!pip install xgboost
!pip install catboost
!pip install lightgbm
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score , confusion_matrix , classification_report


import warnings
warnings.filterwarnings('ignore')




import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


df_train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
df_submission = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')


df_train.shape,df_test.shape


df_train.head()


df_test.head()


df_train.info()
print("\n" * 2)
df_test.info


df_train.describe()




df_test.describe()


# null val in df_train dataset
df_train.isnull().sum()


# null val in df_test dataset
df_test.isnull().sum()


print(df_train['Drained_after_socializing'].value_counts())
print("\n" * 2)
print(df_test['Drained_after_socializing'].value_counts())


print(df_train['Stage_fear'].value_counts())
print("\n" * 2)
print(df_train['Stage_fear'].value_counts())


outliers = []
for feature in df_train.select_dtypes(include = np.number).columns:
    Q1 = df_train[feature].quantile(0.25)
    Q3 = df_train[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - (1.5 * IQR)
    upper_bound = Q3 + (1.5 * IQR)

    if df_train[(df_train[feature] < lower_bound )| (df_train[feature] > upper_bound)].any(axis=None):
        outliers.append(feature)
        df_train[feature] = df_train[feature].clip(lower = lower_bound , upper=upper_bound)

print("outlier:", outliers)


# outlier in test dataset
outliers = []
for feature in df_test.select_dtypes(include = np.number).columns:
    Q1 = df_test[feature].quantile(0.25)
    Q3 = df_test[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - (1.5 * IQR)
    upper_bound = Q3 + (1.5 * IQR)

    if df_test[(df_test[feature] < lower_bound )| (df_test[feature] > upper_bound)].any(axis=None):
        outliers.append(feature)
        df_test[feature] = df_test[feature].clip(lower = lower_bound , upper=upper_bound)

print("outlier:", outliers)


num_col = df_train.select_dtypes(include = 'number').columns
for col in num_col:
    df_train[col].fillna(df_train[col].median(),inplace=True)


# impute teh test dataset
num_col = df_test.select_dtypes(include = 'number').columns
for col in num_col:
    df_test[col].fillna(df_test[col].median(),inplace=True)


# impute categorical dataset in train dataset
cat_col = df_train.select_dtypes(include = 'object').columns
for col in cat_col:
    df_train[col].fillna(df_train[col].mode()[0],inplace = True)


# categorical dataset in test
cat_col = df_test.select_dtypes(include = 'object').columns
for col in cat_col:
    df_test[col].fillna(df_test[col].mode()[0],inplace = True)


df_train.isnull().sum()


df_test.isnull().sum()
df_test.head()


df_train['Drained_after_socializing'] = df_train['Drained_after_socializing'].replace({'Yes': 1, 'No': 0})
df_test['Drained_after_socializing'] = df_test['Drained_after_socializing'].replace({'Yes': 1, 'No': 0})


df_train['Stage_fear'] = df_train['Stage_fear'].replace({'Yes': 1, 'No': 0})
df_test['Stage_fear'] = df_test['Stage_fear'].replace({'Yes': 1, 'No': 0})


df_train['Personality'] = df_train['Personality'].replace({'Extrovert': 1, 'Introvert': 0})


df_train.head()


df_train["Avoidance_score"] = (df_train["Time_spent_Alone"] + df_train["Drained_after_socializing"] - df_train["Social_event_attendance"] - df_train["Going_outside"])
df_test["Avoidance_score"] = (df_test["Time_spent_Alone"] + df_test["Drained_after_socializing"] - df_test["Social_event_attendance"] - df_test["Going_outside"])


df_train["Social_boldness_score"] = (df_train["Social_event_attendance"] + df_train["Going_outside"] - df_train["Stage_fear"])
df_test["Social_boldness_score"] = (df_test["Social_event_attendance"] + df_test["Going_outside"] - df_test["Stage_fear"])


df_train["Extroversion_index"] = df_train["Social_event_attendance"] + df_train["Going_outside"]
df_test["Extroversion_index"] = df_test["Social_event_attendance"] + df_test["Going_outside"]


df_train["Introversion_index"] = df_train["Time_spent_Alone"] + df_train["Drained_after_socializing"]
df_test["Introversion_index"] = df_test["Time_spent_Alone"] + df_test["Drained_after_socializing"]


df_train["Confidence_gap"] = df_train["Social_boldness_score"] - df_train["Stage_fear"]
df_test["Confidence_gap"] = df_test["Social_boldness_score"] - df_test["Stage_fear"]


df_train["Social_anxiety_score"] = df_train["Going_outside"] * df_train["Stage_fear"]
df_test["Social_anxiety_score"] = df_test["Going_outside"] * df_test["Stage_fear"]


print(df_train['Personality'].value_counts())


df_train.head()


X = df_train.drop(['id','Personality'], axis = 1)
y = df_train['Personality']
X_test = df_test.drop(['id'],axis = 1)



# scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)


X_train , X_val , y_train , y_val = train_test_split(X,y, test_size = 0.3, random_state = 42,stratify=y)


model = LogisticRegression(random_state = 42 , class_weight = 'balanced', penalty ='l2', C= 1.0 ,
solver = 'liblinear' , max_iter = 10000 , fit_intercept = True, warm_start = True )



model.fit(X_train,y_train)


y_pred_train = model.predict(X_train)


accuracy = accuracy_score(y_train,y_pred_train)
print('Accuracy of training data = ', accuracy)
print("Classification Report: \n", classification_report(y_train, y_pred_train))
print("Confusion Matrix: \n", confusion_matrix(y_train, y_pred_train))


y_val_pred_logistic = model.predict(X_val)
validation_data_accuracy = accuracy_score(y_val, y_val_pred_logistic)
print("\n\nAccuracy on testing data = ", validation_data_accuracy)
print("Classification Report: \n", classification_report(y_val, y_val_pred_logistic))
print("Confusion Matrix: \n", confusion_matrix(y_val, y_val_pred_logistic))


rf = RandomForestClassifier(n_estimators=6000, random_state=42, class_weight='balanced', max_depth=15, min_samples_split=6, min_samples_leaf=2, max_features='sqrt', bootstrap=True)
rf.fit(X_train, y_train)


rf = RandomForestClassifier(n_estimators=6000, random_state=42, class_weight='balanced', max_depth=15, min_samples_split=6, min_samples_leaf=2, max_features='sqrt', bootstrap=True)
rf.fit(X_train, y_train)

y_train_prediction = rf.predict(X_train)
training_data_accuracy = accuracy_score(y_train, y_train_prediction)
print("Accuracy on training data = ", training_data_accuracy)
print("Classification Report: \n", classification_report(y_train, y_train_prediction))
print("Confusion Matrix: \n", confusion_matrix(y_train, y_train_prediction))

y_val_pred_rf = rf.predict(X_val)
validation_data_accuracy = accuracy_score(y_val, y_val_pred_rf)
print("Accuracy on testing data = ", validation_data_accuracy)
print("Classification Report: \n", classification_report(y_val, y_val_pred_rf))
print("Confusion Matrix: \n", confusion_matrix(y_val, y_val_pred_rf))


svc = SVC(random_state=42, class_weight='balanced', kernel='rbf', C=10, gamma='scale', probability=True, shrinking=True, tol=1e-3)
svc.fit(X_train, y_train)

y_train_prediction = svc.predict(X_train)
training_data_accuracy = accuracy_score(y_train, y_train_prediction)
print("Accuracy on training data = ", training_data_accuracy)
print("Classification Report: \n", classification_report(y_train, y_train_prediction))
print("Confusion Matrix: \n", confusion_matrix(y_train, y_train_prediction))

y_val_pred_svc = svc.predict(X_val)
validation_data_accuracy = accuracy_score(y_val, y_val_pred_svc)
print("Accuracy on testing data = ", validation_data_accuracy)
print("Classification Report: \n", classification_report(y_val, y_val_pred_svc))
print("Confusion Matrix: \n", confusion_matrix(y_val, y_val_pred_svc))


decision_tree = DecisionTreeClassifier(random_state=42, class_weight='balanced', criterion='gini', splitter='best', max_depth=10, min_samples_split=5, min_samples_leaf=3, max_features='sqrt')
decision_tree.fit(X_train, y_train)

y_train_prediction = decision_tree.predict(X_train)
training_data_accuracy = accuracy_score(y_train, y_train_prediction)
print("Accuracy on training data = ", training_data_accuracy)
print("Classification Report: \n", classification_report(y_train, y_train_prediction))
print("Confusion Matrix: \n", confusion_matrix(y_train, y_train_prediction))

y_val_pred_svc = decision_tree.predict(X_val)
validation_data_accuracy = accuracy_score(y_val, y_val_pred_svc)
print("Accuracy on testing data = ", validation_data_accuracy)
print("Classification Report: \n", classification_report(y_val, y_val_pred_svc))
print("Confusion Matrix: \n", confusion_matrix(y_val, y_val_pred_svc))


knn = KNeighborsClassifier(n_neighbors=7, weights='distance', algorithm='auto', leaf_size=30, p=2, metric='minkowski', n_jobs=-1)
knn.fit(X_train, y_train)

y_train_prediction = knn.predict(X_train)
training_data_accuracy = accuracy_score(y_train, y_train_prediction)
print("Accuracy on training data = ", training_data_accuracy)
print("Classification Report: \n", classification_report(y_train, y_train_prediction))
print("Confusion Matrix: \n", confusion_matrix(y_train, y_train_prediction))

y_val_pred_svc = knn.predict(X_val)
validation_data_accuracy = accuracy_score(y_val, y_val_pred_svc)
print("Accuracy on testing data = ", validation_data_accuracy)
print("Classification Report: \n", classification_report(y_val, y_val_pred_svc))
print("Confusion Matrix: \n", confusion_matrix(y_val, y_val_pred_svc))


gbc = GradientBoostingClassifier(n_estimators=1000, learning_rate=0.05, max_depth=5, subsample=0.8, min_samples_split=5, min_samples_leaf=3, max_features='sqrt', random_state=42)
gbc.fit(X_train, y_train)

y_train_prediction = gbc.predict(X_train)
training_data_accuracy = accuracy_score(y_train, y_train_prediction)
print("Accuracy on training data = ", training_data_accuracy)
print("Classification Report: \n", classification_report(y_train, y_train_prediction))
print("Confusion Matrix: \n", confusion_matrix(y_train, y_train_prediction))

y_val_pred_svc = gbc.predict(X_val)
validation_data_accuracy = accuracy_score(y_val, y_val_pred_svc)
print("Accuracy on testing data = ", validation_data_accuracy)
print("Classification Report: \n", classification_report(y_val, y_val_pred_svc))
print("Confusion Matrix: \n", confusion_matrix(y_val, y_val_pred_svc))


abc = AdaBoostClassifier(n_estimators=1000, learning_rate=0.05, algorithm='SAMME', random_state=42)
abc.fit(X_train, y_train)

y_train_prediction = abc.predict(X_train)
training_data_accuracy = accuracy_score(y_train, y_train_prediction)
print("Accuracy on training data = ", training_data_accuracy)
print("Classification Report: \n", classification_report(y_train, y_train_prediction))
print("Confusion Matrix: \n", confusion_matrix(y_train, y_train_prediction))

y_val_pred_svc = abc.predict(X_val)
validation_data_accuracy = accuracy_score(y_val, y_val_pred_svc)
print("Accuracy on testing data = ", validation_data_accuracy)
print("Classification Report: \n", classification_report(y_val, y_val_pred_svc))
print("Confusion Matrix: \n", confusion_matrix(y_val, y_val_pred_svc))


xgb = XGBClassifier(random_state=42, n_estimators=6000, task_type='GPU', learning_rate=0.05, max_depth=8, subsample=0.8, colsample_bytree=0.8, use_label_encoder=False, eval_metric='logloss', class_weight='balanced')
xgb.fit(X_train, y_train)

y_train_prediction = xgb.predict(X_train)
training_data_accuracy = accuracy_score(y_train, y_train_prediction)
print("Accuracy on training data = ", training_data_accuracy)
print("Classification Report: \n", classification_report(y_train, y_train_prediction))
print("Confusion Matrix: \n", confusion_matrix(y_train, y_train_prediction))

y_val_pred_svc = xgb.predict(X_val)
validation_data_accuracy = accuracy_score(y_val, y_val_pred_svc)
print("Accuracy on testing data = ", validation_data_accuracy)
print("Classification Report: \n", classification_report(y_val, y_val_pred_svc))
print("Confusion Matrix: \n", confusion_matrix(y_val, y_val_pred_svc))


lgbm = LGBMClassifier(random_state=42, class_weight='balanced', device="GPU", n_estimators=4000, learning_rate=0.03, max_depth=6, num_leaves=12, min_child_samples=30, subsample=0.8, colsample_bytree=0.8)
lgbm.fit(X_train, y_train)

y_train_pred_lgbm = lgbm.predict(X_train)
training_data_accuracy_lgbm = accuracy_score(y_train, y_train_pred_lgbm)
print("Accuracy on training data = ", training_data_accuracy_lgbm)
print("Classification Report:\n", classification_report(y_train, y_train_pred_lgbm))
print("Confusion Matrix:\n", confusion_matrix(y_train, y_train_pred_lgbm))

y_val_pred_lgbm = lgbm.predict(X_val)
validation_data_accuracy_lgbm = accuracy_score(y_val, y_val_pred_lgbm)
print("Accuracy on validation data =", validation_data_accuracy_lgbm)
print("Classification Report:\n", classification_report(y_val, y_val_pred_lgbm))
print("Confusion Matrix:\n", confusion_matrix(y_val, y_val_pred_lgbm))


from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

# Convert to DataFrame/Series if needed
if isinstance(X_train, np.ndarray):
    X_train = pd.DataFrame(X_train)
if isinstance(X_val, np.ndarray):
    X_val = pd.DataFrame(X_val)
if isinstance(y_train, np.ndarray):
    y_train = pd.Series(y_train)
if isinstance(y_val, np.ndarray):
    y_val = pd.Series(y_val)
    
# Combine training and validation sets
X = pd.concat([X_train, X_val]).reset_index(drop=True)
y = pd.concat([y_train, y_val]).reset_index(drop=True)

# Stratified K-Fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

all_thresholds = []
all_fold_f1 = []
all_fold_reports = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    print(f"\n========== Fold {fold} ==========")

    X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]

    model = CatBoostClassifier(
        random_state=42,
        iterations=6000,
        learning_rate=0.05,
        depth=6,
        loss_function="Logloss",
        eval_metric="Accuracy",
        class_weights=[3.84, 1.35],
        verbose=0,
        task_type="GPU"
    )

    model.fit(X_tr, y_tr)

    # Predict probabilities
    y_val_probs = model.predict_proba(X_va)[:, 1]

    # Find best threshold based on macro F1-score
    best_thresh = 0.5
    best_macro_f1 = 0

    for t in np.arange(0.25, 0.35, 0.005):
        y_val_pred = (y_val_probs > t).astype(int)
        macro_f1 = f1_score(y_va, y_val_pred, average='macro')
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_thresh = t

    all_thresholds.append(best_thresh)

    # Final predictions with best threshold
    final_preds = (y_val_probs > best_thresh).astype(int)
    acc = accuracy_score(y_va, final_preds)
    report = classification_report(y_va, final_preds, digits=4)
    cm = confusion_matrix(y_va, final_preds)
    f1_macro = f1_score(y_va, final_preds, average='macro')

    all_fold_f1.append(f1_macro)
    all_fold_reports.append(report)

    print(f"Best Threshold: {best_thresh:.3f}")
    print(f"Accuracy: {acc:.4f}")
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", report)

# Overall performance
print("\n========== Overall Summary ==========")
print(f"Average Best Threshold: {np.mean(all_thresholds):.4f}")
print(f"Average Macro F1-score: {np.mean(all_fold_f1):.4f}")


# 1. Retrain final model on full training data
final_model = CatBoostClassifier(
    random_state=42,
    iterations=6000,
    learning_rate=0.05,
    depth=6,
    loss_function="Logloss",
    eval_metric="Accuracy",
    class_weights=[3.84, 1.35],
    verbose=0,
    task_type="GPU"
)
final_model.fit(X, y)

# 2. Predict on test set
cat_test_probs = final_model.predict_proba(X_test_scaled)[:, 1]

# 3. Use average threshold from cross-validation
final_thresh = np.mean(all_thresholds)
cat_test_preds = (cat_test_probs > final_thresh).astype(int)

# 4. Prepare submission
submission = pd.DataFrame({
    'id': df_test['id'],
    'Personality': cat_test_preds
})

submission['Personality'] = submission['Personality'].replace({
    1: 'Extrovert',
    0: 'Introvert'
})

submission.to_csv("submission.csv", index=False)


import matplotlib.pyplot as plt

plt.hist(cat_test_probs, bins=50)
plt.title("Distribution of Test Probabilities")
plt.xlabel("Probability of being Extrovert (class 1)")
plt.ylabel("Count")
plt.grid(True)
plt.show()


