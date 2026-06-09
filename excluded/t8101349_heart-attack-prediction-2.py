import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
warnings.simplefilter("ignore")


train_path = "/kaggle/input/unibs-heart-attack-analysis-prediction-2024/train.csv"
test_path = "/kaggle/input/unibs-heart-attack-analysis-prediction-2024/test.csv"


train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)


features = ["st_slope","exercise_angina","chest_pain_type","oldpeak","pulse","max_heart_rate","sex","age","cholesterol","fasting_blood_sugar","resting_bp_s","target"]
train_df=train_df[features]
train_df



cat_cols = ["st_slope", "exercise_angina", "chest_pain_type", "sex", "fasting_blood_sugar"]
num_cols = ["oldpeak", "pulse", "max_heart_rate", "age", "cholesterol","resting_bp_s"]

for col in cat_cols:
    train_df[col].fillna(train_df[col].mode()[0], inplace=True)

for col in num_cols:
    train_df[col].fillna(train_df[col].mean(), inplace=True)


scaler = StandardScaler()
train_df[num_cols] = scaler.fit_transform(train_df[num_cols])


X = train_df.iloc[:,:-1]
y = train_df.iloc[:,-1]


X


y


from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)


model = LogisticRegression(class_weight='balanced')

model.fit(X_train, y_train)
predicted_val = model.predict(X_valid)
print ("The f1_score of Logistic Regression is : ", f1_score(y_valid, predicted_val)*100, "%")


model = GaussianNB()

model.fit(X_train, y_train)
predicted_val = model.predict(X_valid)
print("The f1_score of Gaussian Naive Bayes model is : ", f1_score(y_valid, predicted_val)*100, "%")


model = SVC()

model.fit(X_train, y_train)
predicted_val = model.predict(X_valid)
print("The f1_score of SVM is : ", f1_score(y_valid, predicted_val)*100, "%")


model = BernoulliNB()

model.fit(X_train, y_train)
predicted_val = model.predict(X_valid)
print("The f1_score of Gaussian Naive Bayes model is : ", f1_score(y_valid, predicted_val)*100, "%")


from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb

model = xgb.XGBClassifier(random_state=1, use_label_encoder=False, subsample= 0.8, n_estimators= 50, max_depth= 7, learning_rate= 0.2, gamma= 0.3, colsample_bytree= 0.8)

model.fit(X_train, y_train)
predicted_val = model.predict(X_valid)
print("The f1_score of XGBClassifier model is : ", f1_score(y_valid, predicted_val)*100, "%")


from sklearn.metrics import mean_squared_error
model = xgb.XGBRegressor(random_state=1, subsample= 0.8, n_estimators= 150, max_depth= 9, learning_rate= 0.05, gamma= 0, colsample_bytree= 0.8)

model.fit(X_train, y_train)
predicted_val = model.predict(X_valid)
print("The MSE of XGBRegressor model is : ", mean_squared_error(y_valid, predicted_val))


from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# 使用 SMOTE 進行 Oversampling
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1
)


grid_search.fit(X_train_resampled, y_train_resampled)


best_model = grid_search.best_estimator_

print("Best parameters:", grid_search.best_params_)


predicted_val = best_model.predict(X_valid)
print("The f1_score of SMOTE model is : ", f1_score(y_valid, predicted_val)*100, "%")



model = xgb.XGBClassifier(random_state=1, use_label_encoder=False, subsample= 0.8, n_estimators= 50, max_depth= 7, learning_rate= 0.2, gamma= 0.3, colsample_bytree= 0.8)

model.fit(X_train_resampled, y_train_resampled)
predicted_val = model.predict(X_valid)
print("The f1_score of XGBClassifier model is : ", f1_score(y_valid, predicted_val)*100, "%")


valid_proba = best_model.predict_proba(X_valid)[:, 1]

threshold = 0.5

valid_y_pred = (valid_proba >= threshold).astype(int)



from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# 混淆矩陣
cm = confusion_matrix(valid_y_pred, y_valid)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()


valid_proba = best_model.predict_proba(X_valid)[:, 1]

threshold = 0.38

valid_y_pred = (valid_proba >= threshold).astype(int)



from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# 混淆矩陣
cm = confusion_matrix(valid_y_pred, y_valid)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()


#全部資料訓練
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30]
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1
)


grid_search.fit(X_resampled, y_resampled)


best_model = grid_search.best_estimator_

print("Best parameters:", grid_search.best_params_)


features = ["st_slope","exercise_angina","chest_pain_type","oldpeak","pulse","max_heart_rate","sex","age","cholesterol","fasting_blood_sugar","resting_bp_s"]
test_df=test_df[features]
test_df



cat_cols = ["st_slope", "exercise_angina", "chest_pain_type", "sex", "fasting_blood_sugar"]
num_cols = ["oldpeak", "pulse", "max_heart_rate", "age", "cholesterol","resting_bp_s"]

for col in cat_cols:
    test_df[col].fillna(train_df[col].mode()[0], inplace=True)

for col in num_cols:
    test_df[col].fillna(train_df[col].mean(), inplace=True)


test_df[num_cols] = scaler.transform(test_df[num_cols])


test_df.isnull().sum()


test_df.shape


test_proba = best_model.predict_proba(test_df)[:, 1]

threshold = 0.38

test_y_pred = (test_proba >= threshold).astype(int)
test_y_pred.shape


submission_df = pd.DataFrame({
    "Id": range(1, len(test_y_pred) + 1),
    "target": test_y_pred
})

# 儲存成 CSV
submission_df.to_csv("submission.csv", index=False)

