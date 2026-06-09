import os
import pandas as pd
import numpy as np
import itertools
import matplotlib
import matplotlib.pyplot as plt
#from xgboost import XGBClassifier, plot_importance
#from sklearn.impute import SimpleImputer
#from sklearn.model_selection import cross_val_score

from pandas import Series
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier

#if os.path.isfile("TaipeiSansTCBeta-Regular.ttf"):
#  print("中文化模組存在。")
#else:
#  print("中文化模組不存在。")
#  !wget -O TaipeiSansTCBeta-Regular.ttf https://drive.google.com/uc?id=1eGAsTN1HBpJAkeVM57_C7ccp7hbgSz3_&export=download
#中文化模組
#matplotlib.font_manager.fontManager.addfont("TaipeiSansTCBeta-Regular.ttf")
#matplotlib.rc('font',family='Taipei Sans TC Beta')

output_dir = '/kaggle/working'
os.makedirs(output_dir, exist_ok=True)
train = pd.read_csv('/kaggle/input/playground-series-s3e22/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s3e22/test.csv')

# 數值化
train["surgery"] = train["surgery"].fillna("no").map({'no': 0, 'yes': 1})
train['age']=train['age'].map({'adult':1,'young':0})
train["temp_of_extremities"] = train["temp_of_extremities"].fillna("normal").map({'cold': 0, 'cool': 1, 'normal': 2, 'warm': 3})
train["peripheral_pulse"] = train["peripheral_pulse"].fillna("normal").map({'absent': 0, 'reduced': 1, 'normal': 2, 'increased': 3})
train['mucous_membrane']=train['mucous_membrane'].fillna("normal_pink").map({'pale_cyanotic':1,'pale_pink':2,'normal_pink':3,'bright_red':4,'dark_cyanotic':5,'bright_pink':6})
train["capillary_refill_time"] = train["capillary_refill_time"].fillna("3").map({'less_3_sec': 0, '3': 1, 'more_3_sec': 2})
train["pain"] = train["pain"].fillna("depressed").map({"depressed": 0,"alert": 1,"slight": 2,"mild_pain": 3,"moderate": 4,"severe_pain": 5,"extreme_pain": 6,})
train["peristalsis"] = train["peristalsis"].fillna("hypomotile").map({'hypermotile': 0, 'normal': 1, 'hypomotile': 2, 'absent': 3})
train["abdominal_distention"] = train["abdominal_distention"].fillna("none").map({'none': 0, 'slight': 1, 'moderate': 2, 'severe': 3})
train["nasogastric_tube"] = train["nasogastric_tube"].fillna("none").map({'none': 0, 'slight': 1, 'significant': 2})
train["nasogastric_reflux"] = train["nasogastric_reflux"].fillna("none").map({'less_1_liter': 0, 'none': 1, 'more_1_liter': 2})
train["rectal_exam_feces"] = train["rectal_exam_feces"].fillna("absent").map({'absent': 0, 'decreased': 1, 'normal': 2, 'increased': 3})
train["abdomen"] = train["abdomen"].fillna("distend_small").map({'normal': 0, 'other': 1, 'firm': 2,'distend_small': 3, 'distend_large': 4})
train["abdomo_appearance"] = train["abdomo_appearance"].fillna("serosanguious").map({'clear': 0, 'cloudy': 1, 'serosanguious': 2})
train["cp_data"] = train["cp_data"].fillna("yes").map({'no': 0, 'yes': 1})
train["surgical_lesion"] = train["surgical_lesion"].fillna("yes").map({'no': 0, 'yes': 1})
train['outcome']=train['outcome'].map({'lived':0,'died':1,'euthanized':2})
test["surgery"] = test["surgery"].fillna("no").map({'no': 0, 'yes': 1})
test['age']=test['age'].map({'adult':1,'young':0})
test["temp_of_extremities"] = test["temp_of_extremities"].fillna("normal").map({'cold': 0, 'cool': 1, 'normal': 2, 'warm': 3})
test["peripheral_pulse"] = test["peripheral_pulse"].fillna("normal").map({'absent': 0, 'reduced': 1, 'normal': 2, 'increased': 3})
test['mucous_membrane']=test['mucous_membrane'].fillna("normal_pink").map({'pale_cyanotic':1,'pale_pink':2,'normal_pink':3,'bright_red':4,'dark_cyanotic':5,'bright_pink':6})
test["capillary_refill_time"] = test["capillary_refill_time"].fillna("3").map({'less_3_sec': 0, '3': 1, 'more_3_sec': 2})
test["pain"] = test["pain"].fillna("depressed").map({"depressed": 0,"alert": 1,"slight": 2,"mild_pain": 3,"moderate": 4,"severe_pain": 5,"extreme_pain": 6,})
test["peristalsis"] = test["peristalsis"].fillna("hypomotile").map({'hypermotile': 0, 'normal': 1, 'hypomotile': 2, 'absent': 3})
test["abdominal_distention"] = test["abdominal_distention"].fillna("none").map({'none': 0, 'slight': 1, 'moderate': 2, 'severe': 3})
test["nasogastric_tube"] = test["nasogastric_tube"].fillna("none").map({'none': 0, 'slight': 1, 'significant': 2})
test["nasogastric_reflux"] = test["nasogastric_reflux"].fillna("none").map({'less_1_liter': 0, 'none': 1, 'more_1_liter': 2})
test["rectal_exam_feces"] = test["rectal_exam_feces"].fillna("absent").map({'absent': 0, 'decreased': 1, 'normal': 2, 'increased': 3})
test["abdomen"] = test["abdomen"].fillna("distend_small").map({'normal': 0, 'other': 1, 'firm': 2,'distend_small': 3, 'distend_large': 4})
test["abdomo_appearance"] = test["abdomo_appearance"].fillna("serosanguious").map({'clear': 0, 'cloudy': 1, 'serosanguious': 2})
test["cp_data"] = test["cp_data"].fillna("yes").map({'no': 0, 'yes': 1})
test["surgical_lesion"] = test["surgical_lesion"].fillna("yes").map({'no': 0, 'yes': 1})
# 保存測試資料 ID 與 hospital_number
test_id_cols = test[['id']]
# 分割 X / y
X_train = train.drop(['outcome','hospital_number','id'],axis=1)
# 移除與馬匹健康情況無關的欄位
y_train = train['outcome']
X_test = test.drop(['hospital_number', 'id'], axis=1)
# 缺失值處理
imputer = SimpleImputer(strategy='most_frequent')
X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

# 特徵重要性取得(XGB)
#xgb_base = XGBClassifier(
#    eval_metric='mlogloss',
#    learning_rate=0.02,     # 學習率（控制每棵樹影響程度）
#    max_depth=10,        # 樹的最大深度（控制模型複雜度）
#    n_estimators=300,      # 樹的數量（越多模型越強，但風險過擬合）
#    subsample=0.8,       # 每棵樹使用100%的資料（防止過擬合）
#    colsample_bytree=0.8,  # 每棵樹使用100%的特徵（增加多樣性）
#    random_state=42
#)
#xgb_base.fit(X_train_imputed, y_train)

# 所有特徵的重要性排序
#importances = xgb_base.feature_importances_
#feature_importance_df = pd.DataFrame({
#    'feature': X_train.columns,
#    'importance': importances
#}).sort_values(by='importance', ascending=False)

# 顯示前10名
#top_features = feature_importance_df['feature'].head(10).tolist()
#print("\n前10名特徵:", top_features)

# 畫圖
#plt.figure(figsize=(10,12))
#plt.barh(feature_importance_df['feature'], feature_importance_df['importance'], color='salmon')
#plt.xlabel('Feature Importance')
#plt.title('All Feature Importances (XGBoost)')
#plt.gca().invert_yaxis()
#plt.tight_layout()
#plt.show()

# 初步取得特徵重要性
base_model = RandomForestClassifier(n_estimators=100,max_depth=10,random_state=42)
base_model.fit(X_train_imputed, y_train)

importances = base_model.feature_importances_
indices = np.argsort(importances)[::-1]
#  取頂五最相關的特徵
top_features = [X_train.columns[i] for i in indices[:10]]

print("\n前10名特徵:")
print(top_features)

# 繪製所有特徵重要性圖表
feature_importance_df = pd.DataFrame({
    'feature': X_train.columns,
    'importance': importances
}).sort_values(by='importance', ascending=False)

plt.figure(figsize=(10, 12))
plt.barh(feature_importance_df['feature'], feature_importance_df['importance'], color='skyblue')
plt.xlabel('Feature Importance')
plt.title('All Feature Importances from RandomForest')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


# 交叉驗證TIME
#results = {}
#for n in [2, 4]:
#    for combo in itertools.combinations(top_features, n):
#        X_subset = X_train_imputed[list(combo)]
#        model = XGBClassifier(eval_metric='mlogloss',
#        learning_rate=0.05,max_depth=10,n_estimators=1000,
#        subsample=0.8,colsample_bytree=0.8,random_state=42)
#        scores = cross_val_score(model, X_subset, y_train, cv=5, scoring='accuracy')
#        avg_score = np.mean(scores)
#        results[combo] = avg_score
#        print(f"Features: {combo}, CV Accuracy: {avg_score:.4f}")

#best_combo = max(results, key=results.get)
#print(f"\n最佳特徵組合: {best_combo}")
#print(f"平均準確率: {results[best_combo]:.4f}")

#比較隨機森林+梯度提升機做比較
results={}
for n in [2, 4]:
    for combo in itertools.combinations(top_features, n):
        X_subset=X_train_imputed[list(combo)]

        rf_model=RandomForestClassifier(n_estimators=100,max_depth=10,random_state=42)
        gb_model=GradientBoostingClassifier(loss="log_loss",learning_rate=0.1,n_estimators=100,random_state=42)

        rf_score=cross_val_score(rf_model,X_subset,y_train,cv=5,scoring='accuracy').mean()
        gb_score=cross_val_score(gb_model,X_subset,y_train,cv=5,scoring='accuracy').mean()

        results[combo] = {'RF': rf_score, 'GB':gb_score}
        print(f"Features: {combo}, RF Accuracy: {rf_score:.4f}, GBDT Accuracy: {gb_score:.4f}")
# 取得最佳組合
best_combo,best_model_name,best_score=None,None,0
for combo,scores in results.items():
    for model_name,score in scores.items():
        if score > best_score:
            best_score=score
            best_combo=combo
            best_model_name=model_name
print(f"\n最佳模型:{best_model_name}，特徵:{best_combo}，準確率:{best_score:.4f}")


# 使用最佳特徵集重新訓練
X_train_final = X_train_imputed[list(best_combo)]
X_test_final = X_test_imputed[list(best_combo)]

#哪個成績比較好就用哪個
if best_model_name=='RF':
  final_model=RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
else:
  final_model=GradientBoostingClassifier(loss="log_loss",learning_rate=0.1,n_estimators=100,random_state=42)

final_model.fit(X_train_final, y_train)


# 預測測試集
test_preds = final_model.predict(X_test_final)

# 將 outcome 數字轉回英文標籤
label_map = {0:'lived',1:'died',2:'euthanized'}
test_labels = pd.Series(test_preds).map(label_map)


# ========= 輸出 CSV =========
submission = test_id_cols.copy()
submission['outcome'] = test_labels
submission.to_csv('/kaggle/working/submission.csv', index=False)
submission

