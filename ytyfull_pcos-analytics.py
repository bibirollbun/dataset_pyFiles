# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Просто вручную ставлю в соответствие словам циферки) категорий мало, так что так

import pandas as pd

df = pd.read_csv('/kaggle/input/exploring-predictive-health-factors/train.csv')

df['Age'] = df['Age'].apply(lambda x: 
                            22.5 if x == "20-25" else 
                            17.5 if x == "15-20" else 
                            52.5 if x == "45 and above" else
                            38.5 if x == "35-44" else
                            32.5 if x == "30-35" else 
                            15.5 if x == "Less than 20" else
                            27.5 if x == "25-30" else
                            27.5 if x == "30-25" else
                            None)
df['Age'] = df['Age'].astype(float)

df['PCOS'] = df['PCOS'].apply(lambda x: 0 if x == 'No' else 1)
df['Hyperandrogenism'] = df['Hyperandrogenism'].apply(lambda x: 0 if x == 'No' else 1)
df['Insulin_Resistance'] = df['Insulin_Resistance'].apply(lambda x: 0 if x == 'No' else 1)

df['Hormonal_Imbalance'] = df['Hormonal_Imbalance'].map({
    'No': 0,
    'Yes': 50,
    'No, Yes, not diagnosed by a doctor': 30,
    'Yes Significantly': 100
})

df['Hirsutism'] = df['Hirsutism'].map({
    'No': 0,
    'Yes': 60,
    'No, Yes, not diagnosed by a doctor': 30
})

df['Conception_Difficulty'] = df['Conception_Difficulty'].map({
    'No': 0,
    'Yes, diagnosed by a doctor': 70,
    'Yes': 50
})

df['Exercise_Frequency'] = df['Exercise_Frequency'].map({
    'Never': 0,
    'Rarely': 20,
    '1-2 Times a Week': 40,
    '3-4 Times a Week': 60,
    '6-8 Times a Week': 80,
    'Less than usual': 30,
    'Less than 6 hours': 10,
    '6-8 hours': 90
})

df['Exercise_Type'] = df['Exercise_Type'].map({
    'No Exercise': 0,
    'Cardio (e.g., running, cycling, swimming)': 30,
    'Strength training (e.g., weightlifting, resistance exercises)': 40,
    'Flexibility and balance (e.g., yoga, pilates)': 20,
    'High-intensity interval training (HIIT)': 50,
    'Cardio, Strength training': 60,
    'Cardio, Flexibility and balance': 65,
    'Strength training, Flexibility and balance': 70,
    'Cardio, Strength training, Flexibility and balance': 80
})

df['Exercise_Duration'] = df['Exercise_Duration'].map({
    'Not Applicable': 0,
    'Less than 30 minutes': 20,
    '20 minutes': 30,
    '30 minutes': 50,
    '30 minutes to 1 hour': 70,
    '45 minutes': 80,
    'More than 30 minutes': 90
})

df['Sleep_Hours'] = df['Sleep_Hours'].map({
    'Less than 6 hours': 10,
    '3-4 hours': 0,
    '6-8 hours': 60,
    '9-12 hours': 80,
    'More than 12 hours': 100
})

df['Exercise_Benefit'] = df['Exercise_Benefit'].map({
    'Not at All': 0,
    'Not Much': 30,
    'Somewhat': 60,
    'Yes Significantly': 100
})


df = df.dropna()

for column in df.columns:
    unique_values = df[column].unique()
    print(f"Столбец '{column}' содержит следующие классы: {unique_values}")
    
print(df.head())

df.to_csv('modified_train.csv', index=False)



# Аналолгичные преобразования для тест. Никаких фич не выделяю, просто преобразование данных

import pandas as pd
import numpy as np

df2 = pd.read_csv('/kaggle/input/exploring-predictive-health-factors/test.csv')

def safe_map(series, mapping, default):
    return series.map(mapping).fillna(default)

age_mapping = {
    "20-25": 22.5,
    "15-20": 17.5,
    "45 and above": 52.5,
    "35-44": 38.5,
    "30-35": 32.5,
    "Less than 20": 15.5,
    "25-30": 27.5,
}

df2['Age'] = df2['Age'].apply(lambda x: age_mapping.get(x, np.nan))
age_mean = df2['Age'].mean()
df2['Age'] = df2['Age'].fillna(age_mean)

weight_median = df2['Weight_kg'].median()
df2['Weight_kg'] = df2['Weight_kg'].fillna(weight_median)

df2['Hyperandrogenism'] = df2['Hyperandrogenism'].apply(lambda x: 0 if x == 'No' else 1)
df2['Insulin_Resistance'] = df2['Insulin_Resistance'].apply(lambda x: 0 if x == 'No' else 1)

df2['Hormonal_Imbalance'] = safe_map(df2['Hormonal_Imbalance'], {
    'No': 0,
    'Yes': 50,
    'No, Yes, not diagnosed by a doctor': 30,
    'Yes Significantly': 100
}, 0)  

df2['Hirsutism'] = safe_map(df2['Hirsutism'], {
    'No': 0,
    'Yes': 60,
    'No, Yes, not diagnosed by a doctor': 30
}, 0)  

df2['Conception_Difficulty'] = safe_map(df2['Conception_Difficulty'], {
    'No': 0,
    'Yes, diagnosed by a doctor': 70,
    'Yes': 50
}, 0)  

df2['Exercise_Frequency'] = safe_map(df2['Exercise_Frequency'], {
    'Never': 0,
    'Rarely': 20,
    '1-2 Times a Week': 40,
    '3-4 Times a Week': 60,
    '6-8 Times a Week': 80,
    'Less than usual': 30,
    'Less than 6 hours': 10,
    '6-8 hours': 90
}, 0)  

df2['Exercise_Type'] = safe_map(df2['Exercise_Type'], {
    'No Exercise': 0,
    'Cardio (e.g., running, cycling, swimming)': 30,
    'Strength training (e.g., weightlifting, resistance exercises)': 40,
    'Flexibility and balance (e.g., yoga, pilates)': 20,
    'High-intensity interval training (HIIT)': 50,
    'Cardio, Strength training': 60,
    'Cardio, Flexibility and balance': 65,
    'Strength training, Flexibility and balance': 70,
    'Cardio, Strength training, Flexibility and balance': 80
}, 0) 

df2['Exercise_Duration'] = safe_map(df2['Exercise_Duration'], {
    'Not Applicable': 0,
    'Less than 30 minutes': 20,
    '20 minutes': 30,
    '30 minutes': 50,
    '30 minutes to 1 hour': 70,
    '45 minutes': 80,
    'More than 30 minutes': 90
}, 0)  

df2['Sleep_Hours'] = safe_map(df2['Sleep_Hours'], {
    'Less than 6 hours': 10,
    '3-4 hours': 0,
    '6-8 hours': 60,
    '9-12 hours': 80,
    'More than 12 hours': 100
}, 0)  

df2['Exercise_Benefit'] = safe_map(df2['Exercise_Benefit'], {
    'Not at All': 0,
    'Not Much': 30,
    'Somewhat': 60,
    'Yes Significantly': 100
}, 0) 

print("Количество пропущенных значений в каждом столбце:")
print(df2.isnull().sum())

df2 = df2.fillna({
    'Age': age_mean,
    'Weight_kg': weight_median,
    'Hormonal_Imbalance': 0,
    'Hirsutism': 0,
    'Conception_Difficulty': 0,
    'Exercise_Frequency': 0,
    'Exercise_Type': 0,
    'Exercise_Duration': 0,
    'Sleep_Hours': 0,
    'Exercise_Benefit': 0
})

print("Количество пропущенных значений после заполнения:")
print(df2.isnull().sum())

df2['ID'] = np.arange(len(df2))

cols = df2.columns.tolist()
cols.insert(0, cols.pop(cols.index('ID')))
df2 = df2[cols]

for column in df2.columns:
    unique_values = df2[column].unique()
    print(f"Столбец '{column}' содержит следующие классы: {unique_values}")

df2.to_csv('modified_test.csv', index=False)

print("Модифицированный тестовый набор данных успешно сохранен в 'modified_test.csv'.")
print(df2.head())



# Сдесь логистическая регрессия на данных без фич

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

train_df = pd.read_csv('modified_train.csv')
test_df = pd.read_csv('modified_test.csv')

target = 'PCOS'
features = [col for col in train_df.columns if col != target]

X_train = train_df[features].copy()  
y_train = train_df[target]
X_test = test_df[features].copy()    

num_cols = ['Age', 'Weight_kg', 'Sleep_Hours', 'Exercise_Duration']

scaler = StandardScaler()

X_train.loc[:, num_cols] = scaler.fit_transform(X_train[num_cols])
X_test.loc[:, num_cols] = scaler.transform(X_test[num_cols])

model_params = {
    'solver': 'lbfgs',         
    'max_iter': 1000,          
    'random_state': 42,
    'C': 1.0,
    'tol': 1e-4,
    'fit_intercept': True,
    'class_weight': 'balanced', 
    'n_jobs': -1                
}

model = LogisticRegression(**model_params)
model.fit(X_train, y_train)

y_test_pred_prob = model.predict_proba(X_test)[:, 1]  

if 'ID' in test_df.columns:
    df_pred = pd.DataFrame({'ID': test_df['ID'], 'PCOS': y_test_pred_prob})
else:
    df_pred = pd.DataFrame({'ID': test_df.index, 'PCOS': y_test_pred_prob})

df_pred.to_csv('SL_predictions.csv', index=False)

print("Предсказания вероятностей успешно сохранены в 'SL_predictions.csv'.")
print(df_pred.head())



# Та же самая логистическая регрессия, только с подором параметров

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score

train_df = pd.read_csv('modified_train_automated.csv')
test_df = pd.read_csv('modified_test_automated.csv')

target = 'PCOS'
features = [col for col in train_df.columns if col != target]

X_train = train_df[features].copy()
y_train = train_df[target]
X_test = test_df[features].copy()

num_cols = ['Age', 'Weight_kg', 'Sleep_Hours', 'Exercise_Duration']

scaler = StandardScaler()
X_train.loc[:, num_cols] = scaler.fit_transform(X_train[num_cols])
X_test.loc[:, num_cols] = scaler.transform(X_test[num_cols])

log_reg = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)

param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],          # Коэффициент регуляризации
    'penalty': ['l1', 'l2'],               # Тип регуляризации
    'solver': ['liblinear', 'saga']        # Солверы, поддерживающие выбранные типы регуляризации
}

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator=log_reg,
    param_grid=param_grid,
    scoring='roc_auc',
    cv=cv,
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_train, y_train)

print(f"Лучшие параметры: {grid_search.best_params_}")
print(f"Лучший ROC AUC: {grid_search.best_score_:.4f}")

best_model = grid_search.best_estimator_
y_test_pred_prob = best_model.predict_proba(X_test)[:, 1]

if 'ID' in test_df.columns:
    df_pred = pd.DataFrame({'ID': test_df['ID'], 'PCOS': y_test_pred_prob})
else:
    df_pred = pd.DataFrame({'ID': np.arange(len(test_df)), 'PCOS': y_test_pred_prob})

df_pred.to_csv('SL_mod_predictions.csv', index=False)

print("Предсказания вероятностей успешно сохранены в 'SL_mod_predictions.csv'.")
print(df_pred.head())



# Всто то же самое, что и с логистической регрессией

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

train_df = pd.read_csv('modified_train.csv')
test_df = pd.read_csv('modified_test.csv')

target = 'PCOS'
features = [col for col in train_df.columns if col != target]

X_train = train_df[features].copy()
y_train = train_df[target]
X_test = test_df[features].copy()

num_cols = ['Age', 'Weight_kg', 'Sleep_Hours', 'Exercise_Duration']

scaler = StandardScaler()
X_train.loc[:, num_cols] = scaler.fit_transform(X_train[num_cols])
X_test.loc[:, num_cols] = scaler.transform(X_test[num_cols])

rf_params = {
    'n_estimators': 100,          
    'max_depth': None,             
    'min_samples_split': 2,        
    'min_samples_leaf': 1,         
    'bootstrap': True,             
    'random_state': 42,
    'class_weight': 'balanced',    
    'n_jobs': -1                   
}

rf_model = RandomForestClassifier(**rf_params)

rf_model.fit(X_train, y_train)

y_test_pred_prob = rf_model.predict_proba(X_test)[:, 1] 

if 'ID' in test_df.columns:
    df_pred = pd.DataFrame({'ID': test_df['ID'], 'PCOS': y_test_pred_prob})
else:
    df_pred = pd.DataFrame({'ID': np.arange(len(test_df)), 'PCOS': y_test_pred_prob})

df_pred.to_csv('RF_predictions.csv', index=False)

print("Предсказания вероятностей успешно сохранены в 'RF_predictions.csv'.")
print(df_pred.head())



import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import xgboost as xgb

train_df = pd.read_csv('modified_train.csv')
test_df = pd.read_csv('modified_test.csv')

target = 'PCOS'
features = [col for col in train_df.columns if col != target]

X_train = train_df[features].copy()
y_train = train_df[target]
X_test = test_df[features].copy()

num_cols = ['Age', 'Weight_kg', 'Sleep_Hours', 'Exercise_Duration']
scaler = StandardScaler()
X_train.loc[:, num_cols] = scaler.fit_transform(X_train[num_cols])
X_test.loc[:, num_cols] = scaler.transform(X_test[num_cols])

xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    use_label_encoder=False,
    random_state=42
)

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    cv=cv,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_train, y_train)

print("Лучшие параметры:", grid_search.best_params_)
print("Лучший ROC AUC:", grid_search.best_score_)

best_xgb = grid_search.best_estimator_
y_test_pred_prob = best_xgb.predict_proba(X_test)[:, 1]

if 'ID' in test_df.columns:
    df_pred = pd.DataFrame({'ID': test_df['ID'], 'PCOS': y_test_pred_prob})
else:
    df_pred = pd.DataFrame({'ID': np.arange(len(test_df)), 'PCOS': y_test_pred_prob})

df_pred.to_csv('XGB_predictions.csv', index=False)
print("Предсказания вероятностей успешно сохранены в 'XGB_predictions.csv'.")
print(df_pred.head())



import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from catboost import CatBoostClassifier

train_df = pd.read_csv('modified_train.csv')
test_df = pd.read_csv('modified_test.csv')

target = 'PCOS'
features = [col for col in train_df.columns if col != target]

X_train = train_df[features].copy()
y_train = train_df[target]
X_test = test_df[features].copy()

num_cols = ['Age', 'Weight_kg', 'Sleep_Hours', 'Exercise_Duration']
scaler = StandardScaler()
X_train.loc[:, num_cols] = scaler.fit_transform(X_train[num_cols])
X_test.loc[:, num_cols] = scaler.transform(X_test[num_cols])

cat_features = []  

base_catboost = CatBoostClassifier(
    verbose=0,
    random_state=42
)

param_grid = {
    'iterations': [100, 200, 300],
    'depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1],
    'l2_leaf_reg': [1, 3, 5, 7]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator=base_catboost,
    param_grid=param_grid,
    cv=cv,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_train, y_train, cat_features=cat_features)

print("Лучшие параметры:", grid_search.best_params_)
print("Лучший ROC AUC:", grid_search.best_score_)

best_catboost = grid_search.best_estimator_
y_test_pred_prob = best_catboost.predict_proba(X_test)[:, 1]

if 'ID' in test_df.columns:
    df_pred = pd.DataFrame({'ID': test_df['ID'], 'PCOS': y_test_pred_prob})
else:
    df_pred = pd.DataFrame({'ID': np.arange(len(test_df)), 'PCOS': y_test_pred_prob})

df_pred.to_csv('CatBoost_predictions.csv', index=False, float_format='%.10f')

print("Предсказания вероятностей успешно сохранены в 'CatBoost_predictions.csv'.")
print(df_pred.head())



# Объединяю модели, показавшие неплохие результаты, главная - логистическая регрессия, имеющая самый высокий скор. 

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold

train_df = pd.read_csv('modified_train.csv')
test_df = pd.read_csv('modified_test.csv')

target = 'PCOS'
features = [col for col in train_df.columns if col != target]

X_train = train_df[features].copy()
y_train = train_df[target]
X_test = test_df[features].copy()

num_cols = ['Age', 'Weight_kg', 'Sleep_Hours', 'Exercise_Duration']

scaler = StandardScaler()
X_train.loc[:, num_cols] = scaler.fit_transform(X_train[num_cols])
X_test.loc[:, num_cols] = scaler.transform(X_test[num_cols])

rf_params = {
    'n_estimators': 100,          
    'max_depth': None,             
    'min_samples_split': 2,        
    'min_samples_leaf': 1,         
    'bootstrap': True,             
    'random_state': 42,
    'class_weight': 'balanced',    
    'n_jobs': -1                   
}

lr_params = {
    'solver': 'lbfgs',         
    'max_iter': 1000,          
    'random_state': 42,
    'C': 1.0,
    'tol': 1e-4,
    'fit_intercept': True,
    'class_weight': 'balanced', 
    'n_jobs': -1                
}

estimators = [
    ('lr', LogisticRegression(**lr_params)),
    ('rf', RandomForestClassifier(**rf_params)),
    ('xgb', xgb.XGBClassifier(objective='binary:logistic', 
                              use_label_encoder=False, 
                              eval_metric='auc',
                              random_state=42,
                              depth=5,
                              iterations=100,
                              l2_leaf_reg=7,
                              learning_rate=0.01
                              ))
]

final_estimator = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)

stacking_clf = StackingClassifier(
    estimators=estimators,
    final_estimator=final_estimator,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    n_jobs=-1,
    passthrough=True  
)

stacking_clf.fit(X_train, y_train)

y_test_pred_prob = stacking_clf.predict_proba(X_test)[:, 1]

if 'ID' in test_df.columns:
    df_pred = pd.DataFrame({'ID': test_df['ID'], 'PCOS': y_test_pred_prob})
else:
    df_pred = pd.DataFrame({'ID': np.arange(len(test_df)), 'PCOS': y_test_pred_prob})

df_pred.to_csv('Stacking_predictions.csv', index=False, float_format='%.10f')

print("Предсказания метамодели успешно сохранены в 'Stacking_predictions.csv'.")
print(df_pred.head())



# Выделяю фичи

import pandas as pd
import numpy as np
import category_encoders as ce

df = pd.read_csv('/kaggle/input/exploring-predictive-health-factors/train.csv')

def map_age(age_str):
    mapping = {
        "20-25": 22.5,
        "15-20": 17.5,
        "45 and above": 52.5,
        "35-44": 38.5,
        "30-35": 32.5,
        "Less than 20": 15.5,
        "25-30": 27.5,
        "30-25": 27.5  
    }
    return mapping.get(age_str, np.nan)

df['Age'] = df['Age'].apply(map_age).astype(float)

df['PCOS'] = df['PCOS'].apply(lambda x: 0 if x == 'No' else 1)

df['Hormonal_Imbalance'] = df['Hormonal_Imbalance'].map({
    'No': 0,
    'Yes': 50,
    'No, Yes, not diagnosed by a doctor': 30,
    'Yes Significantly': 100
})
df['Hirsutism'] = df['Hirsutism'].map({
    'No': 0,
    'Yes': 60,
    'No, Yes, not diagnosed by a doctor': 30
})
df['Conception_Difficulty'] = df['Conception_Difficulty'].map({
    'No': 0,
    'Yes, diagnosed by a doctor': 70,
    'Yes': 50
})
df['Exercise_Frequency'] = df['Exercise_Frequency'].map({
    'Never': 0,
    'Rarely': 20,
    '1-2 Times a Week': 40,
    '3-4 Times a Week': 60,
    '6-8 Times a Week': 80,
    'Less than usual': 30,
    'Less than 6 hours': 10,
    '6-8 hours': 90
})
df['Exercise_Type'] = df['Exercise_Type'].map({
    'No Exercise': 0,
    'Cardio (e.g., running, cycling, swimming)': 30,
    'Strength training (e.g., weightlifting, resistance exercises)': 40,
    'Flexibility and balance (e.g., yoga, pilates)': 20,
    'High-intensity interval training (HIIT)': 50,
    'Cardio, Strength training': 60,
    'Cardio, Flexibility and balance': 65,
    'Strength training, Flexibility and balance': 70,
    'Cardio, Strength training, Flexibility and balance': 80
})
df['Exercise_Duration'] = df['Exercise_Duration'].map({
    'Not Applicable': 0,
    'Less than 30 minutes': 20,
    '20 minutes': 30,
    '30 minutes': 50,
    '30 minutes to 1 hour': 70,
    '45 minutes': 80,
    'More than 30 minutes': 90
})
df['Sleep_Hours'] = df['Sleep_Hours'].map({
    'Less than 6 hours': 10,
    '3-4 hours': 0,
    '6-8 hours': 60,
    '9-12 hours': 80,
    'More than 12 hours': 100
})
df['Exercise_Benefit'] = df['Exercise_Benefit'].map({
    'Not at All': 0,
    'Not Much': 30,
    'Somewhat': 60,
    'Yes Significantly': 100
})

# =========================
# Добавляем новые признаки
# =========================

df['Hormonal_Index'] = (df['Hormonal_Imbalance'] + df['Hirsutism']) / 2

df['Activity_Index'] = (df['Exercise_Frequency'] + df['Exercise_Type'] + 
                        df['Exercise_Duration'] + df['Exercise_Benefit']) / 4

def categorize_sleep(s):
    if s <= 30:
        return 'Недостаточный'
    elif s <= 70:
        return 'Нормальный'
    else:
        return 'Избыточный'
    
df['Sleep_Category'] = df['Sleep_Hours'].apply(categorize_sleep)

df['Age_Activity_Interaction'] = df['Age'] * df['Activity_Index']

def age_group(age):
    if age < 20:
        return 'Молодой'
    elif age < 35:
        return 'Юный взрослый'
    elif age < 50:
        return 'Средний возраст'
    else:
        return 'Пожилой'
    
df['Age_Group'] = df['Age'].apply(age_group)

df['High_Conception_Risk'] = df['Conception_Difficulty'].apply(lambda x: 1 if x >= 50 else 0)

df['Health_Index'] = (df['Hormonal_Index'] + df['Activity_Index'] + df['Sleep_Hours']) / 3

# -------------------------------------------------
# Заполняем пропущенные значения в числовых столбцах
# медианой
# -------------------------------------------------
df.fillna(df.median(numeric_only=True), inplace=True)

cat_columns = df.select_dtypes(include=['object']).columns.tolist()

encoder = ce.OrdinalEncoder(cols=cat_columns, handle_unknown='impute')
df_encoded = encoder.fit_transform(df)

print("Столбцы после преобразования:", df_encoded.columns.tolist())
print(df_encoded.head())

df_encoded.to_csv('modified_train_automated.csv', index=False)



import pandas as pd
import numpy as np

df2 = pd.read_csv('/kaggle/input/exploring-predictive-health-factors/test.csv')

def safe_map(series, mapping, default):
    return series.map(mapping).fillna(default)

age_mapping = {
    "20-25": 22.5,
    "15-20": 17.5,
    "45 and above": 52.5,
    "35-44": 38.5,
    "30-35": 32.5,
    "Less than 20": 15.5,
    "25-30": 27.5
}
df2['Age'] = df2['Age'].apply(lambda x: age_mapping.get(x, np.nan))
age_median = df2['Age'].median()
df2['Age'] = df2['Age'].fillna(age_median)

weight_median = df2['Weight_kg'].median()
df2['Weight_kg'] = df2['Weight_kg'].fillna(weight_median)

df2['Hyperandrogenism'] = df2['Hyperandrogenism'].apply(lambda x: 0 if x == 'No' else 1)
df2['Insulin_Resistance'] = df2['Insulin_Resistance'].apply(lambda x: 0 if x == 'No' else 1)

df2['Hormonal_Imbalance'] = safe_map(df2['Hormonal_Imbalance'], {
    'No': 0,
    'Yes': 50,
    'No, Yes, not diagnosed by a doctor': 30,
    'Yes Significantly': 100
}, 0)

df2['Hirsutism'] = safe_map(df2['Hirsutism'], {
    'No': 0,
    'Yes': 60,
    'No, Yes, not diagnosed by a doctor': 30
}, 0)

df2['Conception_Difficulty'] = safe_map(df2['Conception_Difficulty'], {
    'No': 0,
    'Yes, diagnosed by a doctor': 70,
    'Yes': 50
}, 0)

df2['Exercise_Frequency'] = safe_map(df2['Exercise_Frequency'], {
    'Never': 0,
    'Rarely': 20,
    '1-2 Times a Week': 40,
    '3-4 Times a Week': 60,
    '6-8 Times a Week': 80,
    'Less than usual': 30,
    'Less than 6 hours': 10,
    '6-8 hours': 90
}, 0)

df2['Exercise_Type'] = safe_map(df2['Exercise_Type'], {
    'No Exercise': 0,
    'Cardio (e.g., running, cycling, swimming)': 30,
    'Strength training (e.g., weightlifting, resistance exercises)': 40,
    'Flexibility and balance (e.g., yoga, pilates)': 20,
    'High-intensity interval training (HIIT)': 50,
    'Cardio, Strength training': 60,
    'Cardio, Flexibility and balance': 65,
    'Strength training, Flexibility and balance': 70,
    'Cardio, Strength training, Flexibility and balance': 80
}, 0)

df2['Exercise_Duration'] = safe_map(df2['Exercise_Duration'], {
    'Not Applicable': 0,
    'Less than 30 minutes': 20,
    '20 minutes': 30,
    '30 minutes': 50,
    '30 minutes to 1 hour': 70,
    '45 minutes': 80,
    'More than 30 minutes': 90
}, 0)

df2['Sleep_Hours'] = safe_map(df2['Sleep_Hours'], {
    'Less than 6 hours': 10,
    '3-4 hours': 0,
    '6-8 hours': 60,
    '9-12 hours': 80,
    'More than 12 hours': 100
}, 0)

df2['Exercise_Benefit'] = safe_map(df2['Exercise_Benefit'], {
    'Not at All': 0,
    'Not Much': 30,
    'Somewhat': 60,
    'Yes Significantly': 100
}, 0)

df2.fillna(df2.median(numeric_only=True), inplace=True)

# --------------------------
# Создание новых признаков
# --------------------------

df2['Hormonal_Index'] = (df2['Hormonal_Imbalance'] + df2['Hirsutism']) / 2

df2['Activity_Index'] = (
    df2['Exercise_Frequency'] + 
    df2['Exercise_Type'] + 
    df2['Exercise_Duration'] + 
    df2['Exercise_Benefit']
) / 4

def categorize_sleep(s):
    if s <= 30:
        return 0
    elif s <= 70:
        return 1
    else:
        return 2
df2['Sleep_Category'] = df2['Sleep_Hours'].apply(categorize_sleep)

df2['Age_Activity_Interaction'] = df2['Age'] * df2['Activity_Index']

def age_group(age):
    if age < 20:
        return 0
    elif age < 35:
        return 1
    elif age < 50:
        return 2
    else:
        return 3
df2['Age_Group'] = df2['Age'].apply(age_group)

df2['High_Conception_Risk'] = df2['Conception_Difficulty'].apply(lambda x: 1 if x >= 50 else 0)

df2['Health_Index'] = (df2['Hormonal_Index'] + df2['Activity_Index'] + df2['Sleep_Hours']) / 3

df2.fillna(df2.median(numeric_only=True), inplace=True)

# --------------------------
# Подготовка к сохранению
# --------------------------

df2['ID'] = np.arange(len(df2))
cols = df2.columns.tolist()
cols.insert(0, cols.pop(cols.index('ID')))
df2 = df2[cols]

for column in df2.columns:
    print(f"Столбец '{column}' содержит следующие классы: {df2[column].unique()}")

df2.to_csv('modified_test_automated.csv', index=False)
print(df2.head())



import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

def load_data():
    df_train = pd.read_csv('modified_train_new.csv')
    df_test = pd.read_csv('modified_test_new.csv')
    
    if 'ID' in df_train.columns:
        df_train = df_train.drop(columns=['ID'])
    return df_train, df_test

def prepare_data(df_train, df_test, target='PCOS'):
    features = [col for col in df_train.columns if col != target]
    X_train = df_train[features].copy()
    y_train = df_train[target].copy()
    
    X_test = df_test.copy()
    if 'ID' in X_test.columns:
        X_test = X_test.drop(columns=['ID'])
    return X_train, y_train, X_test

def build_pipeline():
    lr_params = {
        'solver': 'lbfgs',
        'max_iter': 1000,
        'random_state': 42,
        'C': 1.0,
        'tol': 1e-4,
        'fit_intercept': True,
        'class_weight': 'balanced'
    }
    
    rf_params = {
        'n_estimators': 100,
        'max_depth': None,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'bootstrap': True,
        'random_state': 42,
        'class_weight': 'balanced',
        'n_jobs': -1
    }
    
    xgb_params = {
        'objective': 'binary:logistic',
        'use_label_encoder': False,
        'eval_metric': 'auc',
        'random_state': 42,
        'max_depth': 5,
        'n_estimators': 100,
        'reg_lambda': 7,
        'learning_rate': 0.01
    }
    
    estimators = [
        ('lr', LogisticRegression(**lr_params)),
        ('rf', RandomForestClassifier(**rf_params)),
        ('xgb', xgb.XGBClassifier(**xgb_params))
    ]
    
    final_estimator = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    
    stacking_clf = StackingClassifier(
        estimators=estimators,
        final_estimator=final_estimator,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        n_jobs=-1,
        passthrough=True
    )
    
    pipeline = Pipeline([
        ('scaler', RobustScaler()),
        ('stacking', stacking_clf)
    ])
    
    return pipeline

def main():
    df_train, df_test = load_data()
    X_train, y_train, X_test = prepare_data(df_train, df_test, target='PCOS')
    
    pipeline = build_pipeline()
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
    print("CV ROC AUC: {:.4f} ± {:.4f}".format(cv_scores.mean(), cv_scores.std()))
    
    pipeline.fit(X_train, y_train)
    
    y_test_pred_prob = pipeline.predict_proba(X_test)[:, 1]
    
    if 'ID' in df_test.columns:
        df_pred = pd.DataFrame({'ID': df_test['ID'], 'PCOS': y_test_pred_prob})
    else:
        df_pred = pd.DataFrame({'ID': np.arange(len(df_test)), 'PCOS': y_test_pred_prob})
    
    df_pred.to_csv('Stacking_predictions.csv', index=False, float_format='%.10f')
    print("Предсказания метамодели успешно сохранены в 'Stacking_predictions.csv'.")
    print(df_pred.head())

if __name__ == '__main__':
    main()


import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV 

def load_data():
    df_train = pd.read_csv('modified_train_new.csv')
    df_test = pd.read_csv('modified_test_new.csv')
    
    if 'ID' in df_train.columns:
        df_train = df_train.drop(columns=['ID'])
    return df_train, df_test

def prepare_data(df_train, df_test, target='PCOS'):
    features = [col for col in df_train.columns if col != target]
    X_train = df_train[features].copy()
    y_train = df_train[target].copy()
    
    X_test = df_test.copy()
    if 'ID' in X_test.columns:
        X_test = X_test.drop(columns=['ID'])
    return X_train, y_train, X_test

def build_pipeline():
    lr_params = {
        'solver': 'lbfgs',
        'max_iter': 1000,
        'random_state': 42,
        'C': 1.0,
        'tol': 1e-4,
        'fit_intercept': True,
        'class_weight': 'balanced'
    }
    
    rf_params = {
        'n_estimators': 100,
        'max_depth': None,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'bootstrap': True,
        'random_state': 42,
        'class_weight': 'balanced',
        'n_jobs': -1
    }
    
    xgb_params = {
        'objective': 'binary:logistic',
        'use_label_encoder': False,
        'eval_metric': 'auc',
        'random_state': 42,
        'max_depth': 5,
        'n_estimators': 100,
        'reg_lambda': 7,
        'learning_rate': 0.01
    }

    lr_model = LogisticRegression(**lr_params)
    rf_model = RandomForestClassifier(**rf_params)
    xgb_model = xgb.XGBClassifier(**xgb_params)
    
    lr_calibrated = CalibratedClassifierCV(lr_model, cv=5, method='isotonic')
    rf_calibrated = CalibratedClassifierCV(rf_model, cv=5, method='isotonic')
    xgb_calibrated = CalibratedClassifierCV(xgb_model, cv=5, method='isotonic')
    
    estimators = [
        ('lr', lr_calibrated),
        ('rf', rf_calibrated),
        ('xgb', xgb_calibrated)
    ]
    
    final_estimator = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    
    stacking_clf = StackingClassifier(
        estimators=estimators,
        final_estimator=final_estimator,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        n_jobs=-1,
        passthrough=True  
    )
    
    pipeline = Pipeline([
        ('scaler', RobustScaler()),
        ('stacking', stacking_clf)
    ])
    
    return pipeline

def main():
    df_train, df_test = load_data()
    X_train, y_train, X_test = prepare_data(df_train, df_test, target='PCOS')
    
    pipeline = build_pipeline()
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
    print("CV ROC AUC: {:.4f} ± {:.4f}".format(cv_scores.mean(), cv_scores.std()))
    
    pipeline.fit(X_train, y_train)
    
    y_test_pred_prob = pipeline.predict_proba(X_test)[:, 1]
    
    if 'ID' in df_test.columns:
        df_pred = pd.DataFrame({'ID': df_test['ID'], 'PCOS': y_test_pred_prob})
    else:
        df_pred = pd.DataFrame({'ID': np.arange(len(df_test)), 'PCOS': y_test_pred_prob})
    
    df_pred.to_csv('Stacking_predictions.csv', index=False, float_format='%.10f')
    print("Предсказания метамодели успешно сохранены в 'Stacking_predictions.csv'.")
    print(df_pred.head())

if __name__ == '__main__':
    main()


