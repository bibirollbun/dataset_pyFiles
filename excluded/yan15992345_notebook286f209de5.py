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


# 学号: 2024423320125, 姓名: 严俊浩

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

def load_data():
    train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
    test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
    return train, test

def preprocess_data(train, test):
    train_id = train['id']
    test_id = test['id']
    y = train['Fertilizer Name']
    
    numerical_cols = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
    categorical_cols = ['Soil Type', 'Crop Type']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=[
                ('scaler', StandardScaler())
            ]), numerical_cols),
            ('cat', Pipeline(steps=[
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ]), categorical_cols)
        ])
    
    X = train.drop(['id', 'Fertilizer Name'], axis=1)
    X_test = test.drop('id', axis=1)
    
    X_processed = preprocessor.fit_transform(X)
    X_test_processed = preprocessor.transform(X_test)
    
    return X_processed, X_test_processed, y, test_id, preprocessor

def train_model(X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    model = LGBMClassifier(
        objective='multiclass',
        metric='multi_logloss',
        num_class=len(y.unique()),
        n_estimators=500,
        learning_rate=0.05,
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
    )
    
    y_val_probs = model.predict_proba(X_val)
    y_val_onehot = pd.get_dummies(y_val).values
    map5_score = compute_map5(y_val, y_val_probs, model)  
    print(f"验证集MAP@5得分: {map5_score:.4f}")
    
    return model
def compute_map5(y_true, y_pred_probs, model):
    U = len(y_true)
    map5_total = 0.0
    for i in range(U):
        true_label = y_true.iloc[i]
        probs = y_pred_probs[i]
        top5_indices = np.argsort(probs)[::-1][:5]
        top5_labels = model.classes_[top5_indices]
        
        relevant_count = 0
        precision_sum = 0
        for k in range(5):
            pred_label = top5_labels[k]
            if pred_label == true_label:
                relevant_count += 1
                precision_sum += relevant_count / (k + 1)
        
        map5_total += precision_sum
    return map5_total / U
def analyze_feature_importance(model, preprocessor, numerical_cols, categorical_cols):
    cat_features = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_cols)
    all_features = numerical_cols + cat_features.tolist()
    
    plt.figure(figsize=(10, 6))
    plt.barh(all_features, model.feature_importances_)
    plt.xlabel('Score of importance')
    plt.ylabel('Name of features')
    plt.title('The analysis of importance of soil features')
    plt.savefig('feature_importance.png')
    plt.show()

def generate_predictions(model, X_test, test_id):
    probs = model.predict_proba(X_test)
    
    top5_indices = np.argsort(probs, axis=1)[:, -5:][:, ::-1]
    top5_labels = model.classes_[top5_indices]
    
    submission = pd.DataFrame({
        'id': test_id,
        'Fertilizer1': top5_labels[:, 0],
        'Fertilizer2': top5_labels[:, 1],
        'Fertilizer3': top5_labels[:, 2],
        'Fertilizer4': top5_labels[:, 3],
        'Fertilizer5': top5_labels[:, 4]
    })
    
    submission.to_csv('submission.csv', index=False)
    print("测试集预测结果已保存至submission.csv")
    return submission

def main():
    train, test = load_data()
    
    X, X_test, y, test_id, preprocessor = preprocess_data(train, test)
    
    model = train_model(X, y)
    
    numerical_cols = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
    categorical_cols = ['Soil Type', 'Crop Type']
    analyze_feature_importance(model, preprocessor, numerical_cols, categorical_cols)
    
    submission = generate_predictions(model, X_test, test_id)
    data=pd.read_csv('submission.csv',encoding='utf-8')
    print(data)

if __name__ == "__main__":
    main()


