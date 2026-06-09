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


# Импорт необходимых библиотек
import pandas as pd  
import numpy as np   
from sklearn.ensemble import RandomForestClassifier  # Модель случайного леса
from sklearn.model_selection import cross_val_score  # Кросс-валидация
from sklearn.preprocessing import StandardScaler    # Масштабирование признаков
from sklearn.metrics import accuracy_score          # Метрика точности
import warnings
warnings.filterwarnings('ignore')  


train_data = pd.read_csv('/kaggle/input/forest-cover-type-prediction/train.csv')  
test_data = pd.read_csv('/kaggle/input/forest-cover-type-prediction/test.csv')    


# Разделение на признаки и целевую переменную
X_train = train_data.drop(['Id', 'Cover_Type'], axis=1)  # Признаки (все кроме ID и целевой переменной)
y_train = train_data['Cover_Type']  # Целевая переменная (тип лесного покрова)
X_test = test_data.drop('Id', axis=1)  # Признаки тестовых данных

# Сохраняем Id для submission
test_ids = test_data['Id']  

# Предобработка данных
scaler = StandardScaler()  # Инициализация стандартизатора

# Масштабируем числовые признаки (исключая бинарные)
numeric_features = ['Elevation', 'Aspect', 'Slope', 
                   'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',
                   'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 
                   'Hillshade_Noon', 'Hillshade_3pm', 
                   'Horizontal_Distance_To_Fire_Points'] 

# Применяем масштабирование только к числовым признакам
X_train_scaled = X_train.copy()  
X_test_scaled = X_test.copy()    

X_train_scaled[numeric_features] = scaler.fit_transform(X_train[numeric_features]) 
X_test_scaled[numeric_features] = scaler.transform(X_test[numeric_features])        

# Создание и обучение модели Random Forest
print("\nTraining Random Forest model...")
model = RandomForestClassifier(
    n_estimators=200,      
    max_depth=30,          
    min_samples_split=5,   
    min_samples_leaf=2,    
    random_state=42,       
    n_jobs=-1             
)

# Кросс-валидация
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=3, scoring='accuracy')  
print(f"Cross-validation scores: {cv_scores}")          
print(f"Mean CV accuracy: {cv_scores.mean():.4f}")      

# Обучение модели на всех тренировочных данных
model.fit(X_train_scaled, y_train)  

# Предсказание на тренировочных данных для проверки
train_predictions = model.predict(X_train_scaled)        
train_accuracy = accuracy_score(y_train, train_predictions)  
print(f"Training accuracy: {train_accuracy:.4f}")       

# Предсказание на тестовых данных
print("\nMaking predictions on test data...")
test_predictions = model.predict(X_test_scaled)  

# Создание submission файла
submission = pd.DataFrame({
    'Id': test_ids,              
    'Cover_Type': test_predictions  
})


print("\nPrediction distribution:")
print(submission['Cover_Type'].value_counts().sort_index())  

# Сохранение результатов
submission.to_csv('submission.csv', index=False)  
print("\nSubmission file saved as 'submission.csv'")


feature_importance = pd.DataFrame({
    'feature': X_train.columns,              
    'importance': model.feature_importances_ 
}).sort_values('importance', ascending=False)  

print("\nTop 10 most important features:")
print(feature_importance.head(10))  




