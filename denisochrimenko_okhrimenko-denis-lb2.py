# Forest Cover Type Prediction - Улучшенная версия
# score с 0.74 до 0.75+

# ==================== ИМПОРТ БИБЛИОТЕК ====================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import warnings
warnings.filterwarnings('ignore')


# ==================== ЗАГРУЗКА ДАННЫХ ====================

df = pd.read_csv("/kaggle/input/forest-cover-type-prediction/train.csv")
test_df = pd.read_csv("/kaggle/input/forest-cover-type-prediction/test.csv")

print('Размерность обучающей выборки:', df.shape)
print('Размерность тестовой выборки:', test_df.shape)
print('\nПервые строки данных:')
print(df.head())


# ==================== FEATURE ENGINEERING ====================
# Создаём новые признаки для улучшения качества модели

def create_features(data):
    """
    Creates additional features based on domain knowledge
    """
    df = data.copy()
    
    # 1. Euclidean distance to hydrology (комбинация горизонтального и вертикального расстояния)
    df['Distance_To_Hydrology'] = np.sqrt(
        df['Horizontal_Distance_To_Hydrology']**2 + 
        df['Vertical_Distance_To_Hydrology']**2
    )
    
    # 2. Mean hillshade (средняя освещённость в течение дня)
    df['Mean_Hillshade'] = (
        df['Hillshade_9am'] + 
        df['Hillshade_Noon'] + 
        df['Hillshade_3pm']
    ) / 3
    
    # 3. Hillshade difference (изменение освещённости от утра к вечеру)
    df['Hillshade_Difference'] = df['Hillshade_3pm'] - df['Hillshade_9am']
    
    # 4. Total distance (суммарное расстояние до ключевых объектов)
    df['Total_Distance'] = (
        df['Horizontal_Distance_To_Hydrology'] + 
        df['Horizontal_Distance_To_Roadways'] + 
        df['Horizontal_Distance_To_Fire_Points']
    )
    
    # 5. Elevation and slope interaction (взаимодействие высоты и крутизны)
    df['Elevation_Slope'] = df['Elevation'] * df['Slope']
    
    # 6. Aspect in radians for circular features (преобразование аспекта в циклические признаки)
    df['Aspect_Sin'] = np.sin(df['Aspect'] * np.pi / 180)
    df['Aspect_Cos'] = np.cos(df['Aspect'] * np.pi / 180)
    
    # 7. Distance to roadways vs fire points ratio
    df['Road_Fire_Ratio'] = df['Horizontal_Distance_To_Roadways'] / (
        df['Horizontal_Distance_To_Fire_Points'] + 1
    )
    
    return df


# Применяем feature engineering
print("\n" + "="*60)
print("СОЗДАНИЕ НОВЫХ ПРИЗНАКОВ")
print("="*60)

df = create_features(df)
test_df = create_features(test_df)

print(f"Новая размерность обучающей выборки: {df.shape}")
print(f"Добавлено признаков: {df.shape[1] - 56}")


# ==================== ПОДГОТОВКА ДАННЫХ ====================

def prepare_data(df, test_df):
    """
    Splits data into train/validation and prepares features
    """
    # Separate target variable
    y = df["Cover_Type"].copy()
    X = df.drop(["Cover_Type", "Id"], axis=1).copy()
    
    # Test data
    test_ids = test_df['Id'].copy()
    X_test = test_df.drop('Id', axis=1).copy()
    
    # Train-validation split (80-20)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y  # maintain class distribution
    )
    
    print("\nРазмеры выборок:")
    print(f"Train: {X_train.shape}")
    print(f"Valid: {X_valid.shape}")
    print(f"Test: {X_test.shape}")
    
    return X_train, X_valid, y_train, y_valid, X_test, test_ids

X_train, X_valid, y_train, y_valid, X_test, test_ids = prepare_data(df, test_df)



# ==================== ПОСТРОЕНИЕ МОДЕЛЕЙ ====================

print("\n" + "="*60)
print("ОБУЧЕНИЕ МОДЕЛЕЙ")
print("="*60)

# Model 1: Random Forest with optimized parameters
print("\n1. Random Forest Classifier")
rf_model = RandomForestClassifier(
    n_estimators=300,      # More trees for better performance
    max_depth=25,          # Deeper trees to capture complex patterns
    min_samples_split=2,
    min_samples_leaf=1,
    criterion='entropy',   # Information gain
    max_features='sqrt',   # Randomness for robustness
    random_state=42,
    n_jobs=-1,
    verbose=0
)

rf_model.fit(X_train, y_train)
rf_score = rf_model.score(X_valid, y_valid)
print(f"Random Forest Validation Accuracy: {rf_score:.4f}")



# Model 2: Extra Trees (often better than RF for this type of data)
print("\n2. Extra Trees Classifier")
et_model = ExtraTreesClassifier(
    n_estimators=300,
    max_depth=25,
    min_samples_split=2,
    min_samples_leaf=1,
    criterion='entropy',
    max_features='sqrt',
    random_state=42,
    n_jobs=-1,
    verbose=0
)

et_model.fit(X_train, y_train)
et_score = et_model.score(X_valid, y_valid)
print(f"Extra Trees Validation Accuracy: {et_score:.4f}")


# Model 3: Voting Ensemble (combining both models)
print("\n3. Voting Ensemble")
voting_model = VotingClassifier(
    estimators=[
        ('rf', rf_model),
        ('et', et_model)
    ],
    voting='soft',  # Use probabilities for better results
    n_jobs=-1
)

voting_model.fit(X_train, y_train)
voting_score = voting_model.score(X_valid, y_valid)
print(f"Voting Ensemble Validation Accuracy: {voting_score:.4f}")



# Select best model
models = {
    'Random Forest': (rf_model, rf_score),
    'Extra Trees': (et_model, et_score),
    'Voting Ensemble': (voting_model, voting_score)
}

best_model_name = max(models, key=lambda k: models[k][1])
best_model = models[best_model_name][0]
best_score = models[best_model_name][1]

print("\n" + "="*60)
print(f"ЛУЧШАЯ МОДЕЛЬ: {best_model_name}")
print(f"Точность на валидации: {best_score:.4f} ({best_score*100:.2f}%)")
print("="*60)


# ==================== ОЦЕНКА КАЧЕСТВА ====================

def evaluate_model(model, X_val, y_val):
    """
    Evaluates model performance with confusion matrix and classification report
    """
    y_pred = model.predict(X_val)
    
    # Confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_val, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.xlabel('Предсказанный класс', fontsize=12)
    plt.ylabel('Истинный класс', fontsize=12)
    plt.title(f'Confusion Matrix - {type(model).__name__}', fontsize=14)
    plt.show()
    
    # Classification report
    print("\nОтчёт по классам:")
    print(classification_report(y_val, y_pred))
    
    return accuracy_score(y_val, y_pred)

print("\nДетальная оценка лучшей модели:")
final_accuracy = evaluate_model(best_model, X_valid, y_valid)


# ==================== FEATURE IMPORTANCE ====================

if hasattr(best_model, 'feature_importances_'):
    # Get feature importances
    if isinstance(best_model, VotingClassifier):
        # Average importances from ensemble
        importances = np.mean([
            est.feature_importances_ 
            for name, est in best_model.named_estimators_.items()
        ], axis=0)
    else:
        importances = best_model.feature_importances_
    
    # Create dataframe
    feature_imp = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    print("\nТоп-15 наиболее важных признаков:")
    print(feature_imp.head(15))
    
    # Plot top 20 features
    plt.figure(figsize=(10, 8))
    top_features = feature_imp.head(20)
    plt.barh(range(len(top_features)), top_features['Importance'])
    plt.yticks(range(len(top_features)), top_features['Feature'])
    plt.xlabel('Важность признака')
    plt.title('Топ-20 наиболее важных признаков')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()


# ==================== СОЗДАНИЕ SUBMISSION ====================

print("\n" + "="*60)
print("СОЗДАНИЕ ФАЙЛА ДЛЯ ОТПРАВКИ")
print("="*60)

# Make predictions on test set
test_predictions = best_model.predict(X_test)

# Create submission file
submission = pd.DataFrame({
    'Id': test_ids,
    'Cover_Type': test_predictions
})

# Save to CSV
submission_file = '/kaggle/working/submission.csv'
submission.to_csv(submission_file, index=False)

print(f"\n✓ Файл submission.csv успешно создан")
print(f"✓ Модель: {best_model_name}")
print(f"✓ Точность на валидации: {best_score:.4f}")
print(f"✓ Количество предсказаний: {len(submission)}")
print("\nРаспределение предсказанных классов:")
print(submission['Cover_Type'].value_counts().sort_index())

print("\n" + "="*60)
print("ГОТОВО! Файл готов для отправки на Kaggle")
print("="*60)

