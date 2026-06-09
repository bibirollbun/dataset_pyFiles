# ライブラリインポート
import numpy as np
import pandas as pd
import os

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
# from xgboost import XGBClassifier
from catboost import CatBoostClassifier

import plotly.express as px


# ファイル名を出力
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# 学習データ
df_train = pd.read_csv('/kaggle/input/exploring-predictive-health-factors/train.csv')
df_train.head()


df_train.info()


df_train.describe()


# カテゴリ変数の可視化（例: Ageの分布）
fig_age = px.histogram(df_train, x="Age", title="Ageの分布", text_auto=True)
fig_age.show()


# 連続変数の可視化
fig_weight = px.histogram(df_train, x="Weight_kg", title="体重の分布", text_auto=True)
fig_weight.show()


# カテゴリ変数の可視化（例: PCOSの分布）
fig_pcos = px.histogram(df_train, x="PCOS", title="PCOSの分布", text_auto=True)
fig_pcos.show()


# カテゴリ変数の可視化
fig_Hormonal_Imbalance = px.histogram(df_train, x="Hormonal_Imbalance", title="Hormonal_Imbalanceの分布", text_auto=True)
fig_Hormonal_Imbalance.show()


# カテゴリ変数の可視化
fig_Hyperandrogenism = px.histogram(df_train, x="Hyperandrogenism", title="Hyperandrogenismの分布", text_auto=True)
fig_Hyperandrogenism.show()


# カテゴリ変数の可視化
fig_Hirsutism = px.histogram(df_train, x="Hirsutism", title="Hirsutismの分布", text_auto=True)
fig_Hirsutism.show()


# カテゴリ変数の可視化
fig_Conception_Difficulty = px.histogram(df_train, x="Conception_Difficulty", title="Conception_Difficultyの分布", text_auto=True)
fig_Conception_Difficulty.show()


# カテゴリ変数の可視化
fig_Insulin_Resistance = px.histogram(df_train, x="Insulin_Resistance", title="Insulin_Resistanceの分布", text_auto=True)
fig_Insulin_Resistance.show()


# Exercise_Frequencyの頻度を棒グラフで表示
fig_exercise = px.bar(df_train, x="Exercise_Frequency", title="運動頻度の分布", text_auto=True)
fig_exercise.show()


# カテゴリ変数の可視化
fig_Exercise_Type = px.histogram(df_train, x="Exercise_Type", title="Exercise_Typeの分布", text_auto=True)
fig_Exercise_Type.show()


# カテゴリ変数の可視化
fig_Exercise_Duration = px.histogram(df_train, x="Exercise_Duration", title="Exercise_Durationの分布", text_auto=True)
fig_Exercise_Type.show()




# Sleep_Hours の分布を円グラフで表示
fig_sleep = px.pie(df_train, names="Sleep_Hours", title="睡眠時間の割合")
fig_sleep.show()


# カテゴリ変数の可視化
fig_Exercise_Benefit = px.histogram(df_train, x="Exercise_Benefit", title="Exercise_Benefitの分布", text_auto=True)
fig_Exercise_Benefit.show()




 # テストデータの読み込み
df_test = pd.read_csv('/kaggle/input/exploring-predictive-health-factors/test.csv')


# Process age ranges
def process_age(age_str):
    if pd.isna(age_str):
        return np.nan
    age_str = str(age_str).lower().strip()
    
    age_mapping = {
        '20-25': 22.5,
        '15-20': 17.5,
        '45 and above': 47.5,
        '22-25': 23.5,
        '50-60': 55,
        '30-35': 32.5,
        '35-44': 39.5,
        '25-30': 27.5,
        '25-25': 25,
        'less than 20': 19,
        'less than 20)': 19,
        'less than 20-25': 19,
        '30-25': 27.5,
        '30-40': 35,
        '30-30': 30,
        'less than 20-25': 22.5,
        '45-49': 47
    }
    
    # First try the mapping
    if age_str in age_mapping:
        return age_mapping[age_str]
    
    # Handle ranges
    if '-' in age_str:
        low, high = map(float, age_str.split('-'))
        return (low + high) / 2
        
    # Try direct conversion
    try:
        return float(age_str)
    except ValueError:
        # Default value if nothing else works
        return 25.0  # median age as fallback


# Define category mappings
category_mappings = {
    'Hormonal_Imbalance': {'No': 0, 'Yes': 1, 'Yes Significantly': 1, 'No, Yes, not diagnosed by a doctor': 0.5},
    'Hyperandrogenism': {'No': 0, 'Yes': 1},
    'Hirsutism': {'No': 0, 'Yes': 1, 'No, Yes, not diagnosed by a doctor': 0.5},
    'Conception_Difficulty': {'No': 0, 'Yes': 1, 'Yes, diagnosed by a doctor': 1, 'No, Yes, not diagnosed by a doctor': 0.5},
    'Insulin_Resistance': {'No': 0, 'Yes': 1, 'No, Yes, not diagnosed by a doctor': 0.5},
    'Exercise_Frequency': {
        'Never': 0, 'Rarely': 1, '1-2 Times a Week': 2, 
        '3-4 Times a Week': 3, '6-8 Times a Week': 4
    },
    'Exercise_Duration': {
        'Not Applicable': 0, 'Less than 30 minutes': 1, '30 minutes': 2,
        '45 minutes': 3, 'More than 30 minutes': 3, '30 minutes to 1 hour': 3
    },
    'Sleep_Hours': {
        '3-4 hours': 3.5, 'Less than 6 hours': 5, '6-8 hours': 7,
        '9-12 hours': 10.5, 'More than 12 hours': 13
    },
    'Exercise_Benefit': {
        'Not at All': 0, 'Not Much': 1, 'Somewhat': 2, 'Yes Significantly': 3
    }
}


# Process Exercise Type
def process_exercise_type(x):
    if pd.isna(x):
        return 0
    if isinstance(x, str):
        if 'No Exercise' in x:
            return 0
        elif ',' in x:
            return 4  # Multiple types
        elif 'Cardio' in x:
            return 1
        elif 'Strength' in x:
            return 2
        elif 'Flexibility' in x:
            return 3
    return 0


# Preprocess data
def preprocess_data(df):
    df = df.copy()
    
    # Process Age
    df['Age'] = df['Age'].apply(process_age)
    
    # Process Exercise Type
    df['Exercise_Type'] = df['Exercise_Type'].apply(process_exercise_type)
    
    # Apply category mappings
    for col, mapping in category_mappings.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)
    
    return df


# Apply preprocessing
df_train = preprocess_data(df_train)
df_test = preprocess_data(df_test)


df_train.info()


def prepro_nan(df):
    df["Age"] = df["Age"].fillna(df["Age"].median()) 
    df["Weight_kg"] = df["Weight_kg"].fillna(df["Weight_kg"].median()) 
    df["Hormonal_Imbalance"] = df["Hormonal_Imbalance"].fillna(0) 
    df["Hyperandrogenism"] = df["Hyperandrogenism"].fillna(0) 
    df["Hormonal_Imbalance"] = df["Hormonal_Imbalance"].fillna(0) 
    df["Hirsutism"] = df["Hirsutism"].fillna(0) 
    df["Conception_Difficulty"] = df["Conception_Difficulty"].fillna(0) 
    df["Insulin_Resistance"] = df["Insulin_Resistance"].fillna(0) 
    df["Exercise_Frequency"] = df["Exercise_Frequency"].fillna(0)
    df["Sleep_Hours"] = df["Sleep_Hours"].fillna(0) 
    df["Exercise_Duration"] = df["Exercise_Duration"].fillna(0) 
    df["Exercise_Benefit"] = df["Exercise_Benefit"].fillna(0) 

    return df


df_train = prepro_nan(df_train)
df_test = prepro_nan(df_test)


df_train.info()


# Define features
numeric_features = ['Age', 'Weight_kg']
categorical_features = [col for col in df_train.columns if col not in numeric_features + ['ID', 'PCOS']]
categorical_features


# Create preprocessing pipeline　中央値で補完
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median'))
])


categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value=-1)),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])



preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])


# Create model pipeline
model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', CatBoostClassifier(
        depth=4, iterations=300, l2_leaf_reg=9, learning_rate=0.03
    ))
])


# model_pipeline = Pipeline([
#     ('preprocessor', preprocessor),
#     ('classifier', XGBClassifier(
#         learning_rate=0.05,
#         n_estimators=200,
#         max_depth=5,
#         min_child_weight=2,
#         gamma=0.1,
#         subsample=0.8,
#         colsample_bytree=0.8,
#         random_state=42
#     ))
# ])



# Prepare data for modeling
X = df_train[numeric_features + categorical_features]
y = df_train['PCOS'].map({'Yes': 1, 'No': 0})


# Fit model
model_pipeline.fit(X, y)


# Generate predictions
test_features = df_test[numeric_features + categorical_features]
test_features



predictions = model_pipeline.predict_proba(test_features)[:, 1]
predictions


# Create submission
submission = pd.DataFrame({
    'ID': df_test['ID'],
    'PCOS': predictions
})
submission


# Save results
submission.to_csv('pcos_predictions.csv', index=False)


# Display results
print("Submission Preview:")
print(submission.head())
print("\nPrediction Statistics:")
print(f"Number of predictions: {len(predictions)}")
print(f"Prediction range: {predictions.min():.3f} to {predictions.max():.3f}")
print(f"Mean prediction: {predictions.mean():.3f}")


