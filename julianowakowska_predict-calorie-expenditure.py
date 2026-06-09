import numpy as np
import pandas as pd

training_dataset = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
training_dataset.head()


training_dataset.info()


categorical_features = ['Sex']
numerical_features = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']


training_dataset = training_dataset.drop(columns=['id'])


from sklearn.model_selection import train_test_split

X = training_dataset.iloc[:, :-1]
y = training_dataset.iloc[:, -1]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


import matplotlib.pyplot as plt
import seaborn as sns

for col in numerical_features:
    sns.scatterplot(x=X[col], y=y)
    plt.title(f'{col} vs Target')
    plt.show()


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


class DistanceFeatureGenerator(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = pd.DataFrame(X, columns=['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp'])
        dist_height_weight = np.abs(df['Height'] - df['Weight']).values.reshape(-1, 1)
        dist_age_duration = np.abs(df['Age'] - df['Duration']).values.reshape(-1, 1)
        dist_heart_temp = np.abs(df['Heart_Rate'] - df['Body_Temp']).values.reshape(-1, 1)
        return np.hstack([dist_height_weight, dist_age_duration, dist_heart_temp])

    def get_feature_names_out(self, input_features=None):
        return np.array([
            'dist_height_weight',
            'dist_age_duration',
            'dist_heart_temp'
        ])


distance_and_scaling = Pipeline([
    ('distance_features', DistanceFeatureGenerator()),
    ('scaler', StandardScaler())
])


class CombinedFeatureGenerator(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self 
        
    def transform(self, X):
        df = pd.DataFrame(X, columns=['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp'])
        
        combined_sum_height_weight = (df['Height'] + df['Weight']).values.reshape(-1, 1)
        combined_ratio_height_weight = (df['Weight'] / (df['Height'] + 1e-5)).values.reshape(-1, 1)
        combined_multip_height_weight = (df['Height']*100 + df['Weight']).values.reshape(-1, 1)
        combined_multip_weight_height = (df['Weight']*100 + df['Height']).values.reshape(-1, 1)
    
        combined_sum_age_duration = (df['Age'] + df['Duration']).values.reshape(-1, 1)
        combined_ratio_age_duration  = (df['Age'] / (df['Duration'] + 1e-5)).values.reshape(-1, 1)
        combined_multip_age_duration = (df['Age']*100 + df['Duration']).values.reshape(-1, 1)
        combined_multip_duration_age = (df['Duration']*100 + df['Age']).values.reshape(-1, 1)
    
        combined_sum_heart_body = (df['Heart_Rate'] + df['Body_Temp']).values.reshape(-1, 1)
        combined_ratio_heart_body = (df['Heart_Rate'] / (df['Body_Temp'] + 1e-5)).values.reshape(-1, 1)
        combined_multip_heart_body = (df['Heart_Rate']*100 + df['Body_Temp']).values.reshape(-1, 1)
        combined_multip_body_heart = (df['Body_Temp']*100 + df['Heart_Rate']).values.reshape(-1, 1)
        
        return np.hstack([combined_sum_height_weight, combined_ratio_height_weight, combined_multip_height_weight, combined_multip_weight_height,
                          combined_sum_age_duration, combined_ratio_age_duration, combined_multip_age_duration, combined_multip_duration_age,
                          combined_sum_heart_body, combined_ratio_heart_body, combined_multip_heart_body, combined_multip_body_heart])

    def get_feature_names_out(self, input_features=None):
        return np.array([
            'combined_sum_height_weight',
            'combined_ratio_height_weight',
            'combined_multip_height_weight',
            'combined_multip_weight_height',
            'combined_sum_age_duration',
            'combined_ratio_age_duration',
            'combined_multip_age_duration',
            'combined_multip_duration_age',
            'combined_sum_heart_body',
            'combined_ratio_heart_body',
            'combined_multip_heart_body',
            'combined_multip_body_heart'
        ])


combine_and_scaling = Pipeline([
    ('combine_features', CombinedFeatureGenerator()),
    ('scaler', StandardScaler())
])


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures


preprocessor = ColumnTransformer(
    transformers=[
        ('sex_ohe', OneHotEncoder(drop='if_binary'), categorical_features),
        ('num_scaler', StandardScaler(), numerical_features),
        ('distance', distance_and_scaling, numerical_features),
        ('combination', combine_and_scaling, numerical_features),
        ('poly_temp', PolynomialFeatures(degree=2, include_bias=False), ['Body_Temp']) 
    ],
    remainder='passthrough'
)


from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', DecisionTreeRegressor(max_depth=20))
])


evaluation = {}


from sklearn.metrics import mean_squared_log_error, r2_score
from sklearn.model_selection import KFold

kf = KFold(n_splits=3)

total_r2, total_msle = [], []
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    predictions = np.maximum(predictions, 0)


    msle = mean_squared_log_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    total_r2.append(r2)
    total_msle.append(msle)
    
    print(f"R²: {r2:.6f}, MSLE: {msle:.6f}")


eval_dict = {
    'LinearRegression()': [0.9827575777612246, 0.15238995310268266],
    'Ridge()': [0.9827438855667877, 0.15256878436341714],
    'Lasso()': [0.9690479000146474, 0.29291966226444816],
    'KNeighborsRegressor()': [0.990020003918492, 0.010232348993266166],
    'DecisionTreeRegressor()': [0.9929439951803151, 0.007519796360730058],
    'DecisionTreeRegressor(max_depth=10)': [0.9923650550415015, 0.007485803366431208],
    'DecisionTreeRegressor(max_depth=20)': [0.9932530368283617, 0.006649053576176347],
    'RandomForestRegressor(n_jobs=-1)': [0.9963019233836871, 0.004038057184643943]
}

models = list(eval_dict.keys())
r2_scores = [eval_dict[model][0] for model in models]
msle_scores = [eval_dict[model][1] for model in models]


x = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots()
bars1 = ax.bar(x - width/2, r2_scores, width, label='R²')
bars2 = ax.bar(x + width/2, msle_scores, width, label='MSLE')

# Labels and titles
ax.set_ylabel('Scores')
ax.set_title('Model Evaluation: R² and MSLE')
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=90)
ax.legend()

# Bar labels
ax.bar_label(bars1, fmt='%.4f', padding=3)
ax.bar_label(bars2, fmt='%.4f', padding=3)

plt.show()


evaluation


pipeline.fit(X_train, y_train)
model = pipeline.named_steps['model']


importances = model.feature_importances_
feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()

importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
importance_df = importance_df.sort_values('Importance', ascending=False)


plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.gca().invert_yaxis()
plt.xlabel("Importance Score")
plt.title("Feature Importances")
plt.tight_layout()
plt.show()



test_dataset = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")


predictions = pipeline.predict(test_dataset)

output_df = pd.DataFrame({
    'index': test_dataset.id,
    'prediction': predictions
})

output_df.to_csv('submission.csv', index=False)

