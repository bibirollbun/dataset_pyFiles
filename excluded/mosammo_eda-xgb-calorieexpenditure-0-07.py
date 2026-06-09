import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, RobustScaler, PolynomialFeatures
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


train_df = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv', index_col='id')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv', index_col='id')


train_df.info()


train_df.describe()


train_df['Height'] = train_df['Height']/100. #convert to meters
train_df['Duration_Heart_Rate'] = train_df['Duration'] * train_df['Heart_Rate']
train_df['BMI'] = train_df['Weight'] / (train_df['Height'] ** 2)


test_df['Height'] = test_df['Height'] / 100.
test_df['Duration_Heart_Rate'] = test_df['Duration'] * test_df['Heart_Rate']
test_df['BMI'] = test_df['Weight'] / (test_df['Height'] ** 2)


features_for_poly = ['Age', 'Weight', 'Duration']  


poly = PolynomialFeatures(degree=2, include_bias=False)

poly_features = poly.fit_transform(train_df[features_for_poly])
poly_feature_names = poly.get_feature_names_out(features_for_poly)

train_df = train_df.drop(features_for_poly, axis=1)  

poly_df = pd.DataFrame(poly_features, columns=poly_feature_names)
train_df = pd.concat([train_df, poly_df], axis=1)


poly_features_test = poly.transform(test_df[features_for_poly])

test_df = test_df.drop(features_for_poly, axis=1)  


poly_df_test = pd.DataFrame(poly_features_test, columns=poly_feature_names)

poly_df_test.index = test_df.index
test_df = pd.concat([test_df, poly_df_test], axis=1)


train_df['Sex'] = train_df['Sex'].map({'female': 0, 'male': 1})
test_df['Sex'] = test_df['Sex'].map({'female': 0, 'male': 1})


train_df.columns


numerical_features = ['Height', 'Heart_Rate', 'Body_Temp', 'Calories',
       'Duration_Heart_Rate', 'BMI', 'Age', 'Weight', 'Duration', 'Age^2',
       'Age Weight', 'Age Duration', 'Weight^2', 'Weight Duration',
       'Duration^2']


for feature in numerical_features:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Boxplot
    sns.boxplot(y=train_df[feature], ax=axes[0])
    axes[0].set_ylabel(feature)
    axes[0].set_title(f'Boxplot of {feature}')

    # Violin plot
    sns.violinplot(y=train_df[feature], ax=axes[1])
    axes[1].set_ylabel(feature)
    axes[1].set_title(f'Violin Plot of {feature}')

    # Distribution plot with KDE
    sns.histplot(train_df[feature], bins=30, kde=True, ax=axes[2], color='steelblue')
    axes[2].set_title(f'Distribution Plot of {feature}')
    axes[2].set_xlabel(feature)
    axes[2].set_ylabel('Count')

    plt.tight_layout()
    plt.show()


for col in numerical_features:
    lower = train_df[col].quantile(0.01)
    upper = train_df[col].quantile(0.99)
    
    train_df[col] = np.clip(train_df[col], lower, upper)
    if col == 'Calories':
        continue
    test_df[col] = np.clip(test_df[col], lower, upper)



train_df['Sex'] = train_df['Sex'].map({'female': 0, 'male': 1})
test_df['Sex'] = test_df['Sex'].map({'female': 0, 'male': 1})


numerical_features.remove('Calories')


scaler = RobustScaler()
train_df[numerical_features] = scaler.fit_transform(train_df[numerical_features])
test_df[numerical_features] = scaler.transform(test_df[numerical_features])


train_df['Calories'] = np.log1p(train_df[['Calories']])


X = train_df.drop('Calories', axis=1)
y = train_df['Calories']
X_test = test_df.copy()


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 0.9, 1.0],
}


grid_search = GridSearchCV(estimator=xgb.XGBRegressor(random_state=42),
                           param_grid=param_grid,
                           scoring='neg_mean_squared_error',
                           cv=3,
                           verbose=1)


grid_search.fit(X_train, y_train)


best_model = grid_search.best_estimator_


print("Best hyperparameters:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)


sub = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')


sub['Calories'] = np.expm1(best_model.predict(X_test))
sub.to_csv('sub.csv', index=False)




