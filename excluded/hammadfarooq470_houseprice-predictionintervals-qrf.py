import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# Load Data
try:
    train_df = pd.read_csv('dataset.csv')
    test_df = pd.read_csv('test.csv')
    sample_submission_df = pd.read_csv('sample_submission.csv')
except FileNotFoundError:
    print("Demo data being generated...")
    train_df = pd.DataFrame({
        'id': range(1000),
        'feature1': np.random.rand(1000),
        'feature2': np.random.randint(0, 5, 1000),
        'feature3': np.random.rand(1000) * 100,
        'categorical_feature': [f'cat_{i%3}' for i in range(1000)],
        'sale_price': np.random.rand(1000) * 100000 + 150000
    })
    test_df = pd.DataFrame({
        'id': range(200000, 200020),
        'feature1': np.random.rand(20),
        'feature2': np.random.randint(0, 5, 20),
        'feature3': np.random.rand(20) * 100,
        'categorical_feature': [f'cat_{i%3}' for i in range(20)],
    })
    sample_submission_df = pd.DataFrame({'id': test_df['id'], 'pi_lower': 0, 'pi_upper': 0})



# ignore Warnings 
import warnings
warnings.filterwarnings("ignore")


# EDA
print(train_df.info())
print(train_df.describe())
sns.histplot(train_df['sale_price'], kde=True)
plt.title("Target Distribution")
plt.show()

sns.boxplot(x='categorical_feature', y='sale_price', data=train_df)
plt.title("Sale Price by Category")
plt.show()


# Features & Target
TARGET = 'sale_price'
ID_COL = 'id'
y = train_df[TARGET]
X = train_df.drop(columns=[TARGET, ID_COL])
X_test = test_df.drop(columns=[ID_COL])


# Preprocessor
num_features = X.select_dtypes(include=['int64','float64']).columns
cat_features = X.select_dtypes(include=['object']).columns

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
])


# Custom Quantile RF
class QuantileRandomForestRegressor(RandomForestRegressor):
    def predict_quantiles(self, X, quantiles=[0.05, 0.95]):
        all_preds = np.array([tree.predict(X) for tree in self.estimators_])
        return np.percentile(all_preds, [q*100 for q in quantiles], axis=0).T

ALPHA = 0.1
LOWER_Q, UPPER_Q = ALPHA/2, 1-ALPHA/2

qrf_model = Pipeline([
    ('preprocessor', preprocessor),
    ('qrf', QuantileRandomForestRegressor(n_estimators=300, random_state=42, min_samples_leaf=5, n_jobs=-1))
])


# Train
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
qrf_model.fit(X_train, y_train)


# Validate
X_val_preprocessed = qrf_model.named_steps['preprocessor'].transform(X_val)
qrf = qrf_model.named_steps['qrf']
val_preds = qrf.predict_quantiles(X_val_preprocessed, quantiles=[LOWER_Q, UPPER_Q])


# Interval evaluation
val_lower, val_upper = val_preds[:,0], val_preds[:,1]
coverage = np.mean((y_val >= val_lower) & (y_val <= val_upper))
print(f"Validation coverage: {coverage:.2f}")


# Predict Test
X_test_preprocessed = qrf_model.named_steps['preprocessor'].transform(X_test)
test_preds = qrf.predict_quantiles(X_test_preprocessed, quantiles=[LOWER_Q, UPPER_Q])

submission_df = pd.DataFrame({
    'id': test_df[ID_COL],
    'pi_lower': test_preds[:,0],
    'pi_upper': test_preds[:,1]
})


submission_df.to_csv('submission.csv', index=False)
print("submission.csv saved.")
print(submission_df.head())

