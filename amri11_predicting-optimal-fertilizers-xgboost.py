import pandas as pd
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


%%time

train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
sample = pd.read_csv('/kaggle/input/playground-series-s5e6/sample_submission.csv')


train.head()


train.info()


test.info()


print(train.duplicated().sum())
print(test.duplicated().sum())


train.describe()


test.describe()


train.describe(include = 'object')


test.describe(include = 'object')


train.info()


plt.figure(figsize=(8, 6))
sns.boxplot(x=train['Fertilizer Name'], y=train['Temparature'], palette='coolwarm')
plt.xlabel('Fertilizer Name')
plt.ylabel('Temparature')
plt.title('Boxplot Temparature vs Fertilizer Name')
plt.show()


plt.figure(figsize=(8, 6))
sns.boxplot(x=train['Fertilizer Name'], y=train['Humidity'], palette='coolwarm')
plt.xlabel('Fertilizer Name')
plt.ylabel('Humidity')
plt.title('Boxplot Humidity vs Fertilizer Name')
plt.show()


plt.figure(figsize=(8, 6))
sns.boxplot(x=train['Fertilizer Name'], y=train['Moisture'], palette='coolwarm')
plt.xlabel('Fertilizer Name')
plt.ylabel('Moisture')
plt.title('Boxplot Moisture vs Fertilizer Name')
plt.show()


soil_counts = train['Soil Type'].value_counts()
palette_soil = sns.color_palette("Blues", n_colors=len(soil_counts))
soil_color_map = dict(zip(soil_counts.index, palette_soil[::-1])) 
ordered_soil = sorted(train['Soil Type'].unique(), key=lambda x: soil_counts.get(x, 0), reverse=True)
colors_soil = [soil_color_map.get(e, "#cccccc") for e in ordered_soil]
plt.figure(figsize=(8, 5))
sns.countplot(x='Fertilizer Name', hue='Soil Type', data=train, palette=soil_color_map)
plt.title('Fertilizer Name by Soil type')
plt.xlabel('Fertilizer Name')
plt.ylabel('Count')
plt.legend(title='Soil Type', bbox_to_anchor=(1.05, 1), loc='upper left') 
plt.tight_layout()
plt.show()


crop_counts = train['Crop Type'].value_counts()
palette_crop = sns.color_palette("Blues", n_colors=len(crop_counts))
crop_color_map = dict(zip(crop_counts.index, palette_crop[::-1])) 
ordered_crop = sorted(train['Crop Type'].unique(), key=lambda x: crop_counts.get(x, 0), reverse=True)
colors_crop = [crop_color_map.get(e, "#cccccc") for e in ordered_crop]
plt.figure(figsize=(8, 5))
sns.countplot(x='Fertilizer Name', hue='Crop Type', data=train, palette=crop_color_map)
plt.title('Fertilizer Name by Crop type')
plt.xlabel('Fertilizer Name')
plt.ylabel('Count')
plt.legend(title='Crop Type', bbox_to_anchor=(1.05, 1), loc='upper left') 
plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 6))
sns.boxplot(x=train['Fertilizer Name'], y=train['Nitrogen'], palette='coolwarm')
plt.xlabel('Fertilizer Name')
plt.ylabel('Nitrogen')
plt.title('Boxplot Nitrogen vs Fertilizer Name')
plt.show()


plt.figure(figsize=(8, 6))
sns.boxplot(x=train['Fertilizer Name'], y=train['Potassium'], palette='coolwarm')
plt.xlabel('Fertilizer Name')
plt.ylabel('Potassium')
plt.title('Boxplot Potassium vs Fertilizer Name')
plt.show()


plt.figure(figsize=(8, 6))
sns.boxplot(x=train['Fertilizer Name'], y=train['Phosphorous'], palette='coolwarm')
plt.xlabel('Fertilizer Name')
plt.ylabel('Phosphorous')
plt.title('Boxplot Phosphorous vs Fertilizer Name')
plt.show()


train['Fertilizer Name'].value_counts()


class FertilizerPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, drop_outliers=True):
        self.drop_outliers = drop_outliers
        self.cat_columns = ['Soil Type', 'Crop Type']
        self.encoders = {}

    def fit(self, X, y=None):
        # Save category for one-hot
        for col in self.cat_columns:
            self.encoders[col] = sorted(X[col].dropna().unique())
        return self

    def transform(self, X):
        df = X.copy()

        # Drop outliers only for train
        if self.drop_outliers:
            Q1 = df['Moisture'].quantile(0.25)
            Q3 = df['Moisture'].quantile(0.75)
            IQR = Q3 - Q1
            df = df[(df['Moisture'] >= Q1 - 1.5 * IQR) & (df['Moisture'] <= Q3 + 1.5 * IQR)]

        # Feature engineering
        df['N_to_P'] = df['Nitrogen'] / (df['Phosphorous'] + 1)
        df['K_to_P'] = df['Potassium'] / (df['Phosphorous'] + 1)
        df['Temp_Humidity'] = df['Temparature'] * df['Humidity']
        df['Total_NPK'] = df['Nitrogen'] + df['Phosphorous'] + df['Potassium']

        # One-hot encoding 
        for col in self.cat_columns:
            for cat in self.encoders[col]:
                df[f"{col}_{cat}"] = (df[col] == cat).astype(int)
            df.drop(columns=[col], inplace=True)

        return df


preprocessor = FertilizerPreprocessor(drop_outliers=True)
train_processed = preprocessor.fit_transform(train)
test_processed = preprocessor.transform(test)


X_train = train.drop(columns=['id', 'Fertilizer Name'])
y_train = train['Fertilizer Name']

# Label encoding untuk target
le = LabelEncoder()
y_encoded = le.fit_transform(y_train)

pipeline = Pipeline([
    ('preprocess', FertilizerPreprocessor(drop_outliers=True)),
    ('xgb', XGBClassifier(
        objective='multi:softprob',
        eval_metric='mlogloss',
        use_label_encoder=False,
        n_estimators=1000,
        learning_rate=0.01,         # learning rate
        max_depth=15,                # max tree depth
        colsample_bytree=0.8,       # fraction of columns used per tree
        colsample_bylevel=0.9,      # fraction of columns per level (alternative to "col bit rate")
        subsample=0.8,
        random_state=42
    ))
])

# Train model
pipeline.fit(X_train, y_encoded)


xgb_model = pipeline.named_steps['xgb']
feature_names = pipeline.named_steps['preprocess'].transform(X_train).columns

# take value importance
importances = xgb_model.feature_importances_

# make DataFrame
fi_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(data=fi_df.head(20), x='Importance', y='Feature', palette='viridis')
plt.title('Top 20 Feature Importances (XGBoost)')
plt.tight_layout()
plt.show()


X_test = test.drop(columns=['id'])
proba = pipeline.predict_proba(X_test)

# Take the top 3 predictions
top3 = np.argsort(proba, axis=1)[:, -3:][:, ::-1]  # take the 3 highest, sorted from highest to lowest

# Convert back to Fertilizer Name label
top3_labels = np.array(le.inverse_transform(np.unique(top3)))  # mapping all possible labels

# Make a prediction result
predictions = []
for row in top3:
    labels = le.inverse_transform(row)
    predictions.append(' '.join(labels))

# Make DataFrame submission
submission = pd.DataFrame({
    'id': test['id'],
    'Fertilizer Name': predictions
})

# Save To CSV
submission.to_csv('submission.csv', index=False)
submission.head()

