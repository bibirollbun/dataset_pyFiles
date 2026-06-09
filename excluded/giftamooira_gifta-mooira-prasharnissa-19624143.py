import pandas as pd
import numpy as np
import random
import os
from sklearn.ensemble import RandomForestRegressor

random.seed(42)
np.random.seed(42)

path_data = '/kaggle/input/sparta-2024-data-science-competition/'
train_df = pd.read_csv(f"{path_data}train.csv")
test_df = pd.read_csv(f"{path_data}test.csv")


numerical_features = ['accommodates', 'bedrooms', 'beds', 'bathrooms', 'latitude', 'longitude', 'host_listings_count',
                      'availability_30', 'availability_60', 'availability_90', 'availability_365',
                      'number_of_reviews', 'review_scores_rating', 'reviews_per_month']

categorical_features = ['room_type', 'property_type', 'city']
target = 'price'

train_clean_df = train_df.copy()
test_processed_df = test_df.copy()

for col in numerical_features:
    median_val = train_clean_df[col].median()
    train_clean_df[col].fillna(median_val, inplace=True)
    test_processed_df[col].fillna(median_val, inplace=True)

all_data = pd.concat([train_clean_df.drop(target, axis=1), test_processed_df], axis=0)
all_data_encoded = pd.get_dummies(all_data, columns=categorical_features, dummy_na=False)

train_processed = all_data_encoded[:len(train_df)].copy()
test_processed = all_data_encoded[len(train_df):].copy()
train_processed[target] = train_clean_df[target]

all_features = numerical_features + list(all_data_encoded.columns[all_data_encoded.columns.str.startswith(tuple(categorical_features))])
features_final = [col for col in all_features if col in train_processed.columns and col != 'price' and col != 'id']

X = train_processed[features_final]
y = train_processed[target]


model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X, y)

predictions = model.predict(test_processed[features_final])

submission_df = pd.DataFrame({'id': test_processed['id'], 'price': predictions})
submission_df.to_csv('/kaggle/working/submission.csv', index=False)


import pandas as pd
import numpy as np
import random
import os
import matplotlib.pyplot as plt
import seaborn as sns

random.seed(42)
np.random.seed(42)

path_data = '/kaggle/input/sparta-2024-data-science-competition/'
train_df = pd.read_csv(f"{path_data}train.csv")
test_df = pd.read_csv(f"{path_data}test.csv")

plt.figure(figsize=(10, 6))
sns.boxplot(x='room_type', y='price', data=train_df)
plt.title('Distribution of Price by Room Type')
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x='accommodates', y='price', data=train_df)
plt.title('Price vs. Accommodates')
plt.show()

