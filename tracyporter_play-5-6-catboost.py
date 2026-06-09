import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import average_precision_score

from catboost import CatBoostClassifier
from catboost import Pool


import matplotlib.pyplot as plt
import seaborn as sns


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")


train


fertiliser = train['Fertilizer Name'].unique()
fertiliser


num_fertiliser_categories = len(fertiliser)
num_fertiliser_categories


crop = train['Crop Type'].unique()
crop


num_crops = len(crop)
num_crops


pivot_table_crop = pd.pivot_table(train, index='Crop Type', columns='Fertilizer Name', aggfunc='size')
print(pivot_table_crop)


pivot_table_soil = pd.pivot_table(train, index='Soil Type', columns='Fertilizer Name', aggfunc='size')
print(pivot_table_soil)


#Count fertilizer usage for each crop
fertilizer_count_crop = train.groupby(['Crop Type', 'Fertilizer Name']).size().reset_index(name='Count')

# Select top 3 fertilizers per crop
top_fertilizer_crop = fertilizer_count_crop.sort_values(['Crop Type', 'Count'], ascending=[True, False])
top_fertilizer_crop = top_fertilizer_crop.groupby('Crop Type').head(3)

print(top_fertilizer_crop)


#Count fertilizer usage for each soil type
fertilizer_count_soil = train.groupby(['Soil Type', 'Fertilizer Name']).size().reset_index(name='Count')

# Select top 3 fertilizers per soil type
top_fertilizer_soil = fertilizer_count_soil.sort_values(['Soil Type', 'Count'], ascending=[True, False])
top_fertilizer_soil = top_fertilizer_soil.groupby('Soil Type').head(3)

print(top_fertilizer_soil)


train.isna().sum().sum()


test


test.isna().sum().sum()


submission


train = train.drop('id', axis=1)
test = test.drop('id', axis=1)

train.shape, test.shape


label_encoder = LabelEncoder()

encoded_labels = label_encoder.fit_transform(train['Fertilizer Name'])
encoded_labels


enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

for col in test:
    if train[col].dtype == 'object':
        train[col] = enc.fit_transform(train[col].values.reshape(-1,1))
        test[col] = enc.transform(test[col].values.reshape(-1,1))



y = train.pop('Fertilizer Name')
X = train
X_test = test


model = CatBoostClassifier(
    iterations = 1000,
    learning_rate = 0.02,
    depth = 6,
    # eval_metric = 'MAPE',
    # objective='multi:softprob',
    random_seed = 42,
    early_stopping_rounds = 50,
    verbose = 100
)


model_catboost = model.fit(X, y)


prediction = model_catboost.predict_proba(X_test)
prediction


top_3 = []
for line in prediction:
    top_3.append(np.argsort(line)[-1:-4:-1])


preds = [' '.join(label_encoder.inverse_transform(row)) for row in top_3]


submission['Fertilizer Name'] = preds
submission.to_csv('submission.csv', index=False)
submission = pd.read_csv('submission.csv')
submission

