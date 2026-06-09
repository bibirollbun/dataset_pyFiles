# Importing necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

import warnings
warnings.filterwarnings("ignore")

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# Train Data
df_train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
df_train.head()


df_train.info()


# Distribution of numerical values
numeric_cols = ['Temparature','Humidity','Moisture','Nitrogen','Potassium','Phosphorous']

for col in numeric_cols:
    plt.figure()
    sns.boxplot(x=df_train[col])
    plt.title(f'{col}')
    plt.show()


# Correlation
plt.figure(figsize=(10, 8))
sns.heatmap(df_train[numeric_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()


# Categorical Column Analysis
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.pie(df_train['Soil Type'].value_counts().values, labels = df_train['Soil Type'].value_counts().index, autopct='%1.1f%%')
ax1.set_title('Ratio of Soil Type')

ax2.pie(df_train['Crop Type'].value_counts().values, labels = df_train['Crop Type'].value_counts().index, autopct='%1.1f%%',)
ax2.set_title('Ratio of Crop Type')

plt.show()


plt.figure(figsize=(10, 5))
sns.countplot(data=df_train, x='Fertilizer Name', order=df_train['Fertilizer Name'].value_counts().index, palette = 'Spectral')
plt.title('Distribution of Fertilizer Types')
plt.xticks(rotation=45)
plt.show()


averages = df_train.groupby('Fertilizer Name')[numeric_cols].mean().round(2)
display(averages)


# NPK Total
#df_train['NPK_Total'] = df_train['Nitrogen'] + df_train['Phosphorous'] + df_train['Potassium']

# Moisture Temperature Ratio
#df_train['MoistureTemp_Ratio'] = df_train['Moisture'] / (df_train['Temparature'] + 1e-5)

# Composite Weather Index
#df_train['WaterStressIndex'] = (df_train['Moisture'] + df_train['Humidity']) / (df_train['Temparature'] + 1e-5)


y = df_train['Fertilizer Name']
y


# Encoding and Scaling
numerical_features = ['Temparature','Humidity','Moisture','Nitrogen','Potassium','Phosphorous']
categorical_features = ['Crop Type','Soil Type']

preprocessor = ColumnTransformer(
    transformers = [
        ('cat', OneHotEncoder(handle_unknown = 'ignore'), categorical_features),
        ('num',StandardScaler(), numerical_features)
    ]
)

pipeline = Pipeline(steps = [
    ('preprocessor', preprocessor)
])


X_data = pipeline.fit_transform(df_train)

columns_cat = pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features)
columns_all = list(columns_cat) + numerical_features
X_train_new = pd.DataFrame(X_data, columns=columns_all)
X_train_new.head()


le_fertilizer = LabelEncoder()
encoded_y = le_fertilizer.fit_transform(y)

y_encoded = pd.Series(encoded_y)

y_encoded


# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_train_new, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)


from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = []
fold_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_new, y_encoded)):
    print(f"Fold {fold + 1}")

    X_train, X_val = X_train_new.iloc[train_idx], X_train_new.iloc[val_idx]
    y_train, y_val = y_encoded.iloc[train_idx], y_encoded.iloc[val_idx]

    model = XGBClassifier(
    objective='multi:softprob',
    num_class=len(np.unique(y_encoded)),
    eval_metric='mlogloss',
    max_depth = 12,
    colsample_bytree = 0.467,
    subsample = 0.86,
    gamma = 0.26,
    learning_rate = 0.05,
    n_estimators = 500,
    random_state = 42
    )

    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              early_stopping_rounds=30,
              verbose=False)

    preds = model.predict(X_val)
    score = f1_score(y_val, preds, average="macro") 
    print(f"Fold {fold + 1} F1 Macro: {score:.4f}")

    fold_scores.append(score)
    models.append(model)

print(f"\nAverage F1 Macro: {np.mean(fold_scores):.4f}")


df_test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
df_test_new = pipeline.transform(df_test)
test_df = pd.DataFrame(df_test_new, columns=columns_all)

test_df.head()


# Feature Engineering
# df_test_new['NPK_Total'] = df_test_new['Nitrogen'] + df_test_new['Phosphorous'] + df_test_new['Potassium']

# df_test_new['MoistureTemp_Ratio'] = df_test_new['Moisture'] / (df_test_new['Temparature'] + 1e-5)

# df_test_new['WaterStressIndex'] = (df_test_new['Moisture'] + df_test_new['Humidity']) / (df_test_new['Temparature'] + 1e-5)

# Label Encoding
# le_soil2 = LabelEncoder()
# df_test_new['Soil Type_LE'] = le_soil2.fit_transform(df_test_new['Soil Type'])
# le_crop2 = LabelEncoder()
# df_test_new['Crop Type_LE'] = le_crop2.fit_transform(df_test_new['Crop Type'])

# One Hot Encoding

#df_test_encoded = pd.get_dummies(df_test_new, columns=['Crop Type', 'Soil Type'], drop_first=True)

# Scaling
#scaler = StandardScaler()
#scaling_columns = ['Temparature','Humidity','Moisture','Nitrogen','Potassium','Phosphorous']

#df_test_encoded[scaling_columns] = scaler.fit_transform(df_test_encoded[scaling_columns])

#df_test_encoded.head()


model_new = XGBClassifier(
    objective='multi:softprob',
    num_class=len(np.unique(y_encoded)),
    eval_metric='mlogloss',
    max_depth = 12,
    colsample_bytree = 0.467,
    subsample = 0.86,
    gamma = 0.26,
    learning_rate = 0.05,
    n_estimators = 500,
    random_state = 42
    )
model_new.fit(X_train, y_train)


y_test_pred = model_new.predict(test_df)

y_test_labels = le_fertilizer.inverse_transform(y_test_pred)

y_test_labels


probabilities = model_new.predict_proba(test_df)


top3_indices = np.argsort(probabilities, axis=1)[:, -3:][:, ::-1]

print(top3_indices)


top3_labels = np.array([
    le_fertilizer.inverse_transform(row) for row in top3_indices
])

pred_strings = [' '.join(row) for row in top3_labels]


submission_df = pd.DataFrame({
    'id': df_test['id'], 
    'Fertilizer Name': pred_strings
})

submission_df.to_csv('submission.csv', index=False)

