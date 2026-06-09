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


train = pd.read_csv("/kaggle/input/playground-series-s5e6/train.csv")


test = pd.read_csv("/kaggle/input/playground-series-s5e6/test.csv")


sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e6/sample_submission.csv")


train


train.shape


train.info()


train.describe()


import matplotlib.pyplot as plt 
import seaborn as sns


numerical_features = train.select_dtypes(['int64','float64']).columns
numerical_features


categorical_features = train.select_dtypes(['object']).columns
categorical_features


sns.heatmap(
    train[numerical_features].corr(), 
    cmap='Spectral', 
    annot=True, 
    fmt='.2f',             
    linewidths=0.5,       
    linecolor='gray',      
    cbar_kws={'shrink': 0.8, 'label': 'Correlation Coefficient'},  
    vmin=-1, vmax=1, 
    square=True           
)


# Set up subplots
fig, axes = plt.subplots(len(numerical_features), 2, figsize=(12, 5 * len(numerical_features)))

for i, col in enumerate(numerical_features):
    # Histogram
    sns.histplot(train[col], bins=30, kde=True, ax=axes[i, 0])
    axes[i, 0].set_title(f"Distribution of {col}")

    # Boxplot
    sns.boxplot(x=train[col], ax=axes[i, 1])
    axes[i, 1].set_title(f"Boxplot of {col}")

plt.tight_layout()
plt.show()


categorical_features = ['Soil Type', 'Crop Type']
for feature in categorical_features:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.countplot(x=feature,data=train,order=train[feature].value_counts().index,palette='Spectral',edgecolor='black',ax=ax)
    ax.set_title(f'Distribution of {feature}', fontsize=15, weight='bold')
    ax.set_xlabel(feature, fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.tick_params(axis='x', rotation=40)
    ax.grid(visible=True, axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()
    print(f"\nğŸ”� Proportions in '{feature}':")
    print(train[feature]
          .value_counts(normalize=True)
          .round(3)
          .rename_axis('Category')
          .reset_index(name='Proportion'), '\n' + '='*50)


cat_feats = ['Soil Type', 'Crop Type']
for col in cat_feats:
    plt.figure(figsize=(8, 4))
    sns.countplot(data=train,x=col,hue='Fertilizer Name',palette='husl',edgecolor='black')
    plt.title(f'{col} by Fertilizer Name', fontsize=14)
    plt.xlabel(col, fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Fertilizer Name', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()
    print(f'\nğŸ“Š Proportions of Fertilizer within "{col}":\n')
    prop_table = train.groupby(col)['Fertilizer Name'].value_counts(normalize=True).unstack().round(3)
    print(prop_table, '\n' + '-'*50)


train.head()


# Choose a vibrant color palette
colors = sns.color_palette("husl", 6)  # 6 distinct, bright colors[3][5]

fig, ax = plt.subplots(2, 3, figsize=(16, 8), facecolor="#f7f7f7")

# List of columns and subplot positions
features = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']

for i, feature in enumerate(features):
    row = i // 3
    col = i % 3
    sns.kdeplot(
        train[feature],
        ax=ax[row, col],
        color=colors[i],
        fill=True,
        linewidth=2.5,
        alpha=0.8
    )
    ax[row, col].set_title(
        feature,
        fontsize=14,
        fontweight='bold',
        color=colors[i]
    )
    ax[row, col].set_facecolor("#ffffff")
    ax[row, col].grid(True, linestyle='--', alpha=0.3)
    ax[row, col].tick_params(axis='both', which='major', labelsize=10)

# Remove axis labels for a cleaner look
for axes in ax.flat:
    axes.set_xlabel('')
    axes.set_ylabel('')

fig.suptitle("Feature Distributions (KDE)", fontsize=20, fontweight='bold', color="#333333")
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()



from sklearn.preprocessing import PowerTransformer
import pandas as pd

def normalize_to_gaussian(df, method='yeo-johnson'):
    
    df_transformed = df.copy()
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

    pt = PowerTransformer(method=method, standardize=False)
    df_transformed[numeric_cols] = pt.fit_transform(df[numeric_cols])

    return df_transformed


train_norm_dist = normalize_to_gaussian(train)
test_norm_dist = normalize_to_gaussian(test)


# Choose a vibrant color palette
colors = sns.color_palette("husl", 6)  # 6 distinct, bright colors[3][5]

fig, ax = plt.subplots(2, 3, figsize=(16, 8), facecolor="#f7f7f7")

# List of columns and subplot positions
features = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']

for i, feature in enumerate(features):
    row = i // 3
    col = i % 3
    sns.kdeplot(
        test_norm_dist[feature],
        ax=ax[row, col],
        color=colors[i],
        fill=True,
        linewidth=2.5,
        alpha=0.8
    )
    ax[row, col].set_title(
        feature,
        fontsize=14,
        fontweight='bold',
        color=colors[i]
    )
    ax[row, col].set_facecolor("#ffffff")
    ax[row, col].grid(True, linestyle='--', alpha=0.3)
    ax[row, col].tick_params(axis='both', which='major', labelsize=10)

# Remove axis labels for a cleaner look
for axes in ax.flat:
    axes.set_xlabel('')
    axes.set_ylabel('')

fig.suptitle("Feature Distributions (KDE)", fontsize=20, fontweight='bold', color="#333333")
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()



train_norm_dist


from sklearn.preprocessing import StandardScaler
import pandas as pd

scaler = StandardScaler()

# Define columns to scale
cols_to_scale = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']

# Fit and transform train
train_scaled_array = scaler.fit_transform(train_norm_dist[cols_to_scale])
train_scaled = pd.DataFrame(train_scaled_array, columns=cols_to_scale, index=train_norm_dist.index)

# Transform test
test_scaled_array = scaler.transform(test_norm_dist[cols_to_scale])
test_scaled = pd.DataFrame(test_scaled_array, columns=cols_to_scale, index=test_norm_dist.index)


train_transformed = pd.concat([train[['id']],train_scaled, train_norm_dist[['Soil Type', 'Crop Type']],train['Fertilizer Name']], axis=1)
test_transformed = pd.concat([test[['id']],test_scaled, test_norm_dist[['Soil Type', 'Crop Type']]], axis=1)


train_transformed


test_transformed


import pandas as pd

def one_hot_encode_column(df, column_name, drop_first=True):
    
    # if column_name not in df.columns:
    #     raise ValueError(f"Column '{column_name}' not found in DataFrame.")

    # One-hot encode the column
    dummies = pd.get_dummies(df[column_name], prefix=column_name, drop_first=drop_first, dtype=int)

    # Drop original column and add encoded columns
    df_encoded = df.drop(column_name, axis=1).join(dummies)

    return df_encoded


train_transformed = one_hot_encode_column(train_transformed,['Soil Type','Crop Type'])
test_transformed = one_hot_encode_column(test_transformed,['Soil Type','Crop Type'])


train_transformed


from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Step 1: Apply Label Encoding
le = LabelEncoder()
train_transformed_array = le.fit_transform(train_transformed['Fertilizer Name'])

# Step 2: Convert to DataFrame
fertilizer_encoded_df = pd.DataFrame(train_transformed_array, columns=['Fertilizer Name'])

# Step 3: Drop original column and concatenate the new one
train_transformed_final = pd.concat(
    [train_transformed.drop('Fertilizer Name', axis=1).reset_index(drop=True),
     fertilizer_encoded_df.reset_index(drop=True)],
    axis=1
)


train_transformed_final


train_transformed = train_transformed_final.drop(columns='id')
test_transformed = test_transformed.drop(columns='id')


train_transformed


test_transformed


sample_submission = sample_submission.drop(columns='id')


sample_submission.value_counts()


from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# 1. Split features and target
X = train_transformed.drop('Fertilizer Name', axis=1)
y = train_transformed['Fertilizer Name']

# # 2. Train-test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train XGBoost model
model = XGBClassifier(objective='multi:softprob', num_class=len(set(y)), eval_metric='mlogloss', use_label_encoder=False)
model.fit(X, y)

# 4. Predict probabilities
y_pred_probs = model.predict_proba(test_transformed)  # shape: [n_samples, n_classes]

# Get top 3 predicted class indices
top_3_preds = np.argsort(y_pred_probs, axis=1)[:, -3:][:, ::-1]

# Decode label indices to actual fertilizer names
top_3_labels = le.inverse_transform(np.unique(y))  # get mapping from label encoder

# Convert top_3_preds to actual class names
top_3_class_names = np.array([le.classes_[row] for row in top_3_preds])


def mapk(actual, predicted, k=3):

    score = 0.0
    for a, p in zip(actual, predicted):
        if a in p:
            score += 1.0 / (p.tolist().index(a) + 1)
    return score / len(actual)



# Build submission DataFrame
submission = pd.DataFrame({
    'id': test['id'],
    'Fertilizer Name': [' '.join(row) for row in top_3_class_names]
})

# Preview
submission.head()



# Save to CSV
submission.to_csv("submission.csv", index=False)

