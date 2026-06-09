import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


from IPython.display import Image, display
img_path = "/kaggle/input/myfiles/pobierz (1).jpg"
display(Image(filename=img_path))


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay, mean_squared_error
import torch.nn as nn

from xgboost import XGBClassifier
import lightgbm as lgb

%matplotlib inline

warnings.filterwarnings('ignore')


img1_path = '/kaggle/input/myfiles/pobierz.png'
display(Image(filename=img1_path))


train_data = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test_data = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
test_submission = pd.read_csv('/kaggle/input/playground-series-s5e6/sample_submission.csv')


print(f'Data Shape: {train_data.shape}')

print(f'\nData Info: ')
train_data.info()

print(f'\nNumerical Features Summary:')
display(train_data.describe().transpose())

print(f'\nFirst 10 rows of the Dataset:')
train_data.head(10)


#drop id column
train_data.drop('id', axis=1, inplace=True)
test_data.drop('id', axis=1, inplace=True)


#There are no duplicates
train_data.duplicated().value_counts()


train_data.nunique()


plt.figure(figsize=(10,3))
sns.heatmap(train_data.isna(), cbar=False)
plt.title("Missingâ€�value pattern"); plt.show()


sns.set_style('whitegrid')

numerical_features = train_data.select_dtypes(include=['number'])

for feature in numerical_features:
    plt.figure(figsize=(12, 6))

    #Histogram of feature
    plt.subplot(1,2,1)
    sns.histplot(data=train_data, x=feature, kde=True, bins=30)
    plt.axvline(train_data[feature].mean(), color='r', linestyle='--', label='Mean')
    plt.axvline(train_data[feature].median(), color='g', linestyle='-.', label='Median')
    plt.title(f'Histogram of {feature}')
    plt.xlabel(f'{feature}')
    plt.ylabel(f'Frequency')

    #Boxplot of feature
    plt.subplot(1,2,2)
    sns.boxplot(data=train_data, x=feature)
    plt.title(f'Histogram of {feature}')
    plt.xlabel(f'{feature}')
 
    plt.tight_layout()
    plt.show()

    print(f'\nStatistics for {feature}:')
    print(f'\nSkewness: {train_data[feature].skew():.2f}')
    print(f'\nStandard Deviation: {train_data[feature].std():.2f}')
    print(f'\nVariance: {train_data[feature].var():.2f}')
    


categorical_features = ['Soil Type', 'Crop Type']

for feature in categorical_features:
    counts = train_data[feature].value_counts()

    #Plot pie chart
    plt.figure(figsize=(6, 6))
    plt.pie(x=counts, labels=counts.index, startangle=90, autopct='%1.1f%%')

    plt.title(f'Distribution of {feature}')
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

    print(f"\nNumber of Unique {feature}: {train_data[feature].nunique()}")
    print(f"\nMissing Values in {feature}: {train_data[feature].isnull().sum()}")


numerical_features = train_data.select_dtypes(include='number').columns
colors = sns.color_palette('husl', len(numerical_features))

sns.reset_defaults()
plt.style.use('default')

rows = -(-len(numerical_features) // 4)
plt.figure(figsize=(20, 5 * rows))

for i, (col, color) in enumerate(zip(numerical_features, colors), 1):
    plt.subplot(rows, 4, i)
    sns.kdeplot(data=train_data, x=col, fill=True, color=color)
    plt.title(f'KDE Plot of {col}', fontsize=14, color=color)
    plt.xlabel(col)
    plt.ylabel('Density')

plt.tight_layout()
plt.show()


numerical_features = train_data.select_dtypes(include='number')

plt.figure(figsize=(12,8))

sns.pairplot(data=numerical_features.sample(1000))
plt.suptitle('Pairplot of Numerical Features')
plt.tight_layout()
plt.show()


bins = [24, 27, 30, 33, 36, 39]
labels = ['25-27', '28-30', '31-33', '34-36', '37-38']
train_data['Temp_Group'] = pd.cut(train_data['Temparature'], bins=bins, labels=labels)

sns.displot(data=train_data, x='Fertilizer Name', hue='Temp_Group', palette='Set2', multiple='stack')
plt.xticks(rotation=45)

plt.xlabel("Fertilizer Type")
plt.ylabel("Count")
plt.title("Distribution of Fertilizer Types by Temperature Group")
plt.show()


labels = ['53-56', '57-60', '61-64', '65-68', '69-72']
bins = [52, 56, 60, 64, 68, 72]

train_data['Hum_Group'] = pd.cut(train_data['Humidity'], labels=labels, bins=bins)

sns.displot(data=train_data, x='Fertilizer Name', hue='Hum_Group', palette='Set2', multiple='stack')

plt.xticks(rotation=45)
plt.xlabel("Fertilizer Type")
plt.ylabel("Count")
plt.title("Distribution of Fertilizer Types by Humidity Group")
plt.show()


bins = [19, 25, 30, 35, 40, 45]
labels = ['20-25', '26-30', '31-35', '36-40', '41-45']

train_data['Moisture_Group'] = pd.cut(train_data['Moisture'], bins=bins, labels=labels, include_lowest=True)

g = sns.displot(
    data=train_data,
    x='Fertilizer Name',
    hue='Moisture_Group',
    palette='Set3',
    multiple='stack',
    height=6,
    aspect=1.5
)

plt.xticks(rotation=45)
plt.xlabel("Fertilizer Type", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.legend(bbox_to_anchor=(0.6, 0., 0.5, 0.5))
plt.title("Distribution of Fertilizer Types by Moisture Group", fontsize=14)

plt.tight_layout()
plt.show()


train_data['Nitr_group'] = pd.qcut(train_data['Nitrogen'], q=5, labels=['0-9', '10-19', '20-29', '30-39', '40-49'])

g = sns.displot(
    data = train_data,
    x='Fertilizer Name',
    hue='Nitr_group',
    palette='Set2',
    multiple='stack',
    height=6,
    aspect=1.5
)

plt.xticks(rotation=45)
plt.xlabel('Fertilizer', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.legend(bbox_to_anchor=(0.6, 0., 0.5, 0.5))
plt.title("Distribution of Fertilizer Types by Nitrogen Group", fontsize=14)

plt.tight_layout()
plt.show()


train_data['Potas_group'] = pd.qcut(train_data['Potassium'], q=5, labels=['0-4', '5-9', '10-14', '15-19', '20-24'])

g = sns.displot(
    data = train_data,
    x='Fertilizer Name',
    hue='Potas_group',
    palette='Set2',
    multiple='stack',
    height=6,
    aspect=1.5
)

plt.xticks(rotation=45)
plt.xlabel('Fertilizer', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.legend(bbox_to_anchor=(0.6, 0., 0.5, 0.5))
plt.title("Distribution of Fertilizer Types by Potas Group", fontsize=14)

plt.tight_layout()
plt.show()


train_data['Phos_group'] = pd.qcut(train_data['Phosphorous'], q=5, labels=['0-9', '10-19', '20-29', '30-39', '40-49'])

g = sns.displot(
    data = train_data,
    x='Fertilizer Name',
    hue='Phos_group',
    palette='Set2',
    multiple='stack',
    height=6,
    aspect=1.5
)

plt.xticks(rotation=45)
plt.xlabel('Fertilizer', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.legend(bbox_to_anchor=(0.6, 0., 0.5, 0.5))
plt.title("Distribution of Fertilizer Types by Phosphorous Group", fontsize=14)

plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 8))
train_data = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
train_data.drop('id', axis=1, inplace=True)

numerical_features = train_data.select_dtypes(include='number').columns
corr = train_data[numerical_features].corr()
sns.heatmap(corr, annot=True, cmap='plasma')

plt.tight_layout()
plt.show()


num_cols = train_data.select_dtypes(include='number').columns
cat_cols = train_data.select_dtypes(include='object').columns


for df in [train_data, test_data]:
    df['row_mean'] = df[num_cols].mean(axis=1)
    df['row_std'] = df[num_cols].std(axis=1)
    df['row_min'] = df[num_cols].min(axis=1)
    df['row_max'] = df[num_cols].max(axis=1)


for n in cat_cols:
    print(f'Number of unique values in {n}: {train_data[n].nunique()}')


from sklearn.preprocessing import LabelEncoder

fertilizer_encoder = ''

for feature in cat_cols:
    label_enc = LabelEncoder()
    if feature != 'Fertilizer Name':
        train_data[feature] = label_enc.fit_transform(train_data[feature])
        test_data[feature] = label_enc.transform(test_data[feature])
    else:
        train_data['target'] = label_enc.fit_transform(train_data[feature]) 
        fertilizer_encoder = label_enc


train_data.head(5)


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import label_ranking_average_precision_score
from xgboost import XGBClassifier

X = train_data.drop(['Fertilizer Name', 'target'], axis=1)
y = train_data['target']
num_classes = y.nunique()

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.1,
    objective='multi:softprob',
    num_class=num_classes,
    use_label_encoder=False,
    eval_metric='mlogloss',
    random_state=42
)
model.fit(X_train, y_train)

val_proba = model.predict_proba(X_val)

y_val_bin = label_binarize(y_val, classes=np.arange(num_classes))

map3 = label_ranking_average_precision_score(y_val_bin, val_proba)
print(f"Validation MAP@3: {map3:.5f}")


model.fit(X, y)


test_proba = model.predict_proba(test_data)
top3_preds = np.argsort(test_proba, axis=1)[:, -3:][:, ::-1]


true_class_probs = val_proba[np.arange(len(y_val)), y_val]

plt.figure(figsize=(10, 6))
plt.hist(true_class_probs, bins=50, color='orange', edgecolor='black')
plt.title('Distribution of Predicted Probabilities for True Class')
plt.xlabel('Predicted Probability for True Label')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()


import optuna
from sklearn.preprocessing import label_binarize


y_bin = label_binarize(y, classes=np.arange(num_classes))

# Train/Validation split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
y_val_bin = label_binarize(y_val, classes=np.arange(num_classes))


def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'objective': 'multi:softprob',
        'num_class': num_classes,
        'use_label_encoder': False,
        'eval_metric': 'mlogloss',
        'random_state': 42
    }

    model = XGBClassifier(**params)
    model.fit(X_train, y_train)

    probas = model.predict_proba(X_val)
    y_val_bin = label_binarize(y_val, classes=np.arange(num_classes))

    map3 = label_ranking_average_precision_score(y_val_bin, probas)
    return map3


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30)


print("Best MAP@3:", study.best_value)
print("Best hyperparameters:", study.best_params)


final_model = XGBClassifier(
    **study.best_params,
    objective='multi:softprob',
    num_class=num_classes,
    use_label_encoder=False,
    eval_metric='mlogloss',
    random_state=42
)
final_model.fit(X, y)


test_proba = final_model.predict_proba(test_data)

top3_preds = np.argsort(test_proba, axis=1)[:, -3:][:, ::-1]


top3_labels = [
    ' '.join(fertilizer_encoder.inverse_transform(row))
    for row in top3_preds
]

submission = pd.DataFrame({
    'id': test_submission['id'],
    'Fertilizer Name': top3_labels
})

submission.to_csv('submission.csv', index=False)

