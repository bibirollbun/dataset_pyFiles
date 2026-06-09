pip install xgboost lightgbm catboost tensorflow


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
import xgboost as xgb
import lightgbm as lgb
import catboost as cb


train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')


print('--- Missing Values (Train) ---')
print(train.isnull().sum())
print('\n--- Missing Values (Test) ---')
print(test.isnull().sum())


print('\n--- Basic Statistics (Quantitative) ---')
print(train.describe())


print('\n--- Basic Statistics (Categorical) ---')
print(train.describe(include='object'))


def plot_missing(df, title):
    plt.figure(figsize=(10, 4))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
    plt.title(title)
    plt.show()

plot_missing(train, 'Train Missing Value Heatmap')
plot_missing(test, 'Test Missing Value Heatmap')


cat_cols = ['Stage_fear', 'Drained_after_socializing', 'Personality']
for col in cat_cols:
    if col in train.columns:
        plt.figure(figsize=(5,3))
        sns.countplot(x=col, data=train)
        plt.title(f'{col} Distribution')
        plt.show()


num_cols = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency']
train[num_cols].hist(figsize=(12,8), bins=15)
plt.suptitle('Distribution of Numerical Variables')
plt.show()


for col in ['Stage_fear', 'Drained_after_socializing']:
    if col in train.columns:
        plt.figure(figsize=(5,3))
        sns.countplot(x=col, hue='Personality', data=train)
        plt.title(f'{col} and Personality Distribution')
        plt.show()


plt.figure(figsize=(8,6))
corr_matrix = train[num_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation between Numerical Variables')
plt.show()


combo_df = train.dropna(subset=['Stage_fear', 'Drained_after_socializing', 'Personality'])

# Let's define combinations
combinations = [
    ("No", "No"),
    ("Yes", "Yes"),
    ("Yes", "No"),
    ("No", "Yes")
]

for sf_val, das_val in combinations:
    subset = combo_df[(combo_df['Stage_fear'] == sf_val) & (combo_df['Drained_after_socializing'] == das_val)]
    counts = subset['Personality'].value_counts()
    print(f"Stage_fear = {sf_val}, Drained_after_socializing = {das_val}:")
    for personality in ['Introvert', 'Extrovert']:
        print(f"  {personality}: {counts.get(personality, 0)}")
    print("-" * 40)


for df in [train, test]:
    # Drained_after_socializing 'No' ise ve Stage_fear NULL ise Stage_fear = 'No'
    mask_no = (df['Drained_after_socializing'] == 'No') & (df['Stage_fear'].isnull())
    df.loc[mask_no, 'Stage_fear'] = 'No'

    # Drained_after_socializing 'Yes' ise ve Stage_fear NULL ise Stage_fear = 'Yes'
    mask_yes_sf = (df['Drained_after_socializing'] == 'Yes') & (df['Stage_fear'].isnull())
    df.loc[mask_yes_sf, 'Stage_fear'] = 'Yes'

    # Stage_fear 'Yes' ise ve Drained_after_socializing NULL ise Drained_after_socializing = 'Yes'
    mask_yes = (df['Stage_fear'] == 'Yes') & (df['Drained_after_socializing'].isnull())
    df.loc[mask_yes, 'Drained_after_socializing'] = 'Yes'

    # Stage_fear 'No' ise ve Drained_after_socializing NULL ise Drained_after_socializing = 'No'
    mask_no_das = (df['Stage_fear'] == 'No') & (df['Drained_after_socializing'].isnull())
    df.loc[mask_no_das, 'Drained_after_socializing'] = 'No'

    # If both are NULL, make them both 'Yes'
    mask_both_null = df['Stage_fear'].isnull() & df['Drained_after_socializing'].isnull()
    df.loc[mask_both_null, 'Stage_fear'] = 'Yes'
    df.loc[mask_both_null, 'Drained_after_socializing'] = 'Yes'


num_cols = [col for col in train.select_dtypes(include='number').columns if col != 'id']
corrs = {}
for col in num_cols:
    # Let's encode stage_fear in binary
    stage_fear_bin = train['Stage_fear'].map({'No': 0, 'Yes': 1})
    corrs[col] = abs(train[col].corr(stage_fear_bin))
most_corr_col = max(corrs, key=corrs.get)
print(f"The numeric column with the highest absolute correlation with Stage_fear: {most_corr_col} (correlation: {corrs[most_corr_col]:.3f})")


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(6,4))
sns.boxplot(x='Stage_fear', y='Time_spent_Alone', data=train)
plt.title('Relationship between Stage_fear and Time_spent_Alone')
plt.xlabel('Stage_fear')
plt.ylabel('Time_spent_Alone')
plt.show()


bins = [train['Time_spent_Alone'].min()-1, 2, 4, 6, 8, train['Time_spent_Alone'].max()+1]
labels = [f"{int(bins[i]+1)}-{int(bins[i+1])}" for i in range(len(bins)-1)]
train['Time_spent_Alone_bin'] = pd.cut(train['Time_spent_Alone'], bins=bins, labels=labels)

grouped = train.groupby(['Time_spent_Alone_bin', 'Stage_fear']).size().unstack(fill_value=0)
print("Number of Stage_fear (No/Yes) by Time_spent_Alone intervals:")
print(grouped)

# Also show with a bar plot
grouped.plot(kind='bar', stacked=False, figsize=(8,5))
plt.title('Distribution of Stage_fear by Time_spent_Alone Interval')
plt.xlabel('Time_spent_Alone Interval')
plt.ylabel('Number of People')
plt.legend(title='Stage_fear')
plt.show()

# Optionally, revert the Time_spent_Alone_bin column
train.drop('Time_spent_Alone_bin', axis=1, inplace=True)


for df in [train, test]:
    mask_no = df['Stage_fear'].isnull() & (df['Time_spent_Alone'] < 4)
    df.loc[mask_no, 'Stage_fear'] = 'No'

    # Fill the remaining null ones with “Yes” if Time_spent_Alone > 4
    mask_yes = df['Stage_fear'].isnull() & (df['Time_spent_Alone'] > 4)
    df.loc[mask_yes, 'Stage_fear'] = 'Yes'


from sklearn.impute import KNNImputer

# Find numeric columns (including Time_spent_Alone)
numeric_cols = [col for col in train.columns if train[col].dtype != 'O']

# Apply KNN imputer to both train and test
imputer = KNNImputer(n_neighbors=5)

# Impute numeric columns in train and test
train_numeric = train[numeric_cols]
test_numeric = test[numeric_cols]

train_imputed = imputer.fit_transform(train_numeric)
test_imputed = imputer.transform(test_numeric)

# Write the imputed values back to the dataframes
train[numeric_cols] = train_imputed
test[numeric_cols] = test_imputed


X = train.drop(['Personality', 'id'], axis=1)
y = train['Personality']
X_test = test.drop(['id'], axis=1)


# Find the numeric column with the highest correlation with Drained_after_socializing
num_cols = [col for col in X.columns if X[col].dtype != 'O' and col != 'Drained_after_socializing']
# Drained_after_socializing is categorical, let's convert it to numeric
drained_map = {'No': 0, 'Yes': 1}
drained_numeric = train['Drained_after_socializing'].map(drained_map)
corrs = {}
for col in num_cols:
    corrs[col] = train[col].corr(drained_numeric)
max_corr_col = max(corrs, key=lambda k: abs(corrs[k]))
print(f"Numeric column with the highest correlation with Drained_after_socializing: {max_corr_col} (corr={corrs[max_corr_col]:.3f})")

# Let's create bins as we did above for Time_spent_Alone
bins = [train[max_corr_col].min()-1, 2, 4, 6, 8, train[max_corr_col].max()+1]
labels = [f"{int(bins[i]+1)}-{int(bins[i+1])}" for i in range(len(bins)-1)]
bin_col = f"{max_corr_col}_bin"
train[bin_col] = pd.cut(train[max_corr_col], bins=bins, labels=labels)

grouped = train.groupby([bin_col, 'Drained_after_socializing']).size().unstack(fill_value=0)
print(f"Drained_after_socializing (No/Yes) counts by {max_corr_col} bins:")
print(grouped)

# Show with a bar plot
grouped.plot(kind='bar', stacked=False, figsize=(8,5))
plt.title(f'Distribution of Drained_after_socializing by {max_corr_col} Bins')
plt.xlabel(f'{max_corr_col} Bin')
plt.ylabel('Number of People')
plt.legend(title='Drained_after_socializing')
plt.show()

# Remove the bin column
train.drop(bin_col, axis=1, inplace=True)


for df in [train, test]:
    mask_no = df['Drained_after_socializing'].isnull() & (df['Time_spent_Alone'] < 4)
    df.loc[mask_no, 'Drained_after_socializing'] = 'No'

    # Fill the remaining nulls with "Yes" if Time_spent_Alone > 4
    mask_yes = df['Drained_after_socializing'].isnull() & (df['Time_spent_Alone'] > 4)
    df.loc[mask_yes, 'Drained_after_socializing'] = 'Yes'


print("Number of nulls in Drained_after_socializing (train):", train['Drained_after_socializing'].isnull().sum())
print("Number of nulls in Drained_after_socializing (test):", test['Drained_after_socializing'].isnull().sum())
print("Number of nulls in Stage_fear (train):", train['Stage_fear'].isnull().sum())
print("Number of nulls in Stage_fear (test):", test['Stage_fear'].isnull().sum())


# 1. Change rows where Stage_fear = Yes, Friends_circle_size = 0, Personality = Extrovert to Introvert
train = train[~((train['Stage_fear'] == 'Yes') & (train['Friends_circle_size'] == 0) & (train['Personality'] == 'Extrovert'))]

# 2. Delete rows where Time_spent_Alone = 11, Friends_circle_size = 0, Personality = Extrovert (make them Introvert)
train = train[~((train['Time_spent_Alone'] == 11) & (train['Friends_circle_size'] == 0) & (train['Personality'] == 'Extrovert'))]

# 3. Delete rows where Time_spent_Alone = 0, Friends_circle_size = 0, Personality = Introvert (make them Extrovert)
train = train[~((train['Time_spent_Alone'] == 0) & (train['Friends_circle_size'] == 0) & (train['Personality'] == 'Introvert'))]


train.drop('Friends_circle_size', axis=1, inplace=True)
test.drop('Friends_circle_size', axis=1, inplace=True)



def preprocess(X, fit=True, encoders=None, scaler=None):
    X = X.copy()
    cat_cols = ['Stage_fear', 'Drained_after_socializing']
    num_cols = [col for col in X.columns if col not in cat_cols]
    # Encode categorical variables
    if fit:
        encoders = {col: LabelEncoder().fit(X[col]) for col in cat_cols}
    for col in cat_cols:
        X[col] = encoders[col].transform(X[col])
    # Scale numerical variables
    if fit:
        scaler = StandardScaler().fit(X[num_cols])
    X[num_cols] = scaler.transform(X[num_cols])
    return X, encoders, scaler

# Preprocessing
X_proc, encoders, scaler = preprocess(X, fit=True)
X_test_proc, _, _ = preprocess(X_test, fit=False, encoders=encoders, scaler=scaler)


le_target = LabelEncoder()
y_enc = le_target.fit_transform(y)


models = {
    'RandomForest': RandomForestClassifier(random_state=42, n_estimators=200, max_depth=8),
    'ExtraTrees': ExtraTreesClassifier(random_state=42, n_estimators=200, max_depth=8),
    'XGBoost': xgb.XGBClassifier(random_state=42, n_estimators=200, max_depth=8, use_label_encoder=False, eval_metric='mlogloss'),
    'LightGBM': lgb.LGBMClassifier(random_state=42, n_estimators=200, max_depth=8),
    'CatBoost': cb.CatBoostClassifier(random_state=42, iterations=200, depth=8, verbose=0),
    'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
    'KNeighbors': KNeighborsClassifier(n_neighbors=7),
    'GaussianNB': GaussianNB(),
    'DecisionTree': DecisionTreeClassifier(random_state=42, max_depth=8)
}


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}
for name, model in models.items():
    scores = cross_val_score(model, X_proc, y_enc, cv=cv, scoring='accuracy')
    results[name] = float(scores.mean())
    print(f'{name} CV Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})')


best_model_name = max(results, key=lambda k: results[k])
best_model = models[best_model_name]
best_score = results[best_model_name]
print(f'Best model: {best_model_name} (Accuracy: {best_score:.4f})')


# Feature extraction with neural network using PyTorch (to avoid TensorFlow NotFoundError)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import numpy as np

# Convert data to PyTorch tensors
# Fix: Use .values to get numpy arrays from DataFrames/Series
X_tensor = torch.tensor(X_proc.values.astype(np.float32))
y_tensor = torch.tensor(np.array(y_enc).astype(np.longlong))

# Define neural network for feature extraction
class FeatureNet(nn.Module):
    def __init__(self, input_dim, feature_dim=16, num_classes=4):
        super(FeatureNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.feature_layer = nn.Linear(32, feature_dim)
        self.output_layer = nn.Linear(feature_dim, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        features = self.relu(self.feature_layer(x))
        out = self.output_layer(features)
        return out, features

input_dim = X_proc.shape[1]
num_classes = len(np.unique(y_enc))
feature_dim = 16

model = FeatureNet(input_dim, feature_dim, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# DataLoader for batching
dataset = TensorDataset(X_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Train the model
model.train()
for epoch in range(30):
    for xb, yb in loader:
        optimizer.zero_grad()
        outputs, _ = model(xb)
        loss = criterion(outputs, yb)
        loss.backward()
        optimizer.step()

# Extract features from the intermediate layer
model.eval()
with torch.no_grad():
    _, X_features = model(X_tensor)
    X_features_np = X_features.numpy()
    X_test_tensor = torch.tensor(X_test_proc.values.astype(np.float32))
    _, X_test_features = model(X_test_tensor)
    X_test_features_np = X_test_features.numpy()

# Evaluate the best model with these features using cross-validation
from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(best_model, X_features_np, y_enc, cv=cv, scoring='accuracy')
print(f"{best_model_name} (NN feature) CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")


# Hyperparameter optimization for Random Forest model
from sklearn.model_selection import GridSearchCV

# Define hyperparameter grid for RandomForest
rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [6, 8, 10],
    'min_samples_split': [2, 5, 10]
}

# Start optimization with GridSearchCV
print("Starting hyperparameter optimization for Random Forest model...")
rf_grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=rf_param_grid,
    cv=cv,
    scoring='accuracy',
    n_jobs=-1,
    verbose=2
)
rf_grid_search.fit(X_features, y_enc)
best_model = rf_grid_search.best_estimator_
print(f"Best hyperparameters: {rf_grid_search.best_params_}")
print(f"Best GridSearchCV score: {rf_grid_search.best_score_:.4f}")


# Perform feature extraction on the test set using the neural network, then make predictions
model.eval()
with torch.no_grad():
    X_test_tensor = torch.tensor(X_test_proc.values.astype(np.float32))
    _, X_test_features = model(X_test_tensor)
    X_test_features_np = X_test_features.numpy()

y_pred = best_model.predict(X_test_features_np)
y_pred_label = le_target.inverse_transform(y_pred)


submission = pd.DataFrame({'id': test['id'], 'Personality': y_pred_label})
submission.to_csv('submission.csv', index=False)
print('submission.csv file has been saved.')

