#imports 
import plotly.express as px
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import pandas as pd
sample_submission=pd.read_csv('/kaggle/input/playground-series-s5e6/sample_submission.csv')
print(sample_submission.shape)
sample_submission.head()


#loading the data

train=pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test=pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')

print(train.info())


# statistical  information of data 
train.describe()


print(train.shape)
train.head()


print(test.shape)
test.head()


# checking for the NULL VALUES 
train.isnull().sum()



# Count occurrences of each fertilizer
fertilizer_counts = train['Fertilizer Name'].value_counts().reset_index()
fertilizer_counts.columns = ['Fertilizer Name', 'Count']

# Create bar chart with unique colors
fig = px.bar(fertilizer_counts, 
             x='Fertilizer Name', 
             y='Count', 
             color='Fertilizer Name',  # this gives each bar a different color
             title='Count of Each Fertilizer',
             labels={'Count': 'Number of Records'})

fig.show()


correlation=train['Temparature'].corr(train['Humidity'])

print(f"Correlation between Temperature and Humidity: {correlation:.3f}")


import plotly.graph_objects as go
from plotly.subplots import make_subplots

numeric_features = ['Temparature', 'Humidity', 'Nitrogen', 'Phosphorous', 'Potassium']

# Calculate subplot rows and columns dynamically
n_cols = 3
n_rows = -(-len(numeric_features) // n_cols)  # Ceiling division

# Create subplot layout
fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=numeric_features)

# Add histograms
for i, feature in enumerate(numeric_features):
    row = i // n_cols + 1
    col = i % n_cols + 1
    fig.add_trace(
        go.Histogram(
            x=train[feature],
            nbinsx=20,
            marker_color='teal',
            opacity=0.75,
            name=feature,
            hovertemplate=f"{feature}: %{{x}}<br>Count: %{{y}}<extra></extra>"
        ),
        row=row, col=col
    )

# Update layout and appearance
fig.update_layout(
    height=300 * n_rows,
    width=1000,
    title_text="Distribution of Numeric Features",
    title_x=0.5,
    showlegend=False,
    template='plotly_white',
    margin=dict(t=80, b=50, l=40, r=40)
)

# Set consistent axes labels
for i in range(1, len(numeric_features) + 1):
    fig.update_xaxes(title_text=numeric_features[i-1], row=(i - 1) // n_cols + 1, col=(i - 1) % n_cols + 1)
    fig.update_yaxes(title_text="Count", row=(i - 1) // n_cols + 1, col=(i - 1) % n_cols + 1)

fig.show()



numeric_features=['Temparature', 'Humidity', 'Nitrogen', 'Phosphorous', 'Potassium']
for col in numeric_features:
    sns.boxplot(data=train, x=col)
    plt.title(f"Boxplot of {col}")
    plt.show()


numeric_features=['Temparature', 'Humidity', 'Nitrogen', 'Phosphorous', 'Potassium']
for col in numeric_features:
    plt.figure(figsize=(6,4))
    sns.violinplot(x='Fertilizer Name', y=col, data=train)
    #plt.xticks(rotation=45)
    plt.title(f"{col} vs Fertilizer")
    plt.show()


le_soil = LabelEncoder()
le_crop = LabelEncoder()
le_fertilizer = LabelEncoder()

train['Soil Type'] = le_soil.fit_transform(train['Soil Type'])
train['Crop Type'] = le_crop.fit_transform(train['Crop Type'])
train['Fertilizer Name'] = le_fertilizer.fit_transform(train['Fertilizer Name'])
print(train.head())


le_soil1 = LabelEncoder()
le_crop1 = LabelEncoder()
test['Soil Type'] = le_soil.fit_transform(test['Soil Type'])
test['Crop Type'] = le_crop.fit_transform(test['Crop Type'])
print(test.head())


print(train.columns.tolist())


print(test.columns.tolist())


print(train['Fertilizer Name'].value_counts())


train.duplicated().sum()


train.columns




import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X = train.drop('Fertilizer Name', axis=1)
y = train['Fertilizer Name']

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

model = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss')
model.fit(X_train, y_train)


xgb.plot_importance(model, importance_type='weight', 
                    max_num_features=10,  # top 10
                    height=0.5,  # bar height
                    grid=False)
plt.title("Top 10 Feature Importances")
plt.show()



model = xgb.XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric='mlogloss'
)

model.fit(X_train, y_train)
y_pred = model.predict(X_valid)
accuracy = accuracy_score(y_valid, y_pred)
print(f"Improved Accuracy: {accuracy:.4f}")


from catboost import CatBoostClassifier
cat_model = CatBoostClassifier(verbose=0, random_state=42)
cat_model.fit(X_train, y_train)
y_pred = cat_model.predict(X_valid)
print("CatBoost Accuracy:", accuracy_score(y_valid, y_pred))


import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

# Prepare data
X = train.drop(['Fertilizer Name', 'id'], axis=1)
y = train['Fertilizer Name']
X_test = test.drop(['id'], axis=1)
test_ids = test['id']

n_classes = len(np.unique(y))

# 10-Fold Stratified CV
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
oof_preds = np.zeros((len(X), n_classes))
test_preds = np.zeros((len(X_test), n_classes))
fold_accs = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\nğŸŒŸ Fold {fold+1}/10")

    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    model = xgb.XGBClassifier(
        n_estimators=1000,
        learning_rate=0.02,
        max_depth=10,
        subsample=0.9,
        colsample_bytree=0.9,
        gamma=0.3,
        min_child_weight=2,
        objective='multi:softprob',
        num_class=n_classes,
        random_state=fold,
        tree_method='hist',
        eval_metric='mlogloss'
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=100,
        verbose=200
    )

    val_pred = model.predict(X_val)
    acc = accuracy_score(y_val, val_pred)
    fold_accs.append(acc)
    print(f"âœ… Fold {fold+1} Accuracy: {acc:.4f}")

    # OOF preds for meta-analysis (optional)
    oof_preds[val_idx] = model.predict_proba(X_val)

    # Average test predictions
    test_preds += model.predict_proba(X_test) / skf.n_splits

mean_acc = np.mean(fold_accs)
print(f"\nğŸŒŸ Average 10-Fold CV Accuracy: {mean_acc:.4f}")

# Predict Top-3 classes from averaged probabilities
top3_indices = np.argsort(test_preds, axis=1)[:, -3:][:, ::-1]

# Join indices to fertilizer names (strings)
top3_names = []
for row in top3_indices:
    actual_names = [str(y.cat.categories[i]) if hasattr(y, 'cat') else str(np.unique(y)[i]) for i in row]
    top3_names.append(' '.join(actual_names))

submission = pd.DataFrame({'id': test_ids, 'Fertilizer Name': top3_names})
submission.to_csv("submission.csv", index=False)
print("\nâœ… Submission saved as 'submission.csv'")
print(submission.head())

