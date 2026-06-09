import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import missingno as msno
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn')
sns.set_palette("husl")


train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv", index_col='id')
test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv", index_col='id')


train.head()


train.info()


train.describe().T


train.describe(include=['O']).T


train.duplicated().sum()


train.isna().sum() / len(train) * 100


msno.matrix(train, figsize=(10, 5), color=(0.2, 0.5, 0.7))
plt.title("Missing Values Pattern", fontsize=16)
plt.show()


features = test.columns
print(features)


numerical_features = test.select_dtypes(exclude="object").columns
print(numerical_features)


categorical_features = test.select_dtypes(include="object").columns
print(categorical_features)


target = 'Personality'


plt.figure(figsize=(18, 10))
for i, col in enumerate(numerical_features, 1):
    plt.subplot(2, 3, i)
    sns.histplot(data=train, x=col, hue=target, kde=True, bins=30, 
                element='step', stat='density', common_norm=False)
    plt.title(f'Distribution of {col}', pad=10, weight='bold')
plt.tight_layout()
plt.show()


plt.figure(figsize=(18, 8))
for i, col in enumerate(numerical_features, 1):
    plt.subplot(2, 3, i)
    sns.boxplot(x=target, y=col, data=train)
    plt.title(f'{col} by Personality', pad=10, weight='bold')
plt.tight_layout()
plt.show()


plt.figure(figsize=(15, 6))
for i, col in enumerate(categorical_features, 1):
    plt.subplot(1, 2, i)
    sns.countplot(x=col, hue=target, data=train)
    plt.title(f'{col} Distribution', pad=10)
    plt.legend(title='Personality')
plt.tight_layout()
plt.show()


plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
ax = sns.countplot(x=target, data=train)
plt.title("Personality Distribution", pad=15)
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}', (p.get_x()+p.get_width()/2., p.get_height()),
                ha='center', va='center', xytext=(0, 10), textcoords='offset points')

plt.subplot(1, 2, 2)
train[target].value_counts().plot(kind='pie', autopct='%1.1f%%', 
                                     explode=[0.05, 0], startangle=90)
plt.title("Personality Proportion", pad=15)
plt.ylabel("")
plt.tight_layout()
plt.show()


from sklearn.impute import SimpleImputer

analysis_df = train.copy()

num_imputer = SimpleImputer(strategy='median')
analysis_df[numerical_features] = num_imputer.fit_transform(analysis_df[numerical_features])

cat_imputer = SimpleImputer(strategy='most_frequent')
analysis_df[categorical_features] = cat_imputer.fit_transform(analysis_df[categorical_features])


for col in categorical_features:
    analysis_df[col], _ = pd.factorize(analysis_df[col])

analysis_df[target], personality_labels = pd.factorize(analysis_df[target])

cor_mat = analysis_df.corr()
mask = np.triu(np.ones_like(cor_mat))

plt.figure(figsize=(12, 10))
sns.heatmap(cor_mat, mask=mask, fmt=".2f", cmap="winter", annot=True)
plt.title("Feature Correlation Matrix", pad=20)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.show()


plt.figure(figsize=(18, 12))

plt.subplot(2, 2, 1)
sns.violinplot(x='Personality', y='Social_event_attendance', data=train, inner='quartile')
plt.title("Social Event Attendance by Personality", pad=15)

plt.subplot(2, 2, 2)
sns.scatterplot(x='Social_event_attendance', y='Going_outside', hue='Personality', 
                data=train, alpha=0.7)
plt.title("Social Activity Relationship", pad=15)

plt.subplot(2, 2, 3)
sns.boxplot(x='Drained_after_socializing', y='Friends_circle_size', hue='Personality', 
            data=train)
plt.title("Friends Circle Size by Energy Drain", pad=15)

plt.subplot(2, 2, 4)
sns.kdeplot(data=train, x='Time_spent_Alone', hue='Personality', fill=True, 
            common_norm=False, alpha=0.5)
plt.title("Time Spent Alone Distribution", pad=15)

plt.tight_layout()
plt.show()


plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
sns.boxplot(x='Personality', y='Time_spent_Alone', data=analysis_df)
plt.title('Time Spent Alone by Personality')

plt.subplot(1, 2, 2)
sns.violinplot(x='Personality', y='Time_spent_Alone', 
               hue='Stage_fear', data=analysis_df)
plt.title('Time Spent Alone by Personality & Stage Fear')
plt.tight_layout()
plt.show()


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Prepare data
X = analysis_df[features]
y = analysis_df[target]

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Feature importance
feature_imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=feature_imp, y=feature_imp.index)
plt.title('Feature Importance for Personality Prediction')
plt.xlabel('Relative Importance')
plt.show()


import sklearn
sklearn.set_config(transform_output='pandas')
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, f1_score


from sklearn.base import clone, BaseEstimator, TransformerMixin

class ConvertToCategory(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.astype("category")
        


from sklearn.preprocessing import MinMaxScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer

preprocessing1 = ColumnTransformer([
    ('num_imputer', make_pipeline(SimpleImputer(strategy='median'), MinMaxScaler()), numerical_features),
    ('cat_imputer', make_pipeline(SimpleImputer(strategy='most_frequent'),
                                  OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
     categorical_features)
], remainder='drop')

preprocessing2 = make_pipeline(
    ColumnTransformer([
        ('num_imputer', 
         SimpleImputer(strategy='constant', fill_value=-1, add_indicator=True), 
         numerical_features),
        ('cat_imputer', 
         SimpleImputer(strategy='constant', fill_value='missing', add_indicator=True), 
         categorical_features)
    ], remainder='drop'),
    ConvertToCategory()
)

# preprocessing3 = ColumnTransformer([
#     ('num_imputer', KNNImputer(), numerical_features),
#     ('cat_imputer', SimpleImputer(strategy='constant', fill_value='missing'), categorical_features)
# ], remainder='drop')
    


orig1 = pd.read_csv("/kaggle/input/extrovert-vs-introvert-behavior-data/personality_datasert.csv")
orig2 = pd.read_csv("/kaggle/input/extrovert-vs-introvert-behavior-data/personality_dataset.csv")


train_concat = pd.concat([train, orig1, orig2], axis=0).reset_index(drop=True)
train_concat.head()


train_concat.duplicated().sum()


train_concat = train_concat.drop_duplicates()


len(train), len(train_concat)


X = train_concat.copy()
y = X.pop(target)


label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)


from catboost import CatBoostClassifier

X_processed = preprocessing2.fit_transform(X).astype(str)
X_train, X_valid, y_train, y_valid = train_test_split(X_processed, y_encoded, 
                                                      test_size=1/7, 
                                                      shuffle=True, 
                                                      random_state=0, 
                                                      stratify=y)

cat_params = {
    'loss_function': 'Logloss',
    'eval_metric': 'Accuracy',
    'n_estimators': 1000,
    'learning_rate': 0.01, 
    'depth': 8,
    'l2_leaf_reg': 3,  # L2 regularization
    'border_count': 254,  # For numerical features (GPU optimized)
    'random_strength': 1,  # Randomness for scoring splits
    'bagging_temperature': 0.8,  # Controls Bayesian bootstrap
    'od_type': 'Iter',  # Overfitting detector type
    'od_wait': 50,  # Early stopping patience
    'grow_policy': 'SymmetricTree',
    'random_seed': 42,
    'task_type': 'GPU', 
    'verbose': False, 
    'allow_writing_files': False,
    'cat_features': X_processed.columns.tolist(),
    'use_best_model': True,
    'boost_from_average': True,
    'max_ctr_complexity': 7,
    'custom_metric': ['F1']
}


cat_clf = CatBoostClassifier(**cat_params)
cat_clf.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], early_stopping_rounds=100, verbose=100)

y_pred_proba = cat_clf.predict_proba(X_valid)[:, 1]
y_pred = (y_pred_proba > 0.5).astype(int)
f1 = f1_score(y_valid, y_pred)
acc = accuracy_score(y_valid, y_pred)
print(f"F1 Score: {f1:.5f}")
print(f"Accuracy: {acc:.5f}")


from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix

print(classification_report(y_valid, y_pred, target_names=label_encoder.classes_.tolist()))

cm = confusion_matrix(y_valid, y_pred)

disp = ConfusionMatrixDisplay(cm, display_labels=label_encoder.classes_.tolist())
disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix")
plt.grid(False)
plt.show()


from sklearn.calibration import CalibrationDisplay

CalibrationDisplay.from_predictions( 
    y_valid, 
    y_pred_proba, 
    n_bins=10, 
    name='Calibrated CatBoost',
    color='darkorange'
)

plt.title('Calibration Curve (Reliability Diagram)', fontsize=14, fontweight='bold')
plt.xlabel('Mean Predicted Probability', fontsize=12)
plt.ylabel('Fraction of Positives', fontsize=12)
plt.legend(loc='best')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()


from xgboost import XGBClassifier


X_processed = preprocessing2.fit_transform(X)
X_train, X_valid, y_train, y_valid = train_test_split(X_processed, y_encoded, 
                                                      test_size=1/7, 
                                                      shuffle=True, 
                                                      random_state=0, 
                                                      stratify=y)



xgb_params = {
    'device': 'cuda',
    'tree_method': 'hist',  # Optimized for GPU and large datasets
    'enable_categorical': True,
    'n_estimators': 2000,  # Increased with early stopping
    'learning_rate': 0.01,  # Lower rate with more trees
    'max_depth': 15,  # Controls tree complexity
    'subsample': 0.95,  # Random subset of samples
    'colsample_bytree': 0.8,  # Random subset of features
    'eval_metric': ['logloss', 'auc'],
    'early_stopping_rounds': 200,
    'random_state': 42,
    'verbosity': 1
}

xgb_clf = XGBClassifier(**xgb_params)

xgb_clf.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=100)

y_pred_proba = xgb_clf.predict_proba(X_valid, iteration_range=(0, xgb_clf.best_iteration+1))[:, 1]
y_pred = (y_pred_proba > 0.50).astype(int)
f1 = f1_score(y_valid, y_pred)
acc = accuracy_score(y_valid, y_pred)
print(f"F1 Score: {f1:.5f}")
print(f"Accuracy: {acc:.5f}")


print(classification_report(y_valid, y_pred, target_names=label_encoder.classes_.tolist()))

cm = confusion_matrix(y_valid, y_pred)

disp = ConfusionMatrixDisplay(cm, display_labels=label_encoder.classes_.tolist())
disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix")
plt.grid(False)
plt.show()


CalibrationDisplay.from_predictions( 
    y_valid, 
    y_pred_proba, 
    n_bins=10, 
    name='Calibrated XGBoost',
    color='darkorange'
)

plt.title('Calibration Curve (Reliability Diagram)', fontsize=14, fontweight='bold')
plt.xlabel('Mean Predicted Probability', fontsize=12)
plt.ylabel('Fraction of Positives', fontsize=12)
plt.legend(loc='best')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()


test_processed = preprocessing2.transform(test).astype(str)

# final_model = clone(cat_clf)
# final_model.fit(X_processed, y_encoded)
test_pred_proba = cat_clf.predict_proba(test_processed)[:, 1]
test_pred = (test_pred_proba >= 0.55).astype(int)


sub = pd.read_csv("/kaggle/input/playground-series-s5e7/sample_submission.csv")
sub[target] = label_encoder.inverse_transform(test_pred)
sub.to_csv("submission.csv", index=False)
sub.head()

