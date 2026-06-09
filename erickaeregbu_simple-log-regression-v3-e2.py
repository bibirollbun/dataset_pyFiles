import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score
warnings.filterwarnings('ignore')


train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test  = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
original = pd.read_csv('/kaggle/input/extrovert-vs-introvert-behavior-data-backup/personality_datasert.csv')

print("Train shape :", train.shape)
print("Test  shape :", test.shape)
print("original  shape :", original.shape)
display(train.head())
display(test.head())
display(original.head())


print("ğŸ”� Missing Values in Train Dataset:")
display(train.isnull().sum().to_frame(name='Missing Values').query('`Missing Values` > 0').sort_values(by='Missing Values', ascending=False))

print("\nğŸ”� Missing Values in Test Dataset:")
display(test.isnull().sum().to_frame(name='Missing Values').query('`Missing Values` > 0').sort_values(by='Missing Values', ascending=False))


print("\n[INFO] Starting preprocessing...")
df_original = original.rename(columns={'Personality': 'match_p'})
drop_cols = [col for col in df_original.columns if col != 'match_p']
df_original = df_original.drop_duplicates(subset=drop_cols)

print(f"Original train shape: {train.shape}, test shape: {test.shape}")
train = train.merge(df_original, how='left')
test = test.merge(df_original, how='left')
print(f"Merged train shape: {train.shape}, test shape: {test.shape}")
display(train.head()), display(test.head())


X = train.drop(columns=['Personality'])
y = train['Personality']


print(y.shape)
target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y)
y = pd.Series(y)


def preprocessing_fold(X_train, X_val):
    X_train = X_train.drop(columns=['id'])
    X_val = X_val.drop(columns=['id'])
    X_train.columns = X_train.columns.str.lower().str.replace(' ', '_')
    X_val.columns = X_val.columns.str.lower().str.replace(' ', '_')
    
    cat_cols = X_train.select_dtypes(include='object').columns.tolist()
    num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()

    for col in num_cols:
        X_train[col] = X_train[col].fillna(X_train[col].mean())
        X_val[col] = X_val[col].fillna(X_train[col].mean())

    for col in cat_cols:
        X_train[col] = X_train[col].fillna("Unknown")
        X_val[col] = X_val[col].fillna("Unknown")

    ord_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X_train[cat_cols] = ord_encoder.fit_transform(X_train[cat_cols])
    X_val[cat_cols] = ord_encoder.transform(X_val[cat_cols])

    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_val[num_cols] = scaler.transform(X_val[num_cols])
    return X_train.reset_index(drop=True), X_val.reset_index(drop=True)


models, scores = [], []

params = {'penalty': 'l2', 'C': 0.002560670857520105, 'solver': 'lbfgs', 
    'max_iter': 1200, 'random_state': 42}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print('FOLD', fold + 1)
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    X_train_prep, X_val_prep = preprocessing_fold(X_train, X_val)
    model = LogisticRegression(**params)
    model.fit(X_train_prep, y_train)
    preds = model.predict(X_val_prep)
    score = accuracy_score(y_val, preds)
    scores.append(score)
    models.append(model)
    print('Accuracy score:', score)
print('Mean accuracy score:', np.mean(scores))


X_train = X.drop(['id', 'match_p'], axis=1).copy()
cat_cols = X_train.select_dtypes(include='object').columns.tolist()
X_train[cat_cols] = X_train[cat_cols].fillna("unknown")
num_cols = X_train.select_dtypes(include='number').columns.tolist()
for col in num_cols:
    X_train[col] = X_train[col].fillna(X_train[col].mean())

encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X_train[cat_cols] = encoder.fit_transform(X_train[cat_cols])
tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1200, learning_rate='auto')
X_tsne = tsne.fit_transform(X_train)
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y, palette="tab10", s=60)
plt.title("t-SNE Visualization")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.legend(title="ĞšĞ»Ğ°Ñ�Ñ�", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


coefs = np.mean([model.coef_[0] for model in models], axis=0)
importance = pd.Series(coefs, index=X_train_prep.columns).sort_values(key=abs, ascending=False)

plt.figure(figsize=(10, 6))
importance.head(15).sort_values().plot(kind='barh', color='salmon', edgecolor='black')
plt.title("Top Feature Importances (Logistic Regression Coefficients)")
plt.xlabel("Coefficient Value")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()


def preprocess_final(X_train, X_test):
    X_train, X_test = X_train.copy(), X_test.copy()
    X_train.drop(columns='id', inplace=True)
    X_test.drop(columns='id', inplace=True)

    X_train.columns = X_train.columns.str.lower().str.replace(' ', '_')
    X_test.columns = X_test.columns.str.lower().str.replace(' ', '_')

    cat_cols = X_train.select_dtypes('object').columns.tolist()
    num_cols = X_train.select_dtypes(include='number').columns.tolist()

    X_train[num_cols] = X_train[num_cols].fillna(X_train[num_cols].mean())
    X_test[num_cols] = X_test[num_cols].fillna(X_train[num_cols].mean())
    X_train[cat_cols] = X_train[cat_cols].fillna('Unknown')
    X_test[cat_cols] = X_test[cat_cols].fillna('Unknown')

    enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X_train[cat_cols] = enc.fit_transform(X_train[cat_cols])
    X_test[cat_cols] = enc.transform(X_test[cat_cols])

    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])
    return X_train.reset_index(drop=True), X_test.reset_index(drop=True), enc, scaler
X, X_test, _, _ = preprocess_final(X ,test)


def submission(X_test, models):
  num_classes = len(np.unique(y))
  pred_probs = np.zeros((X_test.shape[0], num_classes))
  for model in models:
    pred_probs += model.predict_proba(X_test) / len(models)

  final_preds = np.argmax(pred_probs, axis=1)
  final_lavels = target_encoder.inverse_transform(final_preds)
  submission = pd.DataFrame({'id': test['id'],
                             'personality': final_lavels})
  submission.to_csv('submission.csv', index=False)
  return submission


submission =  submission(X_test, models)
submission.head()

