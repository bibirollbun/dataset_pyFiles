import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from scipy.stats import mode


train = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv', index_col='id')


train.dropna().sample(10)


train.info()


train.describe()


plt.pie(train['Personality'].value_counts(), labels=pd.Series(train['Personality'].unique()), autopct='%1.1f%%', startangle=90);


# Duplicated Rows

print(f'There are {train.duplicated().sum()} duplicates')
train.drop_duplicates(inplace = True)


# Number of Missing Values

cols = train.columns
row_length = train.shape[0]
missing_cols = []

for col in cols:
    missing_values = train[col].isnull().sum()
    if missing_values > 0:
        missing_cols.append(col)
        print('-'*10)
        print(f'\"{col}\" has {missing_values} missing values')
        print(f'{round(missing_values * 100 / row_length, 2)} | 100%')

del missing_values



cols_to_plot = cols[:-1]
ncols = 3
nrows = math.ceil(len(cols_to_plot) / ncols)

fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, nrows * 4))
ax = ax.flatten()

for i, col in enumerate(cols_to_plot):
    total_per_class = train['Personality'].value_counts()
    missing_per_class = train[train[col].isna()]['Personality'].value_counts()
    ratio = (missing_per_class / total_per_class).fillna(0)

    bars = ax[i].bar(ratio.index.astype(str), ratio.values, color='salmon')
    ax[i].bar_label(bars, fmt='%.2f')  # <- This adds value labels above bars

    ax[i].set_title(f"Missing ratio in '{col}'")
    ax[i].set_ylabel("Missing ratio")
    ax[i].set_ylim(0, 1)
    ax[i].set_xlabel("Personality")

for j in range(i + 1, len(ax)):
    ax[j].axis("off")

fig.suptitle("Ratio of Missing Values per Personality Class", fontsize=18)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()



# Columns Cardinality

for col in cols[1:]:
    print('-'*10)
    print(f'\"{col}\" has {train[col].nunique()} unique values')


train.dropna().dtypes


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
import pandas as pd

class CustomPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, 
                 binary_cols=None, 
                 first_group_cols=None, 
                 second_group_cols=None,
                 target_col='Personality'):
        self.binary_cols = binary_cols
        self.first_group_cols = first_group_cols
        self.second_group_cols = second_group_cols
        self.target_col = target_col
        self.label_encoders = {}
        self.classwise_modes = {}
        self.global_modes = {}

    def fit(self, X, y=None):
        
        for col in X.drop(columns=['Personality']).columns:
            X[f"{col}_is_missing"] = 0
        
        # Fit LabelEncoders
        for col in self.binary_cols:
            le = LabelEncoder()
            le.fit(X[col].astype(str))
            self.label_encoders[col] = le

        # Save classwise and global modes for first_group_cols
        for col in self.first_group_cols:
            self.classwise_modes[col] = {}
            self.classwise_modes[col]['__global__'] = X[col].mode().iloc[0]  # always store global
            if self.target_col in X.columns:
                for cls in X[self.target_col].dropna().unique():
                    mask = (X[self.target_col] == cls)
                    if X.loc[mask, col].notna().sum() > 0:
                        mode_value = X.loc[mask, col].mode().iloc[0]
                    else:
                        mode_value = self.classwise_modes[col]['__global__']
                    self.classwise_modes[col][cls] = mode_value

        # Save global modes for second_group_cols
        for col in self.second_group_cols:
            self.global_modes[col] = X[col].mode().iloc[0]

        return self

    def transform(self, X):
        X = X.copy()

        # Label encode binary columns
        for col in self.binary_cols:
            X[col] = self.label_encoders[col].transform(X[col].astype(str))

        # If target_col available — use class-wise imputation
        if self.target_col in X.columns:
            for col in self.first_group_cols:
                for cls, mode_value in self.classwise_modes[col].items():
                    if cls == '__global__':
                        continue
                    mask = (X[self.target_col] == cls)
                    X.loc[mask & X[col].isna(), col] = mode_value

            for col in self.second_group_cols:
                X[f"{col}_is_missing"] = X[col].isna().astype(int)
                X[col] = X[col].fillna(self.global_modes[col])
        else:
            # Test case: no target_col — fallback to stored global modes
            for col in self.first_group_cols:
                mode_value = self.classwise_modes[col].get('__global__')
                X[f"{col}_is_missing"] = X[col].isna().astype(int)
                X[col] = X[col].fillna(mode_value)

            for col in self.second_group_cols:
                mode_value = self.global_modes.get(col)
                X[f"{col}_is_missing"] = X[col].isna().astype(int)
                X[col] = X[col].fillna(mode_value)

        return X



binary_cols = ['Stage_fear', 'Drained_after_socializing']
first_group_cols = ['Time_spent_Alone', 'Going_outside', 'Friends_circle_size']
second_group_cols = [col for col in cols_to_plot if col not in first_group_cols]

preprocessor = CustomPreprocessor(
    binary_cols=binary_cols,
    first_group_cols=first_group_cols,
    second_group_cols=second_group_cols,
    target_col='Personality'
)

df = preprocessor.fit_transform(train.copy())



df.sample(10)


df.isnull().sum()


le_target = LabelEncoder()

df['Personality'] = le_target.fit_transform(df['Personality'])


x, y = df.drop(columns=['Personality']), df['Personality']


skf = StratifiedKFold(n_splits=5)

oof_preds = np.zeros_like(y)

for i, (train_index, test_index) in enumerate(skf.split(x, y)):
    clf = SVC()
    x_train, x_test = x.iloc[train_index], x.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]

    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    oof_preds[test_index] = y_pred


accuracy_score(y, oof_preds)


test = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv', index_col='id')

test_df = preprocessor.transform(test.copy())

test_df = test_df[x.columns]


skf = StratifiedKFold(n_splits=7, shuffle=True, random_state=42)

test_preds = []

for fold, (train_idx, val_idx) in enumerate(skf.split(x, y)):
    model = SVC()
    
    x_train, y_train = x.iloc[train_idx], y[train_idx]
    
    model.fit(x_train, y_train)
    
    fold_test_preds = model.predict(test_df)
    test_preds.append(fold_test_preds)

test_preds = np.array(test_preds)


final_test_preds = mode(test_preds, axis=0).mode

final_test_preds = le_target.inverse_transform(final_test_preds)


submission = pd.read_csv('/kaggle/input/playground-series-s5e7/sample_submission.csv')

submission['Personality'] = final_test_preds

submission.to_csv('submission.csv', index=False)

