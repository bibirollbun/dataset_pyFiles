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


import pandas as pd

train = pd.read_csv("/kaggle/input/cat-in-the-dat-ii/train.csv")
display(train.head())
train.info()


train.isnull().sum()


train.duplicated().sum()


# Handle null by dropping it
train_clean = train.dropna(ignore_index=True)
train_clean.info()


for col in train_clean.columns:
    print(f"{col} - {len(train_clean[col].value_counts())}")


import numpy as np
import pandas as pd
import scipy.stats as stats

# import dataset
train = pd.read_csv("/kaggle/input/cat-in-the-dat-ii/train.csv")

# Handle null by dropping it
train_clean = train.dropna(ignore_index=True)

# Computes Cramér’s V statistic to measure association between categorical features
def cramers_v(cat_feature, target):
    """Computes Cramér’s V statistic to measure association between categorical features."""
    confusion_matrix = pd.crosstab(cat_feature, target)
    chi2 = stats.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    r, k = confusion_matrix.shape
    return np.sqrt(chi2 / (n * (min(r, k) - 1)))


print("Biggest Cramer'V in train\n-------------------------")
lfeat = [v for v in list(train_clean.columns) if v not in ["id", "target"]]
done=[]
for v1 in lfeat:
    done.append(v1)
    for v2 in [v for v in lfeat if v not in done]:
        c = cramers_v(train_clean[v1], train_clean[v2])
        if c > 0.08:
            print("{}, {}, {:.5f}".format(v1, v2, c))


# Computes Cramér’s V statistic to measure association between categorical features and target
res = {}
for col in train_clean.columns:
    if col not in ['id', 'target']:
        c = cramers_v(train_clean[col], train_clean['target'])
        res[col] = round(c, 5)

sorted_res = dict(sorted(res.items(), key=lambda item: item[1], reverse=True))
print("Biggest Cramer'V in train\n-------------------------")
for col, val in sorted_res.items():
    print(f"{col} - {val}")


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# import dataset
train = pd.read_csv("/kaggle/input/cat-in-the-dat-ii/train.csv")

def color_and_top(nb_mod, feature, typ, top_n=None):
    if top_n is None:
        resu = ["g", nb_mod]
    elif nb_mod > 2 * top_n:
        resu = ["r", top_n]
    elif nb_mod > top_n:
        resu = ["orange", top_n]
    else:
        resu = ["g", nb_mod]
    
    title = feature[:20] + " (" + typ[:3] + "-{})".format(nb_mod)
    resu.append(title)
    
    return resu


def plot_multiple_categorical(df, features, col_target=None, top_n=None,
                              nb_subplots_per_row=4, hspace=1.3, wspace=0.5,
                              figheight=15, m_figwidth=4.2, landmark=0.01):
    
    if col_target is not None:
        ref = df[col_target].mean()  # Reference
    
    plt.figure()
    if len(features) % nb_subplots_per_row > 0:
        nb_rows = int(np.floor(len(features) / nb_subplots_per_row) + 1)
    else:
        nb_rows = int(np.floor(len(features) / nb_subplots_per_row))
    
    fig, ax = plt.subplots(nb_rows, nb_subplots_per_row, figsize=(figheight, m_figwidth * nb_rows))
    plt.subplots_adjust(hspace=hspace, wspace=wspace)
    
    i = 0
    n_row = 0
    n_col = 0
    
    for feature in features:
        i += 1
        plt.subplot(nb_rows, nb_subplots_per_row, i)
        
        dff = df[[feature, col_target]].copy()
        
        # Handling missing values
        if dff[feature].dtype.name in ["float16", "float32", "float64"]:
            dff[feature] = dff[feature].fillna(-997)
        
        if dff[feature].dtype.name == "object":
            dff[feature] = dff[feature].fillna("_NaN")
        
        if dff[feature].dtype.name == "category" and dff[feature].isnull().sum() > 0:
            dff[feature] = dff[feature].astype(str).replace('', '_NaN', regex=False).astype("category")
        
        # Colors and title
        bar_colr, top_nf, title = color_and_top(dff[feature].nunique(), feature, str(dff[feature].dtype), top_n)
        
        # Stats
        tdf = dff.groupby([feature]).agg({col_target: ['count', 'mean']})
        tdf = tdf.sort_values((col_target, 'count'), ascending=False).head(top_nf).sort_index()
        
        tdf.index = tdf.index.map(str)
        tdf = tdf.rename(index={'-997.0': 'NaN'})
        
        if top_n is not None:
            tdf = tdf.head(top_nf)
        
        tdf["ref"] = ref
        tdf["ref-"] = ref - landmark
        tdf["ref+"] = ref + landmark
        
        # Bar plot for count
        plt.bar(tdf.index, tdf[col_target]['count'].values, color=bar_colr)
        
        plt.title(title, fontsize=11)
        plt.xticks(rotation=90)
        
        # Line plot for mean
        xx = plt.xlim()
        if nb_subplots_per_row == 1:
            ax2 = fig.add_subplot(nb_rows, nb_subplots_per_row, i, sharex=ax[n_row, n_col], frameon=False)
        else:
            ax2 = fig.add_subplot(nb_rows, nb_subplots_per_row, i, sharex=ax[n_row, n_col], frameon=False)
        
        if col_target is not None:
            ax2.plot(tdf[col_target]['mean'].values, marker='x', color='b', linestyle="solid")
            ax2.plot(tdf["ref"].values, marker='_', color='black', linestyle="solid", linewidth=4.0)
            ax2.plot(tdf["ref-"].values, marker='_', color='black', linestyle="solid", linewidth=1.0)
            ax2.plot(tdf["ref+"].values, marker='_', color='black', linestyle="solid", linewidth=1.0)
        
        ax2.yaxis.tick_right()
        ax2.axes.get_xaxis().set_visible(False)
        plt.xlim(xx)
        
        n_col += 1
        if n_col == nb_subplots_per_row:
            n_col = 0
            n_row += 1
    
    plt.show()

# Call the plotting function
plot_multiple_categorical(
    train_clean,
    features=[col for col in train.columns if col not in ['id', 'target']],  # List of features to plot
    col_target='target',                # Target variable
    top_n=17,                            # Top 3 categories to display for each feature
    nb_subplots_per_row=2,              # 2 subplots per row
    figheight=10,                       # Height of the figure
    m_figwidth=4,                       # Width of the figure
    landmark=0.01                       # Deviation for reference lines
)


from sklearn.compose import make_column_transformer
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
import pandas as pd

train = pd.read_csv("/kaggle/input/cat-in-the-dat-ii/train.csv")

train_clean = train.dropna(ignore_index=True)

# Define features (X) and target (y)
X = train_clean.drop(columns=['id', 'target', 'bin_3'])
y = train_clean['target']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, random_state=4)

# Define the preprocessor for OneHotEncoding categorical variables
preprocessor = make_column_transformer(
    (OneHotEncoder(categories='auto', sparse_output=True, dtype='uint8', handle_unknown="ignore"), [f for f in X.columns]),  # Exclude target' columns
    remainder='passthrough'  # Keep the other columns (numeric columns) as they are
)

# Create a pipeline with preprocessor and logistic regression
pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('logreg', LogisticRegression(C= 0.123456789, max_iter=500))
])

# Define cross-validation strategy
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=4)

# Perform cross-validation and get the scores
scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring="roc_auc", return_train_score=True)
cv_score = scores["test_score"].mean()

print(f"Cross-validation AUC score: {cv_score:.7f}")

# Eval on the test-set
# Fit the model on the training data (it has already been done during cross-validation)
pipe.fit(X_train, y_train)

# Make predictions on the test set
y_pred_test = pipe.predict(X_test)  # Predicted labels
y_pred_prob = pipe.predict_proba(X_test)[:, 1]  # Predicted probabilities for the positive class (if binary classification)

# Step 3: Evaluate the model on the test set using ROC AUC and accuracy
roc_auc = roc_auc_score(y_test, y_pred_prob)  # ROC AUC score

# Print the results
print(f"Test ROC AUC score: {roc_auc:.7f}")


# Load test set
test = pd.read_csv("/kaggle/input/cat-in-the-dat-ii/test.csv")
X_test_final = test.drop(columns=['id', 'bin_3'])

X = train_clean.drop(columns=['id', 'target', 'bin_3'])
y = train_clean['target']

# Fit Model
pipe.fit(X, y)

# Make predictions with probabilities (roc_auc is based on probabilities)
y_pred_proba_test = pipe.predict_proba(X_test_final)[:, 1]  # Get probability of class 1

# Prepare the submission dataframe with the ID and predicted target (probability)
submission = pd.DataFrame({'id': test['id'], 'target': y_pred_proba_test})

# Save the submission to a CSV file
submission.to_csv('submission.csv', index=False)

