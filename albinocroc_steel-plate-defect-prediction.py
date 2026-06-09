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
import numpy as np 
import tensorflow as tf
import seaborn as sns 
import matplotlib.pyplot as plt

df = pd.read_csv('/kaggle/input/playground-series-s4e3/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s4e3/test.csv')


df.head(10)


df.shape


df.isna().sum()


delta_x = df['X_Maximum'] - df['X_Minimum']

sns.histplot(delta_x, bins=20, color='blue', kde=False)


delta_x.describe()


neg_x_deltas = df[df['X_Maximum'] - df['X_Minimum'] < 0]

neg_x_indices = neg_x_deltas.index.tolist()

neg_x_deltas


mask = df['X_Maximum'] < df['X_Minimum']

df.loc[mask, ['X_Minimum', 'X_Maximum']] = df.loc[mask, ['X_Maximum', 'X_Minimum']].values


# Sanity check 

df.loc[neg_x_indices]


delta_y = df['Y_Maximum'] - df['Y_Minimum']
delta_y.head(10)


delta_y.describe()


neg_y_deltas = df[df['Y_Maximum'] < df['Y_Minimum']]

neg_y_indices = neg_y_deltas.index.tolist()

print(neg_y_deltas.shape)
neg_y_deltas.head(10)


df.loc[neg_y_indices, ['Y_Minimum', 'Y_Maximum']] = df.loc[neg_y_indices, ['Y_Maximum', 'Y_Minimum']].values

df.loc[neg_y_indices].head(10)


new_del_x = df['X_Maximum'] - df['X_Minimum']
new_del_y = df['Y_Maximum'] - df['Y_Minimum']

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
sns.histplot(new_del_x, color='yellow', kde=False, bins=10, label='delta x', ax=axes[0])
sns.histplot(new_del_y, color='magenta', kde=False, bins=10, label='delta y', ax=axes[1])
plt.tight_layout()
plt.show()


suspicious_del_x = new_del_x[new_del_x > 250]
suspicious_del_y = new_del_y[new_del_y > 3000]

print(f"Number of del_x points > 250: {suspicious_del_x.shape[0]}")
print(f"Number of del_y points > 3000: {suspicious_del_y.shape[0]}")

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
sns.histplot(suspicious_del_x, color='yellow', kde=False, bins=10, label='delta x', ax=axes[0])
sns.histplot(suspicious_del_y, color='magenta', kde=False, bins=10, label='delta y', ax=axes[1])
plt.tight_layout()
plt.show()


q_98 = np.percentile(new_del_y, q=98)
q_985 = np.percentile(new_del_y, q=98.5)
q_99 = np.percentile(new_del_y, q=99)
print(f"{q_98=:.3f}, {q_985=:.3f}, {q_99=:.3f}")


suspicious_del_y.head(20)


trunc_df = df.drop(suspicious_del_y.index)
sns.histplot(trunc_df['Y_Maximum'] - trunc_df['Y_Minimum'], color='orange', kde=False, bins=20, label='delta y')


pd.Series(trunc_df['Y_Maximum']-trunc_df['Y_Minimum']).describe()


# Visualising the distributions of data

import matplotlib.pyplot as plt 
import seaborn as sns 

minmax_feats = ['X_Minimum', 'X_Maximum', 'Y_Minimum', 'Y_Maximum', 'Minimum_of_Luminosity', 'Maximum_of_Luminosity',]

fig1, axes1 = plt.subplots(1, 3, figsize=(12,6))

for i in range(int(len(minmax_feats)/2)):
    sns.histplot(trunc_df[minmax_feats[2*i]], bins=20, kde=False, alpha=0.5, color='magenta', ax=axes1[i], label=minmax_feats[2 * i])
    sns.histplot(trunc_df[minmax_feats[2*i+1]], bins=20, kde=False, alpha=0.5, color='green', ax=axes1[i], label=minmax_feats[2 * i + 1])
    axes1[i].legend()

plt.tight_layout()
plt.show()


x_max_zero = trunc_df[trunc_df['X_Maximum'] == 0]
print(x_max_zero.shape)
x_max_zero.head()


low_x_max = trunc_df[trunc_df['X_Maximum'] < 50]
print(low_x_max.shape)
low_x_max.head(20)


#Visualising all data from the test and train distributions

import warnings

targets = ['Pastry','Z_Scratch','K_Scatch','Stains','Dirtiness','Bumps','Other_Faults']

feats_to_plot = list(set(df.columns.tolist()) - set(targets) - set(['TypeOfSteel_A300', 'TypeOfSteel_A400', 'id']))

trunc_df['set'] = 'train'
test['set'] = 'test'

comb_df = pd.concat([trunc_df, test], axis=0)

palette = ['blue', 'orange']

def plot_data(feature, cls='set', outliers=True):
    fig, axes = plt.subplots(1, 2, figsize=(12,6))
    for i, label in enumerate(comb_df[cls].unique()):
        selection = comb_df.loc[comb_df[cls] == label, feature]
        #Include only the central 95% of the data 
        q_025, q_975 = np.percentile(selection, [2.5, 97.5])
        sel_filtered = selection[(selection > q_025) & (selection < q_975)]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=FutureWarning)
            warnings.simplefilter('ignore', category=UserWarning)
            sns.histplot(sel_filtered, palette = palette[i], alpha=0.5, ax=axes[0], label=feature, bins=50)
            sns.boxplot(data=comb_df, x = cls, y=feature, palette = palette, ax=axes[1], showfliers=outliers)
    axes[0].set_title(f"Distribution of {feature}")
    axes[1].set_title(f"Box plot of {feature}")
    axes[0].legend()
    axes[1].legend()
    plt.tight_layout()
    plt.show()

for feature in feats_to_plot:
    plot_data(feature)


small_iqr = ['Pixels_Areas', 'Sum_of_Luminosity', 'X_Perimeter', 'Y_Perimeter', 'Outside_X_Index']

for feature in small_iqr:
    plot_data(feature, outliers=False)


# Investigating outliers of these variables 

def create_outlier_df(feature, threshold):
    return df[df[feature] > threshold]

for f in small_iqr:
    p = np.percentile(df[f], q=85)
    outlier_df = create_outlier_df(f, p)
    print(outlier_df.head())
    print('Number of outliers: ', outlier_df.shape[0])


ix_dict = {}

for i, f in enumerate(small_iqr):
    p = np.percentile(df[f], q=85)
    outlier_df = create_outlier_df(f, p)
    ix_dict[i] = outlier_df['id']

s0 = ix_dict[0]
s1 = ix_dict[1]
s2 = ix_dict[2]
s3 = ix_dict[3]
s4 = ix_dict[4]

# Reindex all series to have the same index
common_index = s0.index.intersection(s1.index).intersection(s2.index).intersection(s3.index).intersection(s4.index)

equal_mask = (
    s0.loc[common_index] == s1.loc[common_index]
) & (
    s1.loc[common_index] == s2.loc[common_index]
) & (
    s2.loc[common_index] == s3.loc[common_index]
) & (
    s3.loc[common_index] == s4.loc[common_index]
)

same_ix = equal_mask.astype(int)

num_identical_rows = same_ix.sum()

print('Number of identical rows in the outlier dataframes: ', num_identical_rows)


# Visualising some of the identical rows 

outlier_ids = common_index[equal_mask]
print(outlier_ids)

#I want to specifically look at the outlier columns 

common_outliers = df.loc[outlier_ids]
common_outliers[small_iqr].describe()


def clip_extremes(df, features, upper_q=0.99):
    df_clipped = df.copy()
    for f in features:
        upper = df_clipped[f].quantile(upper_q)
        df_clipped[f] = df_clipped[f].clip(lower=None, upper=upper)
    return df_clipped


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler 

targets = ['Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps', 'Other_Faults']

numeric_cols = list(set(df.columns.tolist()) - set(['id', 'TypeOfSteel_A300', 'TypeOfSteel_A400']) - set(targets))

#Clipping outliers so as to not influence the PCA 
df_clipped = clip_extremes(df, numeric_cols)

#Standardise
scaled = StandardScaler().fit_transform(df_clipped[numeric_cols])

pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled)

df_pca = pd.DataFrame(pca_result, columns=['PCA1', 'PCA2'])
df_pca['is_outlier_group'] = df.index.isin(outlier_ids).astype(int)

sns.scatterplot(data=df_pca, x='PCA1', y='PCA2', hue='is_outlier_group', alpha=0.5)


# Visualising the target distributions 

targets = ['Pastry','Z_Scratch','K_Scatch','Stains','Dirtiness','Bumps','Other_Faults']

targets_sum = df[targets].sum()

sns.barplot(x=targets, y=targets_sum.values, palette='viridis')
plt.xlabel('Defect Type')
plt.ylabel('Counts')
plt.title("Frequency of Each Steel Defect Type")
plt.tight_layout()
plt.show()


# Out of curiosity, I also want to visualise the target distributions without the outliers

no_outlr_df = df.drop(outlier_ids)

no_outlr_targets_sum = no_outlr_df[targets].sum()

sns.barplot(x=targets, y=no_outlr_targets_sum.values, palette='viridis')
plt.xlabel('Defect Type')
plt.ylabel('Counts')
plt.title("Frequency of Each Steel Defect Type without Outliers")
plt.tight_layout()
plt.show()


# Feature Engineering
# Replacing Max and Min values with del = Max - Min

# from sklearn.base import BaseEstimator, TransformerMixin

# class DeltaTransformer(BaseEstimator, TransformerMixin):
#     def __init__(self, max_col, min_col, name):
#         self.max_col = max_col
#         self.min_col = min_col
#         self.name = name

#     def fit(self, X, y=None):
#         return self

#     def transform(self, X):
#         X = X.copy()
#         X[f'del_{self.name}'] = X[self.max_col] - X[self.min_col]
#         return X.drop(columns=[self.max_col, self.min_col])

#     def get_feature_names_out(self, input_features=None):
#         return np.array([f'del_{self.name}'])


# # Testing whether this works 
# x_transformed = DeltaTransformer('X_Maximum', 'X_Minimum', 'x').fit_transform(df)
# x_transformed.head()


from sklearn.base import BaseEstimator, TransformerMixin

class DeltaCreator(BaseEstimator, TransformerMixin):
    def __init__(self, delta_specs):
        self.delta_specs = delta_specs

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for max_col, min_col, name in self.delta_specs:
            X[f'del_{name}'] = X[max_col] - X[min_col]
        return X[[f'del_{name}'for _, _, name in self.delta_specs]].astype(float)

    def get_feature_names_out(self, input_features=None):
        return np.array([f'del_{name}' for _, _, name in self.delta_specs])


X = df.drop(columns=targets+["id"])

delta_specs = [
    ('X_Maximum', 'X_Minimum', 'x'),
    ('Y_Maximum', 'Y_Minimum', 'y'),
    ('Maximum_of_Luminosity', 'Minimum_of_Luminosity', 'lum')
]

cat_features = ['TypeOfSteel_A400', 'TypeOfSteel_A300', 'Outside_Global_Index', 'Steel_Plate_Thickness']

dropped = ['X_Maximum', 'X_Minimum', 'Y_Maximum', 'Y_Minimum', 'Maximum_of_Luminosity', 'Minimum_of_Luminosity']
num_features = [col for col in X.columns if col not in cat_features and col not in dropped]


from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler

delta_creator = DeltaCreator(delta_specs)

preprocessor_new = ColumnTransformer(transformers=[
    ('deltas', Pipeline([
        ('create_deltas', delta_creator),
        # ('scale', RobustScaler())
    ]), [col for spec in delta_specs for col in spec[:2]]),
    ('numeric', 'passthrough', num_features),
    ('categoric', 'passthrough', cat_features)
])

pipeline = Pipeline([
    ('preprocess', preprocessor_new),
])


X = df.drop(columns=targets+["id"])

X_transformed = pipeline.fit_transform(X)

delta_feature_names = delta_creator.get_feature_names_out()
all_feature_names = (list(delta_feature_names) + num_features + cat_features)

X_transf_df = pd.DataFrame(X_transformed, columns=all_feature_names, index=X.index)
X_transf_df


from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from xgboost import XGBClassifier 
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import make_scorer, roc_auc_score

targets = ['Pastry','Z_Scratch','K_Scatch','Stains','Dirtiness','Bumps','Other_Faults']
X = df.drop(columns=targets+["id"])
y = df[targets]

models = [
    (
        OneVsRestClassifier(XGBClassifier(eval_metric='logloss')),
        {
            'clf__estimator__n_estimators' : [100, 200],
            'clf__estimator__max_depth': [3, 6],
            'clf__estimator__learning_rate' : [0.01, 0.1],
        }
    ),
    (
        OneVsRestClassifier(HistGradientBoostingClassifier()), 
        { 
            'clf__estimator__learning_rate' : [0.01, 0.1],
            'clf__estimator__l2_regularization' : [0.0, 1.0],
            'clf__estimator__max_depth' : [None, 10],
        }
    ),
    (
        OneVsRestClassifier(RandomForestClassifier()), 
        {
            'clf__estimator__n_estimators' : [100, 200],
            'clf__estimator__max_depth' : [10, 20],
        }
    )
]

def multilabel_auc(y_true, y_pred):
    return roc_auc_score(y_true, y_pred, average='macro')

auc_scorer = make_scorer(multilabel_auc, needs_proba=True)


y = df[targets]

label_sums = y.sum(axis=1)
print(set(label_sums))
counts = pd.Series(label_sums).value_counts()
print(counts)
label_sums[:20]


from sklearn.model_selection import StratifiedKFold 

best_results=[]

cv = StratifiedKFold(n_splits=4)
label_counts = y.sum(axis=1)

for clf, param_grid in models:
    pipe = Pipeline(steps=[
        ('preprocess', preprocessor_new),
        ('clf', clf)
    ])

    grid = GridSearchCV(
        estimator = pipe,
        param_grid = param_grid,
        scoring=auc_scorer,
        cv=cv.split(X, label_counts),
        n_jobs=-1,
        verbose=1
    )

    grid.fit(X, y)

    best_results.append({
        'model' : clf.estimator.__class__.__name__,
        'best_score' : grid.best_score_,
        'best_params' : grid.best_params_, 
        'best_estimator' : grid.best_estimator_, 
    })

for result in best_results:
    print(f"\nModel: {result['model']}")
    print("Best AUC:", result['best_score'])
    print("Best Params:", result['best_params'])


X_no_outliers = no_outlr_df.drop(columns=targets+["id"])
X_no_outliers


y_no_outliers = no_outlr_df[targets]
y_no_outliers


# What happens if we remove the outliers 

# from sklearn.model_selection import StratifiedKFold 

# best_results=[]

# for clf, param_grid in models:
#     pipe = Pipeline(steps=[
#         ('preprocess', preprocessor),
#         ('clf', clf)
#     ])

#     grid = GridSearchCV(
#         estimator = pipe,
#         param_grid = param_grid,
#         scoring=auc_scorer,
#         cv=4,
#         n_jobs=-1,
#         verbose=1
#     )

#     grid.fit(X_no_outliers, y_no_outliers)

#     best_results.append({
#         'model' : clf.estimator.__class__.__name__,
#         'best_score' : grid.best_score_,
#         'best_params' : grid.best_params_, 
#         'best_estimator' : grid.best_estimator_, 
#     })

# for result in best_results:
#     print(f"\nModel: {result['model']}")
#     print("Best AUC:", result['best_score'])
#     print("Best Params:", result['best_params'])


from sklearn.model_selection import train_test_split 

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=123) 

classifier = OneVsRestClassifier(XGBClassifier(learning_rate=0.1, max_depth = 3, n_estimators=100))
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_val)
print(y_pred[:5, :])


from sklearn.metrics import classification_report 

print(classification_report(y_true=y_val,
                           y_pred=y_pred,
                           target_names=targets))


X_no_train, X_no_val, y_no_train, y_no_val = train_test_split(X_no_outliers, y_no_outliers, test_size=0.3, random_state=123) 

classifier = OneVsRestClassifier(XGBClassifier(learning_rate=0.1, max_depth = 3, n_estimators=100))
classifier.fit(X_no_train, y_no_train)

y_no_pred = classifier.predict(X_no_val)
print(y_no_pred[:5, :])


print(classification_report(y_true=y_no_val,
                           y_pred=y_no_pred,
                           target_names=targets))


from xgboost import DMatrix, XGBClassifier
import shap 
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt 

targets = ['Pastry','Z_Scratch','K_Scatch','Stains','Dirtiness','Bumps','Other_Faults']

X = df.drop(columns=targets+["id"])
y = df[targets]

# X[num_features] = X[num_features].apply(pd.to_numeric, errors='coerce')
# X[cat_features] = X[cat_features].astype('category')

xgb = XGBClassifier(learning_rate=0.1, max_depth = 3, n_estimators=100)

preproc_pipe = Pipeline(steps=[
    ('preprocess', preprocessor_new) 
])

X_transformed = preproc_pipe.fit_transform(X)

delta_feature_names = delta_creator.get_feature_names_out()
all_feature_names = (list(delta_feature_names) + num_features + cat_features)

X_transf_df = pd.DataFrame(X_transformed, columns=all_feature_names, index=X.index)

xgb.fit(X_transformed, y)


X_transf_df


print(X_transf_df.dtypes)


from xgboost import DMatrix

booster = xgb.get_booster()

feature_names = X_transf_df.columns
Xd = DMatrix(X_transf_df)

shap_values = booster.predict(Xd, pred_contribs=True)
preds = booster.predict(Xd)


shap.summary_plot(
    [shap_values[:, k, :-1] for k in range(shap_values.shape[1])],
    X_transf_df.values, 
    plot_type="bar",
    feature_names=feature_names,
    class_names=targets,
    show=False,
)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
plt.show()
print()


for k, target_label in enumerate(targets):
    print(f"VIOLIN PLOT REPRESENTING <{target_label}>")
    shap.summary_plot(shap_values[:,k, :-1], X_transf_df.values,
                      plot_type="violin",
                      feature_names=feature_names, show=False)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.show()
    print()


explainer = shap.TreeExplainer(xgb)
shap_interaction_values = explainer.shap_interaction_values(X_transf_df)


shap_interaction_values = np.mean(np.abs(np.array(shap_interaction_values)), axis=0)
print(type(shap_interaction_values))
mean_interactions = np.abs(shap_interaction_values).mean(axis=0)
print(mean_interactions.shape)
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(12,10))
sns.heatmap(mean_interactions, xticklabels=feature_names, yticklabels=feature_names, cmap='viridis', annot=True, fmt='.2f', annot_kws={'size' : 6})
plt.title('Mean absolute interactions')
plt.show()


X_transf_df.nunique()


interact_df = df[num_features+['Steel_Plate_Thickness']]

corr_mtx = interact_df.corr()

plt.figure(figsize = (12,10))

sns.heatmap(
    corr_mtx,
    annot=True,    
    fmt=".2f",
    cmap="coolwarm",   
    center=0,
    linewidths=0.5,
    square=True,
    cbar_kws={"shrink": 0.75}
)

plt.title("Correlation Heatmap of Numeric Features")
plt.tight_layout()
plt.show()


print(df['Steel_Plate_Thickness'].nunique())
print(df['Steel_Plate_Thickness'].unique())
print(df['Steel_Plate_Thickness'].value_counts())


# I will now attempt to group together the highly correlated geometric features of the dataset
# to reduce dimensionality, in order to hopefully reduce noise/help performance 

drop_features = ['Outside_X_Index', 'Pixels_Areas', 'SigmoidOfAreas']

engineered_df = df.drop(drop_features, axis=1)

engineered_df.head()


class Log_Area_Perimeter_Ratio(BaseEstimator, TransformerMixin):
    def __init__(self, log_area, x_perim, y_perim):
        self.log_area = log_area
        self.x_perim = x_perim
        self.y_perim = y_perim

    def fit(self, X, y=None):
        return self 

    def transform(self, X):
        result = X[self.log_area] / (np.log(X[self.x_perim] + X[self.y_perim]))
        return result.to_frame(name='Log_Area_Perimeter_Ratio')

    def get_feature_names_out(self, input_features=None):
        return ['Log_Area_Perimeter_Ratio']


class Log_Area_Index_Ratio(BaseEstimator, TransformerMixin):
    def __init__(self, log_area, log_x_ix, log_y_ix):
        self.log_area = log_area
        self.log_x_ix = log_x_ix
        self.log_y_ix = log_y_ix

    def fit(self, X, y=None):
        return self 

    def transform(self, X):
        result = X[self.log_area] / (X[self.log_x_ix] + X[self.log_y_ix])
        return result.to_frame(name='Log_Area_Index_Ratio')

    def get_feature_names_out(self, input_features=None):
        return ['Log_Area_Index_Ratio']


targets = ['Pastry','Z_Scratch','K_Scatch','Stains','Dirtiness','Bumps','Other_Faults']

engineered_X = engineered_df.drop(columns=targets+["id"])

delta_specs = [
    ('X_Maximum', 'X_Minimum', 'x'),
    ('Y_Maximum', 'Y_Minimum', 'y'),
    ('Maximum_of_Luminosity', 'Minimum_of_Luminosity', 'lum')
]

cat_features = ['TypeOfSteel_A400', 'TypeOfSteel_A300', 'Outside_Global_Index', 'Steel_Plate_Thickness']

dropped_new = ['X_Maximum', 'X_Minimum', 'Y_Maximum', 'Y_Minimum', 'Maximum_of_Luminosity', 'Minimum_of_Luminosity', 'X_Perimeter', 'Y_Perimeter', 'Log_X_Index', 'Log_Y_Index']
num_features_new = [col for col in engineered_X.columns if col not in cat_features and col not in dropped_new]


from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

ap_cols = ['LogOfAreas', 'X_Perimeter', 'Y_Perimeter']
aix_cols = ['LogOfAreas', 'Log_X_Index', 'Log_Y_Index']

delta_creator = DeltaCreator(delta_specs)
log_area_perim = Log_Area_Perimeter_Ratio('LogOfAreas', 'X_Perimeter', 'Y_Perimeter')
log_area_ix = Log_Area_Index_Ratio('LogOfAreas', 'Log_X_Index', 'Log_Y_Index')

preprocessor_ap = ColumnTransformer(transformers=[
    ('deltas', Pipeline([
        ('create_deltas', delta_creator)
    ]), [col for spec in delta_specs for col in spec[:2]]),
    ('log_area_perim', log_area_perim, ap_cols),
    ('log_area_ix', log_area_ix, aix_cols),
    ('numeric', 'passthrough', num_features_new),
    ('categoric', 'passthrough', cat_features)
])

pipeline_ap = Pipeline([
    ('preprocess', preprocessor_ap)
])


eng_X_transformed = pipeline_ap.fit_transform(engineered_X)
y = df[targets]

delta_feature_names = delta_creator.get_feature_names_out()
ap_feature_names = log_area_perim.get_feature_names_out()
aix_feature_names = log_area_ix.get_feature_names_out()
all_feature_names_new = (list(delta_feature_names) + ap_feature_names + aix_feature_names + num_features_new + cat_features)

eng_X_tf_df = pd.DataFrame(eng_X_transformed, columns=all_feature_names_new)
eng_X_tf_df.head()


from xgboost import XGBClassifier

full_pipeline = Pipeline([
    ('preprocess', preprocessor_ap),
    ('xgb', XGBClassifier(learning_rate=0.1, max_depth = 3, n_estimators=100, use_label_encoder=False, eval_metric='logloss'))
])

from sklearn.model_selection import cross_val_score

scores = cross_val_score(
    full_pipeline,
    engineered_X, y,
    cv=5,
    scoring=auc_scorer
)

print("ROC AUC scores:", scores)


# Binning Thickness and processing categorical variables as such

df_copy = df.copy()
df_copy['Binned_Thickness'] = df_copy['Steel_Plate_Thickness'].round(-1)
print(df_copy['Binned_Thickness'].value_counts())


def thickness_bin(val):
    if val <= 40:
        return 0 

    elif val <= 60:
        return 1 

    elif val <= 70:
        return 2 

    elif val <= 80:
        return 3 

    elif val <= 100:
        return 4

    elif val <= 150:
        return 5

    elif val <= 200:
        return 6 

    elif val > 250:
        return 7 


from sklearn.base import BaseEstimator, TransformerMixin

class Binner(BaseEstimator, TransformerMixin):
    def __init__(self, col_bin, func_bin):
        self.col_bin = col_bin
        self.func_bin = func_bin

    def fit(self, X, y=None):
        return self 

    def transform (self, X):
        return X[self.col_bin].apply(self.func_bin).to_frame(name=f'{self.col_bin}_Bin')


df_copy['Hand_Binned_Thickness'] = df_copy['Steel_Plate_Thickness'].apply(thickness_bin)
print(df_copy['Hand_Binned_Thickness'].value_counts())


df['Outside_Global_Index'].value_counts()


def OGI_bin(val):
    if val < 0.5:
        return 0
    else:
        return 1

ogi_new = df['Outside_Global_Index'].apply(OGI_bin)
print(ogi_new.value_counts())


ap_cols = ['LogOfAreas', 'X_Perimeter', 'Y_Perimeter']
aix_cols = ['LogOfAreas', 'Log_X_Index', 'Log_Y_Index']

types_of_steel = ['TypeOfSteel_A400', 'TypeOfSteel_A300']

delta_creator = DeltaCreator(delta_specs)
log_area_perim = Log_Area_Perimeter_Ratio('LogOfAreas', 'X_Perimeter', 'Y_Perimeter')
log_area_ix = Log_Area_Index_Ratio('LogOfAreas', 'Log_X_Index', 'Log_Y_Index')
thickness_binner = Binner('Steel_Plate_Thickness', thickness_bin)
ogi_binner = Binner('Outside_Global_Index', OGI_bin)

preprocessor_bin = ColumnTransformer(transformers=[
    ('deltas', Pipeline([
        ('create_deltas', delta_creator)
    ]), [col for spec in delta_specs for col in spec[:2]]),
    ('log_area_perim', log_area_perim, ap_cols),
    ('log_area_ix', log_area_ix, aix_cols),
    ('thickness', thickness_binner, ['Steel_Plate_Thickness']),
    ('ogi', ogi_binner, ['Outside_Global_Index']),
    ('numeric', 'passthrough', num_features_new),
    ('categoric', 'passthrough', types_of_steel)
])


from xgboost import XGBClassifier

full_pipeline_new = Pipeline([
    ('preprocess', preprocessor_bin),
    ('xgb', XGBClassifier(learning_rate=0.1, max_depth = 3, n_estimators=100, use_label_encoder=False, eval_metric='logloss'))
])

from sklearn.model_selection import cross_val_score

scores_new = cross_val_score(
    full_pipeline_new,
    engineered_X, y,
    cv=5,
    scoring=auc_scorer
)

print("ROC AUC scores:", scores_new)

