import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import datetime
import warnings

from sklearn import clone
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error, mean_squared_log_error, r2_score, roc_auc_score, make_scorer
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.feature_selection import SelectFromModel, RFE, SelectKBest
from sklearn.linear_model import LogisticRegression, Lasso
from lightgbm import LGBMRegressor

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)


DATA_PATH = "/kaggle/input/playground-series-s5e5/"
WORKING_PATH = "/kaggle/working/"  
RESULTS_PATH = WORKING_PATH + "results"
RSEED = 42
N_CV_SPLITS = 5

os.makedirs(RESULTS_PATH, exist_ok=True)

np.random.seed(RSEED)
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("deep")

global_results = pd.DataFrame()
target = 'calories'


def get_current_time():
    return datetime.datetime.strftime(datetime.datetime.now(), "%Y-%m-%d_%H-%M-%S")

def read_data(fileName, dataPath=DATA_PATH):
    return pd.read_csv(dataPath + fileName)

def root_mean_squared_error(ytrue, ypred):
    return np.sqrt(mean_squared_error(ytrue, ypred))

def root_mean_squared_log_error(ytrue, ypred, sample_weight=None):
    ypred_clipped = np.maximum(ypred, 1e-15)  # Prevent negative values
    return np.sqrt(mean_squared_log_error(ytrue, ypred_clipped, sample_weight=sample_weight))

root_mean_squared_log_error_scorer = make_scorer(root_mean_squared_log_error, \
                                                 greater_is_better=False)


def plot_histplot(df, column, bins="auto", title=None, figsize=(10, 6), kde=True, hist=True,
                 color='steelblue', stat='count', log_scale=False, ax=None, ylabel="", label=None):
    """
    Creates a histogram plot for numerical data.
    """
    # Don't create a new figure if ax is provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    if kde and not hist:
        sns.kdeplot(data=df, x=column, ax=ax, color=color, label=label)
    else:
        sns.histplot(data=df, x=column, bins=bins, kde=kde, 
                    element="step" if not hist else "bars",
                    color=color, stat=stat, ax=ax, label=label)
    
    # Set title and labels on the provided axis
    if title: ax.set_title(title, fontsize=14)
    if ylabel: ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlabel(column, fontsize=12)
    
    # Apply log scale if requested
    if log_scale:
        ax.set_yscale('log')
    
    # Add grid for better readability
    ax.grid(axis='y', alpha=0.3)
    
    # Only do tight_layout if we created the figure
    if ax is None:
        plt.tight_layout()
        
    return ax


def plot_pairplot(df, hue_column="", title=""):
    colors = sns.color_palette("viridis", 2)

    plt.figure(figsize=(12, 10))


    if hue_column:
        pairplot = sns.pairplot(
            df, 
            hue=hue_column,
            palette=colors, 
            corner=True,  # Shows only the lower triangle to reduce redundancy
            plot_kws={'alpha': 0.6, 's': 30, 'edgecolor': 'k', 'linewidth': 0.5},
            diag_kws={'alpha': 0.6}
        )
    else:
        pairplot = sns.pairplot(
            df,
            corner=True,  
            plot_kws={'alpha': 0.6, 's': 30, 'edgecolor': 'k', 'linewidth': 0.5},
            diag_kws={'alpha': 0.6}
        )

    pairplot.figure.suptitle(title, y=1.02, fontsize=16)
    plt.tight_layout()


def plot_heatmap(corr, title="", annot=True, figsize=(12, 10), *args, **kwargs):
    plt.figure(figsize=figsize)

    # Create a mask for the upper triangle to avoid redundancy
    mask = np.triu(np.ones_like(corr, dtype=bool))

    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0, square=True, linewidths=.5, 
                annot=annot, fmt=".2f", cbar_kws={"shrink": .8}, **kwargs)

    plt.title(title, fontsize=16)
    plt.tight_layout()


def plot_horizontal_bar(x, y, title="", xlabel="", ylabel=""):
    plt.figure(figsize=(10, 8))
    sns.barplot(x=x, y=y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()


def print_metrics(ytrue, yhat):
    rmse = root_mean_squared_error(ytrue, yhat)
    r2 = r2_score(ytrue, yhat)

    print("RMSE:", rmse)
    print("R2:", r2)

    return {"r2": r2, "rmse": rmse}


def save_submission(id, yhat, column=target, filename="", path="results/"):
    if not filename:
        filename = get_current_time() + ".csv"

    res = pd.DataFrame({
        "id": id,
        column: yhat
    })

    path += filename
    res.to_csv(path, index=False)


def handle_prediction(X, y, model, Xval=None, yval=None, storing="", print_yhat=False, 
                      fold=None, full_set=False, cvn=5, score_fn=None):
    X = X.copy()
    y = y.copy()

    model.fit(X, y)

    yhat = model.predict(X)
    train_res = print_metrics(y, yhat)

    print()

    if Xval is not None and yval is not None:
        yhat = model.predict(Xval)
        val_res = print_metrics(yval, yhat)
    else:
        full_set = None
        val_res = None

    # cross_val
    if full_set:
        X = pd.concat([X, Xval])
        y = pd.concat([y, yval])

    if not fold:
        cross_res = abs(cross_val_score(model, X, y, cv=cvn, n_jobs=-1, 
                    scoring=score_fn if callable(score_fn) or isinstance(score_fn, str) else "neg_root_mean_squared_error"))
    else:
        cross_res = np.zeros((cvn, 1))
        for i, (train_idx, val_idx) in enumerate(fold.split(X, y)):
            model = clone(model)
            model.fit(X.iloc[train_idx, :], y.iloc[train_idx])
            yhat = model.predict(X.iloc[val_idx, :])
            cross_res[i] = score_fn(y.iloc[val_idx], yhat)
            
    print("Cross_val RMSE:", cross_res)

    if print_yhat:
        print(yhat)

    if storing:
        cur = []
        for res in [train_res] + ([val_res] if val_res else []):
            for name, metric in res.items():
                cur.append(f"{name}_{metric:.4f}")
        cur.append(f"cross_rmse_{cross_res.mean()}")
        global_results[storing] = cur

    return model


class AdversarialCV:
    def __init__(self, model, rseed=42):
        self.model = model
        self.rseed = rseed

    def make_cv(self, train, test, cv=5, shuffle=True, fold_method=None):
        train = train.copy()
        test = test.copy()

        train["test"] = 0
        test["test"] = 1

        df = pd.concat([train, test], axis=0)
        X, y = df.drop("test", axis=1), df["test"]
        
        if isinstance(fold_method, str):
            if fold_method == 'skf':
                fold_method = StratifiedKFold(n_splits=5, shuffle=shuffle, random_state=self.rseed)
            elif fold_method == 'kf':
                fold_method = KFold(n_splits=5, shuffle=shuffle, random_state=self.rseed)
       
        res = np.zeros(shape=(1, cv))

        for i, (train_ix, test_ix) in enumerate(fold_method.split(X, y)):
            self.model.fit(X.iloc[train_ix], y.iloc[train_ix])
            
            if hasattr(self.model, 'predict_proba'):
                yhat = self.model.predict_proba(X.iloc[test_ix])[:, 1]
            else:
                yhat = self.model.predict(X.iloc[test_ix])
                
            res[0, i] = roc_auc_score(y.iloc[test_ix], yhat)

        return res


train = read_data("train.csv")
test = read_data("test.csv")
orig = read_data("calories.csv", dataPath="/kaggle/input/calories-burnt-prediction/")
sampleSubmission = read_data("sample_submission.csv")

train.columns = train.columns.str.lower().str.replace(" ", "_")
test.columns = test.columns.str.lower().str.replace(" ", "_")
orig.columns = orig.columns.str.lower().str.replace(" ", "_")

train = train.drop("id", axis=1)
test, test_id = test.drop("id", axis=1), test["id"]

Xtrain = train.drop(target, axis=1)
ytrain = train[target].astype(np.uint8)

cv    = KFold(shuffle = True, random_state = RSEED, n_splits = N_CV_SPLITS)

orig = orig.drop('user_id', axis=1)


if 'male' in train['sex'].unique():
    train['sex'] = train['sex'].apply(lambda x: 1 if x == 'male' else 0)
    test['sex'] = test['sex'].apply(lambda x: 1 if x == 'male' else 0)
    
    orig.rename(columns={'gender': 'sex'}, inplace=True)
    orig['sex'] = orig['sex'].apply(lambda x: 1 if x == 'male' else 0) 


train.shape, test.shape, orig.shape


train


orig


train.info()


for p in [train, test, orig]:
    if p.isnull().sum().sum():
        print(p.isnull().sum())


train.describe()


orig.describe()


test.describe()


from itertools import combinations

adv_cv = AdversarialCV(
    LogisticRegression(random_state=RSEED),
    rseed=RSEED
)

outcome = []
cols = test.columns
for p1, p2 in combinations([["train", train], ["test", test], ["orig", orig]], r=2):
    n1, d1 = p1
    n2, d2 = p2

    res = adv_cv.make_cv(d1[cols], d2[cols], cv=5, shuffle=True, fold_method='kf')
    
    outcome.append([[n1, n2], res.mean()])

outcome


cols = train.columns
fig, axes = plt.subplots(nrows=len(cols), ncols=1, figsize=(10, 4 * len(cols))) 
train_color = 'steelblue'
test_color = 'darkorange'
n_sample = 200_000

for i, c in enumerate(cols):
    plot_histplot(train.sample(n_sample, random_state=RSEED), column=c, ax=axes[i], color=train_color, label='Train')
    if c in test.columns:
        plot_histplot(test.sample(n_sample, random_state=RSEED), column=c, ax=axes[i], color=test_color, label='Test')
    
    axes[i].legend() 

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels)
fig.suptitle("Distribution of data in train and test per column", fontsize=16, y=1.02) 
plt.tight_layout(rect=[0, 0.01, 1, 0.99]) 
plt.show() 


from math import ceil

cols = ['age', 'height', 'weight', 'duration', 'heart_rate', 'body_temp', 'calories']
ncols = 1
fig, axes = plt.subplots(nrows=ceil(len(cols) / ncols) , ncols=ncols, figsize=(10, 4 * len(cols))) 
colors = ['steelblue', 'darkorange']
n_sample = 100_000
unique_sex = train['sex'].unique()

for i, c in enumerate(cols):
    for j, value in enumerate(unique_sex):
        data = train[train['sex'] == value]
        plot_histplot(data.sample(n_sample, random_state=RSEED), column=c, ax=axes[i], 
                      color=colors[j], label=f'sex={value}')
    
    axes[i].legend() 

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels)
fig.suptitle("Distribution of features by sex", fontsize=16, y=1.02) 
plt.tight_layout(rect=[0, 0.01, 1, 0.99]) 
plt.show() 


plot_heatmap(train.corr(method='spearman'))


plot_pairplot(train.drop('sex', axis=1).sample(10_000), title="Pairwise plots of train. ");


def bin_features(df, columns, bins=10):
    df = df.copy()
    for c in columns:
        df[f'binned_{c}'] = pd.cut(df[c], bins=bins)
    return df

train_to_bin_columns = ['age', 'height', 'weight']
train_binned_columns = [f'binned_{c}' for c in train_to_bin_columns]
train_bin = bin_features(train, train_to_bin_columns)


cols = [c for c in train_bin.columns if c != target and c not in train_to_bin_columns]
fig, axes =plt.subplots(nrows=len(cols), ncols=1, figsize=(12, 22))

mean_color = 'steelblue'
median_color = 'darkorange'

for i, c in enumerate(cols):
    aggr_c = train_bin.groupby(c, observed=False).agg({
        target: ['mean', 'median', 'std']
    })

    if c in train_binned_columns:
        x = [f'{interval.left:.0f}-{interval.right:.0f}' for interval in aggr_c.index]
    else: 
        x = aggr_c.index

    sns.lineplot(x=x, y=aggr_c[(target, 'mean')], color=mean_color, label='Mean Calories', ax=axes[i])
    sns.lineplot(x=x, y=aggr_c[(target, 'median')], color=median_color, label='Median Calories', ax=axes[i])
    #sns.lineplot(x=x, y=aggr_c[(target, 'std')], color='forestgreen', label='Std Dev Calories')
    
    axes[i].set_xlabel(c)
    axes[i].set_ylabel(target)
    axes[i].legend() 

#handles, labels = axes[0].get_legend_handles_labels()
#fig.legend(handles, labels)
fig.suptitle("Aggregated target mean, median per feature", fontsize=16, y=1.02) 
plt.tight_layout(rect=[0, 0.01, 1, 0.99]) 
plt.show() 


from math import ceil

pairs = [('weight', 'height'), ('age', 'height'), ('age', 'weight'), ('duration', 'heart_rate'), ('duration', 'body_temp'), ('heart_rate', 'body_temp')]
ncols = 2
fig, axes = plt.subplots(nrows=ceil(len(pairs) / ncols), ncols=ncols, figsize=(12, 12))
n_sample = 100_000

for i, (a, b) in enumerate(pairs):
    sns.scatterplot(
        data=train.sample(n_sample, random_state=RSEED), x=a, y=b, ax=axes[i//2][i%2], alpha=0.3
    )

fig.suptitle("Feature interactions. ", fontsize=13)
plt.tight_layout()


print(train[["weight", "height", "age"]].corr('spearman'))


def create_features(df, lightweight=False, heavy=False):
    """
    Create engineered features for calorie prediction
    
    Args:
        df: DataFrame with raw features
        lightweight: If True, creates only 5 key features that can't be captured by polynomial transforms
                     If False, creates all features (default)
    
    Returns:
        DataFrame with added features
    """
    df = df.copy()
    
    # Calculate max heart rate (used by multiple features)
    df['max_hr'] = 220 - df['age']

    
    if lightweight:
        # 1. Basal Metabolic Rate (sex-specific formula)
        men_bmr = 88.362 + (13.397 * df['weight']) + (4.799 * df['height']) - (5.677 * df['age'])
        women_bmr = 447.593 + (9.247 * df['weight']) + (3.098 * df['height']) - (4.330 * df['age'])
        df['bmr'] = df['sex'] * men_bmr + (1 - df['sex']) * women_bmr
        
        # 2. Heart rate reserve - cardiovascular fitness metric
        df['heart_rate_reserve'] = df['heart_rate'] / df['max_hr']
        
        # 3. MET-based calorie estimation
        df['met_calories'] = (3 + (df['heart_rate'] / df['max_hr']) * 8) * df['weight'] * (df['duration']/60)
        
        # 4. Thermal effect - considers deviation from normal body temperature
        df['thermal_effect'] = (df['body_temp'] - 36.5) * df['weight'] * (df['duration'] / 60)
        
        # 5. Combined cardio-metabolic load
        df['cardio_load'] = df['heart_rate'] * df['duration'] * df['weight'] / 1000
    
    # --- ADDITIONAL FEATURES (added if not lightweight) ---
    if heavy:
        # Body composition features
        df['bmi'] = df['weight'] / ((df['height']/100) ** 2)
        df['body_surface_area'] = np.sqrt(df['height'] * df['weight'] / 3600)
        
        # Additional heart rate and cardiovascular features
        #df['hr_duration'] = df['heart_rate'] * df['duration']
        df['hr_intensity'] = df['heart_rate'] / df['max_hr'] * 100
        
        # Work and energy features
        #df['work_load'] = df['weight'] * df['duration']
        #df['heat_work'] = df['body_temp'] * df['duration']
        df['cardio_efficiency'] = df['heart_rate'] / df['body_temp']
        
        # Demographic interactions
        #df['age_weight'] = df['age'] * df['weight']
        #df['sex_weight'] = df['sex'] * df['weight']
        #df['age_height'] = df['age'] * df['height']
        
        # Complex interactions
        df['intensity_profile'] = df['heart_rate'] * df['body_temp'] / df['weight']
        df['endurance_factor'] = df['duration'] * df['heart_rate'] / df['age']
        df['met_estimate'] = 3 + (df['heart_rate'] / df['max_hr']) * 8
        
        # Non-linear transformations
        df['log_duration'] = np.log1p(df['duration'])
        df['duration_squared'] = df['duration'] ** 2
        df['weight_squared'] = df['weight'] ** 2
        
        # Advanced calorie calculations
        df['expected_calories'] = df['bmr'] * (df['duration']/1440) * (1 + df['heart_rate_reserve'])
        df['thermal_burn_factor'] = (df['body_temp'] - 36.5) * df['weight'] * (df['duration']/60)
        df['metabolic_efficiency'] = df['heart_rate'] / (df['body_temp'] * df['bmi'])
    
    # Always drop the intermediate max_hr feature
    df = df.drop('max_hr', axis=1)
    
    return df


train_bin_enc = train_bin.copy()

encoders = {}

for c in train_binned_columns:
    lb_enc = LabelEncoder()
    train_bin_enc[c] = lb_enc.fit_transform(train_bin_enc[c])
    encoders[c] = lb_enc


pl2 = PolynomialFeatures(degree=2)

temp = train_bin_enc.sample(n=50_000)
Xtrain_bin_enc, ytrain_bin_enc = temp.drop(target, axis=1), temp[target]

Xtrain_bin_enc_eng = create_features(Xtrain_bin_enc, lightweight=True, heavy=False)
Xtrain_poly = pl2.fit_transform(Xtrain_bin_enc_eng)

Xtrain_poly = pd.DataFrame(Xtrain_poly, columns=pl2.get_feature_names_out())
Xtrain_poly_eng = create_features(Xtrain_poly, lightweight=False, heavy=True)

sc = StandardScaler()
X_pl_sc = sc.fit_transform(Xtrain_poly_eng)


X_pl_sc.shape


selectors = [
    SelectKBest(k=30),
    RFE(Lasso(random_state=RSEED, max_iter=2_000)),
    SelectFromModel(LGBMRegressor(random_state=RSEED, verbose=0, force_col_wise=True)),
    RFE(LGBMRegressor(random_state=RSEED, verbose=0, force_col_wise=True))
]

res_datasets = []

for sr in selectors:
   X_sl = sr.fit_transform(X_pl_sc, ytrain_bin_enc)
   res_datasets.append([sr, X_sl])
   print("Iteration completed. ")


def get_important_features(selectors, feature_names):
    results = []
    
    for i, selector in enumerate(selectors):
        # Get the mask of selected features
        mask = selector.get_support()
        selected_features = feature_names[mask]
        
        # Different approach based on selector type
        if isinstance(selector, SelectKBest):
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': selector.scores_
            })

            importance_df = importance_df.sort_values('importance', ascending=False)
            method_name = "SelectKBest"
            
        elif isinstance(selector, RFE) and isinstance(selector.estimator_, Lasso):
            # RFE with Lasso has rankings and coefficients
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'ranking': selector.ranking_
            })

            coef = selector.estimator_.coef_
            temp = {}
            for c in selected_features:
                temp[c] = coef[i]

            importance_df['importance'] = [temp.get(c, 0) for c in feature_names]
            importance_df = importance_df.sort_values('importance', key=abs, ascending=False)
            method_name = "RFE(Lasso)"
            
        elif isinstance(selector, SelectFromModel) and isinstance(selector.estimator_, LGBMRegressor):
            # SelectFromModel with LGBMRegressor uses feature_importances_
            importances = selector.estimator_.feature_importances_
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            })

            importance_df = importance_df.sort_values('importance', ascending=False)
            method_name = "SelectFromModel(LGBMRegressor)"
            
        elif isinstance(selector, RFE) and isinstance(selector.estimator_, LGBMRegressor):
            # RFE with LGBMRegressor has rankings and feature_importances_
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'ranking': selector.ranking_
            })

            importances = selector.estimator_.feature_importances_
            temp = {}
            for c in selected_features:
                temp[c] = importances[i]

            importance_df['importance'] = [temp.get(c, 0) for c in feature_names]
            importance_df = importance_df.sort_values('ranking')
            method_name = "RFE(LGBMRegressor)"
        
        results.append({
            'method': method_name,
            'n_selected': sum(mask),
            'selected_features': selected_features,
            'importance_df': importance_df
        })

    return results

important_features_data = get_important_features([sr for sr, _ in res_datasets], Xtrain_poly_eng.columns)


feature_ranks = pd.DataFrame({'feature': Xtrain_poly_eng.columns})

for d in important_features_data:
    method_name = d['method']
    data = d["importance_df"].copy()
    data['importance'] = data['importance'].abs()

    feature_ranks[method_name] = feature_ranks['feature'].map(
        data.set_index('feature')['importance'])

# Calculate average importance across all methods (after normalizing each column)
for col in feature_ranks.columns[1:]:
    feature_ranks[f'{col}_norm'] = feature_ranks[col] / feature_ranks[col].max()

norm_cols = [col for col in feature_ranks.columns if col.endswith('_norm')]
feature_ranks['avg_importance'] = feature_ranks[norm_cols].mean(axis=1)

feature_ranks = feature_ranks.sort_values('avg_importance', ascending=False)
top_n = feature_ranks.shape[0]

plt.figure(figsize=(6, 10))
plt.barh(feature_ranks['feature'].iloc[:top_n], feature_ranks['avg_importance'].iloc[:top_n], alpha=0.6, color='steelblue')
plt.title('Feature Importance Across All Selected Features')
plt.xlabel('Average Normalized Importance')
plt.tight_layout()
plt.show()


top_n = 30
top_features = feature_ranks['feature'].iloc[:top_n]
top_features = np.concatenate([top_features, [c for c in Xtrain.columns if c not in top_features.values]]) 
                                    # adding original predictors 
                                    # for stability (they can be overshadowed due to multicollinearity)

train_pl_eng_smpl = Xtrain_poly_eng.copy()
train_pl_eng_smpl[target] = ytrain_bin_enc
heat_feat = np.concatenate([top_features, [target]])
plot_heatmap(train_pl_eng_smpl[heat_feat].corr('spearman'), 'Correlation among engineered predictors', 
             figsize=(10, 12), annot=False)


def convert_pl_eng_slc(train, test, target, top_features, dg=2, sample_n=None):
    pl = PolynomialFeatures(degree=dg)

    if sample_n is not None:
        train = train.sample(n=sample_n, random_state=RSEED)

    X, y = train.drop(target, axis=1), train[target]
    
    train_to_bin_columns = ['age', 'height', 'weight']
    train_binned_columns = [f'binned_{c}' for c in train_to_bin_columns]
    
    # Compute bin edges from combined data to ensure consistency
    bin_edges = {}
    for c in train_to_bin_columns:
        all_data = pd.concat([X[c], test[c]])
        bin_edges[c] = pd.cut(all_data, bins=10, retbins=True)[1]
    
    # Apply binning to train data with consistent edges
    X_bin = X.copy()
    for c in train_to_bin_columns:
        X_bin[f'binned_{c}'] = pd.cut(X[c], bins=bin_edges[c])
    
    encoders = {}
    for c in train_binned_columns:
        lb_enc = LabelEncoder()
        X_bin[c] = lb_enc.fit_transform(X_bin[c])
        encoders[c] = lb_enc
    
    X_leng = create_features(X_bin, lightweight=True, heavy=False)
    X_leng_pl = pl.fit_transform(X_leng)

    X_leng_pl = pd.DataFrame(X_leng_pl, columns=pl.get_feature_names_out())
    X_poly_eng = create_features(X_leng_pl, lightweight=False, heavy=True)

    train = X_poly_eng[top_features]
    train[target] = y

    # Apply same binning to test data
    test_bin = test.copy()
    for c in train_to_bin_columns:
        test_bin[f'binned_{c}'] = pd.cut(test[c], bins=bin_edges[c])
    
    for c in train_binned_columns:
        test_bin[c] = encoders[c].transform(test_bin[c])
    
    test_leng = create_features(test_bin, lightweight=True, heavy=False)
    test_poly = pl.transform(test_leng)
    test_poly = pd.DataFrame(test_poly, columns=pl.get_feature_names_out())
    test_poly_eng = create_features(test_poly, lightweight=False, heavy=True)

    return train, test_poly_eng[top_features]


train_poly_eng, test_poly_eng = convert_pl_eng_slc(train, test, target, top_features, dg=2, sample_n=None)
print("Sets are successfuly converted! ")


model = LGBMRegressor(n_estimators=2000, random_state=RSEED, verbose=0)

handle_prediction(
    Xtrain_bin_enc[Xtrain.columns], 
    ytrain_bin_enc,
    model=model,
    storing="lgbm_default",
    score_fn=root_mean_squared_log_error_scorer
)

print()

handle_prediction(
    train_poly_eng.drop(target, axis=1), 
    train_poly_eng[target],
    model=clone(model),
    storing=None,
    score_fn=root_mean_squared_log_error_scorer,
    cvn=7
)


model = LGBMRegressor(
    n_estimators=4000,
    random_state=RSEED,
    importance_type='gain',
    n_jobs=-1
)

model.fit(
    train_poly_eng.drop(target, axis=1), train_poly_eng[target],
    eval_metric='rmsle',
)

yhat = model.predict(test_poly_eng)

save_submission(test_id, yhat, target, filename='lgbm_est5000_pl2_eng_slc34.csv')

