import pandas as pd
import numpy as np
import time
from matplotlib import animation, pyplot as plt
from IPython.display import HTML
from sklearn import preprocessing, pipeline, compose, decomposition, linear_model, metrics
from joblib import Parallel, delayed
start_time = time.time() #Time the whole thing


ml_label = 'diagnosed_diabetes'
dataset_X = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv').replace([np.inf, -np.inf], np.nan).set_index('id')
print('{} rows contain NaN values in a dataframe of {} rows.'.format(len(dataset_X[dataset_X.isna().any(axis='columns')]),len(dataset_X)))
dataset_y = dataset_X[ml_label]
dataset_X = dataset_X.drop(columns=ml_label)
test_dataset = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')
test_dataset = test_dataset.replace([np.inf, -np.inf], np.nan).set_index('id')


# We start by getting the dtype and unique count of each feature
column_descriptions = pd.concat([dataset_X.dtypes.rename('dtype').astype(str).replace(
                                    regex=r'64',value='').str.strip().to_frame(),
                                dataset_X.nunique().rename('nunique').to_frame()],
                               axis='columns')

# Then we add the feature kind. I will use the sort order key for better feature order in the logic further below
def get_feature_type(row):
    numeric_categorical_cutoff = 10
    if row['nunique'] == 2:
        kind = ('boolean',10)
    elif row['dtype'] in ['object']:
        kind = ('categorical',20)
    elif row['nunique'] > numeric_categorical_cutoff and row['dtype'] in ['int','float']:
        kind = ('continuous',40)
    else:
        kind = ('ordinal',30)
    return {'feature_kind':kind[0],'sort_order':kind[1]}

column_descriptions = pd.concat([column_descriptions,
                                column_descriptions.apply(get_feature_type,axis='columns').apply(pd.Series)],
                                axis='columns')

# VIFs
def compute_VIFs(df):
    df = df.select_dtypes(exclude='object').copy()
    df -= df.mean()
    with Parallel(n_jobs=-1,prefer='threads') as p:
        # I think ElasticNet produces better VIFs than OrdinaryLeastSquares. If identical operation to
        # statsmodels.stats.outliers_influence.variance_inflation_factor is desired, the linear_model.LinearRegression
        # should be used here instead
        estimators = p(delayed(linear_model.ElasticNet().fit)(df.drop(columns=i),df[i]) for i in df.columns)
        r2_scores = p(delayed(estimators[i].score)(df.drop(columns=df.columns[i]),df[df.columns[i]]) for i in range(len(estimators)))
        r2_scores = pd.Series(index = df.columns, data = r2_scores)
        vifs = 1/(1-r2_scores)
    return vifs
    
column_descriptions['VIF'] = compute_VIFs(dataset_X)
column_descriptions['VIF'] = column_descriptions['VIF'].fillna(0.0)
column_descriptions


def make_column_preprocessor(df_column_descriptions, continuous_binning=False):
    all_feature_kinds = df_column_descriptions.drop_duplicates(subset=['feature_kind']).sort_values(by='sort_order')['feature_kind'].to_list()
    preprocessors = []
    # For each feature kind present in the descriptions...
    for a_feature_kind in all_feature_kinds:
        # Choose a suitable preprocessor...
        estimator = "passthrough"
        if 'continuous' in a_feature_kind and continuous_binning:
            estimator = preprocessing.KBinsDiscretizer(n_bins=10, encode='ordinal',subsample=None)
        if 'categorical' in a_feature_kind:
            estimator = preprocessing.OrdinalEncoder()
        if 'log-normal' in a_feature_kind:
            estimator = preprocessing.PowerTransformer(standardize=False)
        if 'pca' in a_feature_kind:
            estimator = decomposition.PCA(n_components='mle')
        # ...and select all columns with this feature kind.
        selected_descriptions = df_column_descriptions[df_column_descriptions['feature_kind']==a_feature_kind]
        selected_columns = selected_descriptions.sort_values(by=['sort_order','nunique']).index.to_list()
        # Construct a tuple, suitable for the compose.ColumnTransformer
        preprocessors.append((a_feature_kind, estimator, selected_columns))
    return preprocessors

eda_pipeline = pipeline.Pipeline(steps = [
            ("preprocessor", compose.ColumnTransformer(transformers=make_column_preprocessor(column_descriptions),
                                                       verbose_feature_names_out=False)),
            ("scaler",       preprocessing.StandardScaler()), # Without scaling the downstream model does not converge
            ("model",        linear_model.LogisticRegression())
        ],verbose=True).set_output(transform='pandas').fit(dataset_X,dataset_y)

eda_pipeline


coefs = pd.DataFrame(data=eda_pipeline['model'].coef_.transpose(),
                     index=eda_pipeline['model'].feature_names_in_,
                     columns=['importance']).abs().sort_values(by=['importance'],ascending=False)
coefs['importance'] /= coefs['importance'].sum()  #L1 normalization
coefs.plot(rot=60)
coefs['total_%'] = coefs['importance'].cumsum().round(3)*100
coefs


important_features = coefs[coefs['importance']*100>=0.9].index.to_list()
important_features += ['waist_to_hip_ratio', 'alcohol_consumption_per_week']
important_features = column_descriptions[column_descriptions.index.isin(important_features)].copy()
important_features['importance_%'] = (coefs['importance']*100).round(2)
important_features = important_features.sort_values(by='importance_%',ascending=False)

fig, axes = plt.subplots(nrows=len(important_features),ncols=2,figsize=(15, 3*len(important_features)))

def plot_hist_autobins(ax, a_series):
    bins = a_series.nunique()
    if bins > 300:
        bins = 300
    elif bins > 100:
        bins = bins // 2
    elif bins > 10:
        bins = bins - 2
    return a_series.hist(ax = ax, legend=True, bins = bins)

def do_plot_column(a_feature):
    feature_index = important_features.index.get_loc(a_feature)
    ax = axes[feature_index][0]
    ax.set_ylabel("Importance: {}%".format(important_features.loc[a_feature]['importance_%']))
    if feature_index == 0:
        ax.set_title("Train/test comparison")
    plot_hist_autobins(ax, dataset_X[a_feature].rename(a_feature+' (train)'))
    plot_hist_autobins(ax, test_dataset[a_feature].rename(a_feature+' (test)'))
    ax = axes[feature_index][1]
    if feature_index == 0:
        ax.set_title("Target variable influence")
    plot_hist_autobins(ax, dataset_X[dataset_y>=0.5][a_feature].rename(a_feature+" (has diabetes)"))
    plot_hist_autobins(ax, dataset_X[dataset_y<0.5][a_feature].rename(a_feature+" (no diabetes)"))

with Parallel(n_jobs=-1,prefer='threads') as p:
     p(delayed(do_plot_column)(i) for i in important_features.index)

plt.show(fig)

important_features = important_features[~important_features.index.isin(['cardiovascular_history'])]
important_features


important_features.loc[important_features.index.str.contains('cholesterol'),['feature_kind']] = 'pca'


dataset_y.hist(bins=3)
plt.show()


transformed_dataset = None
def expose_dataset(X):
    global transformed_dataset
    transformed_dataset = X.copy()
    return X
    
ml_pipeline = pipeline.Pipeline(steps = [
            ("preprocessor", compose.ColumnTransformer(transformers=make_column_preprocessor(important_features),
                                                       remainder='drop',
                                                       verbose_feature_names_out=False)),
            ("scaler",       preprocessing.StandardScaler()),
            ("expose",       preprocessing.FunctionTransformer(expose_dataset, feature_names_out="one-to-one")),
            ("model",        linear_model.LogisticRegression())
        ],verbose=True).set_output(transform='pandas').fit(dataset_X,dataset_y)

ml_pipeline


fig = plt.figure()

def do_plot(a_col):
    axes = fig.add_subplot(111) #Plot each feature on its own graphic, otherwise all get stacked up together
    axes.set_xlim([-5,5]) #All features are scaled to standard gaussian distribution, so they can be plotted in the same interval
    return plt.getp(plot_hist_autobins(axes,transformed_dataset[a_col]),"children") #get the matplotlib artists of the graphic

display(HTML(animation.ArtistAnimation(fig,
                                       [do_plot(i) for i in transformed_dataset.columns],
                                       interval=2000 #2-second delay between frames
                                      ).to_html5_video())) 
plt.close(fig)


final_descriptions = transformed_dataset.describe(percentiles=[i/10 for i in range(1,10,2)]).T
final_descriptions['VIF'] = compute_VIFs(transformed_dataset)
final_descriptions.round(5)


print("Shape of training data: {} samples by {} features".format(len(transformed_dataset),ml_pipeline['model'].n_features_in_))
# Get the index of the "1.0" target variable label
target_index = ml_pipeline['model'].classes_.argmax()  


y_pred = ml_pipeline.predict_proba(dataset_X)[:,target_index]
clf_disp = metrics.RocCurveDisplay.from_predictions(dataset_y, y_pred)
auc = metrics.roc_auc_score(dataset_y, y_pred)
plt.show()
leaderboard_top = 0.707
print("ROC-AUC:",auc,f"or about {1-auc/leaderboard_top:.2%} away from current leaderboard top of {leaderboard_top}")


test_dataset[ml_label] = ml_pipeline.predict_proba(test_dataset)[:,target_index].round(2)
print('Mean of prediction:', test_dataset[ml_label].mean())
print('Mean of training:  ', dataset_y.mean())
test_dataset.isna().any().rename('has_NaNs?').to_frame()


submission = test_dataset[[ml_label]]
submission.to_csv('/kaggle/working/submission.csv', index=True)
print("Submission file successfully generated.")
submission


print(f"Done in {time.time() - start_time:.1f} seconds")

