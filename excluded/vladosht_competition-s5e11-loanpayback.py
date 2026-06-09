import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import preprocessing, pipeline, linear_model, metrics, base, utils, model_selection
import time
pd.set_option('future.no_silent_downcasting', True)  #Silence some pandas warnings
pd.set_option('display.max_rows', 100)


ml_label = 'loan_paid_back'
dataset = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv').replace([np.inf, -np.inf], np.nan)
print('Count of NaN values present in dataframe:', len(dataset[dataset.isna().any(axis='columns')]))
start_time = time.time() #Start timing the model creation
pd.concat([dataset.dtypes.rename('dtypes').to_frame(),dataset.nunique().rename('nunique').to_frame()],axis='columns')


def make_frequency(a_col,df_input):
    category = str(a_col)
    df = pd.pivot_table(df_input,index=category,columns=ml_label,values='id',aggfunc='count').reset_index()
    df['category'] = category
    df = df.rename(columns={category:"class"}).set_index(['category','class'])
    df['freq'] = (df.iloc[:,1] / df.sum(axis='columns')).round(10)
    return df
paid_back_frequencies = pd.concat(map(lambda x:make_frequency(x,dataset), dataset.select_dtypes('object')))
paid_back_frequencies


paid_back_relative_std = paid_back_frequencies.groupby(by='category')['freq']
paid_back_relative_std = paid_back_relative_std.std() / paid_back_relative_std.mean()
paid_back_relative_std = paid_back_relative_std.sort_values()
print(paid_back_relative_std)
paid_back_relative_std_columns = paid_back_relative_std[paid_back_relative_std<0.05].index.to_list()
paid_back_relative_std_columns


[dataset.boxplot(column=i,by='grade_subgrade',figsize=(15,5)).get_figure().suptitle("")
     for i in dataset.select_dtypes(exclude='object') if i not in [ml_label]]
plt.show()


def feature_engineering_eda(X):
    X = X.copy()
    X['grade'] = X['grade_subgrade'].str[:1]
    X = X.drop(columns=['grade_subgrade']+paid_back_relative_std_columns)
    return X
eda_features = preprocessing.FunctionTransformer(func=feature_engineering_eda,validate=False,check_inverse=False).fit_transform(dataset)
leading_categories = pd.concat(map(lambda x:make_frequency(x,eda_features), eda_features.select_dtypes('object')))
leading_category_names = leading_categories.index.get_level_values(0).unique().tolist()
print(leading_category_names)
leading_categories


# This dataframe will be the backbone of our per-group logistic regression.
categorical_groups = eda_features.groupby(by=leading_category_names)[ml_label].sum()
categorical_groups = categorical_groups.to_frame()
categorical_groups['n_samples'] = eda_features.groupby(by=leading_category_names)[ml_label].count()
categorical_groups[ml_label] /= categorical_groups['n_samples']
categorical_groups


def estimate_feature_importance(df):
    df = df.drop(columns=['id']+leading_category_names)
    local_model = pipeline.Pipeline([
            ("scale_to_0_mean_and_unit_variance",preprocessing.StandardScaler()),
            ("model",linear_model.LogisticRegression())
        ]).fit(X=df.drop(columns=[ml_label]),y=df[ml_label])
    # I assume more important features will have larger coefficients (i.e.,abs(coefficient)>0)
    # So we extract here the coefficients from each regression
    feature_coeffs = pd.Series(local_model['model'].coef_[0],index=df.drop(columns=[ml_label]).columns)
    return feature_coeffs
    
def run_importance_estimation(a_group):
    key, df = a_group
    try:
        feature_coefs = estimate_feature_importance(df)
    except:
        return
    feature_coefs = feature_coefs.to_frame().T
    for i in range(len(key)):
        feature_coefs[leading_category_names[i]] = key[i]
    return feature_coefs.set_index(leading_category_names)

feature_coefficients = pd.concat(list(map(run_importance_estimation,eda_features.groupby(by=leading_category_names))))
feature_coefficients


feature_importance_raw = pd.merge(categorical_groups, feature_coefficients, left_index=True, right_index=True, how='left')
feature_importance_raw = feature_importance_raw.fillna(0.0).abs()
feature_importance_raw


feature_columns = feature_importance_raw.drop(columns=[ml_label,'n_samples']).columns.to_list()
feature_importance_scaled = preprocessing.StandardScaler().fit_transform(feature_importance_raw[feature_columns].fillna(0.0).abs().T)


feature_importances = feature_importance_raw.copy()
feature_importances.loc[:,feature_columns] = feature_importance_scaled.transpose()
feature_importances[feature_columns].plot(figsize=(15,5))
plt.show()
feature_importances


feature_importances_cutoff = feature_importances[feature_importances[ml_label]<0.98].dropna()
feature_importances_cutoff['% uncertainty'] = (0.5 - (feature_importances_cutoff[ml_label] - 0.5).abs()).abs()*2
feature_importances_cutoff['% samples'] = feature_importances_cutoff['n_samples']
feature_importances_cutoff['% samples'] /= feature_importances_cutoff['% samples'].max()
ax = feature_importances_cutoff[['% uncertainty','% samples']].plot.bar(figsize=(15,5))
feature_importances_cutoff[feature_columns].plot(ax=ax,rot=90)
ax.legend(ncols=len(feature_columns),bbox_to_anchor=(0.1, 1.0))
plt.show()


leading_features = feature_importances.dropna()[feature_columns].median()
display(leading_features)
leading_features = leading_features[leading_features>-0.5].index.to_list()
leading_features


# We need these to hold the row and column indexes, because they are lost after the standard scikit-learn transformations,
# which work with numpy arrays
index_holder, column_holder = None, None

# Function for a custom preprocessor, which leaves in the dataset only the categorical and continuous features
# which we identified above as important.
def feature_selection_from_eda(X):
    global index_holder, column_holder
    X = X.copy()[leading_category_names+leading_features].set_index(leading_category_names)
    index_holder = X.index
    column_holder = X.columns
    return X

# Re-implant the column and row indexes after the transformation chain
def final_feature_preparation(X):
    global index_holder, column_holder
    df = pd.DataFrame(X,index=index_holder,columns=column_holder).reset_index()
    return df

class CustomEstimator(base.BaseEstimator, base.RegressorMixin):
    
    def __init__(self,sample_cutoff_n=30):
        self.sample_cutoff_n = sample_cutoff_n

    def _get_target_col(self,i):
        return f"{self.target_col}_{i}"

    def fit(self, X, y):

        self.feature_names_ = list(X.columns)
        self.n_features_in_ = len(self.feature_names_)
        self.target_col = y.name

        data = X
        data[self.target_col] = y

        # Here we compute the payback frequency tables per group of feature values.
        # Most of the groups will not have enough samples to calculate a reliable per-group probability.
        # For these, we will use a coarser grouping.
        self.grouped_data_ = []
        for left_n in range(len(self.feature_names_)):
            left_n += 1
            a_grouping = data.groupby(self.feature_names_[:left_n])[self.target_col].aggregate(['sum','count'])
            a_grouping = a_grouping[a_grouping['count'] >= self.sample_cutoff_n]
            a_grouping[self._get_target_col(left_n)] = a_grouping['sum'] / a_grouping['count']
            a_grouping = a_grouping.drop(columns=['sum','count']).reset_index()
            self.grouped_data_.append(a_grouping)

        self.is_fitted_ = True
        return self

    def predict(self, X):
        
        utils.validation.check_is_fitted(self, 'is_fitted_')

        predictions_df = X
        for i in self.grouped_data_:
            predictions_df = predictions_df.merge(i,how='left')

        predictions = None
        for i in range(len(self.feature_names_),0,-1):
            if predictions is None:
                predictions = predictions_df[self._get_target_col(i)]
                continue
            predictions = predictions.combine_first(predictions_df[self._get_target_col(i)])

        return predictions.values
        
    def score(self, X, y):
        y_pred = self.predict(X)
        return metrics.roc_auc_score(y_true=y, y_score=y_pred)
        
ml_pipeline = pipeline.Pipeline(steps = [
        ("feature_engineering",preprocessing.FunctionTransformer(func=feature_engineering_eda,validate=False,check_inverse=False)),
        ("feature_selection",preprocessing.FunctionTransformer(func=feature_selection_from_eda,validate=False,check_inverse=False)),
        ("discretize",preprocessing.KBinsDiscretizer(n_bins=26,encode='ordinal',subsample=None)),
        ("feature_prep",preprocessing.FunctionTransformer(func=final_feature_preparation,validate=False,check_inverse=False)),
        ("model",CustomEstimator())
    ],verbose=True).fit(dataset.drop(columns=[ml_label]),dataset[ml_label])

display(ml_pipeline)
print(f"Done in {time.time() - start_time:.1f} seconds. Score:",
      ml_pipeline.score(dataset.drop(columns=[ml_label]),dataset[ml_label]))
for i in ml_pipeline['model'].grouped_data_:
    display(i.head(3))


test_dataset = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')
test_dataset[ml_label] = ml_pipeline.predict(test_dataset)
print('Mean of prediction:', test_dataset[ml_label].mean())
print('Mean of training:  ', dataset[ml_label].mean())
test_dataset.isna().any()


submission = test_dataset[['id',ml_label]]
submission.to_csv('/kaggle/working/submission.csv', index=False)
print("Submission file successfully generated.")
submission

