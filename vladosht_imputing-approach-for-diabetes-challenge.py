import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn import preprocessing, pipeline, compose, decomposition, ensemble, impute
import time
start_time = time.time() #Time the whole thing


ml_label = 'diagnosed_diabetes'
ml_features = ['family_history_diabetes', 'physical_activity_minutes_per_week', 'age',
               'systolic_bp', 'diastolic_bp', 'heart_rate',
               'diet_score', 'bmi', 'waist_to_hip_ratio',
               'hdl_cholesterol', 'ldl_cholesterol', 'cholesterol_total', 'triglycerides',
               ml_label]
dataset = pd.concat([pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv'),
                     pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')]).set_index('id')[ml_features]
dataset.info()


class MyClassifierProba(ensemble.HistGradientBoostingClassifier):
    def predict(self, X):
        return self.predict_proba(X)[:,list(self.classes_).index(1.0)]

def binarizer_with_nan(X, threshold=0.0, copy=True):
    """
    Signature and behavior copied from sklearn.preprocessing.Binarizer
    """
    if copy:
        X = X.copy()
    for a_col in X.columns:
        above_threshold = X[~X[a_col].isna()][a_col] > threshold
        X.loc[above_threshold[above_threshold].index, a_col] = 1
        X.loc[above_threshold[~above_threshold].index, a_col] = 0
    return X

ml_pipeline = pipeline.Pipeline(steps = [
            ("scaler",       preprocessing.StandardScaler()),
            # This binarizer is needed, because the StandardScaler standardizes the diagnosed_diabetes column too,
            # and because sklearn.preprocessing.Binarizer does not work with NaNs
            ("binarizer",    compose.ColumnTransformer(transformers=[
                                ('target',
                                 preprocessing.FunctionTransformer(func=binarizer_with_nan,feature_names_out='one-to-one',),
                                 [ml_label])
                             ],remainder='passthrough', verbose_feature_names_out=False)),
            ("impute",       impute.IterativeImputer(estimator = MyClassifierProba(),
                                                     tol=0.015,
                                                     skip_complete=True,
                                                     verbose=2,
                                                     add_indicator=True))
        ],verbose=True).set_output(transform='pandas')

ml_pipeline


transformed = ml_pipeline.fit_transform(dataset).rename(columns={'missingindicator_' + ml_label:"is_prediction"})
transformed['is_prediction'] = transformed['is_prediction'] > 0.5


display(transformed.info())
display(transformed.nunique())
display(transformed.describe(percentiles=[i/10 for i in range(1,10,2)]).T.round(5))


transformed.groupby(by='is_prediction')[ml_label].describe().round(4)


results = transformed[transformed['is_prediction']][ml_label].round(2)
results.hist()


submission = results.to_frame().reset_index()
print("Submission does not contain NaNs?","❌" if submission.isna().any().any() else "✔️")
print("All predictions are between 0 and 1?","✔️" if submission[ml_label].between(0,1).all() else "❌")
submission.to_csv('/kaggle/working/submission.csv', index=False)
print("Submission file successfully generated.")
submission


print(f"Done in {time.time() - start_time:.1f} seconds")

