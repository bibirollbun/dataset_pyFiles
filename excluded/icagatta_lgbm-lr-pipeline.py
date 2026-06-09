import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import lightgbm as lgb
from sklearn.base import BaseEstimator, TransformerMixin


DATA_PATH = '/kaggle/input/widsdatathon2025/'


class SafeFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.selector = None
        self.fallback = False
        
    def fit(self, X, y):
        try:
            self.selector = Pipeline([
                ('variance', VarianceThreshold(threshold=0.01)),
                ('select', SelectFromModel(
                    LogisticRegression(
                        penalty='l1', 
                        solver='liblinear',
                        class_weight='balanced',
                        max_iter=1000
                    ),
                    threshold="median"
                ))
            ])
            self.selector.fit(X, y)
        except ValueError:
            self.fallback = True
        return self
    
    def transform(self, X):
        if self.fallback:
            return RobustScaler().fit_transform(X)
        return self.selector.transform(X)


def load_data(split):
    if split == 'TEST':
        categorical = pd.read_excel(f"{DATA_PATH}/{split}/{split}_CATEGORICAL.xlsx")
        connectome = pd.read_csv(f"{DATA_PATH}/{split}/{split}_FUNCTIONAL_CONNECTOME_MATRICES.csv")
        quantitative = pd.read_excel(f"{DATA_PATH}/{split}/{split}_QUANTITATIVE_METADATA.xlsx")
        df = connectome.merge(quantitative, on='participant_id').merge(categorical, on='participant_id')
    if split == 'TRAIN':
        categorical = pd.read_excel(f"{DATA_PATH}/{split}_NEW/{split}_CATEGORICAL_METADATA_new.xlsx")
        connectome = pd.read_csv(f"{DATA_PATH}/{split}_NEW/{split}_FUNCTIONAL_CONNECTOME_MATRICES_new_36P_Pearson.csv")
        quantitative = pd.read_excel(f"{DATA_PATH}/{split}_NEW/{split}_QUANTITATIVE_METADATA_new.xlsx")
        df = connectome.merge(quantitative, on='participant_id').merge(categorical, on='participant_id')
    
        solutions = pd.read_excel(f"{DATA_PATH}/TRAIN_NEW/TRAINING_SOLUTIONS.xlsx")
        df = df.merge(solutions, on='participant_id')
        
    return df


train_df = load_data('TRAIN')
test_df = load_data('TEST')

critical_features = ['MRI_Track_Age_at_Scan', 'SDQ_SDQ_Hyperactivity', 'Barratt_Barratt_P1_Edu']
train_df = train_df.dropna(subset=critical_features)

connectome_cols = [c for c in train_df.columns if 'throw' in c]

adhd_features = train_df.drop(['participant_id', 'ADHD_Outcome', 'Sex_F'], axis=1)
y_adhd = train_df['ADHD_Outcome']

sex_features = train_df[connectome_cols]
y_sex = train_df['Sex_F']

adhd_preprocessor = ColumnTransformer([
    ('connectome', RobustScaler(), connectome_cols),
    ('quant', RobustScaler(), ['MRI_Track_Age_at_Scan', 'SDQ_SDQ_Hyperactivity']),
    ('cat', OneHotEncoder(handle_unknown='ignore'), ['Basic_Demos_Study_Site'])
])

adhd_model = lgb.LGBMClassifier(
    objective='binary',
    num_leaves=63,
    max_depth=6,
    learning_rate=0.05,
    n_estimators=1200,
    class_weight='balanced',
    reg_alpha=0.05,
    reg_lambda=0.05,
    min_child_samples=20,
    verbosity=-1,
    force_row_wise=True,
)

adhd_pipeline = Pipeline([
    ('preprocessor', adhd_preprocessor),
    ('classifier', adhd_model)
])

sex_pipeline = Pipeline([
    ('preprocessing', SafeFeatureSelector()),
    ('scaler', RobustScaler()),
    ('classifier', LogisticRegression(
        penalty='elasticnet',
        solver='saga',
        l1_ratio=0.5,
        class_weight='balanced',
        max_iter=2000,
    ))
])

adhd_pipeline.fit(adhd_features, y_adhd)
sex_pipeline.fit(sex_features, y_sex)

test_adhd_features = test_df.drop('participant_id', axis=1)
test_sex_features = test_df[connectome_cols]

adhd_preds = adhd_pipeline.predict(test_adhd_features)
sex_preds = sex_pipeline.predict(test_sex_features)

submission = pd.DataFrame({
    'participant_id': test_df['participant_id'],
    'ADHD_Outcome': adhd_preds.astype(int),
    'Sex_F': sex_preds.astype(int)
})
submission.to_csv('submission.csv', index=False)




