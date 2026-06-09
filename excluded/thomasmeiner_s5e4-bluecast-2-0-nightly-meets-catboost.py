!pip install scikit-learn --upgrade -q


%%capture
!pip install bluecast --find-links=file:/kaggle/input/bluecast-nightly/bluecast-2.0.0-py3-none-any.whl


import numpy as np 
import pandas as pd 
from bluecast.blueprints.cast_cv_regression import BlueCastCVRegression
from bluecast.config.training_config import (
    CatboostTuneParamsRegressionConfig,
    TrainingConfig,
)
from bluecast.ml_modelling.catboost_regression import CatboostModelRegression


train = pd.read_csv(r'/kaggle/input/playground-series-s5e4/train.csv')
test = pd.read_csv(r'/kaggle/input/playground-series-s5e4/test.csv')
submission = pd.read_csv(r'/kaggle/input/playground-series-s5e4/sample_submission.csv')
target = "Listening_Time_minutes"


train.info()


# taken from: https://www.kaggle.com/code/litsea/ps5e04-quality-eda-fe-xgboost

def segment_percentage(percentage):
    if percentage <= 20:
        return '0-20'
    elif percentage <= 40:
        return '20-40'
    elif percentage <= 60:
        return '40-60'
    elif percentage <= 80:
        return '60-80'
    elif percentage <= 100:
        return '80-100'
    else:
        return '100+'

def feature_engineering(df):
    
    columns = ['Podcast_Name', 'Episode_Title']
    for col in columns:
        counts = df[col].value_counts(normalize=True)
        df[f'{col}_popularity'] = df[col].map(counts)
        
    df['Host_Popularity_percentage_segment'] = df['Host_Popularity_percentage'].apply(segment_percentage)
    df['Guest_Popularity_percentage_segment'] = df['Guest_Popularity_percentage'].apply(segment_percentage)
    
    df['Host_vs_Guest_Popularity'] = df['Host_Popularity_percentage'] / (df['Guest_Popularity_percentage'] + 1)
    df['Total_Popularity'] = df['Host_Popularity_percentage'] + df['Guest_Popularity_percentage']
    df['Very_Popular_Guest'] = ((df['Guest_Popularity_percentage'] > 50) & (df['Guest_Popularity_percentage'] < 55)).astype(int)
    df['Avg_Episode_Length_By_Genre'] = df.groupby('Genre')['Episode_Length_minutes'].transform('mean')
    
    df['Ads_per_Minute'] = df['Number_of_Ads'] / (df['Episode_Length_minutes'] + 1)
    df['Has_Ads'] = (df['Number_of_Ads'] > 0).astype(int)
    df['Ads_Intensity'] = pd.cut(df['Number_of_Ads'], bins=[-1, 0, 2, 3], labels=['No Ads', 'Few Ads', 'Many Ads'])
    
    df['Has_Sport'] = df['Podcast_Name'].str.contains(r'\b(Sports|Sport|Arena)\b', case=False, na=False).astype(int)
    df['Has_Music'] = df['Podcast_Name'].str.contains(r'\b(Music|Melody|Sound|Tune)\b', case=False, na=False).astype(int)
    df['Has_Comedy'] = df['Podcast_Name'].str.contains(r'\b(Joke|Funny|Comedy|Laugh|Humor)\b', case=False, na=False).astype(int)
    df['Has_Crime'] = df['Podcast_Name'].str.contains(r'\b(Mystery|Criminal|Crime|Detective)\b', case=False, na=False).astype(int)
    df['Has_Education'] = df['Podcast_Name'].str.contains(r'\b(Study|Educational|Learning|Brain)\b', case=False, na=False).astype(int)
    df['Has_Finance'] = df['Podcast_Name'].str.contains(r'\b(Business|Market|Money|Finance)\b', case=False, na=False).astype(int)
    
    # by yunsuxiaozi (https://www.kaggle.com/code/yunsuxiaozi/pss5e4-xgb-baseline)
    df['Episode_Title_num']=df['Episode_Title'].apply(lambda x:int(x[len('Episode '):]))
    df['sin_Episode_Title_num']=np.sin(2*np.pi*df['Episode_Title_num']/12)
    df['cos_Episode_Title_num']=np.cos(2*np.pi*df['Episode_Title_num']/12)
    df['sin_Episode_Length_minutes']=np.sin(2*np.pi*df['Episode_Length_minutes']/60)
    df['cos_Episode_Length_minutes']=np.cos(2*np.pi*df['Episode_Length_minutes']/60)
    
    new_features = ['Podcast_Name_popularity', 'Episode_Title_popularity', 'Host_Popularity_percentage_segment', 'Guest_Popularity_percentage_segment', 'Host_vs_Guest_Popularity', 'Ads_per_Minute', 'Total_Popularity', 'Avg_Episode_Length_By_Genre', 'Very_Popular_Guest', 'Has_Ads', 'Ads_Intensity', 'Episode_Title_num', 'sin_Episode_Title_num', 'cos_Episode_Title_num', 'sin_Episode_Length_minutes', 'cos_Episode_Length_minutes', 'Has_Sport', 'Has_Music', 'Has_Comedy', 'Has_Crime', 'Has_Education', 'Has_Finance']
    
    return df, new_features


train, new_features = feature_engineering(train)
test, new_features = feature_engineering(test)


train


from bluecast.preprocessing.feature_types import FeatureTypeDetector
from bluecast.preprocessing.feature_creation import AddRowLevelAggFeatures, GroupLevelAggFeatures


ignore_cols = [target, "id"]

feat_type_detector = FeatureTypeDetector()
train_data = feat_type_detector.fit_transform_feature_types(train.drop(ignore_cols, axis=1))


agg_feat_creator = AddRowLevelAggFeatures()

train_num = agg_feat_creator.add_row_level_agg_features(train.loc[:, feat_type_detector.num_columns])
test_num = agg_feat_creator.add_row_level_agg_features(test.loc[:, feat_type_detector.num_columns])

train_num = train_num.drop(agg_feat_creator.original_features, axis=1)
test_num = test_num.drop(agg_feat_creator.original_features, axis=1)


#train = pd.concat([train, train_num], axis=1)
#test = pd.concat([test, test_num], axis=1)


from typing import List, Optional, Dict, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, RepeatedStratifiedKFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostRegressor
from tqdm import tqdm


class CatBoostForwardSelector:
    def __init__(self):
        self.interaction_features: List[str] = []
        self.selected_interactions: List[str] = []
        self.base_features: List[str] = []
        self.best_rmse: float = np.inf
        self.categorical_features: List[str] = []

    def _create_interaction_features(
        self, df: pd.DataFrame, features: List[str]
    ) -> pd.DataFrame:
        for i, c1 in enumerate(features):
            for c2 in features[i + 1:]:
                new_col = f"{c1}_{c2}"
                df[new_col] = df[c1] * df[c2]
                if new_col not in self.interaction_features:
                    self.interaction_features.append(new_col)
        return df

    def fit_transform(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        features: List[str],
        target_col: str,
        group_col: Optional[str] = None,
        categorical_features: Optional[List[str]] = None,
        catboost_params: Optional[Dict[str, Union[int, float, str]]] = None,
    ) -> pd.DataFrame:
        self.base_features = features.copy()
        self.categorical_features = categorical_features or []

        train = self._create_interaction_features(train, features)
        test = self._create_interaction_features(test, features)

        if catboost_params is None:
            catboost_params = {
                "iterations": 50,
                "learning_rate": 0.1,
                "depth": 3,
                "loss_function": "RMSE",
                "verbose": 0,
                "random_seed": 42,
                "use_best_model": True,
                "task_type": "GPU"
            }

        all_features = ["baseline"] + self.interaction_features
        selected = []

        # After creating the interaction features
        if group_col is not None:
            folds = GroupKFold(n_splits=train[group_col].nunique())
            splits = list(folds.split(train, groups=train[group_col]))
        else:
            y_binned = LabelEncoder().fit_transform(
                pd.qcut(train[target_col], q=10, duplicates="drop")
            )
            folds = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)
            splits = list(folds.split(train, y_binned))
        
        all_features = ["baseline"] + self.interaction_features
        selected = []
        
        for col in tqdm(all_features):
            if col != "baseline":
                selected.append(col)
        
            oof = np.zeros(len(train))
            preds = np.zeros(len(test))
            fold_count = 0
        
            for train_idx, val_idx in splits:
                fold_count += 1
        
                x_train = train.iloc[train_idx][features + selected]
                y_train = train.iloc[train_idx][target_col]
                x_val = train.iloc[val_idx][features + selected]
                y_val = train.iloc[val_idx][target_col]
                x_test = test[features + selected]
        
                cat_features_indices = [
                    x_train.columns.get_loc(cf)
                    for cf in self.categorical_features
                    if cf in x_train.columns
                ]
        
                model = CatBoostRegressor(**catboost_params)
                model.fit(
                    x_train,
                    y_train,
                    eval_set=(x_val, y_val),
                    cat_features=cat_features_indices,
                    early_stopping_rounds=20
                )
        
                oof[val_idx] = model.predict(x_val)
                preds += model.predict(x_test)
        
            preds /= fold_count
            rmse = mean_squared_error(train[target_col], oof, squared=False)
        
            if rmse < self.best_rmse:
                print(f"NEW BEST with {col} at RMSE={rmse:.5f}")
                self.best_rmse = rmse
            else:
                print(f"Worse with {col} at RMSE={rmse:.5f}")
                if col != "baseline":
                    selected.remove(col)

        self.selected_interactions = selected
        return train[features + self.selected_interactions + [target_col]]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._create_interaction_features(df, self.base_features)
        return df[self.base_features + self.selected_interactions]



categorical_cols = train.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_cols = train.select_dtypes(exclude=['object', 'category']).columns.tolist()
numerical_cols.remove(target)

print(f"Numerical features: {numerical_cols}")

selector = CatBoostForwardSelector()

train = selector.fit_transform(
    train=train,
    test=test,
    features=numerical_cols,
    target_col=target,
    #group_col="year_group",  # or None
    categorical_features=categorical_cols
)

test = selector.transform(test)


# refresh with the columns
ignore_cols = [target, "id"]

feat_type_detector = FeatureTypeDetector()
train_data = feat_type_detector.fit_transform_feature_types(train.drop(ignore_cols, axis=1))


automl = BlueCastCVRegression(
    class_problem="regression", 
    ml_model=CatboostModelRegression(
            class_problem="regression",
            cat_columns=None,
            #conf_training=train_config,
            conf_catboost=CatboostTuneParamsRegressionConfig(),
        ),
        conf_xgboost=CatboostTuneParamsRegressionConfig(),
)
automl.conf_training.cat_encoding_via_ml_algorithm = True
automl.conf_training.hyperparameter_tuning_max_runtime_secs = 60 * 60 * 1.5 # 1h tuning time per model 
automl.conf_training.autotune_on_device = "gpu"

automl.fit(train.copy(), target_col=target)
y_preds = automl.predict(test)
submission[target] = y_preds
submission.to_csv("submission.csv", index=False)
submission

