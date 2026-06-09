"""
Goal: Learn how to write the feature engineering Class.

Author: Rudra Prasad Bhuyan
V1: 22-10-2025 09:43 IST
"""
print("")


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import List, Dict

import warnings
warnings.filterwarnings('ignore')


sub_path = '/kaggle/input/playground-series-s5e10/sample_submission.csv'
train_path = '/kaggle/input/playground-series-s5e10/train.csv'
test_path = '/kaggle/input/playground-series-s5e10/test.csv'

extra_2k_path = '/kaggle/input/simulated-roads-accident-data/synthetic_road_accidents_2k.csv'
extra_10k_path = '/kaggle/input/simulated-roads-accident-data/synthetic_road_accidents_10k.csv'
extra_100k_path = '/kaggle/input/simulated-roads-accident-data/synthetic_road_accidents_100k.csv'

sub_df = pd.read_csv(sub_path)
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

extra_2k_df = pd.read_csv(extra_2k_path)
extra_10k_df = pd.read_csv(extra_10k_path)
extra_100k_df = pd.read_csv(extra_100k_path)


train_df.sample(3)


extra_2k_df.sample(3)


train_df.shape


train_df.info()


train_df.select_dtypes(include='bool').columns.to_list()


train_df['road_signs_present'].map({True:1, False: 0, 'True':1, 'False':0})


train_df.select_dtypes(include=['float64', 'int64']).columns.to_list()


train_df.select_dtypes(include='object').columns.to_list()


class FeatureEngineer:
    """
    Feature engineering Pipeline:
        - Converts binary columns to int
        - Label-encodes multiclass categorical cols
        - Create engineered meta features (meat_*) 
        - Scales numeric columns 
    """

    def __init__(self,
                 binary_columns: List[str] = None,
                 multiclass_columns: List[str] = None,
                 numeric_columns: List[str] = None):
        
        self.binary_columns = binary_columns or ['road_signs_present', 'public_road', 'holiday', 'school_season']
        self.numeric_columns = numeric_columns or ['num_lanes', 'curvature', 'speed_limit', 'num_reported_accidents']
        self.multiclass_columns = multiclass_columns or ['road_type', 'lighting', 'weather', 'time_of_day']
        
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scaler: StandardScaler = None
        self.fitted = False

    # ==========================================================
    # ----------------------- Encode Binary --------------------
    # ==========================================================
    
    def _encode_binaries(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert binary columns to Integer columns."""
        df = df.copy()
        for col in self.binary_columns:
            if col in df.columns:
                # Convert boolean -> Integer
                df[col] = df[col].map({True: 1, False: 0, 'True': 1, 'False': 0}).fillna(df[col])

                # If still not integer, cast
                try:
                    df[col] = df[col].astype(int)
                except Exception:
                    df[col] = df[col].apply(lambda x: 1 if str(x).lower() in ['1', 'true'] else 0)

        return df

    # ==========================================================
    # ------------------- Fit Label Encoders -------------------
    # ==========================================================
    
    def fit_label_encoders(self, df: pd.DataFrame):
        """Perform the label Encoding"""
        for col in self.multiclass_columns:
            if col in df.columns:
                le = LabelEncoder()
                le.fit(df[col])
                self.label_encoders[col] = le

    def transform_label_encoders(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform the labels using fitted label encoders."""
        df = df.copy()
        for col, le in self.label_encoders.items():
            if col in df.columns:
                df[col] = le.transform(df[col])
        return df

    # ==========================================================
    # --------------------------- Fit Scaler -------------------------
    # ==========================================================
    
    def fit_scaler(self, df: pd.DataFrame):
        """Fit the StandardScaler on numeric columns."""
        self.scaler = StandardScaler()
        num_cols = [c for c in self.numeric_columns if c in df.columns]
        if len(num_cols) > 0:
            self.scaler.fit(df[num_cols].fillna(0).values)

    def transform_scaler(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform numeric columns using the fitted scaler."""
        df = df.copy()
        num_cols = [c for c in self.numeric_columns if c in df.columns]
        if len(num_cols) > 0 and self.scaler is not None:
            scaled = self.scaler.transform(df[num_cols].fillna(0).values)
            scaled_df = pd.DataFrame(scaled, columns=[f'scaled_{c}' for c in num_cols], index=df.index)
            df = pd.concat([df, scaled_df], axis=1)
        return df

    # ==========================================================
    # -------------- Create Engineered Features ---------------
    # ==========================================================
    
    def create_meta_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create meta features based on existing columns."""
        df = df.copy()

        # BASE meta features 
        if 'curvature' in df.columns:
            df['meta_curvature'] = 0.3 * df['curvature']
        else:
            df['meta_curvature'] = 0

        df['meta_night'] = 0.2 * ((df.get('lighting', '') == 'night') | (df.get('lighting', '') ==  'Night')).astype(int)
        df['meta_weather'] = 0.1 * (df.get('weather', '') != 'clear').astype(int)
        df['meta_speed'] = 0.2 * (df.get('speed_limit', 0) >= 60).astype(int)
        df['meta_accidents'] = 0.1 * (df.get('num_reported_accidents', 0) > 2).astype(int)
        df['meta_total'] = df['meta_curvature'] + df['meta_night'] + df['meta_weather'] + df['meta_speed'] + df['meta_accidents']

        # Interaction features (meta_*)
        df['meta_night_weather'] = df['meta_night'] * df['meta_weather']
        df['meta_night_curvature'] = df['meta_night'] * df['meta_curvature']
        df['meta_speed_curvature'] = df['meta_speed'] * df['meta_curvature']
        df['meta_weather_speed'] = df['meta_weather'] * df['meta_speed']
        df['meta_accidents_speed'] = df['meta_accidents'] * df['meta_speed']

        # Non-linear transforms
        df['meta_total_sq'] = df['meta_total'] ** 2
        df['meta_total_sqrt'] = np.sqrt(np.clip(df['meta_total'], a_min=0, a_max=None))
        df['meta_curvature_sq'] = df['meta_curvature'] ** 2

        # Lighting & weather granular
        df['meta_dawn_dusk'] = 0.15 * df['lighting'].isin(['dawn','dusk']).astype(int)
        df['meta_poor_visibility'] = 0.25 * ((df['lighting'] == 'night') | (df['weather'].isin(['fog', 'rain']))).astype(int)
        df['meta_severe_weather'] = 0.2 * df['weather'].isin(['snow','fog']).astype(int)
        df['meta_wet_conditions'] = 0.12 * df['weather'].isin(['rain','snow']).astype(int)

        # Speed zones
        df['meta_very_high_speed'] = 0.25 * (df['speed_limit'] >= 80).astype(int)
        df['meta_moderate_speed'] = 0.1 * ((df['speed_limit'] >= 40) & (df['speed_limit'] < 60)).astype(int)
        df['meta_low_speed'] = 0.05 * (df['speed_limit'] < 40).astype(int)

        # Accident history
        df['meta_high_accidents'] = 0.15 * (df['num_reported_accidents'] > 5).astype(int)
        df['meta_moderate_accidents'] = 0.08 * ((df['num_reported_accidents'] >= 3) & (df['num_reported_accidents'] <= 5)).astype(int)
        if 'num_reported_accidents' in df.columns:
            df['meta_accident_rate'] = df['num_reported_accidents'] / (df['num_reported_accidents'].max() + 1)

        # Curvature zones
        if 'curvature' in df.columns:
            q75 = df['curvature'].quantile(0.75)
            q25 = df['curvature'].quantile(0.25)
            df['meta_sharp_curve'] = 0.35 * (df['curvature'] > q75).astype(int)
            df['meta_moderate_curve'] = 0.2 * ((df['curvature'] > q25) & (df['curvature'] <= q75)).astype(int)
        else:
            df['meta_sharp_curve'] = 0
            df['meta_moderate_curve'] = 0

        # Combined risk scenarios
        df['meta_night_curve_risk'] = df['meta_night'] * df['meta_curvature'] * 1.5
        df['meta_weather_speed_risk'] = df['meta_weather'] * df['meta_speed'] * 1.3
        df['meta_triple_risk'] = df['meta_night'] * df['meta_weather'] * df['meta_speed']

        # Enhanced total
        df['meta_enhanced_total'] = (df['meta_total'] + df['meta_poor_visibility'] + df['meta_severe_weather'] +
                                     df['meta_very_high_speed'] + df['meta_high_accidents'] + df['meta_sharp_curve'])

        # Ratio features
        df['meta_curvature_per_speed'] = df['curvature'] / (df['speed_limit'] + 1)
        df['meta_accidents_per_curvature'] = df['num_reported_accidents'] / (df['curvature'] + 1)

        return df

    # ==========================================================
    # ------------ Curvature Interaction Features --------------
    # ==========================================================
    
    def create_curvature_cross_features(self, df: pd.DataFrame, target_columns: List[str]) -> pd.DataFrame:
        """Create curvature-based interaction features."""
        df = df.copy()
        for col in target_columns:
            if col not in df.columns:
                continue
            if np.issubdtype(df[col].dtype, np.number):
                df[f'eng_curvature_x_{col}_mult'] = df['curvature'] * df[col]
                df[f'eng_curvature_x_{col}_div'] = df['curvature'] / (df[col] + 1e-7)
            else:
                uniques = df[col].astype(str).unique()[:10]
                for val in uniques:
                    name = f"eng_curvature_x_{col}_{str(val).replace(' ', '_')}"
                    df[name] = df['curvature'] * (df[col].astype(str) == str(val)).astype(int)
        return df

    # ==========================================================
    # ---------------------- Fit & Transform  -----------------
    # ==========================================================
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform the data."""
        df = df.copy().reset_index(drop=True)
        df = self._encode_binaries(df)
        self.fit_label_encoders(df)
        df = self.transform_label_encoders(df)
        df = self.create_meta_features(df)
        cross_cols = list(self.numeric_columns) + [c for c in self.multiclass_columns if c in df.columns]
        df = self.create_curvature_cross_features(df, cross_cols)
        self.fit_scaler(df)
        df = self.transform_scaler(df)
        self.fitted = True
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform the data after fitting."""
        if not self.fitted:
            raise RuntimeError("FeatureEngineer must be fitted on train before transform(test).")
        df = df.copy().reset_index(drop=True)
        df = self._encode_binaries(df)
        df = self.transform_label_encoders(df)
        df = self.create_meta_features(df)
        cross_cols = list(self.numeric_columns) + [c for c in self.multiclass_columns if c in df.columns]
        df = self.create_curvature_cross_features(df, cross_cols)
        df = self.transform_scaler(df)
        return df



def merge_extra_data(train_df: pd.DataFrame,
                     extra_list: List[pd.DataFrame],
                     use_all: bool = True,
                     sample_frac: float = 1.0,
                     random_state: int = 42) -> pd.DataFrame:
    """
    Optionally append extra synthetic datasets to the train dataframe.
    - extra_list: list of dataframes to append
    - use_all: if False, will sample `sample_frac` from each extra df (useful to avoid memory issues)
    """
    frames = [train_df]
    for df in extra_list:
        if df is None or df.shape[0] == 0:
            continue
        if not use_all and 0 < sample_frac < 1.0:
            frames.append(df.sample(frac=sample_frac, random_state=random_state))
        else:
            frames.append(df)
    merged = pd.concat(frames, axis=0, ignore_index=True).reset_index(drop=True)
    return merged


# 1) Merge extras into train if desired (set use_all=False to sample smaller portion)
extra_list = [extra_2k_df, extra_10k_df, extra_100k_df]
train_aug = merge_extra_data(train_df, extra_list=extra_list, use_all=False, sample_frac=0.5)  
# If you don't want augmentation, just do: train_aug = train_df.copy()


# 2) Initialize FE
fe = FeatureEngineer(
    binary_columns=['road_signs_present','public_road','holiday','school_season'],
    multiclass_columns=['road_type','lighting','weather','time_of_day'],
    numeric_columns=['num_lanes','curvature','speed_limit','num_reported_accidents']
)


# 3) Fit-transform on train & transform on test
eng_train_df = fe.fit_transform(train_df)
eng_test_df  = fe.transform(test_df)


# 4) Quick checks
print("Train shape:", eng_train_df.shape)
print("Test shape: ", eng_test_df.shape)


eng_train_df.sample(3)


train_aug.columns.to_list()


print("Sample engineered cols: \n")
[c for c in eng_train_df.columns if c.startswith('meta_')]


print("Sample curvature-engineered cols: \n")
[c for c in eng_train_df.columns if c.startswith('eng_curvature')]

