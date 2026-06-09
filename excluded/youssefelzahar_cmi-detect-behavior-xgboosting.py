import pandas as pd
#pd.set_option('display.width', None)
#pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)
from scipy.stats import shapiro
import numpy as np
from scipy.stats import skew
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import kaggle_evaluation.cmi_inference_server
from sklearn.model_selection import RandomizedSearchCV



data=pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv")
data.columns


gdata=pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/train_demographics.csv")
gdata.info()


data["sequence_type"].value_counts()


result = data.groupby('sequence_id').agg({
    'sequence_type': lambda x: ','.join(x),  # Concatenate types as string
    'sequence_counter': 'sum'                # Sum of counters
}).reset_index()

result


data.isnull().sum()


gdata.isnull().sum()


number_data=data.select_dtypes(["int64","float64"])


total_rows=number_data.shape[0]
null_data = number_data.isna().sum().reset_index().rename(columns = {0: "Nulls_Count", "index": "Column_Name"}).sort_values(by="Nulls_Count", ascending=False)
null_data['Percentage']=(null_data['Nulls_Count']/total_rows)*100
null_data[null_data["Nulls_Count"] > 0]


stat, p = shapiro(number_data)
print('Statistics=%.3f, p=%.3f' % (stat, p))
if p > 0.05:
    print("Data looks normally distributed (fail to reject H0)")
else:
    print("Data does not look normally distributed (reject H0)")


median = number_data.median(numeric_only=True)

clean_data = number_data.fillna(median)

clean_data.isnull().sum()


object_data=data.select_dtypes(["object"])
object_data


total_rows=object_data.shape[0]
null_data = object_data.isna().sum().reset_index().rename(columns = {0: "Nulls_Count", "index": "Column_Name"}).sort_values(by="Nulls_Count", ascending=False)
null_data['Percentage']=(null_data['Nulls_Count']/total_rows)*100
null_data[null_data["Nulls_Count"] > 0]


data=pd.concat([clean_data,object_data],axis=1)
data.isnull().sum()


object_data = data.select_dtypes(include='object').columns.tolist()
if 'sequence_id' in object_data:
    object_data.remove('sequence_id')

sensor_data = [col for col in data.select_dtypes(include='number').columns if col != 'gesture']

def extract_features_per_squence(data, sensor_cols):
    feature_list = []
    sequence_ids = data["sequence_id"].unique()
    for seq_id in sequence_ids:
        seq_data = data[data['sequence_id'] == seq_id]
        features = {'sequence_id': seq_id}
        for col in sensor_cols:
            values = seq_data[col]
            features[f"{col}_mean"] = values.mean()
            features[f"{col}_std"] = values.std()
            features[f"{col}_median"] = values.median()
            features[f"{col}_min"] = values.min()
            features[f"{col}_max"] = values.max()
            features[f"{col}_range"] = values.max() - values.min()
            #features[f"{col}_iqr"] = values.quantile(0.75) - values.quantile(0.25)

        feature_list.append(features)
    return pd.DataFrame(feature_list)

fixed_length_df = extract_features_per_squence(data, sensor_data)
object_cols_df = data.groupby('sequence_id')[object_data].first().reset_index()
gesture_map = data.groupby('sequence_id')['gesture'].first().reset_index()

data = fixed_length_df.merge(object_cols_df, on='sequence_id', how='left')
data = data.merge(gesture_map, on='sequence_id', how='left')

data.head()



def check_outliers(df):
    for col in df.columns:
        if df[col].dtype != 'object':
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - (1.5 * iqr)
            upper_bound = q3 + (1.5 * iqr)
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            #print(f"Column: {col}")
            #print(f"Outliers: {len(outliers)}")
    return df
check_outliers(data)   


def solve_outliers(df):
    for col in df.columns:
        if df[col].dtype != 'object':
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - (1.5 * iqr)
            upper_bound = q3 + (1.5 * iqr)
            median=df[col].median()
            df[col] = np.where(df[col] < lower_bound, median, df[col])
            df[col] = np.where(df[col] > upper_bound, median, df[col])
            #print(f"Column: {col}")
            #print(f"Outliers: {len(df[(df[col] < lower_bound) | (df[col] > upper_bound)])}")
           
    return df
data=solve_outliers(data)        


data.select_dtypes(exclude='object').skew()




skewed_features = data.apply(
    lambda x: skew(x.dropna()) if x.dtype != 'object' else 0
)
high_skew = skewed_features[abs(skewed_features) > 0.75].index.tolist()

print(f"عدد الأعمدة اللي فيها skew عالي: {len(high_skew)}")

pt = PowerTransformer(method='yeo-johnson')

data[high_skew] = pt.fit_transform(data[high_skew])


from scipy.stats import chi2_contingency

chi2_stat, p_val, dof, ex = chi2_contingency(pd.crosstab(data['subject'], data['gesture_x']))
print(f"P-value: {p_val}")
if p_val < 0.05:  # Typically, a p-value < 0.05 suggests significance
    print("The column is significant.")


data['subject'] = data['subject'].str.replace('SUBJ_', '').astype(int)



gdata['subject'] = gdata['subject'].str.replace('SUBJ_', '').astype(int)



data=data.drop("gesture_y",axis=1)


from sklearn.preprocessing import LabelEncoder

object_cols = data.select_dtypes(include='object').columns.tolist()

object_cols = [col for col in object_cols if col != 'gesture_x']  

le_dict = {}
for col in object_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))
    le_dict[col] = le  



data.head()


data["gesture_x"].value_counts()


import joblib

le = LabelEncoder()
data['gesture_encoded'] = le.fit_transform(data['gesture_x'])


joblib.dump(le, 'gesture_encoder.pkl')



X=data.drop(["gesture_encoded","gesture_x"],axis=1)
y=data["gesture_encoded"]
"""""
xgb_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='mlogloss',tree_method='hist',device="cuda"))
])
xgb_scores = cross_val_score(xgb_pipeline, X, y, cv=5, scoring='accuracy')
print("XGBoost CV Accuracy:", xgb_scores.mean())
"""""


"""""
param_dist = {
    'xgb__n_estimators': [100, 200, 300],
    'xgb__max_depth': [3, 5, 7],
    'xgb__learning_rate': [0.01, 0.05, 0.1],
    'xgb__subsample': [0.7, 0.8, 1],
    'xgb__colsample_bytree': [0.7, 0.9, 1.0],
    'xgb__gamma': [0, 0.1, 0.3]
}

random_search = RandomizedSearchCV(
    xgb_pipeline,
    param_distributions=param_dist,
    n_iter=20,
    scoring='accuracy',
    cv=5,
    verbose=1,
    random_state=42,
    n_jobs=-1
)
random_search.fit(X, y)
"""""


best_params = {
    'subsample': 0.8,
    'n_estimators': 300,
    'max_depth': 7,
    'learning_rate': 0.05,
    'gamma': 0.1,
    'colsample_bytree': 0.7
}

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('xgb', XGBClassifier(
        use_label_encoder=False,
        eval_metric='mlogloss',
        device='cuda',
        **best_params
    ))
])
xgb_scores = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')
print("XGBoost CV Accuracy:", xgb_scores.mean())
pipeline.fit(X, y)
joblib.dump(pipeline, "xgb_best_model.pkl")
print("✅ Model saved as xgb_best_model.pkl")


gesture = joblib.load('/kaggle/working/gesture_encoder.pkl')



feature_cols = X.columns.tolist()

np.save("/kaggle/working/sensor_data_cols.npy", feature_cols, allow_pickle=True)




import polars as pl
import joblib
import numpy as np
import pandas as pd

def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    sequence_df = sequence.to_pandas()
    demographics_df = demographics.to_pandas()

    sensor_data = [col for col in sequence_df.select_dtypes(include='number').columns]

    features = {}
    for col in sensor_data:
        values = sequence_df[col]
        features[f"{col}_mean"] = values.mean()
        features[f"{col}_std"] = values.std()
        features[f"{col}_median"] = values.median()
        features[f"{col}_min"] = values.min()
        features[f"{col}_max"] = values.max()
    
    combined = {**features, **demographics_df.iloc[0].to_dict()}
    input_df = pd.DataFrame([combined])

    if 'subject' in input_df.columns:
        input_df['subject'] = input_df['subject'].astype(str).str.replace('SUBJ_', '').astype(int)

    model = joblib.load("/kaggle/working/xgb_best_model.pkl")
    encoder = joblib.load("/kaggle/working/gesture_encoder.pkl")

    try:
        feature_cols = np.load("/kaggle/working/sensor_data_cols.npy", allow_pickle=True).tolist()
        missing_cols = [col for col in feature_cols if col not in input_df.columns]

        if missing_cols:
            zero_df = pd.DataFrame(0, index=input_df.index, columns=missing_cols)
            input_df = pd.concat([input_df, zero_df], axis=1)

        input_df = input_df[feature_cols]
    except FileNotFoundError:
        pass

    pred_encoded = model.predict(input_df)
    pred = encoder.inverse_transform(pred_encoded)

    return pred[0]



import os
inference_server = kaggle_evaluation.cmi_inference_server.CMIInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(
        data_paths=(
            '/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv',
            '/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv',
        )
    )

