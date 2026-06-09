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


original  = pd.read_csv('/kaggle/input/fertilizer-prediction/Fertilizer Prediction.csv')


original.head()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import  TransformerMixin
from sklearn.preprocessing import  MinMaxScaler, StandardScaler, RobustScaler
from sklearn.preprocessing import  LabelEncoder
from collections import defaultdict
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from matplotlib.lines import Line2D
from sklearn.preprocessing import LabelEncoder


import warnings
warnings.filterwarnings('ignore')


df  = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
test  = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')


df  =  pd.concat([original, df], axis='rows')


df.head()


df.rename({'Temparature':'Temperature'},axis='columns', inplace=True)
test.rename({'Temparature':'Temperature'},axis='columns', inplace=True)


import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

numerical_cols = df.select_dtypes(include='number').columns.difference(['id'])

num_cols = 3  # Number of columns in the grid
num_rows = (len(numerical_cols) + num_cols - 1) // num_cols

fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows), constrained_layout=True)
axes = axes.flatten()

for i, col in enumerate(numerical_cols):
    sns.kdeplot(df[col], ax=axes[i], color='blue', fill=True, label='Train')
    if col in test.columns:
        sns.kdeplot(test[col], ax=axes[i], color='red', fill=True, label='Test')
    axes[i].set_title(col)

    ax_box = axes[i].inset_axes([0.2, -0.4, 0.6, 0.2])  # [x, y, width, height]
    sns.boxplot(x=df[col], ax=ax_box, orient='h')
    ax_box.set(xlabel='')

# Remove unused subplots
for j in range(len(numerical_cols), len(axes)):
    fig.delaxes(axes[j])

# Add a legend in the top right corner of the figure
custom_lines = [
    Line2D([0], [0], color='blue', lw=4, label='Train'),
    Line2D([0], [0], color='red', lw=4, label='Test')
]
fig.legend(
    handles=custom_lines,
    loc='upper right',
    bbox_to_anchor=(1, 1),  # x=1 (right), y=1 (top)
    frameon=False
)

plt.show()



numerical_cols = df.select_dtypes(include='number').columns.difference(['id'])


corr_matrix = df[numerical_cols].corr()

# Affichage avec Seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation matrix")
plt.show()


cat_columns = df.select_dtypes(exclude=np.number).columns
num_cols = 3  # Number of columns in the grid
num_rows = (len(cat_columns) + num_cols - 1) // num_cols

# Create the subplots
fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows), constrained_layout=True)
axes = axes.flatten()
palette = sns.color_palette("Set2", len(df.iloc[:, 0].value_counts()))

for i, col in enumerate(cat_columns):
    df[col].value_counts().plot(kind='bar', ax=axes[i], color=palette)
    axes[i].set_title(col)
    axes[i].tick_params(axis='x', rotation=45, labelsize=8)
for j in range(len(cat_columns), len(axes)):
    fig.delaxes(axes[j])

plt.show()


def generate_fertilizer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # Interaction features
    df["Temp_Humidity"] = df["Temperature"] * df["Humidity"]
    df["Nitro_Phos"] = df["Nitrogen"] * df["Phosphorous"]
    df["Moisture_Potassium"] = df["Moisture"] * df["Potassium"]


    # Normalized nutrient features
    total_nutrients = df["Nitrogen"] + df["Phosphorous"] + df["Potassium"] + 1e-5
    df["Norm_N"] = df["Nitrogen"] / total_nutrients
    df["Norm_P"] = df["Phosphorous"] / total_nutrients
    df["Norm_K"] = df["Potassium"] / total_nutrients

    # Domain-specific features
    df["Soil_Fertility_Index"] = (df["Nitrogen"] + df["Phosphorous"] + df["Potassium"]) / 3
    df["Water_Stress"] = df["Temperature"] / (df["Humidity"] + 1e-5)
    df["Nutrient_Stress"] = df["Moisture"] / (df["Nitrogen"] + df["Phosphorous"] + df["Potassium"] + 1e-5)

    return df


#df  =  generate_fertilizer_features(df)
#test  =  generate_fertilizer_features(test)


class Custom_Scaler(TransformerMixin):
    def __init__(self, except_col=[], cols=[], strategy="MinMax"):
        super().__init__()
        self.except_col=except_col
        self.cols = cols if cols else []
        self.strategy = strategy

    def fit(self, df, y=None):
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        final_col =  numerical_cols.difference(self.except_col)
        self.col  =  final_col if not self.cols else self.cols
        if self.strategy=="RBT":
            self.scaler = RobustScaler().fit(df[self.col]) 
        elif self.strategy=="STD" :
            self.scaler = StandardScaler().fit(df[self.col])
        else :
            self.scaler =  MinMaxScaler().fit(df[self.col]) 
        return self
    
    def transform(self, data, y=None):
        df =data.copy()
        scaler_data =  self.scaler.transform(df[self.col])
        scaler_data_df = pd.DataFrame(scaler_data, columns=self.col, index=df.index)
        others_cols  =  df.columns.difference(self.col)
        return pd.concat([scaler_data_df, df[others_cols]], axis='columns')

class MultiColumnLabelEncoder(TransformerMixin):
    def __init__(self, except_col=[]):
        self.except_col = except_col
        self.label_encoders = defaultdict(LabelEncoder)

    def fit(self,X , y=None):
        df  = X.copy()
        cat_col =  df.select_dtypes(exclude=[np.number]).columns
        final_col =  cat_col.difference(self.except_col)
        self.columns = final_col
        for col in self.columns:
            self.label_encoders[col]
            self.label_encoders[col].fit(df[col])
        return self

    def transform(self, X):
        X_copy = X.copy()  # To avoid modifying the original dataframe
        for col in self.columns:
            X_copy[col] = X_copy[col].apply(lambda s: '<unknown>' if s not in self.label_encoders[col].classes_ else s)
            self.label_encoders[col].classes_ = np.append(self.label_encoders[col].classes_, '<unknown>')
            X_copy[col] = self.label_encoders[col].transform(X_copy[col])
        return X_copy

    def inverse_transform(self, X):
        X_copy = X.copy()  # To avoid modifying the original dataframe
        for col in self.columns:
            X_copy[col] = self.label_encoders[col].inverse_transform(X_copy[col])
        return X_copy


pipe  = Pipeline([('scaler', Custom_Scaler(except_col=['id'])), ('label_encoder', MultiColumnLabelEncoder(except_col=['Fertilizer Name']))])


transform_data  = pipe.fit_transform(df)


transform_data.head()


test_transform = pipe.transform(test)


test_data  =  test_transform.drop(columns=['id'])


X = transform_data.drop(['id', 'Fertilizer Name'], axis='columns')
y =  transform_data['Fertilizer Name']


encoder  = LabelEncoder()


y =  pd.Series(encoder.fit_transform(y))


LGBM_params = {'learning_rate': 0.04375759116574153, 'n_estimators': 290, 'max_depth': 10, 'num_leaves': 127, 'min_child_samples': 18, 'subsample': 0.6110123156192024, 'colsample_bytree': 0.9086928433636647, 'reg_alpha': 0.007981450304647659, 'reg_lambda': 5.195244483830506,
        "gpu_platform_id":0,  
        "gpu_device_id":0,
        }


def make_predictions(inputs, model):
    probs = model.predict_proba(inputs)
    top3_idx = np.argsort(probs, axis=1)[:, -3:][:, ::-1]
    predictions = encoder.inverse_transform(top3_idx.ravel()).reshape(top3_idx.shape)
    return predictions

def fast_top_k_score(actual, predicted, k=3):
    actual = np.array(actual)
    predicted = np.array(predicted)

    if len(actual) != len(predicted):
        raise ValueError(f"Length mismatch: actual has {len(actual)}, predicted has {len(predicted)}")

    top_k = predicted[:, :k]  # (n_samples, k)

    # Check if actual[i] is in top_k[i]
    matches = (top_k == actual[:, None])  # (n_samples, k), bool

    # Get the index (rank) where the match happens, or -1 if no match
    match_indices = np.argmax(matches, axis=1)  # (n_samples,)
    match_found = matches.any(axis=1)  # (n_samples,)

    # Reciprocal rank: 1 / (rank + 1), only if match is found
    reciprocal_ranks = np.zeros(len(actual))
    reciprocal_ranks[match_found] = 1.0 / (match_indices[match_found] + 1)

    return reciprocal_ranks.mean()


def probability(actual, predictions):
    actual_labels = encoder.inverse_transform(actual)
    actual_labels = actual_labels.tolist()
    predictions = predictions.tolist()
    score = fast_top_k_score(actual_labels, predictions, k=3)
    return score
    print(f"Score: {score:.5f}")


X_train, X_test, y_train, y_test = train_test_split(X,y)


model = LGBMClassifier(**LGBM_params)


model.fit(X_train, y_train)


pd.DataFrame({
    "columns":X.columns,
    "importance":model.feature_importances_ 
}).sort_values('importance')


y_pred =  make_predictions(X_test, model)
prediction = make_predictions(test_data, model)


score = probability(y_test, y_pred)
score


submission  = pd.DataFrame([], columns=['id', 'Fertilizer Name'])
submission.id  =  test.id
submission['Fertilizer Name'] =[' '.join(preds) for preds in prediction]


submission.head()


submission.to_csv('submission.csv', index=False)

