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


train=pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test=pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
train


cols=['road_type', 'num_lanes', 'curvature', 'speed_limit', 'lighting',
       'weather', 'road_signs_present', 'public_road', 'time_of_day',
       'holiday', 'school_season', 'num_reported_accidents']


from sklearn.preprocessing import OneHotEncoder, StandardScaler, Normalizer
import pandas as pd
def preprocess(df, fitted_transformers=None):
    df_processed = pd.DataFrame(index=df.index)

    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    

    is_fitting = fitted_transformers is None

    if is_fitting:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        scaler = StandardScaler()
    else:
        encoder = fitted_transformers.get('encoder')
        scaler = fitted_transformers.get('scaler')
    
    if categorical_cols:
        if is_fitting:
            encoded = encoder.fit_transform(df[categorical_cols])
        else:
            encoded = encoder.transform(df[categorical_cols])
        
        encoded_df = pd.DataFrame(
            encoded, 
            columns=encoder.get_feature_names_out(categorical_cols),
            index=df.index
        )
        df_processed = pd.concat([df_processed, encoded_df], axis=1)
    if numerical_cols:
        if is_fitting:
            scaled = scaler.fit_transform(df[numerical_cols])
        else:
            scaled = scaler.transform(df[numerical_cols])
        
        scaled_df = pd.DataFrame(
            scaled, 
            columns=numerical_cols,
            index=df.index
        )
        df_processed = pd.concat([df_processed, scaled_df], axis=1)
    transformers = {'encoder': encoder, 'scaler': scaler}
    return df_processed, transformers


x=train[cols]
y=train['accident_risk']
x, transformers = preprocess(x)


from sklearn.model_selection import train_test_split as split
x_train, x_test, y_train, y_test=split(x,y,test_size=0.23,random_state=123)


from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from lightgbm import LGBMRegressor, early_stopping
import numpy as np
params = {
    'metric': 'rmse',
    'n_estimators': 3000,
    'learning_rate': 0.007,
    'max_depth': 15,
    'random_state': 42,
    'verbose': -1
}
model = LGBMRegressor(**params)

model.fit(
    x_train, y_train,
    eval_set=[(x_test[:90], y_test[:90])],

)
pred = model.predict(x_test)

r2 = r2_score(y_test, pred)
mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))

print(f"R2 score: {r2:.4f}")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.6f}")




x2=test[cols]
x2, _ = preprocess(x2, fitted_transformers=transformers)


predictions=model.predict(x2)
submission = pd.DataFrame({'id': test['id'], 'accident_risk': predictions })
submission.to_csv('/kaggle/working/submission.csv', index=False)
print("Done!")


num_cols = [i for i in cols if train[i].dtype in ['int64', 'float64']]


import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(7, 6))

numeric_data = train[num_cols]
correlation = numeric_data.corr()

sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Matrix of Numeric Features', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()


