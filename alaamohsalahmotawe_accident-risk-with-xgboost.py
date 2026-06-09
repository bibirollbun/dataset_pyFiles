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


import warnings
warnings.filterwarnings("ignore")


file_path="/kaggle/input/playground-series-s5e10/train.csv"



df=pd.read_csv(file_path)


df.shape


df.head()


df.info()


df=df.drop(columns=['id'])


#after delet id column
df.info()


print(df.duplicated().sum())


df.describe()


for col in df.columns:
    unique_values = df[col].unique() 
    print(f"Column '{col}' has {len(unique_values)} unique values:")
    print(unique_values)
    print("-"*50)


print(df[['curvature', 'accident_risk']].corr())
print('------------------------------------------')
print(df[['num_reported_accidents', 'accident_risk']].corr())
print('------------------------------------------')
print(df[['speed_limit', 'accident_risk']].corr())
print('------------------------------------------')
print(df[['num_lanes', 'accident_risk']].corr())



import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.scatterplot(
    x=df['curvature'],
    y=df['accident_risk'],
    hue=df['lighting'],
    palette='bright',
    alpha=0.8
)
plt.title('accident_risk vs curvature by lighting', fontsize=14)
plt.xlabel('curvature', fontsize=12)
plt.ylabel('accident_risk', fontsize=12)
plt.legend(title='lighting', title_fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()



import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.scatterplot(
    x=df['curvature'],
    y=df['accident_risk'],
    hue=df['speed_limit'],
    palette='bright',
    alpha=0.8
)
plt.title('accident_risk vs curvature by speed_limit ', fontsize=14)
plt.xlabel('curvature', fontsize=12)
plt.ylabel('accident_risk', fontsize=12)
plt.legend(title='speed_limit', title_fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.scatterplot(
    x=df['curvature'],
    y=df['accident_risk'],
    hue=df['weather'],
    palette='bright',
    alpha=0.8
)
plt.title('accident_risk vs curvature by weather', fontsize=14)
plt.xlabel('curvature', fontsize=12)
plt.ylabel('accident_risk', fontsize=12)
plt.legend(title='weather', title_fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.scatterplot(
    x=df['curvature'],
    y=df['accident_risk'],
    hue=df['num_reported_accidents'],
    palette='bright',
    alpha=0.8
)
plt.title('accident_risk vs curvature by num_reported_accidents', fontsize=14)
plt.xlabel('curvature', fontsize=12)
plt.ylabel('accident_risk', fontsize=12)
plt.legend(title='num_reported_accidents', title_fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt


categorical_columns = [
    'road_type', 'num_lanes', 'speed_limit', 'lighting',
    'weather', 'road_signs_present', 'public_road',
    'time_of_day', 'holiday', 'school_season', 'num_reported_accidents'
]

for col in categorical_columns:
   
    counts = df[col].value_counts()
    labels = counts.index.tolist()
    sizes = counts.values.tolist()

   
    colors = plt.cm.tab20.colors[:len(labels)] 
    explode = [0.05 if size < max(sizes)*0.1 else 0 for size in sizes] 
   
    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        autopct='%1.1f%%',
        startangle=90,
        colors=colors,
        explode=explode,
        pctdistance=0.85,
        textprops={'fontsize': 10},
    )

    ax.axis('equal')  
    plt.title(f'Distribution of {col}', fontsize=14, weight='bold')
    ax.legend(wedges, labels, title=col, loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    plt.tight_layout()
    plt.show()



# List of categorical columns
categorical_columns = [
    'road_type', 'num_lanes', 'speed_limit', 'lighting',
    'weather', 'road_signs_present', 'public_road',
    'time_of_day', 'holiday', 'school_season', 'num_reported_accidents'
]

for col in categorical_columns:
    print(f"Column '{col}' frequency count:")
    print(df[col].value_counts())  # Count occurrences of each unique value
    print('-'*50)



import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.hist(df['curvature'], bins=30, color='#4C72B0', edgecolor='black', alpha=0.7)
plt.title('Distribution of Curvature', fontsize=16, weight='bold')
plt.xlabel('Curvature Value', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()



print(df.columns)



import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

train =df

# Split data into train and validation
X = train.drop('accident_risk', axis=1)
y = train['accident_risk']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# Encode categorical columns
categorical_columns = X_train.select_dtypes(include=['object']).columns
for col in categorical_columns:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col])
    X_val[col] = le.transform(X_val[col])



# Train XGBoost model on GPU
xgb_model_gpu = XGBRegressor(
    objective='reg:squarederror',
    eval_metric='rmse',
    learning_rate=0.1,
    max_depth=6,
    n_estimators=500,
    subsample=0.8,
    colsample_bytree=0.8,
    seed=42,
    n_jobs=-1,
    tree_method='hist',
    device='cuda'
)


xgb_model_gpu.fit(X_train, y_train)


# Evaluate on validation set
y_val_pred = xgb_model_gpu.predict(X_val)
rmse_val = mean_squared_error(y_val, y_val_pred, squared=False)
print(f'RMSE on validation set: {rmse_val}')



import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

train = df
test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')

# Save test IDs for submission
test_ids = test['id']


X_train = train.drop('accident_risk', axis=1)
y_train = train['accident_risk']

# Remove ID column from test data (not a feature)
X_test = test.drop('id', axis=1)



categorical_columns = X_train.select_dtypes(include=['object']).columns

for col in categorical_columns:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col])
    X_test[col] = le.transform(X_test[col])


xgb_model = XGBRegressor(
    objective='reg:squarederror',
    eval_metric='rmse',
    learning_rate=0.1,
    max_depth=6,
    n_estimators=500,
    subsample=0.8,
    colsample_bytree=0.8,
    n_jobs=-1,
    tree_method='hist',   
    device='cuda',        
    seed=42
)


xgb_model.fit(X_train, y_train)



#Predict on test data
y_pred_test = xgb_model.predict(X_test)


submission = pd.DataFrame({
    'id': test_ids,
    'accident_risk': y_pred_test
})

#  Create submission file
submission.to_csv('submission.csv', index=False)
print("✅ Training complete — submission.csv saved!")


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.scatterplot(
    x=df['curvature'],
    y=df['accident_risk'],
    hue=df['public_road'],
    palette='bright',
    alpha=0.8
)
plt.title('accident_risk vs curvature by public_road', fontsize=14)
plt.xlabel('curvature', fontsize=12)
plt.ylabel('accident_risk', fontsize=12)
plt.legend(title='public_road', title_fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.scatterplot(
    x=df['curvature'],
    y=df['accident_risk'],
    hue=df['road_signs_present'],
    palette='bright',
    alpha=0.8
)
plt.title('accident_risk vs curvature by road_signs_present', fontsize=14)
plt.xlabel('curvature', fontsize=12)
plt.ylabel('accident_risk', fontsize=12)
plt.legend(title='road_signs_present', title_fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.scatterplot(
    x=df['curvature'],
    y=df['accident_risk'],
    hue=df['time_of_day'],
    palette='bright',
    alpha=0.8
)
plt.title('accident_risk vs curvature by time_of_day', fontsize=14)
plt.xlabel('curvature', fontsize=12)
plt.ylabel('accident_risk', fontsize=12)
plt.legend(title='time_of_day', title_fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.scatterplot(
    x=df['curvature'],
    y=df['accident_risk'],
    hue=df['holiday'],
    palette='bright',
    alpha=0.8
)
plt.title('accident_risk vs curvature by holiday', fontsize=14)
plt.xlabel('curvature', fontsize=12)
plt.ylabel('accident_risk', fontsize=12)
plt.legend(title='holiday', title_fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.scatterplot(
    x=df['curvature'],
    y=df['accident_risk'],
    hue=df['school_season'],
    palette='bright',
    alpha=0.8
)
plt.title('accident_risk vs curvature by school_season', fontsize=14)
plt.xlabel('curvature', fontsize=12)
plt.ylabel('accident_risk', fontsize=12)
plt.legend(title='school_season', title_fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.scatterplot(
    x=df['curvature'],
    y=df['accident_risk'],
    hue=df['num_lanes'],
    palette='bright',
    alpha=0.8
)
plt.title('accident_risk vs curvature by num_lanes ', fontsize=14)
plt.xlabel('curvature', fontsize=12)
plt.ylabel('accident_risk', fontsize=12)
plt.legend(title='num_lanes', title_fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()



import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.scatterplot(
    x=df['curvature'],
    y=df['accident_risk'],
    hue=df['road_type'],
    palette='bright',
    alpha=0.8
)
plt.title('accident_risk vs curvature by road_type ', fontsize=14)
plt.xlabel('curvature', fontsize=12)
plt.ylabel('accident_risk', fontsize=12)
plt.legend(title='road_type', title_fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()

