import pandas as pd
import numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')


df.head().T


df.columns = df.columns.str.lower()
num = [x for x in df.columns if df.dtypes[x] in ('int32', 'int64', 'float32', 'float64')]
cat = [x for x in df.columns if df.dtypes[x] == 'object']
target = 'listening_time_minutes'
num.remove(target)


from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size=.2)


df[num].corrwith(df[target]).sort_values(ascending=False)


from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


cat


num_pip = Pipeline([
    ('impute', SimpleImputer())
    , ('scaler', StandardScaler())
])

cat_pip = Pipeline([
    ('encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
])

pipe = ColumnTransformer([
    ('num', num_pip, ['episode_length_minutes', 'number_of_ads'])
    , ('cat', cat_pip, ['podcast_name', 'genre', 'publication_day', 'publication_time', 'episode_sentiment'])
])

pipe.fit(train)
train_pre = pipe.transform(train)
test_pre = pipe.transform(test)


from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score


from sklearn.linear_model import LinearRegression

lin = LinearRegression()
lin.fit(train_pre, train[target])
preds = lin.predict(train_pre)
print(cross_val_score(lin, train_pre, train[target], scoring='neg_mean_squared_error').mean())



sns.relplot(data=train, x=target, y=preds, hue='genre');


samp = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')
samp.columns = samp.columns.str.lower()

submission = pd.DataFrame({
    'id': samp['id']
    , 'Listening_Time_minutes': lin.predict(pipe.transform(samp))
})

submission.to_csv('submission.csv', index=False)




