import pandas as pd
import numpy as np
import warnings
import seaborn as sns
warnings.filterwarnings('ignore')


df = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
df.columns = df.columns.str.lower()
num = df.select_dtypes(include='number').columns.to_list()
cat = df.select_dtypes(include='object').columns.to_list()
target = 'calories'
num.remove(target)

from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size=.2)


df[num].corrwith(df[target]).sort_values(ascending=False)


df.groupby(['sex'])[target].mean()


sns.pairplot(df)


from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer


num = ['duration', 'heart_rate', 'body_temp', 'age']

processor = make_column_transformer(
    (make_pipeline(SimpleImputer(), StandardScaler(), PolynomialFeatures(2)), num)
    , (OneHotEncoder(sparse_output=False, handle_unknown='ignore'), cat)
)

processor.fit(train)
train_pre = processor.transform(train)


from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.ensemble import RandomForestRegressor


lin = LinearRegression()
lin.fit(train_pre, train[target])
print(-cross_val_score(lin, train_pre, train[target], scoring='neg_root_mean_squared_error').mean())
print(cross_val_score(lin, train_pre, train[target], scoring='r2').mean())


rdg = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0], cv=5)
rdg.fit(train_pre, train[target])
print(-cross_val_score(rdg, train_pre, train[target], scoring='neg_root_mean_squared_error').mean())
print(cross_val_score(rdg, train_pre, train[target], scoring='r2').mean())


rfr = RandomForestRegressor()
rfr.fit(train_pre, train[target])
print(-cross_val_score(rfr, train_pre, train[target], scoring='neg_root_mean_squared_error').mean())
print(cross_val_score(rfr, train_pre, train[target], scoring='r2').mean())


model = rfr
pipe = make_pipeline(processor, model)


preds = pipe.predict(test)
sns.scatterplot(x=test[target], y=preds, hue=test['sex']);


samp = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')


samp.columns = samp.columns.str.lower()


preds = np.array(pipe.predict(samp))
preds[preds < 0] = 0
pd.DataFrame({
    'id': samp['id']
    , 'Calories': preds
}).to_csv('submission.csv', index=False)




