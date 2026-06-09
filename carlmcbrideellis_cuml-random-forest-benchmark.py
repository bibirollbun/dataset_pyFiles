import pandas as pd

X = pd.read_csv('../input/santander-value-prediction-challenge/train.csv', index_col="ID")
y = X.pop("target")


from sklearn.ensemble import RandomForestRegressor


%%time

RandomForestRegressor(n_estimators=200, max_depth=10, random_state=0).fit(X,y)


from cuml.ensemble import RandomForestRegressor as cuRFR


%%time

cuRFR(n_estimators=200, max_depth=10, random_state=0).fit(X,y)

