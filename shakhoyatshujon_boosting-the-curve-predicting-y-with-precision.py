import numpy as np
import pandas as pd
import plotly.io as pio
pio.renderers.default = 'iframe'
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectKBest, mutual_info_regression

from sklearn.metrics import r2_score

import os

# Check if the directory exists
input_dir = "/kaggle/input"
if os.path.exists(input_dir):
    for dirname, _, filenames in os.walk(input_dir):
        for filename in filenames:
            print(os.path.join(dirname, filename))
else:
    print(f"Directory {input_dir} not found or inaccessible.")



train=pd.read_csv('/kaggle/input/mercedes-benz-greener-manufacturing/train.csv.zip')
test=pd.read_csv('/kaggle/input/mercedes-benz-greener-manufacturing/test.csv.zip')


train.head()


train.info()


y=train['y']
X=train.drop(['y','ID'],axis=1)
X_test=test.drop(['ID'],axis=1)


X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42)


X_train.head()


numerical_features=X.select_dtypes(include='number').columns.values
numerical_features


categorical_features=X.select_dtypes(exclude='number').columns.values
categorical_features


enc = OrdinalEncoder(handle_unknown="use_encoded_value",unknown_value=np.nan)
enc.fit(X_train[categorical_features])



#transform Train
X_train[categorical_features]=enc.transform(X_train[categorical_features])

#transform Val
X_val[categorical_features]=enc.transform(X_val[categorical_features])

#transform Test
X_test[categorical_features]=enc.transform(X_test[categorical_features])


X_train[categorical_features]


impute =SimpleImputer(strategy='median')
impute.fit(X_train)



#transform Train
X_train=impute.transform(X_train)

#transform Val
X_val=impute.transform(X_val)

#transform Test
X_test=impute.transform(X_test)



scaler =StandardScaler()
scaler.fit(X_train)


sel=SelectKBest(mutual_info_regression,k=30)
sel.fit(X_train,y_train)


#transform Train
X_train=sel.transform(X_train)

#transform Val
X_val=sel.transform(X_val)

#transform Test
X_test=sel.transform(X_test)


X_train


import plotly.express as px

fig = px.histogram(train, x='y', title='Distribution of Target Variable "y"')
fig.update_layout(xaxis_title='y', yaxis_title='Frequency')
fig.show()


for i, feature in enumerate(categorical_features):
    fig = px.bar(x=X_train[:, i], title=f"Distribution of {feature}")
    fig.update_layout(xaxis_title=feature, yaxis_title="Count")
    fig.show()


import plotly.express as px

for i, feature in enumerate(categorical_features):
    fig = px.box(x=X_train[:, i], y=y_train, title=f"Relationship between {feature} and y")
    fig.update_layout(xaxis_title=feature, yaxis_title="y")
    fig.show()


selected_numerical_features = [f for f in numerical_features if f in sel.get_feature_names_out()]

for feature in selected_numerical_features:
    original_index = np.where(X.columns == feature)[0][0]
    selected_index = np.where(sel.get_feature_names_out() == feature)[0][0]
    fig = px.scatter(x=X_train[:, selected_index], y=y_train,
                     title=f"Relationship between {feature} and y")
    fig.update_layout(xaxis_title=feature, yaxis_title="y")
    fig.show()


impute = SimpleImputer(strategy='median')
X_train = impute.fit_transform(X_train)
X_val = impute.transform(X_val)
X_test = impute.transform(X_test)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)


sel = SelectKBest(mutual_info_regression, k=30)
sel.fit(X_train, y_train)
X_train = sel.transform(X_train)
X_val = sel.transform(X_val)
X_test = sel.transform(X_test)


from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


models = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(),
    "Lasso": Lasso(),
    "Random Forest Regressor": RandomForestRegressor(),
    "Gradient Boosting Regressor": GradientBoostingRegressor()
}


for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    print(f"{name} training complete.")


for name, model in models.items():
    y_val_pred = model.predict(X_val)
    r2 = r2_score(y_val, y_val_pred)
    print(f"{name} R-squared on validation set: {r2}")


best_model_name = ""
best_r2_score = -np.inf

for name, model in models.items():
    y_val_pred = model.predict(X_val)
    r2 = r2_score(y_val, y_val_pred)
    if r2 > best_r2_score:
        best_r2_score = r2
        best_model_name = name

print(f"The best performing model on the validation set is: {best_model_name} with an R-squared score of {best_r2_score:.3f}")


best_model = models[best_model_name]
y_test_pred = best_model.predict(X_test)

submission = pd.DataFrame({
    'ID': test['ID'],
    'y': y_test_pred
})

display(submission.head())


submission.to_csv('Submission_LR_Final.csv',index=False)

