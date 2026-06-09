pip install /kaggle/input/cibmtr-whl-files-for-installation/scikit_survival-0.20.0-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl



!pip install /kaggle/input/cibmtr-whl-files-for-installation/autograd-1.7.0-py3-none-any.whl
!pip install /kaggle/input/cibmtr-whl-files-for-installation/autograd-gamma-0.5.0.tar.gz
!pip install /kaggle/input/cibmtr-whl-files-for-installation/interface_meta-1.3.0-py3-none-any.whl
!pip install /kaggle/input/cibmtr-whl-files-for-installation/formulaic-1.1.1-py3-none-any.whl
!pip install /kaggle/input/cibmtr-whl-files-for-installation/lifelines-0.30.0-py3-none-any.whl


import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
from sklearn.model_selection import train_test_split



test=pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/test.csv")



data=pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/train.csv')
data.head()


data['age_comorbidity_interaction'] = data['age_at_hct'] * data['comorbidity_score']
test['age_comorbidity_interaction'] = test['age_at_hct'] * test['comorbidity_score']
baseline=2008
data["years_since_baseline"]=data["year_hct"]-baseline
data["years_since_baseline"]
test["years_since_baseline"]=test["year_hct"]-baseline
test["years_since_baseline"]



cat_data=data.select_dtypes(["object"])
number_data=data.select_dtypes(["int64","float64"])
cat_test=test.select_dtypes(["object"])
number_test=test.select_dtypes(["int64","float64"])


cat_imputer=SimpleImputer(strategy='most_frequent')
number_impute=SimpleImputer(strategy='mean')
number_impute_test=SimpleImputer(strategy='mean')
cat_data = pd.DataFrame(cat_imputer.fit_transform(cat_data), columns=cat_data.columns)
cat_test = pd.DataFrame(cat_imputer.transform(cat_test), columns=cat_test.columns)
number_data = pd.DataFrame(number_impute.fit_transform(number_data), columns=number_data.columns)
number_test = pd.DataFrame(number_impute_test.fit_transform(number_test), columns=number_test.columns)


scaler = StandardScaler()
number_data=number_data.drop("ID",axis=1)
number_test=number_test.drop("ID",axis=1)

number_data_scaled=scaler.fit_transform(number_data)
number_test_scaled=scaler.fit_transform(number_test)


number_data_scaled = pd.DataFrame(number_data_scaled, columns=number_data.columns)
number_test_scaled = pd.DataFrame(number_test_scaled, columns=number_test.columns)



id_data=data["ID"]
id_test=test["ID"]


data=pd.concat([id_data,number_data_scaled,cat_data],axis=1)
data.head()


test=pd.concat([id_test,number_test_scaled,cat_test],axis=1)
test.head()



def label_encode_datasets(train_df, test_df, categ_fields):
    train_encoded = data.copy()
    test_encoded = test.copy()
    le = LabelEncoder()
    
    for column in cat_data:
        print(f'Encoding: {column} ...')
        le.fit(train_encoded[column])
        
        train_encoded[column] = le.transform(train_encoded[column])
        if column in test_encoded.columns:
            test_encoded[column] = test_encoded[column].map(lambda s: le.transform([s])[0] if s in le.classes_ else None)
            test_encoded[column].fillna(-1, inplace=True)
            test_encoded[column] = test_encoded[column].astype(int)

    return train_encoded, test_encoded





remove_variables = data[['ID', 'efs', 'efs_time']]
features = [feat for feat in data if feat not in remove_variables]
categorical_features = [feat for feat in data[features] if data[feat].dtype == 'object']
numerical_features = [feat for feat in data[features] if feat not in categorical_features]



trn_encoded, tst_encoded = label_encode_datasets(data, test, categorical_features)


trn_encoded["efs"]=trn_encoded["efs"].abs()


trn_encoded['efs'] = (trn_encoded['efs']).astype(int)

print(trn_encoded['efs'].unique()) 



def train_random_survival_forest(df, time_col, event_col, feature_cols, n_estimators=100, random_state=42):

    survival_data = Surv.from_dataframe(event=event_col, time=time_col, data=df)
    X = df[feature_cols]

    X_train, X_test, y_train, y_test = train_test_split(X, survival_data, test_size=0.3, random_state=random_state)

    model = RandomSurvivalForest(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
        max_features='sqrt',
        min_samples_leaf=3
    )
    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)
    print('Score: ', score)

    return model, X_train, X_test, y_train, y_test


time_col = "efs_time"
event_col = "efs"
feature_cols = categorical_features + numerical_features

model, X_train, X_test, y_train, y_test = train_random_survival_forest(trn_encoded, 
                                                                       time_col, 
                                                                       event_col, 
                                                                       feature_cols, 
                                                                       n_estimators=12)



predictions_train = model.predict(trn_encoded[feature_cols])

predictions_train





predictions_test = model.predict(tst_encoded[feature_cols])

predictions_test




from sksurv.metrics import concordance_index_censored
predictions = model.predict(X_test)
c_index = concordance_index_censored(
        y_test["efs"], y_test["efs_time"], predictions
    )[0]
print(f"Model Score (Internal): {model.score(X_test, y_test)}")
print(f"C-index (Test Data): {c_index}")



predictions_test = predictions_test[:len(test['ID'])]



# Save Test Predictions
test_results = pd.DataFrame({
    'ID': test['ID'],  
    'prediction': predictions_test
})

test_results.to_csv('submission.csv', index=False)


test_results

