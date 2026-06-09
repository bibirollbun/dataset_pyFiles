# install the latest version of pycaret
!pip install pycaret


import numpy as np
import pandas as pd


train = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')
sub = pd.read_csv('/kaggle/input/playground-series-s5e3/sample_submission.csv')


train.head()


train.info()


from imblearn.combine import SMOTETomek
SEED = 42
X_train = train.drop(columns=['id', 'rainfall'])
y_train = train['id']
smote = SMOTETomek(random_state=SEED)
X_train_over,y_train_over = smote.fit_resample(X_train,y_train)


from pycaret.classification import setup, compare_models, create_model, tune_model, plot_model, evaluate_model
clf = setup(
    data=train,
    target="rainfall",
    numeric_imputation = 'mean',
    ignore_features=['id'],
    # use_gpu=True,
    session_id = SEED,
    verbose=True,
    fix_imbalance = True,
    fix_imbalance_method=smote
)


test = test.drop(columns=['id'])
test.info()


from sklearn.impute import SimpleImputer
numeric_columns = test.select_dtypes(include=[np.number]).columns
imputer = SimpleImputer(strategy='mean') 
test[numeric_columns] = imputer.fit_transform(test[numeric_columns])
test.info()


model = compare_models()


# from pycaret.classification import pull
# metrics = pull()
# metrics
tuned_model = tune_model(model, optimize="auc")


plot_model(estimator = tuned_model, plot ='learning')


plot_model(estimator=tuned_model, plot='feature')


evaluate_model(tuned_model)


from pycaret.classification import interpret_model
interpret_model(tuned_model)


predictions = tuned_model.predict_proba(test)
sub['rainfall'] = predictions[:, 1]
sub.to_csv('submission.csv',index=False)
sub.head()

