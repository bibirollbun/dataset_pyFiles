import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")



train=pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test=pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')
train_org=pd.read_csv('/kaggle/input/rainfall-prediction-using-machine-learning/Rainfall.csv')


train_org.columns = train_org.columns.str.strip()
train_org['rainfall'] = train_org['rainfall'].str.lower().map({'yes': 1, 'no': 0})
train = train.drop(columns=['id'])
test = test.drop(columns=['id'])


column_order = ['day', 'pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint',
                'humidity', 'cloud', 'sunshine', 'winddirection', 'windspeed', 'rainfall']

train = train[column_order]  
train_org = train_org[column_order]


# day column = [1,2,3,----,364,365,366]
for i in range(366):
    train_org.loc[i, 'day'] = i + 1  



train = pd.concat([train, train_org], ignore_index=True)


from sklearn.impute import KNNImputer
knn=KNNImputer()
train=pd.DataFrame(knn.fit_transform(train),columns=train.columns)
test=pd.DataFrame(knn.fit_transform(test),columns=test.columns)


from sklearn.preprocessing import PolynomialFeatures

def advanced_feature_engineering(df):
    
    # ------------------- Day & Seasonal Features -------------------
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 365)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 365)

    
    # ------------------- Pressure Features -------------------
    df['pressure_rolling_mean'] = df['pressure'].rolling(window=7, min_periods=1).mean()
    df['pressure_rolling_std'] = df['pressure'].rolling(window=7, min_periods=1).std()
    df['pressure_diff'] = df['pressure'].diff()

    # ------------------- Temperature Features -------------------
    df['temp_range'] = df['maxtemp'] - df['mintemp']
    df['temp_ewm'] = df['temparature'].ewm(span=10, adjust=False).mean()
    df['temp_change'] = df['temparature'].diff()

    # Heat Index Approximation (better than temp * humidity)
    df['temp_humidity_interaction'] = df['temparature'] + (0.2 * df['humidity'])

    # ------------------- Dewpoint & Humidity Features -------------------
    df['dewpoint_depression'] = df['temparature'] - df['dewpoint']
    df['rh_approx'] = 100 - (5 * df['dewpoint_depression'])

    # Saturation Vapor Pressure (SVP) - Tetens' Equation
    df['svp'] = 6.1078 * np.exp((17.27 * df['temparature']) / (df['temparature'] + 237.3))

    # Absolute Humidity (AH) in g/mÂ³
    df['abs_humidity'] = (6.112 * np.exp((17.67 * df['temparature']) / (df['temparature'] + 243.5)) * df['humidity'] * 2.1674) / (273.15 + df['temparature'])

    # ------------------- Cloud & Sunshine Features -------------------
    df['cloud_category'] = pd.cut(df['cloud'], bins=[0, 20, 50, 80, 100], labels=[0, 1, 2, 3])
    df['cloud_category'] = df['cloud_category'].astype(float)

    df['sky_opacity'] = df['cloud'] / 100

    # Sunshine fraction (actual sunshine hours divided by daylight hours)
    df['sunshine_pct'] = df['sunshine'] / 24  # Kept as per your original logic
    df['cloud_sun_ratio'] = df['cloud'] / (df['sunshine'] + 1e-6)  # Avoid division by zero
    df['interaction'] = df['cloud'] + df['sunshine'] + df['humidity']

    # ------------------- Wind Features -------------------
    df['winddir_sin'] = np.sin(np.radians(df['winddirection']))
    df['winddir_cos'] = np.cos(np.radians(df['winddirection']))

    # ------------------- Polynomial Features -------------------
    # during EDA i noticed polynomial nature of individual features wrt rain
    poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
    
    poly_features = ['temparature', 'humidity', 'pressure', 'windspeed', 'cloud']
    df_poly = pd.DataFrame(poly.fit_transform(df[poly_features]), columns=poly.get_feature_names_out(poly_features))
    df_poly = df_poly.add_prefix("poly_")
    df = df.join(df_poly)

    return df



train=advanced_feature_engineering(train)
test=advanced_feature_engineering(test)


from sklearn.impute import KNNImputer
knn=KNNImputer()
train=pd.DataFrame(knn.fit_transform(train),columns=train.columns)
test=pd.DataFrame(knn.fit_transform(test),columns=test.columns)


X=train.drop(columns='rainfall')
y=train['rainfall']


from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


Params = {
    'n_estimators': 50,
    'max_depth': 24,
    'learning_rate': 0.2424488764,
    'subsample': 0.616071878022,
    'colsample_bytree': 0.74521067149,
    'gamma': 3.04302648592,
    'reg_alpha': 8.928697492786,
    'reg_lambda': 2.299889992905
}


# Define base models
clf1 = LogisticRegression(C=0.1,solver='liblinear',penalty='l1',max_iter=1000)
clf2 = LogisticRegression(solver='newton-cg',penalty='l2',max_iter=1000,C=1)  
clf3 = LogisticRegression(solver='sag',C=0.01,penalty='l2')
clf4=xgb.XGBClassifier(**Params)
clf5 = KNeighborsClassifier(n_neighbors=10,weights='distance')
clf6 = LogisticRegression(solver='lbfgs',C=0.01,penalty=None)
clf7 = LogisticRegression(solver='saga',C=0.01,penalty='l2',max_iter=1000)


FOLDS = 10
skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)

oof_preds = np.zeros(len(X))
test_preds = np.zeros((len(test), FOLDS))

fold = 1
for train_idx, val_idx in skf.split(X, y):
    print(f"Training fold {fold} ...")
    
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    model = VotingClassifier(estimators=[
    ('lr1', clf1), ('lr2', clf2), ('lr3', clf3),('lr4',clf6),('lr5',clf7)
    ], voting='soft',weights=[5,5,3,1,1])
    model.fit(X_train, y_train)
    
    oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
    fold_auc = roc_auc_score(y_val, oof_preds[val_idx])
    print(f"Fold {fold} AUC: {fold_auc:.4f}")
    
    test_preds[:, fold - 1] = model.predict_proba(test)[:, 1]
    
    fold += 1

overall_auc = roc_auc_score(y, oof_preds)
print(f"\nOverall OOF AUC: {overall_auc:.4f}")

test_pred = test_preds.mean(axis=1)


submission=pd.read_csv('/kaggle/input/playground-series-s5e3/sample_submission.csv')
submission.rainfall=test_pred
submission.to_csv("submission.csv", index=False)




