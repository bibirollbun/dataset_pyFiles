import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, matthews_corrcoef, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectFromModel, RFE
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')


train=pd.read_csv(r'/kaggle/input/playground-series-s5e3/train.csv')
test=pd.read_csv(r'/kaggle/input/playground-series-s5e3/test.csv')
print(train.shape,test.shape)


train.drop_duplicates(inplace=True)
train.isnull().sum()


train.fillna(train.median(), inplace=True)
test.fillna(test.mean(), inplace=True)


numerical_variables = ['pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint', 'humidity', 'cloud', 'sunshine', 'windspeed']
target_variable = 'rainfall' 
categorical_variables = ['winddirection']


custom_palette = ['#3498db', '#e74c3c', '#2ecc71']

train['Source'] = 'Train'
test['Source'] = 'Test'

def generate_feature_visualizations(feature_name):
    sns.set(style='whitegrid')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    plt.subplot(1, 2, 1)
    sns.boxplot(data=pd.concat([train, test]),
                x=feature_name, y="Source", palette=custom_palette)
    plt.xlabel(feature_name)
    plt.title(f"Box Plot for {feature_name} Across Datasets")

    plt.subplot(1, 2, 2)
    sns.histplot(data=train, x=feature_name, color=custom_palette[0], kde=True, bins=30, label="Train", alpha=0.6)
    sns.histplot(data=test, x=feature_name, color=custom_palette[1], kde=True, bins=30, label="Test", alpha=0.6)
    plt.xlabel(feature_name)
    plt.ylabel("Frequency")
    plt.title(f"Histogram for {feature_name} (Train, Test & Original)")
    plt.legend(title="Dataset")

    plt.tight_layout()

    plt.show()

for feature in numerical_variables:
    generate_feature_visualizations(feature)

train.drop('Source', axis=1, inplace=True)
test.drop('Source', axis=1, inplace=True)




from sklearn.feature_selection import mutual_info_regression

X = train.drop(columns=[ 'rainfall'])
y = train['rainfall']
mi=mutual_info_regression(X,y)
mi_df=pd.DataFrame({"Cols":X.columns,'MI':mi})
mi_df.sort_values(ascending=False,inplace=True,by='MI')

plt.figure(figsize=(20,8))
sns.barplot(data=mi_df,x='MI',y='Cols')
plt.show()



def feature_engineering(df):
    df = df.copy()
    df['hci'] = df['humidity'] * df['cloud']
    df['hsi'] = df['humidity'] * df['sunshine']
    df['csr'] = df['cloud'] / (df['sunshine'] + 1e-5)
    df['rd'] = 100 - df['humidity']
    df['sp'] = df['sunshine'] / (df['sunshine'] + df['cloud'] + 1e-5)
    df['wi'] = (0.4 * df['humidity']) + (0.3 * df['cloud']) - (0.3 * df['sunshine'])
    return df

train_comb = feature_engineering(train)
test = feature_engineering(test)


if test.isnull().sum().sum() > 0:
    print("\nHandling missing values in test data...")

    for col in test.columns:
        if test[col].isnull().sum() > 0:
            test[col] = test[col].fillna(train[col].median())


train_comb


xgb_params = {
    'n_estimators': 2407,
    'eta': 0.009462133032592785,
    'gamma': 0.2865859948765318,
    'max_depth': 31,
    'min_child_weight': 47,
    'subsample': 0.6956431754146083,
    'colsample_bytree': 0.3670732604094118,
    'grow_policy': 'lossguide',
    'max_leaves': 73,
    'enable_categorical': True,
    'n_jobs': -1,
    'device': 'cuda',
    'tree_method': 'hist'
}

lgbm_params = {
    'n_estimators': 2500,
    'random_state': 42,
    'max_bin': 1024,
    'colsample_bytree': 0.6,
    'reg_lambda': 80,
    'verbosity': -1
}



X = train_comb.drop(['id', 'rainfall'], axis=1)
y = train_comb['rainfall']
test=test.drop(['id'], axis=1)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)


xgb_model = XGBClassifier(**xgb_params)
lgbm_model = LGBMClassifier(**lgbm_params)

def model_trainer(model, X, y, n_splits=5, random_state=42):

    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(y, pd.Series):
        y = y.values
    
    skfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    oof_probs, oof_mccs, oof_accuracies = [], [], []
    print("="*80)
    print(f"Training {model.__class__.__name__}")
    print("="*80, end="\n")
    
    for fold, (train_idx, test_idx) in enumerate(skfold.split(X, y)):
        X_train_fold, y_train_fold = X[train_idx], y[train_idx]
        X_test_fold, y_test_fold = X[test_idx], y[test_idx]
        
        model.fit(X_train_fold, y_train_fold)
        y_pred = model.predict(X_test_fold)
        
        accuracy = accuracy_score(y_test_fold, y_pred)
        mcc = matthews_corrcoef(y_test_fold, y_pred)
        oof_probs.append(model.predict_proba(X_test_fold))
        oof_mccs.append(mcc)
        oof_accuracies.append(accuracy)
        
        print(f"--- Fold {fold+1} MCC: {mcc:.6f}, Accuracy: {accuracy:.6f}")
        
    print(f"\n---> Mean MCC: {np.mean(oof_mccs):.6f} ± {np.std(oof_mccs):.6f}")
    print(f"---> Mean Accuracy: {np.mean(oof_accuracies):.6f} ± {np.std(oof_accuracies):.6f}")
    return oof_probs, oof_mccs, oof_accuracies

oof_probs_xgb, oof_mccs_xgb, oof_accuracies_xgb = model_trainer(xgb_model, X_train_scaled, y_train, random_state=42)
oof_probs_lgbm, oof_mccs_lgbm, oof_accuracies_lgbm = model_trainer(lgbm_model, X_train_scaled, y_train, random_state=42)

y_val_pred_xgb = xgb_model.predict(X_val_scaled)
y_val_pred_lgbm = lgbm_model.predict(X_val_scaled)

y_val_prob_xgb = xgb_model.predict_proba(X_val_scaled)[:, 1]
y_val_prob_lgbm = lgbm_model.predict_proba(X_val_scaled)[:, 1]




import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

fpr_xgb, tpr_xgb, thresholds_xgb = roc_curve(y_val, y_val_prob_xgb)
roc_auc_xgb = auc(fpr_xgb, tpr_xgb)

fpr_lgbm, tpr_lgbm, thresholds_lgbm = roc_curve(y_val, y_val_prob_lgbm)
roc_auc_lgbm = auc(fpr_lgbm, tpr_lgbm)

plt.figure(figsize=(10, 6))

plt.plot(fpr_xgb, tpr_xgb, color='b', lw=2, label=f'XGBoost (AUC = {roc_auc_xgb:.2f})')
plt.plot(fpr_lgbm, tpr_lgbm, color='r', lw=2, label=f'LightGBM (AUC = {roc_auc_lgbm:.2f})')

plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2)

plt.title('ROC Curve Comparison (XGBoost vs LightGBM)', fontsize=14)
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate (Recall)', fontsize=12)
plt.legend(loc='lower right')
plt.grid(True)

plt.tight_layout()
plt.show()



from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense, Dropout, MaxPooling1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from tensorflow.keras.metrics import AUC


scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(test.drop([], axis=1))

X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], 1))
X_test_scaled = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))



model = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
    MaxPooling1D(pool_size=2),
    Conv1D(filters=32, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid') 
])

from tensorflow.keras.optimizers import SGD

optimizer = SGD(learning_rate=0.001, momentum=0.9, decay=1e-6)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=[AUC(name='auc')])

early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=1e-5, verbose=1)

history = model.fit(
    X_train, y_train,
    epochs=200,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)


train_auc = history.history['auc']
val_auc = history.history['val_auc']

plt.figure(figsize=(10, 6))
plt.plot(train_auc, label='Training AUC', color='b', lw=2)
plt.plot(val_auc, label='Validation AUC', color='r', lw=2)

plt.title('Training and Validation AUC vs Epochs', fontsize=14)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('AUC', fontsize=12)
plt.legend(loc='lower right')
plt.grid(True)

plt.tight_layout()
plt.show()



test_preds = model.predict(X_test_scaled).flatten()

if np.isnan(test_preds).sum() > 0:
    print(f"Found {np.isnan(test_preds).sum()} NaN values in predictions. Fixing them...")
    test_preds = np.nan_to_num(test_preds)  


test=pd.read_csv(r'/kaggle/input/playground-series-s5e3/test.csv')


submission = pd.DataFrame({"id": test['id'], "rainfall": test_preds})
submission.to_csv("submission.csv", index=False)


submission.head()





RMV = ['rainfall','id']
FEATURES = [c for c in train.columns if not c in RMV]
print("Our features are:")
print( FEATURES )


from sklearn.model_selection import KFold
from xgboost import XGBRegressor, XGBClassifier
import xgboost
print("Using XGBoost version",xgboost.__version__)


%%time
FOLDS = 5
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)
    
oof_xgb = np.zeros(len(train))
pred_xgb = np.zeros(len(test))

for i, (train_index, test_index) in enumerate(kf.split(train)):

    print("#"*25)
    print(f"### Fold {i+1}")
    print("#"*25)
    
    x_train = train.loc[train_index,FEATURES].copy()
    y_train = train.loc[train_index,"rainfall"]    
    x_valid = train.loc[test_index,FEATURES].copy()
    y_valid = train.loc[test_index,"rainfall"]
    x_test = test[FEATURES].copy()

    model = XGBClassifier(
        device="cuda",
        max_depth=6,  
        colsample_bytree=0.9, 
        subsample=0.9, 
        n_estimators=10_000,  
        learning_rate=0.1, 
        eval_metric="auc",
        early_stopping_rounds=100,
        alpha=1,
    )
    model.fit(
        x_train, y_train,
        eval_set=[(x_valid, y_valid)],  
        verbose=100 
    )

    # INFER OOF
    oof_xgb[test_index] = model.predict_proba(x_valid)[:,1]
    # INFER TEST
    pred_xgb += model.predict_proba(x_test)[:,1]

# COMPUTE AVERAGE TEST PREDS
pred_xgb /= FOLDS


from sklearn.metrics import roc_auc_score
true = train.rainfall.values
m = roc_auc_score(true, oof_xgb)
print(f"XGBoost CV Score AUC = {m:.3f}")


feature_importance = model.feature_importances_
importance_df = pd.DataFrame({
    "Feature": FEATURES,  
    "Importance": feature_importance
}).sort_values(by="Importance", ascending=False)
plt.figure(figsize=(10, 5))
plt.barh(importance_df["Feature"], importance_df["Importance"])
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("XGBoost Feature Importance")
plt.gca().invert_yaxis()  
plt.show()










