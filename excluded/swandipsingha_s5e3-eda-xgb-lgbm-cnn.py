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





custom_palette = ['#1f77b4', '#ff7f0e']  

def create_grouped_countplot(variable):
    sns.set_style('whitegrid')


    train_data_copy = train.copy()
    test_data_copy = test.copy()

    train_data_copy['Dataset'] = 'Train'
    test_data_copy['Dataset'] = 'Test'

    combined_data = pd.concat([train_data_copy, test_data_copy])

    train_counts = train[variable].value_counts().sort_values(ascending=True).index.tolist()

    plt.figure(figsize=(14, 7))
    sns.countplot(
        data=combined_data, 
        x=variable,  
        hue="Dataset", 
        palette=custom_palette,  
        dodge=True,  
        width=0.85, 
        order=train_counts  
    )

    plt.ylabel("Count")
    plt.xlabel(variable)
    plt.title(f"Grouped Count Plot for {variable} (Train vs Test)")

    plt.xticks(rotation=45, ha="right")

    plt.show()

for variable in categorical_variables:
    create_grouped_countplot(variable)



unique_palette = ['#9b59b6', '#f39c12']

def generate_wind_rose_plot(ax, dataset, name, color):
    wind_direction_radians = np.radians(dataset['winddirection'].dropna())

    bins = np.linspace(0, 2 * np.pi, 37)
    counts, bin_edges = np.histogram(wind_direction_radians, bins=bins)

    bars = ax.bar(bin_edges[:-1], counts, width=np.radians(10), color=color, edgecolor='black', alpha=0.75)

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_xticks(np.radians(np.arange(0, 360, 45)))
    ax.set_xticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'], fontsize=12, fontweight='bold')

    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_yticklabels([])
    ax.set_title(f"Wind Direction - {name}", fontsize=14, fontweight='bold', pad=15)

fig, axes = plt.subplots(1, 2, figsize=(14, 6), subplot_kw={'projection': 'polar'})

generate_wind_rose_plot(axes[0], train, "Training Data", unique_palette[0])
generate_wind_rose_plot(axes[1], test, "Test Data", unique_palette[1])

plt.tight_layout()
plt.show()



fig, ax = plt.subplots(3, 4, figsize=(20, 20))
ax = ax.flatten()
i = 0
for col in train.columns:
    if col != 'rainfall':
        sns.kdeplot(data=train, x=col, ax=ax[i], label="Train", fill=True)
        sns.kdeplot(data=test, x=col, ax=ax[i], label="Test", fill=True)
        ax[i].set_title(col)
        ax[i].legend()
        i += 1
plt.tight_layout()

for j in range(i, len(ax)):
    ax[j].axis("off")

plt.show()


import seaborn as sns
import matplotlib.pyplot as plt
import itertools

features = ['day', 'pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint', 
            'humidity', 'cloud', 'sunshine', 'winddirection', 'windspeed']
pairs = list(itertools.combinations(features, 2))  
n_cols = 3
n_rows = -(-len(pairs) // n_cols)  

fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(20, 5 * n_rows))
axes = axes.flatten()
for i, (x, y) in enumerate(pairs):
    sns.scatterplot(data=train, x=x, y=y, hue='rainfall', palette='coolwarm', ax=axes[i])
    axes[i].set_title(f'{x} vs. {y} (Hue: Rainfall)', fontsize=14)
for j in range(i + 1, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.show()


variables = [col for col in train.columns if col in numerical_variables]+['day']

test_variables = variables
train_variables = variables+ ['rainfall']

corr_train = train[train_variables].corr()
corr_test = test[test_variables].corr()

mask_train = np.triu(np.ones_like(corr_train, dtype=bool))
mask_test = np.triu(np.ones_like(corr_test, dtype=bool))

annot_kws = {"size": 8, "rotation": 45}

plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
ax_train = sns.heatmap(corr_train, mask=mask_train, cmap='viridis', annot=True,
                      square=True, linewidths=.5, xticklabels=1, yticklabels=1, annot_kws=annot_kws)
plt.title('Correlation Heatmap - Train Data')

plt.subplot(1, 2, 2)
ax_test = sns.heatmap(corr_test, mask=mask_test, cmap='viridis', annot=True,
                     square=True, linewidths=.5, xticklabels=1, yticklabels=1, annot_kws=annot_kws)
plt.title('Correlation Heatmap - Test Data')

plt.tight_layout()

plt.show()


variables = [col for col in train.columns if col in numerical_variables]
train_variables = variables + ['rainfall']

corr_train = train[train_variables].corr()[['rainfall']].T  

annot_kws = {"size": 10}  

plt.figure(figsize=(10, 2)) 
ax_train = sns.heatmap(corr_train, cmap='viridis', annot=True, 
                      square=False, linewidths=0.5, annot_kws=annot_kws, 
                      cbar=False) 

plt.xticks(rotation=45, ha="right")  
plt.title('Correlation Heatmap - Train Data (ONLY TARGET)')
plt.yticks(rotation=0)  

# Show plot
plt.show()


from matplotlib.lines import Line2D

train_color = '#8e44ad'  
test_color = '#e67e22'   
rainfall_colors = {0: '#f39c12', 1: '#3498db'} 

numerical_columns = test.select_dtypes(include=['int64', 'float64']).columns.tolist()
for col in ['id', 'day', 'rainfall']:
    if col in numerical_columns:
        numerical_columns.remove(col)

for column in numerical_columns:
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])

    ax0 = fig.add_subplot(gs[0, :])
    ax0.plot(train['id'], train[column], linestyle='-', color=train_color, label='Train Data', alpha=0.8)
    ax0.plot(test['id'], test[column], linestyle='-', color=test_color, label='Test Data', alpha=0.8)

    ax0.set_xlabel('ID', fontsize=14)
    ax0.set_ylabel(column, fontsize=14)
    ax0.set_title(f'Trend Plot: {column} vs ID', fontsize=16, fontweight='bold')
    ax0.legend(fontsize=12)
    ax0.grid(True, linestyle='--', alpha=0.5)

    ax1 = fig.add_subplot(gs[1, 0])
    scatter = ax1.scatter(
        train['day'], train[column],
        c=train['rainfall'].map(rainfall_colors), alpha=0.8
    )
    ax1.set_xlabel('Day', fontsize=14)
    ax1.set_ylabel(column, fontsize=14)
    ax1.set_title(f'Scatter Plot: {column} vs Day (by Rainfall)', fontsize=16, fontweight='bold')

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='No Rainfall',
               markersize=10, markerfacecolor=rainfall_colors[0]),
        Line2D([0], [0], marker='o', color='w', label='Rainfall',
               markersize=10, markerfacecolor=rainfall_colors[1])
    ]
    ax1.legend(handles=legend_elements, title="Rainfall", fontsize=12, title_fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.5)

    ax2 = fig.add_subplot(gs[1, 1])
    sns.kdeplot(data=train, x=column, hue='rainfall', palette=rainfall_colors, ax=ax2, fill=True, common_norm=False, alpha=0.6)

    ax2.set_xlabel(column, fontsize=14)
    ax2.set_ylabel('Density', fontsize=14)
    ax2.set_title(f'Distribution (KDE) of {column} by Rainfall', fontsize=16, fontweight='bold')
    ax2.legend(title='Rainfall', fontsize=12, title_fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout(pad=3.0)
    plt.show()

    plt.figure(figsize=(16, 0.3)) 
    plt.axhline(y=0, color='gray', linewidth=5, linestyle='-') 
    plt.axis('off')
    plt.show()



plt.figure(figsize=(10,10))
sns.heatmap(train.corr(),annot=True)
plt.show()


from sklearn.feature_selection import mutual_info_regression

X = train.drop(columns=[ 'rainfall'])
y = train['rainfall']
mi=mutual_info_regression(X,y)
mi_df=pd.DataFrame({"Cols":X.columns,'MI':mi})
mi_df.sort_values(ascending=False,inplace=True,by='MI')

plt.figure(figsize=(20,8))
sns.barplot(data=mi_df,x='MI',y='Cols')
plt.show()



import numpy as np
import pandas as pd

def feature_engineering(df):
    df = df.copy()
    
    df['hci'] = df['humidity'] * df['cloud']
    df['hsi'] = df['humidity'] * df['sunshine']
    df['csr'] = df['cloud'] / (df['sunshine'] + 1e-5)
    df['rd'] = 100 - df['humidity']
    df['sp'] = df['sunshine'] / (df['sunshine'] + df['cloud'] + 1e-5)
    df['wi'] = (0.4 * df['humidity']) + (0.3 * df['cloud']) - (0.3 * df['sunshine'])
    
    df['temp_range'] = df['maxtemp'] - df['mintemp']
    df['temp_dew_diff'] = df['temparature'] - df['dewpoint']
    df['humidity_cloud_ratio'] = df['humidity'] / (df['cloud'] + 1e-3)
    df['sunshine_cloud_ratio'] = df['sunshine'] / (df['cloud'] + 1e-3)
    df['pressure_wind_interaction'] = df['pressure'] * df['winddirection']
    df['temp_pressure_ratio'] = df['temparature'] / (df['pressure'] + 1e-3)
    df['wind_pressure_ratio'] = df['windspeed'] / (df['pressure'] + 1e-3)
    
    return df

train_comb = feature_engineering(train)
test = feature_engineering(test)



if test.isnull().sum().sum() > 0:
    print("\nHandling missing values in test data...")

    for col in test.columns:
        if test[col].isnull().sum() > 0:
            test[col] = test[col].fillna(train[col].median())


train_comb


plt.figure(figsize=(20,8))
sns.heatmap(train_comb.corr(),annot=True)
plt.show()

X = train_comb.drop(columns=['id', 'rainfall'])
y = train_comb['rainfall']
mi=mutual_info_regression(X,y)
mi_df=pd.DataFrame({"Cols":X.columns,'MI':mi})
mi_df.sort_values(ascending=False,inplace=True,by='MI')

plt.figure(figsize=(20,8))
sns.barplot(data=mi_df,x='MI',y='Cols')
plt.show()



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
        
    print(f"\n---> Mean MCC: {np.mean(oof_mccs):.6f} Â± {np.std(oof_mccs):.6f}")
    print(f"---> Mean Accuracy: {np.mean(oof_accuracies):.6f} Â± {np.std(oof_accuracies):.6f}")
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

