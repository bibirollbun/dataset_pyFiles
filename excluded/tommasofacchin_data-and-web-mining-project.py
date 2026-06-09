import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


#Definitions

#drop_non_numeric(df): Droppa tutte le colonne non numeriche
#get_fitness_index(df): Droppa le colonne categoriche del FitnessGram Child  e crea una nuova feature con la media di queste
#impute_negatives(df): Setta a NaN ogni valore negativo
#fill_BIA(df): Fillo i valori NaN delle colonne BIA con -1 e creo una feature binaria che indica le righe con tutti i valori NaN di queste colonne
#show_nan(df): Mostra le percentuali di valori NaN per ogni colonna
#fill_every_nan(df): Imputa tutti i valori NaN delle feature numeriche
#show_boxplots(df): mostra tutti i boxplots
#show_boxplots_cols(df, cols): mostra i boxplots di un subset di colonne
#remove_outliers_iqr(df, cols): rimuove outliers usando il IQR e sostituendo con la mediana


train = pd.read_csv('/kaggle/input/child-mind-institute-problematic-internet-use/train.csv')
train_tmp = pd.read_csv('/kaggle/input/child-mind-institute-problematic-internet-use/train.csv')
test = pd.read_csv('/kaggle/input/child-mind-institute-problematic-internet-use/test.csv')

print(train.shape)
print(test.shape)


#Droppo subito queste colonne perchè contengono molti valori NaN nel test set, per valutare il modello su kaggle non ha senso processarle
#rimuovo anche l'id dal train
cols = ['id', 'PAQ_A-PAQ_A_Total', 'Fitness_Endurance-Max_Stage', 'Fitness_Endurance-Time_Sec', 
        'Fitness_Endurance-Time_Mins', 'FGC-FGC_GSD', 'FGC-FGC_GSND', 'Physical-Waist_Circumference']
train = train.drop(columns=cols)


#droppo le righe con sii NaN
print(train['sii'].isnull().sum())
train = train.dropna(subset=['sii'])
print(train.shape)


#droppo le colonne in train ma non in test
cols_to_drop = [col for col in train.columns if col not in test.columns and col != 'sii']
train = train.drop(columns=cols_to_drop)

print(cols_to_drop)
print(train.shape)


train.select_dtypes(exclude=['number'])


#droppo le stagioni e l'id
def drop_non_numeric(df):
    cols = df.select_dtypes(exclude=['number']).columns.tolist()
    
    if 'id' in cols:
        cols.remove('id')
        
    print(cols)
    
    df.drop(columns=cols, inplace=True)
    print(df.shape)


drop_non_numeric(train)


train


#FitnessGram Child (categorical)
def get_fitness_index(df):

    cols_FGC = [
        'FGC-FGC_CU_Zone', 'FGC-FGC_GSND_Zone', 'FGC-FGC_GSD_Zone',
        'FGC-FGC_PU_Zone', 'FGC-FGC_SRL_Zone', 'FGC-FGC_SRR_Zone',
        'FGC-FGC_TL_Zone'
    ]

    # Imputazione con moda
    for col in cols_FGC:
        mode = df[col].mode()[0]
        df[col] = df[col].fillna(mode)

    # Calcolo fitness index
    df['fitness_index'] = df[cols_FGC].mean(axis=1)    
    print(df['fitness_index'].describe())
        
    df.drop(columns=cols_FGC, inplace=True)


get_fitness_index(train)


def show_nan(df):
    perc_nan = df.isnull().mean().sort_values(ascending=False)
    print(perc_nan[perc_nan > 0])

show_nan(train)


train


def impute_negatives(df):
    cols = df.select_dtypes(include=['number']).columns
    for col in cols:
        negatives = df[col] < 0
        if negatives.any():
            print(f"{col}: {negatives.sum()}")
            df.loc[negatives, col] = np.nan

        
impute_negatives(train)



#Bio-electric Impedance Analysis
def fill_BIA(df):
    cols_BIA = [
        'BIA-BIA_BMC', 'BIA-BIA_BMI', 'BIA-BIA_BMR', 'BIA-BIA_DEE', 'BIA-BIA_ECW',
        'BIA-BIA_FFM', 'BIA-BIA_FFMI', 'BIA-BIA_FMI', 'BIA-BIA_Fat', 
        'BIA-BIA_ICW', 'BIA-BIA_LDM', 'BIA-BIA_LST', 'BIA-BIA_SMM', 'BIA-BIA_TBW'
    ]
    df['BIA_Missing'] = df[cols_BIA].isnull().all(axis=1).astype(int)
    df[cols_BIA] = df[cols_BIA].fillna(-1) 

fill_BIA(train)


show_nan(train)


def fill_every_nan(df):
    for col in df.columns:
        if df[col].isna().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                if df[col].nunique(dropna=True) > 5:
                    # Colonna numerica continua
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val)
                    print(f"{col}: imputato con mediana ({median_val})")
                else:
                    # Colonna numerica categorica
                    mode_val = df[col].mode()[0]
                    df[col] = df[col].fillna(mode_val)
                    print(f"{col}: imputato con moda ({mode_val})")

fill_every_nan(train)


show_nan(train)


def show_boxplots(df):
    cols = df.select_dtypes(include=['number']).columns.tolist()
    
    n_cols = len(cols)
    n_rows = math.ceil(n_cols / 6)
    
    plt.figure(figsize=(30, 5 * n_rows))
    
    for i, col in enumerate(cols, 1):
        plt.subplot(n_rows, 6, i)
        sns.boxplot(x=df[col])
        plt.title(col)
    
    plt.tight_layout()
    plt.show()

show_boxplots(train)


def show_boxplots_cols(df, cols):
    
    n_cols = len(cols)
    n_rows = math.ceil(n_cols / 6)
    
    plt.figure(figsize=(30, 5 * n_rows))
    
    for i, col in enumerate(cols, 1):
        plt.subplot(n_rows, 6, i)
        sns.boxplot(x=df[col])
        plt.title(col)
    
    plt.tight_layout()
    plt.show()



def remove_outliers_iqr(df, cols):
    
    for col in cols:            
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        median = df[col].median()

        train[col] = train[col].where(
            (train[col] >= lower) & (train[col] <= upper) | train[col].isna(),
            median
        )



cols = [
    'BIA-BIA_BMC', 'BIA-BIA_BMR', 'BIA-BIA_DEE', 'BIA-BIA_ECW', 'BIA-BIA_FFM', 'BIA-BIA_FFMI', 
    'BIA-BIA_Fat', 'BIA-BIA_ICW', 'BIA-BIA_LDM', 'BIA-BIA_LST', 'BIA-BIA_SMM', 'BIA-BIA_TBW', 
]

show_boxplots_cols(train, cols)

remove_outliers_iqr(train, cols)

show_boxplots_cols(train, cols)


from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay

X = train.drop(columns=['sii']) 
y = train['sii']

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)


# GridSearch
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 'log2', None]
}

rf = RandomForestClassifier(random_state=42)

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    scoring='accuracy',
    cv=3,
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV accuracy: {grid_search.best_score_:.4f}")

best_rf = grid_search.best_estimator_
y_val_pred = best_rf.predict(X_val)

print("\n\nClassification report:")
print(classification_report(y_val, y_val_pred, zero_division=0))
print("\nAccuracy:", accuracy_score(y_val, y_val_pred))


# Feature importance
importances = best_rf.feature_importances_
features = X_train.columns

fi_df = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("\nTop 20 feature più importanti:")
print(fi_df.head(20))

plt.figure(figsize=(20, 6))
sns.barplot(x='Importance', y='Feature', data=fi_df.head(10), palette='viridis')
plt.title("Feature Importances - Random Forest")
plt.tight_layout()
plt.savefig("random_forest_feature_importance")
plt.show()


val_results = X_val.copy()
val_results['true_label'] = y_val
val_results['predicted'] = y_val_pred
val_results['correct'] = val_results['true_label'] == val_results['predicted']

error_counts = val_results[val_results['correct'] == False]['true_label'].value_counts()
correct_counts = val_results[val_results['correct'] == True]['true_label'].value_counts()

print("\nWrong:")
print(error_counts)
print("\nCorrect:")
print(correct_counts)

combined = pd.DataFrame({
    'Correct': correct_counts,
    'Wrong': error_counts
}).fillna(0)

combined.plot(kind='bar', figsize=(10, 5), title="Correct vs Wrong Predictions")
plt.ylabel("Numero di istanze")
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()
plt.show()


# Confusion Matrix
cm = confusion_matrix(y_val, y_val_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_rf.classes_)

plt.figure(figsize=(6, 6))
disp.plot(cmap='Blues', values_format='d')
plt.title("Confusion Matrix - Random Forest")
plt.grid(False)
plt.savefig("random_forest_confusion_matrix")
plt.show()


from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from xgboost import XGBClassifier

X = train.drop(columns=['sii'])
y = train['sii']

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 6, 10], 
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.7, 1.0],
    'colsample_bytree': [0.7, 1.0]
}

xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss')

grid_search = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    scoring='accuracy',
    cv=3,
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV accuracy: {grid_search.best_score_:.4f}")


best_xgb = grid_search.best_estimator_
y_val_pred = best_xgb.predict(X_val)

print("\n\nClassification report:")
print(classification_report(y_val, y_val_pred, zero_division=0))
print("\nAccuracy:", accuracy_score(y_val, y_val_pred))


importances = best_xgb.feature_importances_
features = X_train.columns

fi_df = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("\nTop 20 feature più importanti:")
print(fi_df.head(20))

plt.figure(figsize=(20, 6))
sns.barplot(x='Importance', y='Feature', data=fi_df.head(10), palette='viridis')
plt.title("Feature Importances - XGBoost")
plt.tight_layout()
plt.savefig("xgboost_feature_importance.png")
plt.show()


val_results = X_val.copy()
val_results['true_label'] = y_val
val_results['predicted'] = y_val_pred
val_results['correct'] = val_results['true_label'] == val_results['predicted']

error_counts = val_results[val_results['correct'] == False]['true_label'].value_counts()
correct_counts = val_results[val_results['correct'] == True]['true_label'].value_counts()

print("\nWrong:")
print(error_counts)
print("\nCorrect:")
print(correct_counts)

combined = pd.DataFrame({
    'Correct': correct_counts,
    'Wrong': error_counts
}).fillna(0)

combined.plot(kind='bar', figsize=(10, 5), title="Correct vs Wrong Predictions")
plt.ylabel("Numero di istanze")
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()
plt.show()


cm = confusion_matrix(y_val, y_val_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_xgb.classes_)

plt.figure(figsize=(6, 6))
disp.plot(cmap='Blues', values_format='d')
plt.title("Confusion Matrix - XGBoost")
plt.grid(False)
plt.savefig("xgboost_confusion_matrix.png")
plt.show()


test = pd.read_csv('/kaggle/input/child-mind-institute-problematic-internet-use/test.csv')
test.shape


drop_non_numeric(test)


get_fitness_index(test)


#droppo le colonne che non ci sono nel train
cols = [col for col in test.columns if col not in train.columns and col != 'id']
test = test.drop(columns=cols)
test.shape


impute_negatives(test)


show_nan(test)


fill_BIA(test)


fill_every_nan(test)


show_boxplots(test)


cols = [
    #'CGAS-CGAS_Score'
]

show_boxplots_cols(train, cols)

remove_outliers_iqr(train, cols)

show_boxplots_cols(train, cols)


#Lancio XGBoost
X_test = test.drop(columns=['id'])

y_test_pred = best_xgb.predict(X_test)


results = pd.DataFrame({
    'id': test['id'],
    'sii': y_test_pred
})

results.to_csv('submission.csv', index=False)

results




