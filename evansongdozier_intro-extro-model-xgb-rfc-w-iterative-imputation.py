import warnings 

import pandas as pd
import numpy as npy
import matplotlib.pyplot as mpl
import seaborn as sn
from xgboost import XGBClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

print('All standard libraries available, setup successful.')


#Import data as pandas dataframes
df_train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")


df_train.info()
df_train.shape
display(df_train.head())
df_train.tail()


#Ensuring that the missing values are numpy not-a-number. Missing values from Scikitlearn can only interpret numpy.nan
imputer = IterativeImputer(missing_values=npy.nan, max_iter=10, tol=1e-3, n_nearest_features=None, initial_strategy='mean', imputation_order='ascending', random_state=42,)
                            


def preprocess(df):
    df = df.copy()
    
    def yesorno(x):
        if x=="Yes":
            return "1"
        if x=="No":
            return "0"
        else:
            return npy.nan
            
    df["Stage_fear"] = df["Stage_fear"].apply(yesorno).fillna(npy.nan).astype(npy.float64)
    df["Stage_fear"] = pd.to_numeric(df["Stage_fear"],errors='coerce')
    df["Drained_after_socializing"] = df["Drained_after_socializing"].apply(yesorno).fillna(npy.nan).astype(npy.float64)
    df["Drained_after_socializing"] = pd.to_numeric(df["Drained_after_socializing"],errors='coerce')
    
    return df

#Numerizing personality to be represented as a binary set in order to produce proper correlation charts to. 0 = Introvert, 1 = Extrovert
#Since the test does not contain a column for Personality, this function is seperated from the main preprocessing() function. 
def personalitybin(df):
    df = df.copy()
    
    def binary(x):
        if x=="Introvert":
            return "0"
        else:
            return "1"
    df["Personality"]= df["Personality"].apply(binary).astype(npy.float64)
    return df
    
preprocessed_train_df = preprocess(personalitybin(df_train)).astype(npy.float64)
preprocessed_test_df = preprocess(df_test).fillna(npy.nan).astype(npy.float64)

print("preprocessing completed") #Debugging phrase 


#Defining the fit and transform data frame recursively. Output set to be outputted as pandas dataframe. 
imputer.set_output(transform='pandas')

#Converting data set to uniform integer dtype for model  consistency
preprocessed_train_df = imputer.fit_transform(preprocessed_train_df).astype(npy.int64)

#preprocessed_train_df.to_csv('imputedebug.csv', index=False) - For Debugging purposes


preprocessed_train_df.info()
print("preprocessed training data head:")
display(preprocessed_train_df.head())
print("preprocessed training data tail:")
display(preprocessed_train_df.tail())
print("preprocessed test data head:")
preprocessed_test_df.head()



#Ensures that all variables selected for are numeric. Dropping index column.
num_df = preprocessed_train_df.drop("id", axis = 1).select_dtypes(include=[npy.number])

sn.color_palette("colorblind")

if num_df.shape[1] >= 4:
    mpl.figure(figsize=(10,8)) 
    sn.heatmap(num_df.corr(), annot=True, fmt='.2f', cmap='rocket')
    mpl.title('Correlation heatmap of Variables:')
    mpl.tight_layout()
    mpl.show()
else:
    print('Not enough numeric features for correlation analysis')


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
warnings.filterwarnings('ignore')

y_train = preprocessed_train_df["Personality"]
X_train = preprocessed_train_df.drop(["id", "Personality","Going_outside", "Friends_circle_size", "Post_frequency", "Social_event_attendance"], axis=1)

X_test = preprocessed_test_df.drop(["id","Going_outside", "Friends_circle_size", "Post_frequency", "Social_event_attendance"], axis = 1)


#Check for Numpy arrays
X = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
y = y_train.values if isinstance(X_train, pd.Series) else y_train
X_test_npy = X_test.values if isinstance(X_test, pd.DataFrame) else X_test

#Configure Stratified K Fold function for 5-fold cross-validation
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) #Answer to life
test_preds = npy.zeros(len(X_test_npy))
val_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"Training fold {fold + 1}...")

    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    model = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.01, subsample=0.9, 
            colsample_bytree=0.75, colsample_bynode=0.75, colsample_bylevel=0.75, 
            tree_method='gpu_hist', predictor='gpu_predictor', random_state=42, use_label_encoder=False, eval_metric='logloss')

    model.fit(X_tr, y_tr)

    val_pred = model.predict(X_val)
    val_acc = accuracy_score(y_val, val_pred)
    val_scores.append(val_acc)

    test_preds += model.predict(X_test_npy)

final_preds = (test_preds >= 3).astype(int)

#convert binary of predictions back to objects for introvert = 0 and extrovert = 1

def debin(df):
    df = df.copy()
    
    def binary(x):
        if x==0:
            return "Introvert"
        else:
            return "Extrovert"
    df["Personality"]=df["Personality"].apply(binary).astype(object)
    return df

print(f"CV Scores: {val_scores}")
print(f"Average CV Accuracy: {npy.mean(val_scores):.4f}")


#Submission output

output = pd.DataFrame({'id': df_test.id, "Personality": final_preds})
output = debin(output)
output.to_csv('submission.csv', index=False)

