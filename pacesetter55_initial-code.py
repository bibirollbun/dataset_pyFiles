import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns
import warnings 
warnings.filterwarnings("ignore")
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OrdinalEncoder,OneHotEncoder,LabelEncoder
from xgboost import XGBClassifier,XGBRegressor
from lightgbm import LGBMRegressor,LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression
from scipy.stats import mode


df=pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
df_test=pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
sample_submission=pd.read_csv("/kaggle/input/playground-series-s5e7/sample_submission.csv")


df.head()



class EDAHelper:
    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe
    def checking(self):
        df=self.df
        total = len(df)
        check_df = pd.DataFrame(df.isnull().sum(), columns=['#NULLS'])
        check_df['%NULLS'] = round((check_df['#NULLS']/total)*100, 5)
        check_df['#Unique_Valus'] = df.nunique()
        cat_cols = [col for col in df.columns if df[col].dtype == 'object']
        uniques = []
        for col in df.columns:
            if col in cat_cols:
                uniques.append(set(df[col].dropna()))
            else:
                uniques.append(df[col].max() - df[col].min())
        check_df['Unique_Values/Range'] = uniques
        return display(check_df)


    def outliers_detection(self, q1_coef=0.25, q3_coef=0.75):
        df=self.df
        data_out = df.copy()
        outlier_counts = {}
    
        cat_cols = [col for col in data_out.columns if data_out[col].dtype == 'object']
        num_cols = [col for col in data_out.columns if col not in cat_cols]
    
        for col in num_cols:
            q1=data_out[col].quantile(q1_coef)
            q3=data_out[col].quantile(q3_coef)
            iqr = q3 - q1
            outlier_counts[col]=[len(data_out[(data_out[col]<q1-1.5*iqr) | (data_out[col]>q3+1.5*iqr)])]
    
        for col in cat_cols:
            value_counts = data_out[col].value_counts()
            threshold = 0.0001*len(data_out)
            rare_values = value_counts[value_counts<threshold].index
            outlier_counts[col]=[len(data_out[data_out[col].isin(rare_values)])]
    
        outlier_df = pd.DataFrame(outlier_counts).T
        outlier_df.columns = ['Number of Outliers']
        return display(outlier_df)

    def draw_plots(self):
        df=self.df
        numeric_columns = df.select_dtypes(include=['number']).columns
    
        for col in numeric_columns:
            plt.figure(figsize=(14, 6))
    
            # Box plot
            plt.subplot(1, 2, 1)
            sns.boxplot(y=df[col], color='skyblue')
            plt.title(f'Box Plot of {col}')
    
            # Histogram
            plt.subplot(1, 2, 2)
            sns.histplot(df[col], kde=True, color='lightgreen')
            plt.title(f'Histogram of {col}')
    
            plt.tight_layout()
            plt.show()

    def find_skewness(self):
        # Select only numeric columns
        df=self.df
        numeric_columns = df.select_dtypes(include=['number']).columns
        
        # Calculate skewness for each numeric column
        skewness = df[numeric_columns].skew()
        
        # Print skewness for each column
        for col, skew_value in skewness.items():
            print(f'Skewness of {col}: {skew_value:.4f}')
    
        return display(skewness)

    def plot_categorical_columns(self):
        # Select only categorical columns
        df=self.df
        categorical_columns = df.select_dtypes(include=['object']).columns
        
        # Loop through each categorical column and plot
        for col in categorical_columns:
            if len(set(df[col]))<150:
                plt.figure(figsize=(8, 4))
                sns.countplot(y=df[col], palette="Set2", order=df[col].value_counts().index)
                plt.title(f'Distribution of {col}')
                plt.xlabel('Count')
                plt.ylabel(col)
                plt.tight_layout()
                plt.show()
    def run_all(self):
        self.checking()
        self.outliers_detection()
        self.draw_plots()
        self.find_skewness()
        self.plot_categorical_columns()


EDA=EDAHelper(df)


EDA.run_all()


df['kfold']=-1
kf=KFold(n_splits=5,shuffle=True,random_state=24)
for folds,(train_indices,valid_indices) in enumerate(kf.split(X=df)):
    df.loc[valid_indices,'kfold']=folds


df.columns


df_test.head()


useful_features=[c for c in df.columns if c not in ['id','kfold','Personality']]
object_cols=[col for col in useful_features if df[col].dtype=='object']
df_test=df_test[useful_features]
le=LabelEncoder()
df['Personality']=le.fit_transform(df['Personality'])


final_predictions=[]
scores=[]
for fold in range(5):
    xtrain=df[df.kfold != fold].reset_index(drop=True)
    xvalid=df[df.kfold == fold].reset_index(drop=True)
    xtest=df_test.copy()
    ############
    ytrain=xtrain.Personality
    yvalid=xvalid.Personality

    xtrain=xtrain[useful_features]
    xvalid=xvalid[useful_features]
    xtest=xtest[useful_features]
    ordinal_encoder=OrdinalEncoder()
    xtrain[object_cols]=ordinal_encoder.fit_transform(xtrain[object_cols])
    xtest[object_cols]=ordinal_encoder.transform(xtest[object_cols])
    xvalid[object_cols]=ordinal_encoder.transform(xvalid[object_cols])

    model=XGBRegressor()
    model.fit(xtrain,ytrain)
    preds=model.predict(xvalid)
    preds=(preds>=0.5).astype(int)
    print(fold,accuracy_score(preds,yvalid))
    scores.append(accuracy_score(preds,yvalid))
    test_preds=model.predict(xtest)
    final_predictions.append(test_preds)
print(f"Mean score {np.mean(scores)}")
    


final_predictions=[]
scores=[]
for fold in range(5):
    xtrain=df[df.kfold != fold].reset_index(drop=True)
    xvalid=df[df.kfold == fold].reset_index(drop=True)
    xtest=df_test.copy()
    ############
    ytrain=xtrain.Personality
    yvalid=xvalid.Personality

    xtrain=xtrain[useful_features]
    xvalid=xvalid[useful_features]
    xtest=xtest[useful_features]
    ordinal_encoder=OrdinalEncoder()
    xtrain[object_cols]=ordinal_encoder.fit_transform(xtrain[object_cols])
    xtest[object_cols]=ordinal_encoder.transform(xtest[object_cols])
    xvalid[object_cols]=ordinal_encoder.transform(xvalid[object_cols])

    model=XGBClassifier()
    model.fit(xtrain,ytrain)
    preds=model.predict(xvalid)
    print(fold,accuracy_score(preds,yvalid))
    scores.append(accuracy_score(preds,yvalid))
    test_preds=model.predict(xtest)
    final_predictions.append(test_preds)
print(f"Mean score {np.mean(scores)}")
    


final_predictions=[]
scores=[]
for fold in range(5):
    xtrain=df[df.kfold != fold].reset_index(drop=True)
    xvalid=df[df.kfold == fold].reset_index(drop=True)
    xtest=df_test.copy()
    ############
    ytrain=xtrain.Personality
    yvalid=xvalid.Personality

    xtrain=xtrain[useful_features]
    xvalid=xvalid[useful_features]
    xtest=xtest[useful_features]
    ordinal_encoder=OrdinalEncoder()
    xtrain[object_cols]=ordinal_encoder.fit_transform(xtrain[object_cols])
    xtest[object_cols]=ordinal_encoder.transform(xtest[object_cols])
    xvalid[object_cols]=ordinal_encoder.transform(xvalid[object_cols])

    model=LGBMClassifier(verbose=0)
    model.fit(xtrain,ytrain)
    preds=model.predict(xvalid)
    print(fold,accuracy_score(preds,yvalid))
    scores.append(accuracy_score(preds,yvalid))
    test_preds=model.predict(xtest)
    final_predictions.append(test_preds)
print(f"Mean score {np.mean(scores)}")


final_predictions=[]
scores=[]
for fold in range(5):
    xtrain=df[df.kfold != fold].reset_index(drop=True)
    xvalid=df[df.kfold == fold].reset_index(drop=True)
    xtest=df_test.copy()
    ############
    ytrain=xtrain.Personality
    yvalid=xvalid.Personality

    xtrain=xtrain[useful_features]
    xvalid=xvalid[useful_features]
    xtest=xtest[useful_features]
    ordinal_encoder=OrdinalEncoder()
    xtrain[object_cols]=ordinal_encoder.fit_transform(xtrain[object_cols])
    xtest[object_cols]=ordinal_encoder.transform(xtest[object_cols])
    xvalid[object_cols]=ordinal_encoder.transform(xvalid[object_cols])

    model=HistGradientBoostingClassifier(verbose=0,random_state=42)
    model.fit(xtrain,ytrain)
    preds=model.predict(xvalid)
    print(fold,accuracy_score(preds,yvalid))
    scores.append(accuracy_score(preds,yvalid))
    test_preds=model.predict(xtest)
    final_predictions.append(test_preds)
print(f"Mean score {np.mean(scores)}")


final_predictions=[]
scores=[]
for fold in range(5):
    xtrain=df[df.kfold != fold].reset_index(drop=True)
    xvalid=df[df.kfold == fold].reset_index(drop=True)
    xtest=df_test.copy()
    ############
    ytrain=xtrain.Personality
    yvalid=xvalid.Personality

    xtrain=xtrain[useful_features]
    xvalid=xvalid[useful_features]
    xtest=xtest[useful_features]

    ohe = OneHotEncoder(sparse=False, handle_unknown="ignore")
    xtrain_ohe = ohe.fit_transform(xtrain[object_cols])
    xvalid_ohe = ohe.transform(xvalid[object_cols])
    xtest_ohe = ohe.transform(xtest[object_cols])
    
    xtrain_ohe = pd.DataFrame(xtrain_ohe, columns=[f"ohe_{i}" for i in range(xtrain_ohe.shape[1])])
    xvalid_ohe = pd.DataFrame(xvalid_ohe, columns=[f"ohe_{i}" for i in range(xvalid_ohe.shape[1])])
    xtest_ohe = pd.DataFrame(xtest_ohe, columns=[f"ohe_{i}" for i in range(xtest_ohe.shape[1])])
    
    xtrain = pd.concat([xtrain, xtrain_ohe], axis=1)
    xvalid = pd.concat([xvalid, xvalid_ohe], axis=1)
    xtest = pd.concat([xtest, xtest_ohe], axis=1)
    
    # this part is missing in the video:
    xtrain = xtrain.drop(object_cols, axis=1)
    xvalid = xvalid.drop(object_cols, axis=1)
    xtest = xtest.drop(object_cols, axis=1)


    model=HistGradientBoostingClassifier(verbose=0,random_state=42)
    model.fit(xtrain,ytrain)
    preds=model.predict(xvalid)
    print(fold,accuracy_score(preds,yvalid))
    scores.append(accuracy_score(preds,yvalid))
    test_preds=model.predict(xtest)
    final_predictions.append(test_preds)
print(f"Mean score {np.mean(scores)}")


for col in object_cols:
    temp_df=[]
    temp_test_feat=None
    for fold in range(5):
        xtrain=df[df['kfold']!=fold].reset_index(drop=True)
        xvalid=df[df['kfold']==fold].reset_index(drop=True)
        xtest=df_test.copy()
        feat=xtrain.groupby(col)['Personality'].mean()
        feat=feat.to_dict()
        xvalid.loc[:,f"tar_enc_{col}"]=xvalid[col].map(feat)
        temp_df.append(xvalid)
        if temp_test_feat is None:
            temp_test_feat=xtest[col].map(feat)
        else:
            temp_test_feat+=xtest[col].map(feat)
    temp_test_feat/=5
    df_test.loc[:,f"tar_enc_{col}"]=temp_test_feat
    df=pd.concat(temp_df)


df.head()


useful_features=[c for c in df.columns if c not in ['id','kfold','Personality']]
object_cols=[col for col in useful_features if df[col].dtype=='object']
df_test=df_test[useful_features]
le=LabelEncoder()
df['Personality']=le.fit_transform(df['Personality'])


final_predictions=[]
scores=[]
for fold in range(5):
    xtrain=df[df.kfold != fold].reset_index(drop=True)
    xvalid=df[df.kfold == fold].reset_index(drop=True)
    xtest=df_test.copy()
    ############
    ytrain=xtrain.Personality
    yvalid=xvalid.Personality

    xtrain=xtrain[useful_features]
    xvalid=xvalid[useful_features]
    xtest=xtest[useful_features]

    ohe = OneHotEncoder(sparse=False, handle_unknown="ignore")
    xtrain_ohe = ohe.fit_transform(xtrain[object_cols])
    xvalid_ohe = ohe.transform(xvalid[object_cols])
    xtest_ohe = ohe.transform(xtest[object_cols])
    
    xtrain_ohe = pd.DataFrame(xtrain_ohe, columns=[f"ohe_{i}" for i in range(xtrain_ohe.shape[1])])
    xvalid_ohe = pd.DataFrame(xvalid_ohe, columns=[f"ohe_{i}" for i in range(xvalid_ohe.shape[1])])
    xtest_ohe = pd.DataFrame(xtest_ohe, columns=[f"ohe_{i}" for i in range(xtest_ohe.shape[1])])
    
    xtrain = pd.concat([xtrain, xtrain_ohe], axis=1)
    xvalid = pd.concat([xvalid, xvalid_ohe], axis=1)
    xtest = pd.concat([xtest, xtest_ohe], axis=1)
    
    # this part is missing in the video:
    xtrain = xtrain.drop(object_cols, axis=1)
    xvalid = xvalid.drop(object_cols, axis=1)
    xtest = xtest.drop(object_cols, axis=1)


    model=HistGradientBoostingClassifier(verbose=0,random_state=42)
    model.fit(xtrain,ytrain)
    preds=model.predict(xvalid)
    print(fold,accuracy_score(preds,yvalid))
    scores.append(accuracy_score(preds,yvalid))
    test_preds=model.predict(xtest)
    final_predictions.append(test_preds)
print(f"Mean score {np.mean(scores)}")


final_predictions=[]
scores=[]
for fold in range(5):
    xtrain=df[df.kfold != fold].reset_index(drop=True)
    xvalid=df[df.kfold == fold].reset_index(drop=True)
    xtest=df_test.copy()
    ############
    ytrain=xtrain.Personality
    yvalid=xvalid.Personality

    xtrain=xtrain[useful_features]
    xvalid=xvalid[useful_features]
    xtest=xtest[useful_features]

    
    # this part is missing in the video:
    xtrain = xtrain.drop(object_cols, axis=1)
    xvalid = xvalid.drop(object_cols, axis=1)
    xtest = xtest.drop(object_cols, axis=1)


    model=HistGradientBoostingClassifier(verbose=0,random_state=42)
    model.fit(xtrain,ytrain)
    preds=model.predict(xvalid)
    print(fold,accuracy_score(preds,yvalid))
    scores.append(accuracy_score(preds,yvalid))
    test_preds=model.predict(xtest)
    final_predictions.append(test_preds)
print(f"Mean score {np.mean(scores)}")


from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.metrics import accuracy_score
import numpy as np

strategies = ['mean', 'median', 'most_frequent', 'knn']

for strategy in strategies:
    print(f"\nTesting strategy: {strategy}")
    final_predictions = []
    scores = []
    
    for fold in range(5):
        # Split data
        xtrain = df[df.kfold != fold].reset_index(drop=True)
        xvalid = df[df.kfold == fold].reset_index(drop=True)
        xtest = df_test.copy()

        ytrain = xtrain.Personality
        yvalid = xvalid.Personality

        # Select features
        xtrain = xtrain[useful_features]
        xvalid = xvalid[useful_features]
        xtest = xtest[useful_features]

        # Drop object columns
        xtrain = xtrain.drop(object_cols, axis=1)
        xvalid = xvalid.drop(object_cols, axis=1)
        xtest = xtest.drop(object_cols, axis=1)

        # Apply imputation
        if strategy == 'knn':
            imputer = KNNImputer(n_neighbors=5)
        else:
            imputer = SimpleImputer(strategy=strategy)
        
        xtrain = imputer.fit_transform(xtrain)
        xvalid = imputer.transform(xvalid)
        xtest = imputer.transform(xtest)

        # Train and evaluate model
        model = HistGradientBoostingClassifier(verbose=0, random_state=42)
        model.fit(xtrain, ytrain)
        preds = model.predict(xvalid)
        acc = accuracy_score(preds, yvalid)
        print(f"  Fold {fold} accuracy: {acc:.4f}")
        scores.append(acc)

        test_preds = model.predict(xtest)
        final_predictions.append(test_preds)
    
    print(f"Mean score for {strategy}: {np.mean(scores):.9f}")



df=pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
df_test=pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
sample_submission=pd.read_csv("/kaggle/input/playground-series-s5e7/sample_submission.csv")
useful_features=[c for c in df.columns if c not in ['id','kfold','Personality']]
object_cols=[col for col in useful_features if df[col].dtype=='object']
df_test=df_test[useful_features]
df['kfold']=-1
kf=KFold(n_splits=5,shuffle=True,random_state=24)
for folds,(train_indices,valid_indices) in enumerate(kf.split(X=df)):
    df.loc[valid_indices,'kfold']=folds
le=LabelEncoder()
df['Personality']=le.fit_transform(df['Personality'])
final_test_predictions = []
final_valid_predictions = {}
scores = []
for fold in range(5):
    xtrain=df[df.kfold != fold].reset_index(drop=True)
    xvalid=df[df.kfold == fold].reset_index(drop=True)
    xtest=df_test.copy()
    valid_ids = xvalid.id.values.tolist()
    ############
    ytrain=xtrain.Personality
    yvalid=xvalid.Personality

    xtrain=xtrain[useful_features]
    xvalid=xvalid[useful_features]
    xtest=xtest[useful_features]
    ordinal_encoder=OrdinalEncoder()
    xtrain[object_cols]=ordinal_encoder.fit_transform(xtrain[object_cols])
    xtest[object_cols]=ordinal_encoder.transform(xtest[object_cols])
    xvalid[object_cols]=ordinal_encoder.transform(xvalid[object_cols])

    model=HistGradientBoostingClassifier(verbose=0,random_state=42)
    model.fit(xtrain,ytrain)
    preds_valid = model.predict(xvalid)
    test_preds = model.predict(xtest)
    final_test_predictions.append(test_preds)
    final_valid_predictions.update(dict(zip(valid_ids, preds_valid)))
    print(fold,accuracy_score(preds_valid,yvalid))
    scores.append(accuracy_score(preds_valid,yvalid))
    test_preds=model.predict(xtest)
    final_predictions.append(test_preds)
print(f"Mean score {np.mean(scores)}")
final_valid_predictions = pd.DataFrame.from_dict(final_valid_predictions, orient="index").reset_index()
final_valid_predictions.columns = ["id", "pred_1"]
final_valid_predictions.to_csv("train_pred_1.csv", index=False)

sample_submission.Personality = mode(np.column_stack(final_test_predictions), axis=1)[0].ravel()
sample_submission.columns = ["id", "pred_1"]
sample_submission.to_csv("test_pred_1.csv", index=False)


df=pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
df_test=pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
sample_submission=pd.read_csv("/kaggle/input/playground-series-s5e7/sample_submission.csv")
useful_features=[c for c in df.columns if c not in ['id','kfold','Personality']]
object_cols=[col for col in useful_features if df[col].dtype=='object']
df_test=df_test[useful_features]
df['kfold']=-1
kf=KFold(n_splits=5,shuffle=True,random_state=24)
for folds,(train_indices,valid_indices) in enumerate(kf.split(X=df)):
    df.loc[valid_indices,'kfold']=folds
le=LabelEncoder()
df['Personality']=le.fit_transform(df['Personality'])
final_test_predictions = []
final_valid_predictions = {}
scores = []
for fold in range(5):
    xtrain=df[df.kfold != fold].reset_index(drop=True)
    xvalid=df[df.kfold == fold].reset_index(drop=True)
    xtest=df_test.copy()
    valid_ids = xvalid.id.values.tolist()
    ############
    ytrain=xtrain.Personality
    yvalid=xvalid.Personality

    xtrain=xtrain[useful_features]
    xvalid=xvalid[useful_features]
    xtest=xtest[useful_features]
    ordinal_encoder=OrdinalEncoder()
    xtrain[object_cols]=ordinal_encoder.fit_transform(xtrain[object_cols])
    xtest[object_cols]=ordinal_encoder.transform(xtest[object_cols])
    xvalid[object_cols]=ordinal_encoder.transform(xvalid[object_cols])

    model=LGBMClassifier(verbose=0,random_state=42)
    model.fit(xtrain,ytrain)
    preds_valid = model.predict(xvalid)
    test_preds = model.predict(xtest)
    final_test_predictions.append(test_preds)
    final_valid_predictions.update(dict(zip(valid_ids, preds_valid)))
    print(fold,accuracy_score(preds_valid,yvalid))
    scores.append(accuracy_score(preds_valid,yvalid))
    test_preds=model.predict(xtest)
    final_predictions.append(test_preds)
print(f"Mean score {np.mean(scores)}")
final_valid_predictions = pd.DataFrame.from_dict(final_valid_predictions, orient="index").reset_index()
final_valid_predictions.columns = ["id", "pred_2"]
final_valid_predictions.to_csv("train_pred_2.csv", index=False)

sample_submission['Personality'] = mode(np.column_stack(final_test_predictions), axis=1)[0].ravel()
sample_submission.columns = ["id", "pred_2"]
sample_submission.to_csv("test_pred_2.csv", index=False)


df=pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
df_test=pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
sample_submission=pd.read_csv("/kaggle/input/playground-series-s5e7/sample_submission.csv")
useful_features=[c for c in df.columns if c not in ['id','kfold','Personality']]
object_cols=[col for col in useful_features if df[col].dtype=='object']
df_test=df_test[useful_features]
df['kfold']=-1
kf=KFold(n_splits=5,shuffle=True,random_state=24)
for folds,(train_indices,valid_indices) in enumerate(kf.split(X=df)):
    df.loc[valid_indices,'kfold']=folds
le=LabelEncoder()
df['Personality']=le.fit_transform(df['Personality'])
final_test_predictions = []
final_valid_predictions = {}
scores = []
for fold in range(5):
    xtrain=df[df.kfold != fold].reset_index(drop=True)
    xvalid=df[df.kfold == fold].reset_index(drop=True)
    xtest=df_test.copy()
    valid_ids = xvalid.id.values.tolist()
    ############
    ytrain=xtrain.Personality
    yvalid=xvalid.Personality

    xtrain=xtrain[useful_features]
    xvalid=xvalid[useful_features]
    xtest=xtest[useful_features]
    ordinal_encoder=OrdinalEncoder()
    xtrain[object_cols]=ordinal_encoder.fit_transform(xtrain[object_cols])
    xtest[object_cols]=ordinal_encoder.transform(xtest[object_cols])
    xvalid[object_cols]=ordinal_encoder.transform(xvalid[object_cols])

    model=XGBClassifier(verbose=0,random_state=42)
    model.fit(xtrain,ytrain)
    preds_valid = model.predict(xvalid)
    test_preds = model.predict(xtest)
    final_test_predictions.append(test_preds)
    final_valid_predictions.update(dict(zip(valid_ids, preds_valid)))
    print(fold,accuracy_score(preds_valid,yvalid))
    scores.append(accuracy_score(preds_valid,yvalid))
    test_preds=model.predict(xtest)
    final_predictions.append(test_preds)
print(f"Mean score {np.mean(scores)}")
final_valid_predictions = pd.DataFrame.from_dict(final_valid_predictions, orient="index").reset_index()
final_valid_predictions.columns = ["id", "pred_3"]
final_valid_predictions.to_csv("train_pred_3.csv", index=False)

sample_submission['Personality'] = mode(np.column_stack(final_test_predictions), axis=1)[0].ravel()
sample_submission.columns = ["id", "pred_3"]
sample_submission.to_csv("test_pred_3.csv", index=False)


df=pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
df_test=pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")
sample_submission=pd.read_csv("/kaggle/input/playground-series-s5e7/sample_submission.csv")
df['kfold']=-1
kf=KFold(n_splits=5,shuffle=True,random_state=24)
for folds,(train_indices,valid_indices) in enumerate(kf.split(X=df)):
    df.loc[valid_indices,'kfold']=folds
le=LabelEncoder()
df['Personality']=le.fit_transform(df['Personality'])

df1 = pd.read_csv("train_pred_1.csv")
df2 = pd.read_csv("train_pred_2.csv")
df3 = pd.read_csv("train_pred_3.csv")

df_test1 = pd.read_csv("test_pred_1.csv")
df_test2 = pd.read_csv("test_pred_2.csv")
df_test3 = pd.read_csv("test_pred_3.csv")

df = df.merge(df1, on="id", how="left")
df = df.merge(df2, on="id", how="left")
df = df.merge(df3, on="id", how="left")

df_test = df_test.merge(df_test1, on="id", how="left")
df_test = df_test.merge(df_test2, on="id", how="left")
df_test = df_test.merge(df_test3, on="id", how="left")

df.head()


df[['pred_1', 'pred_2', 'pred_3']].corr()


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

useful_features = ["pred_1", "pred_2", "pred_3"]
df_test = df_test[useful_features]

# Encode target labels (if not already done)
le = LabelEncoder()
df["Personality"] = le.fit_transform(df["Personality"])

final_predictions = []
scores = []

for fold in range(5):
    xtrain = df[df.kfold != fold].reset_index(drop=True)
    xvalid = df[df.kfold == fold].reset_index(drop=True)
    xtest = df_test.copy()

    ytrain = xtrain.Personality
    yvalid = xvalid.Personality

    xtrain = xtrain[useful_features]
    xvalid = xvalid[useful_features]

    # Use Logistic Regression instead of Linear Regression
    model = LogisticRegression(max_iter=1000)
    model.fit(xtrain, ytrain)

    preds_valid = model.predict(xvalid)  # class labels
    test_preds = model.predict(xtest)    # class labels
    final_predictions.append(test_preds)

    acc = accuracy_score(yvalid, preds_valid)
    print(f"Fold {fold} Accuracy: {acc:.4f}")
    scores.append(acc)

print(f"\nMean Accuracy: {np.mean(scores):.4f} ± {np.std(scores):.4f}")





