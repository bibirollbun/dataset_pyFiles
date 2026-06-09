import numpy as np 
import pandas as pd 
import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    
    for filename in filenames:
        
        print(os.path.join(dirname, filename))
        
# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go


sample_sub = pd.read_csv("/kaggle/input/playground-series-s5e11/sample_submission.csv")
train_data = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv").drop(columns='id',axis=1)
test_data = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv").drop(columns='id',axis=1)



train_data.head()


test_data.head()


sample_sub.head()


print("train dataset shape:-",train_data.shape)
print("test dataset shape:-",test_data.shape)
print("sample sub dataset shape:-",sample_sub.shape)


train_data.info()


test_data.info()





train_data.isnull().sum().sum()


test_data.isnull().sum().sum()





test_data.describe()


train_data.describe()





train_data.replace([np.inf,-np.inf],np.nan,inplace=True)
test_data.replace([np.inf,-np.inf],np.nan,inplace=True)





numerical_col = train_data.select_dtypes(include=np.number).columns.tolist()
categorical_col = train_data.select_dtypes(include='O').columns.tolist()

print("Numerical column\n",numerical_col)
print()
print("Categorical column\n",categorical_col)





class numerical_analysis:

    def fit(self,data=None,rows=None,col=None,stragey='mean',skew='no',scaler=False):

        self.data = data # original data
        self.data_copy = data.copy() # copy data
        self.column_name = self.data.select_dtypes(include=np.number).columns
        self.rows = rows
        self.col = col
        self.scaler = scaler
        self.skew = skew 

        if stragey in ['mean','median']:
            self.stragey = stragey
            
        else:
            raise ValueError("""
            Default stragey is mean you select median please select valid 
            stragey
            """)    
        
        if self.scaler : self.skew_reduce_processing_analysis()
        
        return self
    
    def skew_reduce_processing_analysis(self):


        from sklearn.preprocessing import StandardScaler,PowerTransformer
        
        for column_name in self.column_name:

            arr = self.data_copy[column_name].values.reshape(-1,1)
            
            if self.scaler == 'boxcox':
                
                shifted = self.data_copy[column_name] - self.data_copy[column_name].min()+1
                arr = shifted.values.reshape(-1,1)
                box_cox = PowerTransformer(method='box-cox')
                transformed = box_cox.fit_transform(arr)            
                
            elif self.scaler == 'standardscaler':
                std = StandardScaler()
                transformed = std.fit_transform(col.values.reshape(-1,1))
                
            # Store the transformed values back
            self.data_copy[column_name] = transformed.flatten()
    
    def compareision_original_reduce_analysis(self):

        if self.scaler == "":
            return
        
        num_cols = self.column_name
        n = len(num_cols)
        
        print("numerical column distributation analysis")

        plt.figure(figsize=(10, 4 * n))
    
        for i, col in enumerate(num_cols, 1):
    
            # --- ORIGINAL DATA ---
            plt.subplot(n, 2, 2 * i - 1)
            ax1 = sns.histplot(self.data[col], kde=True)
            skew_val = self.data[col].skew()

            center_tendency = [np.mean(self.data[col]) if self.stragey == 'mean' else np.median(self.data[column_name])][0]

            plt.axvline(center_tendency,c='red')

            y_max = plt.gca().get_ylim()[1] 
            
            plt.text(x=center_tendency, 
                     y=y_max * 0.9, # Place text at 90% of the chart height
                     s=f"{center_tendency:.2f}", # Correct formatting
                     color='red',
                     fontweight='bold')

    
            ax1.set_title(f"Original Distribution: {col}")
            ax1.set_xlabel(col)
            plt.xticks(rotation=90)
            ax1.legend([f"Skew: {skew_val:.2f}"])
    
            # --- TRANSFORMED DATA ---
            
            plt.subplot(n, 2, 2 * i)
            ax2 = sns.histplot(self.data_copy[col], kde=True)
            skew_new = self.data_copy[col].skew()

            center_tendency = np.var(self.data_copy[col])

            plt.axvline(center_tendency,c='red')

            y_max = plt.gca().get_ylim()[1] 
            
            # FIX 3: F-string formatting
            plt.text(x=center_tendency, 
                     y=y_max * 0.9, # Place text at 90% of the chart height
                     s=f"{center_tendency:.2f}", # Correct formatting
                     color='blue',
                     fontweight='bold')

            
            ax2.set_title(f"Transformed Distribution: {col}")
            ax2.set_xlabel(col)
    
            ax2.legend([f"Skew: {skew_new:.2f}"])
    
        plt.tight_layout()
        plt.show()

    def outlier_method(self):

        rows = len(self.column_name) * 2
        columns = 3

        print(rows)
        
        plt.figure(figsize=(20,5*len(self.column_name)))

        plot_index = 1
                   
        for col in self.column_name:

                     
            # IQR method
            lower_boundries,middle_boundries,upper_boundries = np.percentile(self.data[col],[25,50,100])
            IQR = upper_boundries - lower_boundries
            lower_mix_boundries =  lower_boundries - (1.5 * IQR)
            upper_mix_boundries = upper_boundries + (1.5 * IQR)

            # original data iqr apply
            iqr_clean_data_original = self.data.loc[(self.data[col]>=lower_mix_boundries) &
            (self.data[col]<=upper_mix_boundries),col].values
            
            # transform data iqr        
            lower_boundries,middle_boundries,upper_boundries = np.percentile(self.data_copy[col],[25,50,100])
            IQR = upper_boundries - lower_boundries
            lower_mix_boundries =  lower_boundries - (1.5 * IQR)
            upper_mix_boundries = upper_boundries + (1.5 * IQR)
            iqr_clean_data_transform = self.data_copy.loc[(self.data_copy[col]>=lower_mix_boundries) &
            (self.data_copy[col]<=upper_mix_boundries),col].values
            
            # Z-score method
            z_upper_limit_original = self.data[col].mean() + 3*self.data[col].std()
            z_lower_limit_original = self.data[col].mean() - 3*self.data[col].std()
            # original data z-score
            z_clean_data_original = self.data.loc[(self.data[col]>=z_lower_limit_original) &
            (self.data[col]<=z_upper_limit_original),col].values
            
            # transform data z-score
            z_upper_limit_transform = self.data[col].mean() + 3*self.data[col].std()
            z_lower_limit_transform = self.data[col].mean() - 3*self.data[col].std()
            
            z_clean_data_transofrm = self.data_copy.loc[(self.data_copy[col]>=z_lower_limit_transform) &
            (self.data_copy[col]<=z_upper_limit_transform),col].values

            # Graph
            # original data
            plt.subplot(rows,columns,plot_index)
            sns.histplot(self.data[col],kde=True)
            plt.title(f"Orginal data {col} Box plot")
            plot_index+=1
            
            plt.subplot(rows,columns,plot_index)
            plt.title(f"Orginal IQR data {col} Box plot")
            sns.boxplot(x=iqr_clean_data_original)
            plot_index+=1
            
            plt.subplot(rows,columns,plot_index)
            plt.title(f"Orginal Z-Score data {col} Box plot")
            sns.boxplot(x=z_clean_data_original)
            plot_index+=1
            
            # transform data
            plt.subplot(rows,columns,plot_index)
            plt.title(f"Transform data {col} Box plot")
            sns.histplot(self.data_copy[col],kde=True)
            plot_index+=1
            
            plt.subplot(rows,columns,plot_index)
            plt.title(f"Transform IQR data {col} Box plot")
            sns.boxplot(x=iqr_clean_data_transform)
            plot_index+=1
            
            plt.subplot(rows,columns,plot_index)
            plt.title(f"Transform Z-Score data {col} Box plot")
            sns.boxplot(x=z_clean_data_transofrm)
            plot_index+=1

            
            
        plt.tight_layout()
        plt.show()






# num = numerical_analysis()
# num.fit(data=train_data,
#         rows=len(numerical_col) // 3, 
#         col=3,
#         stragey='mean',
#         skew='yes',
#        scaler='boxcox')


# num.compareision_original_reduce_analysis()


# num.outlier_method()


# num.compareision_original_reduce_analysis()





# num = numerical_analysis()
# num.fit(data=test_data,
#         rows=len(test_data.select_dtypes(include=np.number).columns) // 3, 
#         col=3,
#         stragey='mean',
#         skew='yes',
#        scaler='boxcox')


# num.outlier_method()


from colorama import Fore,Back,Style
print(Fore.YELLOW+Back.BLACK,"Cox-Box preprocessing import to preprocessing using for final pipeline create")





def categorical_analysis(data):
    
    column_name = data.select_dtypes(include="O").columns
  
    for col in column_name:
        
        values = data[col].value_counts().reset_index()
        
        values.loc[data[col].value_counts().values/data.shape[0]*100<=5.0,col] = 'Other'
        values = values.groupby(col).sum().reset_index()

        graph = px.pie(values,
                     values='count',
                     names=col,
                    )
        
        graph.update_layout(height=250,
                            width=350,
                           title=dict(text=f'{col} category',
                                      x=0.5,
                                     font=dict(size=30,color='green',family='comic sans MS',weight=1000)))
        graph.show()
        


# categorical_analysis(train_data)


# categorical_analysis(test_data)


corr = train_data[numerical_col].corr()

mask = np.triu(np.ones_like(corr, dtype=bool))

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
sns.heatmap(
    corr,
    mask=mask,
    annot=True,
    fmt=".3f",
    cmap=sns.color_palette(["lightgrey"] + sns.color_palette("coolwarm", 256)),
    cbar=False
)


corr = test_data[test_data.select_dtypes(include=np.number).columns].corr()

mask = np.triu(np.ones_like(corr, dtype=bool))
plt.subplot(1,2,2)

sns.heatmap(
    corr,
    mask=mask,
    annot=True,
    fmt=".3f",
    cmap=sns.color_palette(["lightgrey"] + sns.color_palette("coolwarm", 256)),
    cbar=False
)
plt.tight_layout()
plt.show()
plt.close('all')


for col,val in zip(numerical_col,train_data[numerical_col].var()):
    print(f'{col}\t=>  {val:.4f}')


for col,val in zip(numerical_col,train_data[numerical_col].std()):
    print(f'{col}\t=>  {val:.4f}')





X = train_data.drop(columns="loan_paid_back",axis=1)
Y = train_data['loan_paid_back']


numerical_col = numerical_col[:-1]


numerical_col


categorical_col = train_data.select_dtypes(include='O').columns
categorical_col








from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder,StandardScaler,PowerTransformer,MinMaxScaler
from sklearn.impute import SimpleImputer 


numerical_pipeline = Pipeline(steps=[
    ("Missing_data_handler",SimpleImputer()),
    ("Numerical_pipeline",PowerTransformer(method='box-cox')),
    ("StandardScaling_pipeline",StandardScaler())
])
numerical_pipeline


categorical_pipeline = Pipeline(steps=[
    ("Encoding",OrdinalEncoder(handle_unknown="use_encoded_value",unknown_value=-1)),
    ("Numerical_scaleing",MinMaxScaler())
])
categorical_pipeline


preprocessing = ColumnTransformer(transformers=[
    ("numerical_handler",numerical_pipeline,numerical_col),
    ("categorical_handler",categorical_pipeline,categorical_col)
]).set_output(transform='pandas')
preprocessing


train_data_preprocessing_data = preprocessing.fit_transform(train_data)


train_data_preprocessing_data.shape





from sklearn.decomposition import PCA


testing_pca = PCA(random_state=42)
testing_pca


testing_pca.fit(train_data_preprocessing_data)


np.cumsum(testing_pca.explained_variance_ratio_)


plt.plot(pd.Series(np.cumsum(testing_pca.explained_variance_ratio_)))

n = 5
plt.axvline(n,c='green')
plt.text(n,np.cumsum(testing_pca.explained_variance_ratio_)[4]+0.02,f'{np.cumsum(testing_pca.explained_variance_ratio_)[n]:.2f}',rotation=45)
plt.scatter(n,np.cumsum(testing_pca.explained_variance_ratio_)[n],c='red')

plt.show()


final_pca = PCA(n_components=n,random_state=42).fit(train_data_preprocessing_data)


final_pca_apply_train_data = final_pca.transform(train_data_preprocessing_data)





pca_pipeline = Pipeline(steps=[
                        ("preprocessing",preprocessing),
                        ('pca',PCA(n_components=n,random_state=42))
                    ])
pca_pipeline





pca_pipeline.fit(train_data)





from sklearn.linear_model import (LogisticRegression,
                                    LassoCV,
                                    SGDClassifier)
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from cuml.svm import SVC as cuSVC
from sklearn.calibration import CalibratedClassifierCV


models_list = {
    "Logistic": LogisticRegression(),
    "LassoClassifier": SGDClassifier(loss="log_loss", penalty="l1"),
    "ElasticNetClassifier": SGDClassifier(loss="log_loss", penalty="elasticnet"),
    "SVC": cuSVC(probability=True),
    "RandomforestClassifier": RandomForestClassifier(),
    "DecisionTreeClassifier": DecisionTreeClassifier(),
    "XGBClassifier": xgb.XGBClassifier(eval_metric="logloss")
}



parameter_tuning = {
    "Logistic": {
        "C":[0.01,0.1,1,10],
        "solver":['lbfgs','liblinear'],
        "max_iter":[500,1000,2000],
        "tol":[1e-4, 1e-3]
    },
    "LassoClassifier":{
        "alpha":[0.0001, 0.001, 0.01, 0.1],
        "max_iter":[500,1000],
        "tol":[1e-4, 1e-3]
    },
    "ElasticNetClassifier":{
        "alpha":[0.0001,0.001,0.01],
        "l1_ratio":[0.2,0.5,0.8],
        "max_iter":[500,1000],
        "tol":[1e-4,1e-3]
    },
    "SVC":{
        "kernel":["linear","rbf"],
        "gamma":["scale","auto"]
    },
    "RandomforestClassifier":{
        "n_estimators":[10,50],
        "max_depth":[5,10]
    },
    "DecisionTreeClassifier":{
        "criterion":["gini","entropy"],
        "max_depth":[2,5,10]
    },
    "XGBClassifier":{
        "learning_rate":[0.01,0.1],
        "max_depth":[2,4,6],
        "n_estimators":[25,50],
        "n_jobs":[-1]
    }
}


from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV,StratifiedKFold,RandomizedSearchCV


final_pca_apply_train_data = pca_pipeline.transform(train_data)
final_pca_apply_train_data.shape,final_pca_apply_train_data.min(),final_pca_apply_train_data.max()


import warnings
from sklearn.exceptions import ConvergenceWarning





import warnings, os, sys

warnings.filterwarnings("ignore")  # hide all Python warnings

# redirect sklearn / cuML console output to nowhere
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, "w")
        sys.stderr = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr



model_analysis = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\nğŸš€ Running Hyperparameter Optimization...\n")

for model_name, model_obj in models_list.items():

    print(f"ğŸ”� Tuning â†’ {model_name}")
    params = parameter_tuning[model_name]

    search = HalvingRandomSearchCV(
        estimator=model_obj,
        param_distributions=params,
        scoring="accuracy",
        cv=cv,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore")

        search.fit(final_pca_apply_train_data, Y)

    best_model = search.best_estimator_

    # Calibration required for probability output:
    if model_name in ["SVC", "LassoClassifier", "ElasticNetClassifier"]:
        print(f"âš™ Applying Calibration to: {model_name}")
        best_model = CalibratedClassifierCV(best_model, cv=3, method="sigmoid")
        best_model.fit(final_pca_apply_train_data, Y)

    model_analysis[model_name] = {
        "best_score": round(search.best_score_, 4),
        "best_params": search.best_params_,
        "final_model": best_model
    }


for model_name,info in model_analysis.items():

    print(f"\n Model is {model_name}")

    print("CV best score:-",info['best_score'])
    print("Best parameter:-",info['best_params'])
    print("Final model:-",info['final_model'])





from sklearn import metrics


def parameter_tuner(dataset, target=None):

    results = {}
    pipeline_data = pca_pipeline.transform(dataset)
    model_with_params = []

    def cross_validation(model):
        folds = 3 if model == "SVC" else 5
        return round(np.mean(cross_val_score(model, pipeline_data, target, cv=folds)), 4)
    
    
    for model_name, params in model_analysis.items():

        print(f"\nğŸ�� Evaluating Final Model â†’ {model_name}")

        # Load the FINAL pretrained model (calibrated where necessary)
        model = params["final_model"]

        
        # Ensure model is fitted (safety)
        try:
            _ = model.predict(pipeline_data[:2])
        except:
            model.fit(pipeline_data, target)

        model_with_params.append((model_name, params))

        # ---------- Universal Probability Logic ----------
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(pipeline_data)[:, 1]

        elif hasattr(model, "decision_function"):
            raw = model.decision_function(pipeline_data)
            prob = MinMaxScaler().fit_transform(raw.reshape(-1, 1)).flatten()

        else:
            raw = model.predict(pipeline_data)
            prob = MinMaxScaler().fit_transform(raw.reshape(-1, 1)).flatten()

        predict = (prob >= 0.5).astype(int)

        # ---------- Metrics ----------
        train_score = model.score(pipeline_data, target)
        cv_score = cross_validation(model)

        results[model_name] = {
            "train_score": train_score,
            "cross_validation": cv_score,
            "accuracy": metrics.accuracy_score(target, predict),
            "f1_score": metrics.f1_score(target, predict),
            "confusion_matrix": metrics.confusion_matrix(target, predict),
            "AUC": metrics.roc_auc_score(target, prob),
            "overfit_gap": abs(train_score - cv_score)
        }

    results_df = pd.DataFrame(results).T.reset_index(names="model")
    return results_df, model_with_params



parameter_tuning , model_with_parameter_tuple = parameter_tuner(train_data,train_data.iloc[:,-1])


parameter_tuning





parameter_tuning_accuracy_analysis = parameter_tuning.melt(id_vars='model')
parameter_tuning_accuracy_analysis = parameter_tuning_accuracy_analysis.loc[~(parameter_tuning_accuracy_analysis['variable']=='confusion_matrix')]
parameter_tuning_accuracy_analysis['value'] = pd.to_numeric(parameter_tuning_accuracy_analysis['value'],
                                                           errors='coerce').astype(np.float32)
parameter_tuning_accuracy_analysis['variable'].unique()


parameter_tuning_accuracy_analysis.head() , parameter_tuning_accuracy_analysis['variable'].unique()





plt.style.use("seaborn-v0_8-darkgrid")

plt.figure(figsize=(10,5))

plt.title("Machine learning model analysis best parameter tuning")
sns.barplot(
    data=parameter_tuning_accuracy_analysis,
    x='model',
    y='value',
    hue='variable'
)
plt.xticks(rotation=-90)
plt.legend(bbox_to_anchor=(1.01,1))
plt.show()






plt.figure(figsize=(15,7))
for i,model in enumerate(models_list.keys()):
    temp = parameter_tuning.loc[parameter_tuning['model']==model,'confusion_matrix'].values[0]
    plt.subplot(4,2,i+1)
    plt.title("Model :-{}".format(model))
    sns.heatmap(temp,annot=True,fmt='.2f',cbar=False)
    plt.xlabel("prediction")
    plt.ylabel("actual")

plt.tight_layout()
plt.show()





from sklearn.ensemble import VotingClassifier


# Extract key metrics
metrics_df = parameter_tuning.set_index("model")[["AUC", "f1_score", "cross_validation"]]

# Normalize each metric (0-1 scale)
normalized = (metrics_df - metrics_df.min()) / (metrics_df.max() - metrics_df.min())

# Weighted combination (you can adjust ratios)
combined_score = (0.5 * normalized["AUC"] +
                  0.3 * normalized["f1_score"] +
                  0.2 * normalized["cross_validation"])

# Convert to weights list
weights = combined_score.round(2).replace(0, 0.1).tolist()

print(weights, sum(weights),combined_score)



final_estimators = [
    (name, details["final_model"])
    for name, details in model_with_parameter_tuple
]
final_estimators


voteing_mechinisim = VotingClassifier(estimators=final_estimators,
                                      voting='soft',
                                      weights=weights
                                     )
voteing_mechinisim


ML_PIPELINE_FLOW = Pipeline(steps=[
    ("stage-1",pca_pipeline),
    ("stage-2",voteing_mechinisim)
])
ML_PIPELINE_FLOW


ML_PIPELINE_FLOW.fit(X,Y)





import joblib


joblib.dump(ML_PIPELINE_FLOW,"ML_Depoly_Pipeline.pkl")





model_pipeline = joblib.load("/kaggle/working/ML_Depoly_Pipeline.pkl")








prediction_sample_data = train_data.sample(10)
prediction_sample_data


prediction_sample = model_pipeline.predict(prediction_sample_data)
prediction_sample


probility_sample = model_pipeline.predict_proba(prediction_sample_data)
probility_sample





full_train_data_prediction = model_pipeline.predict(train_data)
full_train_data_prediction.shape







