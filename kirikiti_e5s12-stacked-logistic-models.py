# Clonar el repositorio
!git clone https://github.com/Kirikiti/Kaggle.git

import os
import shutil

ruta_inicial = os.getcwd()
print(f'Ruta Inicial: {ruta_inicial}')

# Cambiar al directorio del repositorio
os.chdir('Kaggle/modelos') #He importado todo el repositorio en Output de kaggle, navego hasta la carpeta deseada

# Importar la fichero py
import LogReg_ki as lrk

#Volver a ruta incial
os.chdir(ruta_inicial)

#Borrar repositorio clonado
shutil.rmtree("Kaggle")
print(f'Repositorio Kaggle de gitHub borrado')


#--------------------------------------------------------------------------------
#   CONFIGURATION
#--------------------------------------------------------------------------------

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from category_encoders import TargetEncoder

from sklearn.linear_model import LogisticRegression, RANSACRegressor
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.model_selection import train_test_split

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

from xgboost import XGBClassifier

import warnings
warnings.filterwarnings('ignore')


test = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")
train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv", index_col='id')
df_sub = pd.read_csv('/kaggle/input/playground-series-s5e12/sample_submission.csv')


class XGBsLog_ki():
    def __init__(self, train_df, submit_df, target):
        self.train_df = train_df
        self.submit_df = submit_df
        self.TARGET = target
        self.CAT = train_df.select_dtypes(include=['object','category','bool']).columns.tolist()
        self.NUM = [col for col in train_df.select_dtypes(exclude=['object','category','bool']).columns.tolist() if col != target]
        self.FEATURES_0 = [col for col in train_df.columns.tolist() if col != target]
        self.FEATURES_1 = [] #Multicolineal clean
        self.FEATURES_2_in = []
        self.FEATURES_2_out = []
        self.train_df_enc = pd.DataFrame()
        self.submit_df_enc = pd.DataFrame()
        self.train_df_std = pd.DataFrame()
        self.submit_df_std = pd.DataFrame()
        self.submit_df_prd = pd.DataFrame()
        self.train_df_rns = pd.DataFrame()
        self.train_df_in = pd.DataFrame()
        self.train_df_out = pd.DataFrame()
        self.summary_pvalue_in = pd.DataFrame()
        self.summary_pvalue_out = pd.DataFrame()
        self.model_in = LogisticRegression(max_iter=100, C=1, class_weight={0: 1, 1: 1}, penalty='l2', solver='lbfgs')
        self.model_out = LogisticRegression(max_iter=100, C=1, class_weight={0: 1, 1: 1}, penalty='l2', solver='lbfgs')
        self.submitPredictions = []
        self.clf_xgb = XGBClassifier(max_depth=8,
            learning_rate=0.1,
            n_estimators=1000,
            verbosity=0,
            silent=None,
            objective='binary:logistic',
            booster='gbtree',
            n_jobs=-1,
            nthread=None,
            gamma=0,
            min_child_weight=1,
            max_delta_step=0,
            subsample=0.7,
            colsample_bytree=1,
            colsample_bylevel=1,
            colsample_bynode=1,
            reg_alpha=0,
            reg_lambda=1,
            scale_pos_weight=1,
            base_score=0.5,
            random_state=0,
            seed=None,)

    def metodes(self):
        print(f' printCAT() \n extraSubmitCAT() \n dataProcessing() \n cleanFeatures1() \n')
        print(f' cleanOutliers() \n cleanFeatures2() \n modelFit() \n submitPredict \n')
        print(f' classFit()')

    
    def calculate_vif(self, X):
        vif_data = pd.DataFrame()
        vif_data["feature"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        return vif_data
  

    def printCAT(self):
        print(f'UNIQUES VALUES BY CATEGORICAL FEATURES \n' + '-'*40)
        for cat_feature in self.CAT:
            unique_values = self.train_df[cat_feature].unique()
            print(f'Unique Values in {cat_feature}: ')
            print(f'{unique_values} \n')

    def extraSubmitCAT(self):
        print(f'EXTRA uniques values by categorical features \n in submit Data frame \n' + '-'*40)
        for cat_feature in self.CAT:
            extra_unique_values = list(set(self.submit_df[cat_feature].unique()) - set(self.train_df[cat_feature].unique()))
            print(f'Extra Unique Values in {cat_feature}: ')
            print(f'{extra_unique_values} \n')

    def dataProcessing(self):
        # Trade Encodif for Categorical features
        T_enc = TargetEncoder(cols=self.CAT)
        print('Encoding Training Data Frame')
        self.train_df_enc[self.CAT] = T_enc.fit_transform(self.train_df[self.CAT], self.train_df[self.TARGET])
        self.train_df_enc[self.NUM] = self.train_df[self.NUM]
        print('✅')
        print('Encoding Submit Data Frame')
        self.submit_df_enc[self.CAT] = T_enc.transform(self.submit_df[self.CAT])
        self.submit_df_enc[self.NUM] = self.submit_df[self.NUM]
        print('✅')

        train_df_std = self.train_df[self.TARGET]
        
        print('Standardizing Training Data Frame')
        # Standardize only the features
        scaler = StandardScaler()
        standardized_features = pd.DataFrame(scaler.fit_transform(self.train_df_enc), columns=self.train_df_enc.columns)

        # Combine the standardized features with the target variable
        self.train_df_std = pd.concat([standardized_features, train_df_std.reset_index(drop=True)], axis=1)
        print('✅')

        print('Standardizing Submit Data Frame')
        # Standardize the new dataset
        self.submit_df_std = pd.DataFrame(scaler.transform(self.submit_df_enc), columns=self.submit_df_enc.columns)
        print('✅\n')

    def cleanFeatures1(self):
        print(f'Looking for Multicollinearity')
        # Iterate through the VIF values and remove ONLY the highest one, 
        # and once it is removed, recalculate them to see if any other features need to be eliminated.
        X = self.submit_df_std[self.FEATURES_0]
        while True:
            vif_data = self.calculate_vif(X)
            max_vif = vif_data['VIF'].max()
            
            if max_vif < 5.0:
                break
            
            feature_to_remove = vif_data.loc[vif_data['VIF'].idxmax(), 'feature']
            print(f'Removing feature {feature_to_remove} with VIF: {max_vif}\n')
            X = X.drop(columns=[feature_to_remove])
            
        self.FEATURES_1 = np.array(X.columns)

    def cleanOutliers(self, n=5):
        print('Cleaning Outlier form train Data Frame')

        df = self.train_df_std.copy()
        df['inlier'] = False
        all_inlier_mask = np.zeros(df.shape[0], dtype=bool)

        base_model = LogisticRegression()

        for i in range(n):
            print(f'Interation Numer: {i}')
            model = RANSACRegressor(base_model, random_state=42 + i, min_samples=0.5)
            model.fit(df[self.FEATURES_1], df[self.TARGET].astype(int))  # Fit the model
            inlier_mask = model.inlier_mask_
        
            # Update the overall inlier mask
            all_inlier_mask |= inlier_mask  # Combine current inliers with previous ones

        df.loc[all_inlier_mask, 'inlier'] = True
        
        outlier_per= 1 - (df[df['inlier']==True].shape[0] / df.shape[0])
              
        self.train_df_rns = df
        self.train_df_in = self.train_df_rns[self.train_df_rns.inlier==True]
        self.train_df_out = self.train_df_rns[self.train_df_rns.inlier==False]
        
        self.train_df_in = self.train_df_in.drop(columns=['inlier'])
        self.train_df_out = self.train_df_out.drop(columns=['inlier'])

        # Give a name to this datafames
        self.train_df_in.name = 'in'
        self.train_df_out.name = 'out'
        
        df.iloc[:0] #Clean df
        
        print(f'Volume of outliers = {outlier_per*100:.2f}%  ✅\n')

    def cleanFeatures2_code(self,df):
        
        sub_df = df.name # Inlier oder Oulier

        print(sub_df)
        #Clean form Model not significat features
        
        print('Looking for significat features \n')
        
        # CREATE AND TRAIN LOG MODEL (STATMODELS)

        X = df[self.FEATURES_1]
        y = df[self.TARGET].astype(int) 
        
        # Add a constant for the model
        X = sm.add_constant(X)

        # Fit the logistic regression model
        logit_model = sm.Logit(y, X)
        result = logit_model.fit(method='lbfgs', maxiter=100, disp=True)
        # method='newton'
        # method='lbfgs' <-- Usado por defecto por RANDSAC
        # disp=True --> Show information about onvergence process
        

        summary = result.summary()

        # Convert the summary to a DataFrame
        summary_pvalue = pd.DataFrame(summary.tables[1].data[1:], columns=summary.tables[1].data[0])
        if sub_df == 'in':
            self.summary_pvalue_in = summary_pvalue
        else:
            self.summary_pvalue_out = summary_pvalue
            
        # Ensure the p-value column is numeric --> in order to can filter p_values < 0.05
        if  sub_df == 'in':
            self.summary_pvalue_in['P>|z|'] = pd.to_numeric(self.summary_pvalue_in['P>|z|']) 
        else:
            self.summary_pvalue_out['P>|z|'] = pd.to_numeric(self.summary_pvalue_out['P>|z|']) 

        # Filter significant and insignificant features
        # Return field with feature name defined by Summary like ['']
        if  sub_df == 'in':
            significant_features = self.summary_pvalue_in[self.summary_pvalue_in['P>|z|'] < 0.05][''].values
            insignificant_features = self.summary_pvalue_in[self.summary_pvalue_in['P>|z|'] >= 0.05][''].values
        else:
            significant_features = self.summary_pvalue_out[self.summary_pvalue_in['P>|z|'] < 0.05][''].values
            insignificant_features = self.summary_pvalue_out[self.summary_pvalue_in['P>|z|'] >= 0.05][''].values
            
        # Convert to numpy arrays if necessary
        significant_features_array = np.array([f for f in significant_features if f != 'const'])
        insignificant_features_array = np.array(insignificant_features)

        if  sub_df == 'in':
            self.FEATURES_2_in = significant_features_array
        else:
            self.FEATURES_2_out = significant_features_array

        print(f'Significant Features Analysis for: "{sub_df}liers"')
        print("Significant Features:", significant_features_array)
        print("\n")
        print("Insignificant Features:", insignificant_features_array)
        print("\n")

    def cleanFeatures2(self):
        self.cleanFeatures2_code(self.train_df_in)
        self.cleanFeatures2_code(self.train_df_out)

    def modelFit_code(self,df):

        sub_df = df.name # Inlier oder Oulier
        
        print(f'Training Logistic Model: {sub_df}lier')
        
        if sub_df == 'in':
            X = self.train_df_in[self.FEATURES_2_in]
            y = self.train_df_in[self.TARGET].astype(int)
            model = self.model_in
            fit_FEATURES_2=self.FEATURES_2_in
        else:
            X = self.train_df_out[self.FEATURES_2_out]
            y = self.train_df_out[self.TARGET].astype(int)
            model = self.model_out
            fit_FEATURES_2=self.FEATURES_2_in

        cv_scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
        print(f'✅ \n')
        
        print(f'Cross accuracy in Data frame "{sub_df.upper()}LIER" \nROC AUC: {cv_scores} \n')

        # Fit the model with all the data
        model.fit(X, y)

        # Full Train Data Frame, not only clean rows
        X_full = self.train_df_std[fit_FEATURES_2]
        y_full = self.train_df_std[self.TARGET].astype(int)
        
        # Make probability predictions with the new dataset X_full
        y_pred_prob = model.predict_proba(X_full)[:, 1]  # Probabilities of the positive class

        self.train_df_rns[f'y_{sub_df}']=y_pred_prob

        # Calculate the ROC AUC
        roc_auc = roc_auc_score(y_full, y_pred_prob)

        # Calculate the confusion matrix
        y_pred = (y_pred_prob >= 0.5).astype(int)  # Threshold of 0.5 for classification
        conf_matrix = confusion_matrix(y_full, y_pred)

        # Display the result
        print(f'Accuracy over full Training Data Frame')
        print(f'ROC AUC score: {roc_auc} \n')

        # Display the results
        print('Confusion Matrix:')
        print(conf_matrix)
        print('\n')

    def metaFit(self):

        print('Trainin Meta Model \n')
        
        target = 'inlier'

        trainXGB = myLogReg.train_df_rns.copy()

        X = trainXGB.drop([target,self.TARGET], axis=1)  
        y = trainXGB[target].astype(int)

        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)

        X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
        
        self.clf_xgb.fit(X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        early_stopping_rounds=40,
        verbose=10)

        # Make probability predictions with the new dataset X_full
        y_pred_prob = np.array(self.clf_xgb.predict_proba(X))[:, 1]  # Probabilities of the positive class
        self.train_df_rns['meta_preds'] = y_pred_prob

        preds_xgb_valid = np.array(self.clf_xgb.predict_proba(X_valid))
        valid_auc = roc_auc_score(y_score=preds_xgb_valid[:,1], y_true=y_valid)
        print('\n ')
        print(f'Roc Auc Meta model VALID: {valid_auc}')

        preds_xgb_test = np.array(self.clf_xgb.predict_proba(X_test))
        test_auc = roc_auc_score(y_score=preds_xgb_test[:,1], y_true=y_test)
        print(f'Roc Auc Meta model TEST: {test_auc}')

        print(f'✅ \n')

        

    def modelFit(self):
        self.modelFit_code(self.train_df_in)
        self.modelFit_code(self.train_df_out)
        self.train_df_rns['out-in'] = self.train_df_rns['y_out'] - self.train_df_rns['y_in']
        self.metaFit()
        self.train_df_rns['y_preds'] = (self.train_df_rns['y_in'] * self.train_df_rns['meta_preds'])+(self.train_df_rns['y_out'] * (1-self.train_df_rns['meta_preds']))

        #ACCURACY OF THE STACKED MODEL
        print(' ### ACCURACY OF THE STACKED MODEL ###')
        # Calculate the ROC AUC --> Full Train Dataframe
        roc_auc = roc_auc_score(self.train_df_rns[self.TARGET], self.train_df_rns['y_preds'])

        print(f'ACCURACY OVER FULL TRAIN DATA \n' + '-'*40)
        print(f'ROC AUC score: {roc_auc} \n')

        # Define the target variable and the predictions
        y_true = self.train_df_rns[self.TARGET]
        y_pred = self.train_df_rns['y_preds']

        # Initialize StratifiedKFold to split the DataFrame into 10 parts
        skf = StratifiedKFold(n_splits=10)
        roc_auc_scores = []

        # Cross-validation
        for train_index, test_index in skf.split(self.train_df_rns, y_true):
            # Take the test indices
            y_true_test = y_true.iloc[test_index]
            y_pred_test = y_pred.iloc[test_index]

            # Calculate the ROC AUC for this division
            roc_auc = roc_auc_score(y_true_test, y_pred_test)
            roc_auc_scores.append(roc_auc)

        # Display the scores for each division
        print(f'ACCURACY BY FOLD \n' + '-'*40)
        for i, score in enumerate(roc_auc_scores, 1):
            print(f'Fold {i}: ROC AUC score: {score:.4f}')

        # Calculate the mean ROC AUC
        mean_roc_auc = sum(roc_auc_scores) / len(roc_auc_scores)
        print(f'Mean ROC AUC score: {mean_roc_auc:.4f}')
        

    def submitPredict_colde(self,sub_df):
        
        print(f'Caculating submitValues model #{sub_df.upper()}#')

        if sub_df == 'in':
            model = self.model_in
            fit_FEATURES_2=self.FEATURES_2_in
        else:
            model = self.model_out
            fit_FEATURES_2=self.FEATURES_2_in

        X = self.submit_df_std[fit_FEATURES_2]

        y_pred_prob = model.predict_proba(X)[:, 1]  # Probabilities of the positive class

        # Save model Prediction
        self.submit_df_prd[f'y_{sub_df}'] = y_pred_prob
        
        print(f'✅ \n')


    def metaPredictPredict(self):
        print(f'Caculating submitValues for Meta Model')
        X_submit = self.submit_df_prd.copy()
        y_pred_submit = np.array(self.clf_xgb.predict_proba(X_submit))

        self.submit_df_prd['meta_preds'] = y_pred_submit[:, 1]
        print(f'✅ \n')
    
        
    def submitPredict(self):
        self.submit_df_prd = self.submit_df_std.copy()
        self.submitPredict_colde('in')
        self.submitPredict_colde('out')
        self.submit_df_prd['out-in'] = self.submit_df_prd['y_out'] - self.submit_df_prd['y_in']
        self.submitPredictions = self.submit_df_prd['y_in']
        self.metaPredictPredict()
        self.submit_df_prd['y_preds'] = (self.submit_df_prd['y_in'] * self.submit_df_prd['meta_preds']) + (self.submit_df_prd['y_out'] * (1-self.submit_df_prd['meta_preds']))

    def classFit(self):
        self.printCAT()
        self.extraSubmitCAT()
        self.dataProcessing()
        self.cleanFeatures1()
        self.cleanOutliers()
        self.cleanFeatures2()
        self.modelFit()
        self.submitPredict()



target = 'diagnosed_diabetes'
myLogReg = XGBsLog_ki(train,test,target)


myLogReg.metodes()


print(myLogReg.NUM)


myLogReg.printCAT()


#myLogReg.extraSubmitCAT()
myLogReg.dataProcessing()


myLogReg.cleanFeatures1()


myLogReg.FEATURES_1


myLogReg.cleanOutliers(35)


# Save Inlier Outlier list like a binary file
df_inlier=myLogReg.train_df_rns.copy()
df_inlier['index']=df_inlier.index
inlierArray=df_inlier[['index','inlier']].to_numpy
np.save('E5S12inlier.npy', inlierArray)


myLogReg.cleanFeatures2()


myLogReg.modelFit()


myLogReg.submitPredict()


def xgbSwitchThreshold(df):
    # Lista para almacenar los resultados
    roc_auc_scores = []

    # Iterar a través de 102 umbrales
    for threshold in [i / 100 for i in range(101)]:
        # Agregar predicciones basadas en el umbral actual
        df['y_preds'] = df.apply(
            lambda row: row['y_in'] if row['meta_preds'] > threshold else row['y_out'],
            axis=1
        )

        # Calcular el ROC AUC
        roc_auc = roc_auc_score(df['diagnosed_diabetes'], df['y_preds'])

           
        # Almacenar el umbral y su correspondiente ROC AUC
        roc_auc_scores.append((threshold, roc_auc))

        print(f'Threshold: {threshold:.2f}, ROC AUC score: {roc_auc:.4f}')




#xgbSwitchThreshold(myLogReg.train_df_rns)


metaTarget = 'inlier_old'

train_meta = myLogReg.train_df_rns.copy()
submit_meta = myLogReg.submit_df_prd.copy()

train_meta['inlier_old']=train_meta['inlier'].astype(int)
train_meta = train_meta.drop(columns = [myLogReg.TARGET,'inlier'])

metaLog = lrk.LogReg_ki(train_meta,submit_meta,metaTarget)


metaLog.classFit()


def crossVal(df, cv_target, cv_y_preds):

    # Define the target variable and the predictions
    y_true = df[cv_target]
    y_pred = df[cv_y_preds]
    
    # Initialize StratifiedKFold to split the DataFrame into 10 parts
    skf = StratifiedKFold(n_splits=10)
    roc_auc_scores = []

    # Cross-validation
    for train_index, test_index in skf.split(df, y_true):
        # Take the test indices
        y_true_test = y_true.iloc[test_index]
        y_pred_test = y_pred.iloc[test_index]

        # Calculate the ROC AUC for this division
        roc_auc = roc_auc_score(y_true_test, y_pred_test)
        roc_auc_scores.append(roc_auc)

    # Display the scores for each division
    print(f'ACCURACY BY FOLD \n' + '-'*40)
    for i, score in enumerate(roc_auc_scores, 1):
        print(f'Fold {i}: ROC AUC score: {score:.4f}')

    # Calculate the mean ROC AUC
    mean_roc_auc = sum(roc_auc_scores) / len(roc_auc_scores)
    print(f'Mean ROC AUC score: {mean_roc_auc:.4f}')


metaTrain = metaLog.train_df.copy()

# Full Train Data Frame, not only clean rows
X_full = metaLog.train_df_std[metaLog.FEATURES_2]
y_full = metaLog.train_df_std[metaLog.TARGET].astype(int)
        
# Make probability predictions with the new dataset X_full
y_pred_prob = metaLog.model.predict_proba(X_full)[:, 1]  # Probabilities of the positive class

metaTrain['metaLog_preds'] = y_pred_prob
metaTrain[myLogReg.TARGET] = myLogReg.train_df_std[myLogReg.TARGET]
metaTrain['yLog_preds'] = (metaTrain['y_in'] * metaTrain['metaLog_preds']) + (metaTrain['y_out'] * (1 - metaTrain['metaLog_preds']))




crossVal(metaTrain, myLogReg.TARGET, 'yLog_preds')


metaSub = metaLog.submit_df.copy()
metaSub['metaLog_preds'] = metaLog.submitPredictions
metaSub['yLog_preds'] = (metaSub['y_in'] * metaSub['metaLog_preds']) + (metaSub['y_out'] * (1 - metaSub['metaLog_preds']))


metaTrain = myLogReg.train_df_rns.copy()

X_full = metaLog.train_df_std[metaLog.FEATURES_2]
metaTrain['meta_preds'] = metaLog.model.predict_proba(X_full)[:, 1]

#xgbSwitchThreshold(metaTrain)


import matplotlib.pyplot as plt

metaSub['yLog_preds'].hist(bins=2, edgecolor='black')

# Añadir etiquetas y título
plt.title('Histograma de yLog_preds')
plt.xlabel('Clase')
plt.ylabel('Frecuencia')
plt.xticks([0, 1])

# Mostrar el gráfico
plt.show()


best_threshold = 0.01
metaSub['yLog_preds_th'] = metaSub.apply(
    lambda row: row['y_in'] if row['metaLog_preds'] > best_threshold else row['y_out'],
    axis=1
)


# SUB FOR META_XGB
df_sub[target] = myLogReg.submit_df_prd['y_preds']

# SUB FOR META_LOGISTIC
#df_sub[target] = metaSub['yLog_preds_th']

df_sub.to_csv('test_stackLogist_MetaXGB.csv', index=False)

