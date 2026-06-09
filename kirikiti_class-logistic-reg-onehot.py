#--------------------------------------------------------------------------------
#   CONFIGURATION
#--------------------------------------------------------------------------------

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from category_encoders import TargetEncoder

from sklearn.linear_model import LogisticRegression, RANSACRegressor
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import cross_val_score

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

import warnings
warnings.filterwarnings('ignore')


test = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")
train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv", index_col='id')
df_sub = pd.read_csv('/kaggle/input/playground-series-s5e12/sample_submission.csv')


class LogReg_ki():
    def __init__(self, train_df, submit_df, target):
        self.train_df = train_df
        self.submit_df = submit_df
        self.TARGET = target
        self.CAT = train_df.select_dtypes(include=['object','category','bool']).columns.tolist()
        self.NUM = [col for col in train_df.select_dtypes(exclude=['object','category','bool']).columns.tolist() if col != target]
        self.FEATURES_0 = []
        self.FEATURES_1 = [] #Multicolineal clean
        self.FEATURES_2 = []
        self.train_df_enc = pd.DataFrame()
        self.submit_df_enc = pd.DataFrame()
        self.train_df_std = pd.DataFrame()
        self.submit_df_std = pd.DataFrame()
        self.train_df_clean = pd.DataFrame()
        self.summary_pvalue = pd.DataFrame()
        self.model = LogisticRegression(max_iter=100, C=1, class_weight={0: 1, 1: 1}, penalty='l2', solver='lbfgs')
        self.submitPredictions = []

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
        #In this way, I can create the correct encoding without missing labels
        #for test_to_submit

        #train_df, submit_df = self.train_df.copy(), self.submit_df.copy()

        nunique = self.train_df.nunique()
        types = self.train_df.dtypes

        print('Creating New Categorical Features')

        for col in self.train_df.columns:
            if (types[col] == 'object' or nunique[col] < 20) and col not in self.CAT and col != self.TARGET: # Implement  20 < unique[col] < 200 --> range max 20
                print(col, self.train_df[col].nunique())
                self.CAT.append(col)

       
        # One Hot Encodig for Categorical features
        oh_enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='first')
        print('\nEncoding Training Data Frame')
        encoded_features = oh_enc.fit_transform(self.train_df[self.CAT])
        encoded_df = pd.DataFrame(encoded_features, columns=oh_enc.get_feature_names_out(self.CAT))
        self.train_df_enc = pd.concat([encoded_df, self.train_df[self.NUM].reset_index(drop=True)], axis=1)
        print('✅')
        print('Encoding Submit Data Frame')
        encoded_submit_features = oh_enc.transform(self.submit_df[self.CAT])
        encoded_submit_df = pd.DataFrame(encoded_submit_features, columns=oh_enc.get_feature_names_out(self.CAT))
        columns=oh_enc.get_feature_names_out(self.CAT)
        self.submit_df_enc = pd.concat([encoded_submit_df, self.submit_df[self.NUM].reset_index(drop=True)], axis=1)
        print('✅')

        train_df_std = self.train_df[self.TARGET]
        
        print('\nStandardizing Training Data Frame')
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

        self.FEATURES_0 = [col for col in self.train_df_std.columns.tolist() if col != self.TARGET]
        self.NUM = [item for item in self.FEATURES_0 if item not in self.CAT]

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
        print('Cleaning Outliers from Train Data Frame')

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
        outlier_per = 1 - (df[df['inlier'] == True].shape[0] / df.shape[0])
    
        df = df[df['inlier'] == True]
   
        self.train_df_clean = df
        self.train_df_clean = self.train_df_clean.drop(columns=['inlier'])

        print(f'Volume of outliers = {outlier_per * 100:.2f}%  ✅\n')

    def cleanFeatures2(self):
        #Clean form Model not significat features
        
        print('Looking for significat features \n')
        
        # CREATE AND TRAIN LOG MODEL (STATMODELS)

        X = self.train_df_clean[self.FEATURES_1]
        y = self.train_df_clean[self.TARGET].astype(int) 
        
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
        self.summary_pvalue = pd.DataFrame(summary.tables[1].data[1:], columns=summary.tables[1].data[0])

        # Ensure the p-value column is numeric --> in order to can filter p_values < 0.05
        self.summary_pvalue['P>|z|'] = pd.to_numeric(self.summary_pvalue['P>|z|']) 

        # Filter significant and insignificant features
        # Return field with feature name defined by Summary like ['']
        significant_features = self.summary_pvalue[self.summary_pvalue['P>|z|'] < 0.05][''].values
        insignificant_features = self.summary_pvalue[self.summary_pvalue['P>|z|'] >= 0.05][''].values

        # Convert to numpy arrays if necessary
        significant_features_array = np.array([f for f in significant_features if f != 'const'])
        insignificant_features_array = np.array(insignificant_features)

        self.FEATURES_2 = significant_features_array

        print("Significant Features:", significant_features_array)
        print("\n")
        print("Insignificant Features:", insignificant_features_array)
        print("\n")

    def modelFit(self):

        print('Training Logistic Model')
        
        X = self.train_df_clean[self.FEATURES_2]
        y = self.train_df_clean[self.TARGET].astype(int)

        cv_scores = cross_val_score(self.model, X, y, cv=5, scoring='roc_auc')
        #self.model.fit(X, y)
        print(f'✅ \n')
        
        print(f'Cross accuracy in Data frame without Ouliers \nROC AUC: {cv_scores} \n')

        # Fit the model with all the data
        self.model.fit(X, y)

        # Full Train Data Frame, not only clean rows
        X_full = self.train_df_std[self.FEATURES_2]
        y_full = self.train_df_std[self.TARGET].astype(int)
        
        # Make probability predictions with the new dataset X_full
        y_pred_prob = self.model.predict_proba(X_full)[:, 1]  # Probabilities of the positive class

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

    def submitPredict(self):
        print(f'Caculating submitValues')

        X = self.submit_df_std[self.FEATURES_2]

        y_pred_prob = self.model.predict_proba(X)[:, 1]  # Probabilities of the positive class
        self.submitPredictions = y_pred_prob
        print(f'✅ \n')

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
myLogReg = LogReg_ki(train,test,target)


myLogReg.metodes()


print(myLogReg.NUM)


myLogReg.printCAT()


#myLogReg.extraSubmitCAT()
myLogReg.dataProcessing()


print(myLogReg.FEATURES_0)


myLogReg.printCAT()


myLogReg.cleanFeatures1()


myLogReg.FEATURES_1


myLogReg.cleanOutliers(35)


clean_df = myLogReg.train_df_clean.copy()
clean_df['index'] = clean_df.index
clean_df.to_csv('train_df_clean.csv', index=False)


myLogReg.cleanFeatures2()


myLogReg.modelFit()


myLogReg.submitPredict()


df_sub[target] = myLogReg.submitPredictions
df_sub.to_csv('test_logistic_oneHot.csv', index=False)
df_sub.head()

