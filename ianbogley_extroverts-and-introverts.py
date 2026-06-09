#Import packages
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt, warnings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
import tensorflow as tf
from xgboost import XGBClassifier
import xgboost as xgb
warnings.filterwarnings("ignore")


#Import original data
train_full = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")


#Train test split to get a validation dataset for model training
X = train_full.iloc[:,1:8]
y = train_full.iloc[:,8]
X_train, X_val, y_train, y_val = train_test_split(X,y,test_size = .3,random_state=777)
X_test = test.iloc[:,1:8]
#Turn boolean (yes/no) cols into 1/0
num_cols = X_train.select_dtypes('number').columns
cat_cols = X_train.select_dtypes('object').columns
X_train[cat_cols] = X_train.loc[:,cat_cols].replace({'Yes':1,'No':0})
X_val[cat_cols] = X_val.loc[:,cat_cols].replace({'Yes':1,'No':0})
X_test[cat_cols] = X_test.loc[:,cat_cols].replace({'Yes':1,'No':0})
#output variable to 1/0 (1 is introverted)
y_train_encoded = y_train.replace({'Introvert':1,'Extrovert':0}).astype(int)


print("Original data provided (train.csv): ",train_full.shape)
print("Data used to predict submissions (test.csv):",test.shape)
print("Rows in training data:",len(X_train))
print("Rows in validation data:",len(X_val))


train_full.head(3)


y_train.value_counts()


#Categorical variables
train_catdescription = X_train.loc[:,cat_cols].describe()
train_catdescription.loc['missing']=X_train.isna().sum()
train_catdescription


#Numeric variables
train_numdescription = X_train.loc[:,num_cols].describe()
train_numdescription.loc["missing"] = X_train.isna().sum()
train_numdescription.loc["unique"] = X_train.nunique()
train_numdescription.apply(round).astype(int).applymap(lambda x: f"{x:,}")


fig,ax = plt.subplots(1,len(num_cols),figsize = (10,2))

for i in range(len(num_cols)):
    temp_col = num_cols[i]
    temp_data_grouped =[X_train[y_train== g][temp_col] 
                        for g in ['Introvert','Extrovert']]
    ax[i].hist(temp_data_grouped, color = ['blue','red'],stacked = True)
    ax[i].set_title(temp_col,fontsize = 10)
fig.tight_layout()
legend_handles = [
    Patch(color='blue', label='Introvert'),
    Patch(color='red', label='Extrovert')
]
fig.legend(handles=legend_handles, title='Personality', loc='lower center',
          bbox_to_anchor=(.5,-.25))
plt.show()


fig, ax = plt.subplots(1,len(cat_cols),figsize = (10,2))
for i in range(len(cat_cols)):
    cat_col = cat_cols[i]
    crosstab = pd.crosstab(X_train[cat_col], y_train)

    # Plot stacked bar chart
    crosstab.plot(kind='bar', stacked=True, figsize=(10, 3), edgecolor='black',
                 ax = ax[i], color = ('red','blue'))

    ax[i].set_title(cat_col)
    ax[i].legend(title='Personality')
plt.tight_layout()
plt.show()


#Quick linear regression: Given time spent alone above or below 4 hours, 
#what is the likelihood of being an introvert?
Missing = X_train['Time_spent_Alone'].isna()

X = (X_train[~Missing]['Time_spent_Alone']>4).astype(int).to_frame()
y = (y_train[~Missing]=="Introvert").astype(int)

tempmodel = LinearRegression().fit(X, y)

print("Likelihood of being an introvert, time spent alone less than or equal to 4 hours:",
      round(100*tempmodel.intercept_),"%")
print("Likelihood of being an introvert, time spent alone more than 4 hours:",
      np.round(100*(tempmodel.intercept_+tempmodel.coef_[0])).astype(int),"%")


#Get data grouped by target variable
temp_data_grouped =[X_train[y_train== g]['Time_spent_Alone']
                    for g in ['Introvert','Extrovert']]

#Stacked Barchart
plt.hist(temp_data_grouped, color = ['blue','red'],stacked = True)

#Text with % likelihood above and below threshold
plt.text(1.2,4000,f"Likelihood of\nIntroversion: {round(100*tempmodel.intercept_)}%")
plt.text(6,3000,f"Likelihood of\nIntroversion: {round(100*(tempmodel.intercept_+tempmodel.coef_[0]))}%")
plt.axvline(4.4,color = 'black')
plt.title('Time_spent_Alone',fontsize = 10)

#Legend
legend_handles = [
    Patch(color='blue', label='Introvert'),
    Patch(color='red', label='Extrovert')
]
plt.legend(handles=legend_handles, title='Personality', loc='upper right')
plt.show()


predictor_cols = X_train.columns
n_predictors = len(predictor_cols)

fig,ax = plt.subplots(n_predictors,n_predictors)

for row in range(n_predictors):
    for col in range(n_predictors):
        if row == 0:
            ax[row,col].set_title(X_train.columns[col])
        if col == 0:
            ax[row,col].set_ylabel(X_train.columns[row])

        #Get types of cols to determine what type of plot to create
        row_numeric = X_train.columns[row] in num_cols
        col_numeric = X_train.columns[col] in num_cols

        #histogram if comparing var with itself
        if row==col:
            ax[row,col].hist(pd.to_numeric(X_train.iloc[:,row]))
        #scatterplot for 2 numeric vars
        elif row_numeric&col_numeric:
            ax[row,col].scatter(X_train.iloc[:,col],X_train.iloc[:,row],alpha = .01)
        #for 1 numeric and 1 boolean, box and whisker plot
        elif row_numeric|col_numeric:
            #get the 2 columns, with the first being the boolean and the second the numeric
            tempdata = X_train.iloc[:,[row,col]].dropna().iloc[:,[int(row_numeric),int(col_numeric)]]
            #group vars into 2 arrays for boxplot
            tempdata_grouped = tempdata_grouped = [tempdata[tempdata.iloc[:,0]==0].iloc[:,1],tempdata[tempdata.iloc[:,0]==1].iloc[:,1]]
            ax[row,col].boxplot(tempdata_grouped,vert = row_numeric)
            if row_numeric:
                ax[row,col].set_xticks([2,1],['Yes','No'])
            else:
                ax[row,col].set_yticks([2,1],['Yes','No'])
        #Else: stacked barchart
        else:
            temp_crosstab = pd.crosstab(X_train.iloc[:,col],X_train.iloc[:,row])
            temp_crosstab.plot(kind = 'bar',stacked = True,ax = ax[row,col])
            legend_title = X_train.columns[row]
            if len(legend_title)>10:
                legend_title = legend_title[:7]+"..."
            handles,labels = ax[row,col].get_legend_handles_labels()
            ax[row,col].legend(handles,labels,title = legend_title)
            ax[row,col].set_xticks([True,False],['Yes','No'])
        if max(ax[row,col].get_yticks())>1000:
            ax[row,col].set_yticks([tick for tick in ax[row,col].get_yticks() if tick%1000==0])
            ax[row,col].yaxis.set_major_formatter(FuncFormatter(lambda x,_: f'{int(x/1000)}k' if x >= 1000 else f'{int(x)}'))
        
plt.tight_layout()
fig.set_size_inches(20,20)


group_col = 'Personality'

n_cols = len(predictor_cols)

# Layout for subplots (adjust rows if needed)
n_rows = (n_cols + 2) // 3  # 3 columns per row
fig, ax = plt.subplots(n_rows, 3, figsize=(15, 4 * n_rows))
ax = ax.flatten()

for i, col in enumerate(predictor_cols):

    # Counts by group for complete and missing
    complete_counts = y_train[~X_train[col].isna()].value_counts().sort_index()
    missing_counts = y_train[X_train[col].isna()].value_counts().sort_index()

    # Align groups
    all_groups = sorted(set(complete_counts.index).union(missing_counts.index))
    complete_counts = complete_counts.reindex(all_groups, fill_value=0)
    missing_counts = missing_counts.reindex(all_groups, fill_value=0)

    x = np.arange(len(all_groups))
    width = .35

    # Plot stacked bars
    ax[i].bar(x+width/2, missing_counts,width = width, label='Missing', color='salmon')
    ax[i].bar(x-width/2, complete_counts, width = width,label='Complete', color='skyblue')

    ax[i].set_title(col)
    ax[i].set_xticks(x)
    ax[i].set_xticklabels(all_groups, rotation=45)
    ax[i].set_ylabel("Count")

#Hide unused axes
for j in range(i + 1, len(ax)):
    fig.delaxes(ax[j])

# Shared legend and layout
handles, labels = ax[i].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right',bbox_to_anchor=(0.9, 0.87))
fig.suptitle(f'Distribution of {group_col} by Missingness in Predictors', fontsize=16)
fig.tight_layout(rect=[0, 0, 0.95, 0.95])  # leave space for legend and title
plt.show()


#Get dict of variables to impute by column
imputed_vars = {}

#Impute mean for numeric cols
for col in num_cols:
    imputed_vars[col] = X_train[col].mean()
#Impute mode for nonnumeric cols
for col in cat_cols:
    imputed_vars[col] = X_train[col].mode()[0]

#Impute data from imputed cols
for col in X_train.columns:
    X_train[col] = X_train[col].fillna(imputed_vars[col])
    X_val[col] = X_val[col].fillna(imputed_vars[col])
    X_test[col] = X_test[col].fillna(imputed_vars[col])


n_estimators = [x+25 for x in range(30)]
max_depth = [x*15 for x in range(1,10)]


model_results = []
for e in n_estimators:
    for d in max_depth:
        temp_model = XGBClassifier(
            eval_metric = 'logloss',
            n_estimators = e,
            max_depth = d,
            learning_rate = .1,
            random_state=123
        )
        temp_model.fit(X_train,y_train_encoded)
        temp_pred = ['Extrovert' if y==0 else 'Introvert' for y in temp_model.predict(X_val)]
        temp_comp = pd.DataFrame({'actual':y_val,'prediction':temp_pred}).groupby(['actual','prediction']).size().to_frame(name = 'count').reset_index()
        temp_pred_training = ['Extrovert' if y==0 else 'Introvert' for y in temp_model.predict(X_train)]
        temp_comp_training = pd.DataFrame({'actual':y_train,'prediction':temp_pred_training}).groupby(['actual','prediction']).size().to_frame(name = 'count').reset_index()
        temp_accuracy_training =temp_comp_training[temp_comp_training['actual']==temp_comp_training['prediction']]['count'].sum()/temp_comp_training['count'].sum() 
        temp_accuracy =temp_comp[temp_comp['actual']==temp_comp['prediction']]['count'].sum()/temp_comp['count'].sum()
        temp_final = [temp_comp_training,temp_comp,temp_accuracy_training,temp_accuracy,e,d]
        model_results.append(temp_final)


model_results_df= pd.DataFrame([x[2:6] for x in model_results],columns = ['TrainAccuracy','ValidationAccuracy','Estimators','Depth'])
model_results_df=model_results_df.sort_values(['TrainAccuracy','ValidationAccuracy']).reset_index()
model_results_df['model']=model_results_df.index


plt.plot(model_results_df['model'],model_results_df['TrainAccuracy'],label = "Training")
plt.plot(model_results_df['model'],model_results_df['ValidationAccuracy'],label = "Validation")
plt.legend()
plt.show()


temp = model_results_df[(model_results_df['model']>50)&(model_results_df['model']<125)]
plt.plot(temp['model'],temp['TrainAccuracy'],label = "Training")
plt.plot(temp['model'],temp['ValidationAccuracy'],label = "Validation")
plt.legend()
plt.show()


plt.scatter(temp[temp['ValidationAccuracy']>.97130]['Depth'],temp[temp['ValidationAccuracy']>.97130]['Estimators'],c = 'red')
plt.scatter(temp[temp['ValidationAccuracy']<.97130]['Depth'],temp[temp['ValidationAccuracy']<.97130]['Estimators'],)


e = 34
d = 200
final_model = XGBClassifier(
            eval_metric = 'logloss',
            n_estimators = e,
            max_depth = d,
            learning_rate = .1,
            random_state=123
        )
final_model.fit(X_train,y_train_encoded)
final_predictions = ['Extrovert' if y==0 else 'Introvert' for y in final_model.predict(X_test)]


final_submission = pd.DataFrame({
    'id':test['id'],
    'Personality':final_predictions
})
final_submission.to_csv('submission.csv',index = False)

