# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# importing necessary libraries
import tensorflow as tf
import tensorflow.keras as keras
import keras_tuner as kt

# Encoding 
from sklearn.preprocessing import LabelEncoder

# splitting the data 
from sklearn.model_selection import train_test_split

# valuating the features 
from sklearn.feature_selection import mutual_info_classif

# standardizatom
from sklearn.preprocessing import MinMaxScaler

# classification report
from sklearn.metrics import classification_report, confusion_matrix

# plots
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline


# Importing trainset
train_df = pd.read_csv('/kaggle/input/playground-series-s4e11/train.csv')
test_df = pd.read_csv("/kaggle/input/playground-series-s4e11/test.csv")
train_df.name = "train set"
test_df.name = "test set"

# assigning the max columns and certain max rows to display so the veiw could be bigger to veiw and easy
pd.set_option("display.max_columns",None)
pd.set_option("display.max_rows",50)


train_df.index = train_df['id']
train_df.drop(columns=['id'],inplace=True)
test_df.index = test_df['id']
test_df.drop(columns=['id'],inplace=True)
train_df.tail(5)


test_df.head(5)


train_df.info()


# Removing the name column from both the train and test set
train_df.drop(columns=['Name'],axis=1,inplace=True)
test_df.drop(columns=['Name'],axis=1,inplace=True)


train_df.isna().sum()


test_df.isna().sum()


train_df.shape


# This functions primary role is to convert all the non-important categories in a series into 'other' category in the same series
# the inputs we will include in this function will be series , a string to rename category
# and this function will return the new series after categorizing the columns 


def unique_series_values( series ):    
    new_dict_ser= {}
    # if the below condition is true then only proceed else throw an error
    if isinstance(series, pd.Series) :

        # now we will check if the series is a object datatype or a number
        if series.dtype == 'O':
            # print("objects")
            new_series = series.copy()
            unique_ls = new_series.unique()
            for i in unique_ls:
                if i in new_series.values :
                    counts = new_series[new_series == i].count()
                    # print(f"item '{i}' occurences : {counts}")
                    new_dict_ser[i] = counts 
                else:
                    print("!waiting!")
            
        elif series.dtype==('float64' or 'int64'):
            print("numbers")

        return new_dict_ser
    else :
        print(f"The input element is either not a pd.Series OR a string")

def unwanted_series_names(dict_ser):
    # Now when you have the final dictionary we will now evaluate the items 
    # not based on the values but based on the percentage that we will be converting the values into percentage
    # and will only keep the desired percentages and rest of the minimal percentaged values are turned into 'others' category
    
    values_ls = np.array(list(dict_ser.values()))
    
    percen_ls = np.round((values_ls/np.sum(values_ls)*100),3)
    
    ser_percen_ls = {}
    for key , val in zip(dict_ser.keys(),percen_ls):
        # print(i,"->",train_df[train_df['City'] == i]['City'].count())
        ser_percen_ls[key] = val

    no_ls = []
    a=0
    for k , v in ser_percen_ls.items():
        # print(f"k = {k}, v= {v}")
        if v < 1:
            a+=v
            no_ls.append(k)
    return no_ls

def grouping_category(ls_unwanted , series, string):
    if isinstance(ls_unwanted, list) and isinstance(series, pd.Series) and isinstance(string,str):
        #print("proceed")
        for i, item in enumerate(series):
            if item in ls_unwanted:
                #print(i,True)
                series.replace(item, string,inplace=True)
        return series
    else:
        print("Not a valid list or series")
    # return pd.Series(categorized_series)




train_df[train_df.columns].select_dtypes(include=['object'])


# city_ls = list(train_df["City"].unique())
# city_unique_ls = {}
# for i in city_ls:
#     # print(i,"->",train_df[train_df['City'] == i]['City'].count())
#     city_unique_ls[i] = train_df[train_df['City'] == i]['City'].count()

# values_ls = np.array(list(city_unique_ls.values()))

# percen_ls = np.round((values_ls/np.sum(values_ls)*100),3)

# city_percen_ls = {}
# for key , val in zip(city_ls,percen_ls):
#     # print(i,"->",train_df[train_df['City'] == i]['City'].count())
#     city_percen_ls[key] = val

# print(city_percen_ls)

# categorized_city_ls = {}
# a=0
# no_ls = []
# for k , v in city_percen_ls.items():
#     # print(f"k = {k}, v= {v}")
#     if v < 1:
#         a+=v
#         categorized_city_ls['others']=a
#         no_ls.append(k)
#     else :
#         categorized_city_ls[k] = v

# print(categorized_city_ls)
# print("--------------------------------------------")
# print(no_ls)


# train set

unique_city_names_dict = unique_series_values(train_df['City'])
print(type(unique_city_names_dict))
print(unique_city_names_dict)
print("************************************************************")
unwanted_city_names = unwanted_series_names(unique_city_names_dict)
print(type(unwanted_city_names))
print(unwanted_city_names)
print("************************************************************")
new_city_col = grouping_category(unwanted_city_names , train_df['City'], "Others")
print(type(new_city_col))
print(new_city_col)
print("************************************************************")



testing_city_train_col = unique_series_values(new_city_col)
print(testing_city_train_col)



train_df['City'] = new_city_col


# train set

unique_city_names_dict_test_df = unique_series_values(test_df['City'])
print(type(unique_city_names_dict_test_df))
print(unique_city_names_dict_test_df)
print("************************************************************")
unwanted_city_names_test_df = unwanted_series_names(unique_city_names_dict_test_df)
print(type(unwanted_city_names_test_df))
print(unwanted_city_names_test_df)
print("************************************************************")
new_city_col_test_df = grouping_category(unwanted_city_names_test_df , test_df['City'], "Others")
print(type(new_city_col_test_df))
print(new_city_col_test_df)
print("************************************************************")



testing_city_test_col = unique_series_values(new_city_col_test_df)
print(testing_city_test_col)


test_df['City'] = new_city_col_test_df


test_df['City'].unique()


train_df['Profession'].isna().sum()


train_df[(train_df['Working Professional or Student'] == 'Student')]['Profession'].count()


train_df.loc[train_df['Working Professional or Student'] == 'Student', 'Profession'] = train_df.loc[train_df['Working Professional or Student'] == 'Student', 'Profession'].fillna("Student")



train_df['Profession'].isna().sum()


train_df.loc[train_df['Work Pressure'].isna(), 'Profession'] = train_df.loc[train_df['Work Pressure'].isna(), 'Profession'].fillna("Student")



train_df['Profession'].isna().sum()


train_df.loc[train_df['Job Satisfaction'].isna(), 'Profession'] = train_df.loc[train_df['Job Satisfaction'].isna(), 'Profession'].fillna("Student")



train_df['Profession'].isna().sum()


# assigning the profession column 
train_set_profession = unique_series_values(train_df['Profession'])

# valuating the unwanted columns
unwanted_profession_ls = unwanted_series_names(train_set_profession)

# categorizing the column with the other category
new_profession_col = grouping_category(unwanted_profession_ls , train_df['Profession'], "Others")
print(type(new_profession_col))
print(new_profession_col)


new_profession_col.nunique()


train_df['Profession'] =  new_profession_col


train_df['Profession'].isna().sum()


train_df[train_df['Profession'].isna()]['Degree'].unique()


train_df['Profession'] = train_df['Profession'].fillna('Unemployed')


train_df['Profession'].isna().sum()


train_df['Profession'].nunique()


test_df['Profession'].isna().sum()


test_df[(test_df['Working Professional or Student'] == 'Student')]['Profession'].count()


test_df.loc[test_df['Working Professional or Student'] == 'Student', 'Profession'] = test_df.loc[test_df['Working Professional or Student'] == 'Student', 'Profession'].fillna("Student")



train_df['Profession'].isna().sum()


train_df['Profession'].isna().sum()


# assigning the profession column 
test_set_profession_test = unique_series_values(test_df['Profession'])

# valuating the unwanted columns
unwanted_test_profession_ls = unwanted_series_names(test_set_profession_test)

# categorizing the column with the other category
new_profession_col_test = grouping_category(unwanted_test_profession_ls , test_df['Profession'], "Others")
print(type(new_profession_col_test))
print(new_profession_col_test)


new_profession_col_test.nunique()


test_df['Profession'] =  new_profession_col_test


test_df['Profession'].isna().sum()


test_df['Profession'] = test_df['Profession'].fillna('Unemployed')


test_df['Profession'].isna().sum()


test_df['Profession'].unique()


train_df['Sleep Duration'].unique()


train_df['Sleep Duration'].isna().sum()


train_sleepD_list = unique_series_values(train_df["Sleep Duration"])
train_sleepD_N_ls = unwanted_series_names(train_sleepD_list)
new_sleepD_train_ls = grouping_category(train_sleepD_N_ls, train_df['Sleep Duration'], "Others")


new_sleepD_train_ls.unique()


train_df["Sleep Duration"] = new_sleepD_train_ls


train_df["Sleep Duration"].isna().sum()


test_df['Sleep Duration'].unique()


test_df['Sleep Duration'].isna().sum()


test_sleepD_list = unique_series_values(test_df["Sleep Duration"])
test_sleepD_N_ls = unwanted_series_names(test_sleepD_list)
new_sleepD_test_ls = grouping_category(test_sleepD_N_ls, test_df['Sleep Duration'], "Others")


new_sleepD_test_ls.unique()


test_df["Sleep Duration"] = new_sleepD_test_ls


test_df["Sleep Duration"].isna().sum()


train_df["Dietary Habits"].unique()


train_df["Dietary Habits"] = train_df["Dietary Habits"].fillna("null")


train_df["Dietary Habits"].unique()


train_df['Dietary Habits'].isna().sum()


diet_train_ls = unique_series_values(train_df["Dietary Habits"])
unwanted_diet_ls = unwanted_series_names(diet_train_ls)
new_train_diet_H = grouping_category(unwanted_diet_ls,train_df["Dietary Habits"], "Others" )


train_df['Dietary Habits'].isna().sum()


train_df["Dietary Habits"].unique()


test_df["Dietary Habits"].unique()


test_df["Dietary Habits"] = test_df["Dietary Habits"].fillna("null")


test_df["Dietary Habits"].unique()


test_df['Dietary Habits'].isna().sum()


diet_test_ls = unique_series_values(test_df["Dietary Habits"])
unwanted_diet_ls_test = unwanted_series_names(diet_test_ls)
new_test_diet_H = grouping_category(unwanted_diet_ls_test, test_df["Dietary Habits"], "Others" )


test_df['Dietary Habits'].isna().sum()


train_df["Dietary Habits"].unique()


train_df["Degree"].unique()


train_df["Degree"] = train_df["Degree"].fillna("null")


train_df["Degree"].unique()


train_df['Degree'].isna().sum()


degree_train_ls = unique_series_values(train_df["Degree"])
unwanted_degree_ls = unwanted_series_names(degree_train_ls)
new_train_degree = grouping_category( unwanted_degree_ls , train_df["Degree"], "Others" )


train_df['Degree'].isna().sum()


train_df["Degree"].unique()


test_df["Degree"].unique()


test_df["Degree"] = test_df["Degree"].fillna("null")


test_df["Degree"].unique()


test_df['Degree'].isna().sum()


degree_test_ls = unique_series_values(test_df["Degree"])
unwanted_degree_ls_test = unwanted_series_names(degree_test_ls)
new_test_degree = grouping_category(unwanted_degree_ls_test, test_df["Degree"], "Others" )


test_df['Degree'].isna().sum()


test_df["Degree"].unique()


train_df.info()


test_df.info()


train_num_df = train_df.select_dtypes(include=['number'])
train_num_df.drop(columns=['Depression'],inplace=True)
print(train_num_df)
test_num_df = test_df.select_dtypes(include=['number'])
print(test_num_df)


print(train_num_df.isna().sum())
print()
print(test_num_df.isna().sum())


# train set loop 
for i in train_num_df.columns:
    print(f"{i} columns has {train_num_df[i].unique()} items")
    print()

print('#'*100)

# test set loop 
for i in test_num_df.columns:
    print(f"{i} columns has {test_num_df[i].unique()} items")
    print()


train_num_df = train_num_df.fillna(0)
test_num_df = test_num_df.fillna(0)


print(train_num_df.info())
print(test_num_df.info())


train_df[['Age','Academic Pressure', 'Work Pressure', 'CGPA', 
        'Study Satisfaction','Job Satisfaction', 
          'Work/Study Hours', 'Financial Stress']] = train_num_df[['Age','Academic Pressure', 
                                                                  'Work Pressure', 'CGPA', 
                                                                  'Study Satisfaction','Job Satisfaction', 'Work/Study Hours', 
                                                                  'Financial Stress']]

train_df.tail(5)


test_df[['Age','Academic Pressure', 'Work Pressure', 'CGPA', 
        'Study Satisfaction','Job Satisfaction', 
          'Work/Study Hours', 'Financial Stress']] = test_num_df[['Age','Academic Pressure', 
                                                                  'Work Pressure', 'CGPA', 
                                                                  'Study Satisfaction','Job Satisfaction', 'Work/Study Hours', 
                                                                  'Financial Stress']]

test_df.head(5)


train_df.to_csv("Cleaned_train_set.csv")
test_df.to_csv("Cleaned_test_set.csv")


train_obj_df = train_df.select_dtypes(include=['object'])
train_obj_df
test_obj_df = test_df.select_dtypes(include=['object'])
test_obj_df


# Label Encoding train set
train_obj_df = train_obj_df.apply(LabelEncoder().fit_transform)
train_obj_df


# Label Encoding train set
test_obj_df = test_obj_df.apply(LabelEncoder().fit_transform)
test_obj_df


col_obj_ls = list(train_obj_df.columns)
col_obj_ls


train_df[col_obj_ls] = train_obj_df[col_obj_ls]
train_df


test_df[col_obj_ls] = test_obj_df[col_obj_ls]
test_df


# deletting duplicate rows for the trainset
print(f"Duplicates for Train Set : {train_df.duplicated().sum()}")
print(f"Duplicates for Test Set : {test_df.duplicated().sum()}")



train_df.tail(5)


test_df.head(5)


len(train_df.columns)
train_df.iloc[0:0,:17]


# We will only use the train set for making the model as you can see the valid reasons for doing so
X = train_df.iloc[:,:17]
y = train_df['Depression']


scaler = MinMaxScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)  


X


(X_train , X_test , y_train , y_test) = train_test_split(X, y, random_state=2030 , test_size=0.25 )


print(f"SHAPE of X_train :{X_train.shape}")
print(f"SHAPE of X_test : {X_test.shape}")
print(f"SHAPE of y_train : {y_train.shape}")
print(f"SHAPE of y_test : {y_test.shape}")


# 4 = 2+5


result_dict1 = {}
result1 = mutual_info_classif(X,y , random_state = 53, n_neighbors = 2, discrete_features='auto')
print(type(result1))

for i , j in zip(X.columns,result1):
    result_dict1[i] = j

sort_dict1_result = {k : v for k , v in sorted(result_dict1.items(), key = lambda item : item[1] , reverse=True )}
sort_dict1_result


result_dict2 = {}
result2 = mutual_info_classif(X,y , random_state = 53, n_neighbors = 3, discrete_features='auto')
print(type(result2))

for i , j in zip(X.columns,result2):
    result_dict2[i] = j

sort_dict2_result = {k : v for k , v in sorted(result_dict2.items(), key = lambda item : item[1], reverse=True )}
sort_dict2_result


result_dict3 = {}
result3 = mutual_info_classif(X,y , random_state = 53, n_neighbors = 5, discrete_features='auto')
print(type(result3))

for i , j in zip(X.columns,result3):
    result_dict3[i] = j

sort_dict3_result = {k : v for k , v in sorted(result_dict3.items(), key = lambda item : item[1] , reverse=True)}
sort_dict3_result


result_dict4 = {}
result4 = mutual_info_classif(X,y , random_state = 53, n_neighbors = 7, discrete_features='auto')
print(type(result4))

for i , j in zip(X.columns,result4):
    result_dict4[i] = j

sort_dict4_result = {k : v for k , v in sorted(result_dict4.items(), key = lambda item : item[1] , reverse=True)}
sort_dict4_result


17*17


def build_model(hp):
    # initializing the model
    model = keras.Sequential()

    # adding the input layer
    model.add(keras.layers.Input(shape=(len(X.columns),) ) )

    # looping through layers
    # we will keep a maximum of 3 layers and a minimum of 2 layer
    for i in range(hp.Int("num_layers",1,3)):
        model.add(keras.layers.Dense(units = hp.Int(f"units_{i}", max_value = 250, min_value = 5, step = 5 ),
                                    activation = hp.Choice(f"activation_{i}",['relu','leaky_relu']) ))

    # here we will add a dropout layer 
    model.add(keras.layers.Dropout(rate=hp.Float(f"dropout_rate_{i}", 0.1, 0.5, step=0.1)))

    # here we will keep it simple layers on the last layer
    # we will keep a maximum of 3 layers and a minimum of 1 layer
    for i in range(hp.Int("num_layers",1,3)):
        model.add(keras.layers.Dense(units = hp.Int(f"units_{i}", max_value = 96, min_value = 8, step = 8 ),
                                    activation = hp.Choice(f"activation_{i}",['leaky_relu','swish']) ,
                                    kernel_regularizer=keras.regularizers.l2(hp.Float('l2_reg_input', 1e-5, 1e-2, sampling='LOG')) ))
    
    # final output layer
    model.add(keras.layers.Dense(1,activation="sigmoid"))
    
    # defining the optimizer choice
    optimizer_choice = hp.Choice('optmizer',['adam','nadam'])
    optimizer = keras.optimizers.Adam(learning_rate=hp.Choice('learning_rate', [1e-4,5e-3])) if optimizer_choice=='adam' else  keras.optimizers.Nadam(learning_rate=hp.Choice('learning_rate', [1e-4,5e-3]))
    
    # Now we will compile the model with some optimizers and add the metrics to the model with loss functions
    model.compile(
        optimizer = optimizer,
        loss = ['binary_crossentropy'],
        metrics = ['accuracy','Recall','Precision','AUC']
    )

    return model


tuner = kt.Hyperband(
    build_model,
    objective = 'val_accuracy',
    max_epochs = 40,
    factor=3,
    directory="model_tuning_tests",
    project_name='binary_classification_mental_health_data_02'
)



# Define early stopping callback
early_stopping = keras.callbacks.EarlyStopping(monitor='accuracy', patience = 3, restore_best_weights=True)



tuner.search(X_train , y_train , epochs = 30, validation_data=(X_test,y_test),
             verbose =1 , callbacks=[early_stopping])



# Get the best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

# Display best hyperparameters
print("Best Hyperparameters:")
for param in best_hps.values:
    print(f'{param}: {best_hps.get(param)}')
    



# Build and train the best model
best_model = tuner.hypermodel.build(best_hps)
best_model.fit(X_train, y_train, epochs=150, validation_data=(X_test, y_test), verbose=1)



# Evaluate best model
test_loss, test_acc, test_precision, test_recall, test_auc = best_model.evaluate(X_test, y_test)
print(f'Best Test Accuracy: {test_acc:.4f}')
print(f'Precision: {test_precision:.4f}')
print(f'Recall: {test_recall:.4f}')
print(f'AUC: {test_auc:.4f}')



# testingwith the prediction basis
model_yhat = best_model.predict(X_test)



model_yhat_binarized = np.where(model_yhat >= 0.5, 1, 0)


# comparing both the yhat and y_test results
print(classification_report(model_yhat_binarized, y_test, target_names=['Depression','Not Depressed']))


# Calculate confusion matrix
cm = confusion_matrix(model_yhat_binarized, y_test)

# Plot confusion matrix as heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Depressed', "Depressed"], yticklabels=['Not Depressed', "Depressed"])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()


# we will now fit the X_test and y_test to the base model 
best_model.fit(X_test, y_test, epochs= 150, verbose = 1)


test_df.info()


test_df.head(5)


scaler = MinMaxScaler()
test_scaler_df = pd.DataFrame(scaler.fit_transform(test_df), columns=test_df.columns)  


test_scaler_df


test_set_prediction = best_model.predict(test_scaler_df)


test_set_prediction


model_test_set_binarized = np.where(test_set_prediction >= 0.5, 1, 0)


print(type(model_test_set_binarized))
print(model_test_set_binarized)
print(model_test_set_binarized.ndim)
print(len(model_test_set_binarized))


test_df.index


test_df_dict = {"id":test_df.index,
               "Depressed":model_test_set_binarized.ravel()}
test_df_dict


test_df_result = pd.DataFrame(test_df_dict ).set_index('id')
test_df_result


test_df_result.to_csv('submission.csv')

