# Importing the libraries
import tensorflow as tf
import keras_tuner as kt
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline


# Importing the data
train_df = pd.read_csv("/kaggle/input/playground-series-s4e11/train.csv")
train_df.name = "Mental Health Data"
pd.set_option("display.max_columns",None)
pd.set_option("display.max_rows",500)


# dataset
train_df.head(5)


train_df.info()


# function to check the sum of unique features in a column
def features_nunique(df):
    print(f"Unique features in {df.name}")
    for i in df.columns:
        print(f"{i} : {df[i].nunique()}")



# function to list all the uniques values
def list_all_uniques(df):
    print(f"Unique features in {df.name}")
    for i in df.columns:
        print(f"{i} : {df[i].unique()}")



# function to list all unique items with number of items
def ls_val_cnt(cser):
    new_a_ls = list(set(cser))
    items=[]
    counts=[]
    for i in range(0,len(new_a_ls)):
        mask = cser == new_a_ls[i]
        items.append(new_a_ls[i])
        counts.append(mask.sum())
        # cser[new_a_ls[3]]
        # print(f"{i} : for the items {new_a_ls[i]} we have {mask.sum()} no.of occurences")
    # converting the new list into a dataframe and sorting its values into descending order.
    new_pd = pd.DataFrame({"items":items,"counts":counts})
    new_pd.sort_values(by='counts',ascending=False,inplace=True)
    return new_pd



# a function to check and remove or replace the unwanted values or rows in a series with user input category
def rm_rpl_ser_items(ls_unwnt, ser, cate_str):
    # Check if inputs are of the correct types
    if isinstance(ls_unwnt, list) and isinstance(ser, pd.Series):
        new_ser = ser.copy()
        # iterating through all items from the unwanted list
        for item in ls_unwnt:
            # checking whether the values exist in the series
            if item in new_ser.values:
                # print(f"{item} is present in the Series")
                # changing the values into "Others"
                new_ser.loc[new_ser == item] = cate_str
                #print("Value changed successfully")
            else:
                print(f"{item} is not present in the Series")
        return new_ser
    else:
        print("The list provided to check is not a list or the series provided is not a Series")


series_city = train_df['City']


# rm_rpl_ser_items(ls_del_citys, series_city)


series_city.unique()


a = train_df['Sleep Duration']

# print(new_a_ls)
# print(type(new_a_ls[0]))
# print(f"For {new_a_ls[0]} :  times")


# ls_val_cnt(train_df['Sleep Duration'])


# Number of unique values 
features_nunique(train_df)


# function to list all unique features in a dataframe
list_all_uniques(train_df)


train_df.columns


train_df['Name'].unique()


ls_val_cnt(train_df['Working Professional or Student'])


# city columns
train_df['City'].unique()


# unique list assigning for new function ls_del_item
print(ls_val_cnt(train_df['City']))
controlling_city_col = ls_val_cnt(train_df['City'])


ls_del_citys = list(controlling_city_col[controlling_city_col['counts']<10]['items'])
ls_del_citys


# Check if all elements in ls_del_citys are in train_df['City']
if all(item in train_df['City'].values for item in ls_del_citys):
    print(True)
else:
    print(False)


# we have the items to delete here 
# and the series which we want to put the change to 
# and we will assign it into a new variable
other = 'Others'
filtered_city_col = rm_rpl_ser_items(ls_del_citys, train_df['City'], other)

#check if what we have done is correct or not
filtered_city_col.value_counts()



filtered_city_col


train_df['City'].unique()


train_df['Profession'].unique()


# unique list
ls_val_cnt(train_df['Profession'])


# these are all the list that we are going to categorize
ls_other_profession = ['Yuvraj','Moderate','Unveil','Nagpur','Dev','Pranav','Patna','Working Professional',
                       'Visakhapatnam','Profession','Unemployed','Yogesh']
ls_stu_profession = ['B.Ed','BBA','B.Com','MBBS','M.Ed','BE','PhD','MBA','LLM','BCA','Academic']


# if all the items in the list ls_stu_profession is found in the series column train_df['Profession']
# then make all the values present in the ls_stu_profession into the student category

student = 'Student'
hf_filtered_profession_col = rm_rpl_ser_items(ls_stu_profession, train_df['Profession'], student)

#check if what we have done is correct or not
print(hf_filtered_profession_col.value_counts())

type(hf_filtered_profession_col)


# if all the items in the list ls_stu_profession is found in the series column train_df['Profession']
# then make all the values present in the ls_other_profession into the other category

filtered_profession_col = rm_rpl_ser_items(ls_other_profession, hf_filtered_profession_col , other)

#check if what we have done is correct or not
print(filtered_profession_col.value_counts())

type(filtered_profession_col)


# now in column 'Sleep Duration' 
train_df['Sleep Duration'].unique()


# unique list
sleep_duration = ls_val_cnt(train_df['Sleep Duration'])
sleep_duration


# sleep duration categories
ls_others_sleep_duration = ['No','Sleep_Duration','Unhealthy','45','Work_Study_Hours','40-45 hours','Moderate','35-36 hours',
                            '49 hours','than 5 hours','55-66 hours','45-48 hours','Pune','Indore']

less_than_5hrs = ['1-2 hours','3-6 hours','3-4 hours','4-5 hours','4-6 hours','2-3 hours','1-3 hours','1-6 hours']
ls_bw_5_9_hrs = ['5-6 hours','6-7 hours','6-8 hours','8 hours','7-8 hours','9-6 hours','9-5 hours','8-9 hours','9-5']
more_than_9_hrs = ['10-6 hours','More than 8 hours','9-11 hours','10-11 hours']


# others category
others_filtered = rm_rpl_ser_items(ls_others_sleep_duration , train_df['Sleep Duration'], other)

# less than 5 hours category
ls_5hrs = "Less than 5 hours"
less_than_5hrs_filtered = rm_rpl_ser_items(less_than_5hrs, others_filtered, ls_5hrs)

# between 5 and 9 hours
bw_5_9_hrs = "Between 5 - 9 Hours"
bw_5_9 = rm_rpl_ser_items(ls_bw_5_9_hrs, less_than_5hrs_filtered, bw_5_9_hrs)

# more than 9 hours
morethan_9 = "More than 8 hours"
more_than_9 = rm_rpl_ser_items(more_than_9_hrs, bw_5_9, morethan_9)

# assigning it to a new variable 
filtered_sleep_duration_col = more_than_9

filtered_sleep_duration_col.value_counts()


# in column 'Dietary Habits'
train_df['Dietary Habits'].unique()


# unique list
ls_val_cnt(train_df['Dietary Habits'])


# categorizing the deitary habits column
ls_other_diet = ['Yes','No','Male','Class 12','Gender','1.0','M.Tech','Mihir','BSc',
                 'Electrician','2','3','Indoor','Vegas','Hormonal', 'Pratham']

ls_healthy = ['More Healthy','Healthy',]
ls_moderate = ['Moderate','No Healthy','Less than Healthy']
ls_unhealthy = ['Unhealthy','Less Healthy']


# filtering the dietary habit column
other_filtered = rm_rpl_ser_items(ls_other_diet, train_df['Dietary Habits'], other)

# healthy
healthy = 'Healthy'
healthy_filtered = rm_rpl_ser_items(ls_healthy, other_filtered, healthy)

# unhealthy
unhealthy = 'Unhealthy'
unhealthy_filtered = rm_rpl_ser_items(ls_unhealthy, healthy_filtered, unhealthy)

# moderate
mod = 'Moderate'
filtered_diet_col = rm_rpl_ser_items(ls_moderate, unhealthy_filtered, mod)

# value counts of it
filtered_diet_col.value_counts()


# in the column degree lets see what all kinds of unique values are available there
train_df['Degree'].unique()


# unique list
controlling_degree_col = ls_val_cnt(train_df['Degree'])
controlling_degree_col


ls_degree_others_col = list(controlling_degree_col[controlling_degree_col['counts']<5]['items'])
ls_degree_others_col[-1] = str(float(0))
ls_degree_others_col.remove('0.0')
ls_degree_others_col



filtered_degree_col = rm_rpl_ser_items(ls_degree_others_col, train_df['Degree'], other)
filtered_degree_col.value_counts()


# new dataset after cleaning the dataset
new_df = {"Gender":train_df['Gender'].values,
          "Age":train_df['Age'].values,
          "City":filtered_city_col,
          "Worker Professional/Student":train_df['Working Professional or Student'].values,
          "Profession":filtered_profession_col,
          "Academic Pressure":train_df['Academic Pressure'].values,
          "Work Pressure":train_df['Work Pressure'].values,
          "CGPA":train_df['CGPA'].values,
          "Study Satisifaction":train_df['Study Satisfaction'].values,
          "Job Satisfaction":train_df['Job Satisfaction'].values,
          "Sleep Duration":filtered_sleep_duration_col,
          "Dietary Habits":filtered_diet_col,
          "Degree":filtered_degree_col,
          "Suicidal Thoughts?":train_df['Have you ever had suicidal thoughts ?'].values,
          "work/study Hours":train_df['Work/Study Hours'].values,
          "Financial Stress":train_df['Financial Stress'].values,
          "Heritage Illness":train_df['Family History of Mental Illness'].values,
          "Depression":train_df['Depression']}

new_df = pd.DataFrame(new_df)
new_df.name = "Cleaned Mental Health Data"


new_df.head(5)


# Lets fill all the null values with zero
new_df.fillna(0,inplace=True)


new_df.head(5)


features_nunique(new_df)


list_all_uniques(new_df)


new_df.tail(10)


new_df.duplicated().sum()


new_df.drop_duplicates(inplace=True)


new_df.duplicated().sum()


new_df.head(5)


new_df['Gender'].unique()


# Custom naming the gender column
gender = {'Female':0,'Male':1}
new_df['Gender'] = new_df['Gender'].map(gender)
new_df['Gender'].unique()


new_df['Age'].unique()


new_df['City'].unique()


City = list(new_df['City'].unique())
print(City)


city_no = np.arange(0,len(City))
city_no


city_encoder = {}
for i,j in zip(City,city_no):
    city_encoder[i]=j
city_encoder



new_df['City'] = new_df['City'].map(city_encoder)
new_df['City']


new_df['Worker Professional/Student'] = LabelEncoder().fit_transform(new_df['Worker Professional/Student'])
new_df['Worker Professional/Student']


profession = list(new_df['Profession'].unique())
print(len(profession))


profession_no = np.arange(0,len(profession))
profession_no


profession_dict = {}
for i , j in zip(profession, profession_no):
    profession_dict[i]=j
profession_dict


new_df['Profession'] = new_df['Profession'].map(profession_dict)
new_df['Profession']


new_df['Sleep Duration'].unique()


new_df['Sleep Duration'].value_counts()


conditionfor_sleep_duration = new_df[new_df['Sleep Duration'].isin(['Others'])].index
conditionfor_sleep_duration


new_df.drop(conditionfor_sleep_duration, axis=0, inplace=True)


new_df.count()


sleep_duration_dict = {'Less than 5 hours':0,'Between 5 - 9 Hours':1, 'More than 8 hours': 2}

new_df['Sleep Duration'] = new_df['Sleep Duration'].map(sleep_duration_dict)


new_df.head(5)


new_df['Dietary Habits'].unique()


new_df['Dietary Habits'].value_counts()


condition_for_DH = new_df[new_df['Dietary Habits'].isin(['Others', 0])].index
condition_for_DH


new_df.drop(condition_for_DH, axis=0, inplace=True)
new_df.count()


new_df['Dietary Habits'] = LabelEncoder().fit_transform(new_df['Dietary Habits'])
new_df['Dietary Habits']


new_df['Degree'].unique()


new_df['Degree'].value_counts()


condition_for_degree = new_df[new_df['Degree'].isin(['Others',0])].index
condition_for_degree


new_df.drop(condition_for_degree, axis =0 , inplace=True)


new_df['Degree'].unique()


new_df['Degree'] = LabelEncoder().fit_transform(new_df['Degree'])
new_df['Degree']


new_df['Suicidal Thoughts?'].unique()


new_df['Suicidal Thoughts?'] = LabelEncoder().fit_transform(new_df['Suicidal Thoughts?'])
new_df['Suicidal Thoughts?']


new_df['Heritage Illness'].unique()


new_df['Heritage Illness'] = LabelEncoder().fit_transform(new_df['Heritage Illness'])
new_df['Heritage Illness']


new_df.duplicated().sum()


new_df.isna().sum()


new_df.head(5)


new_df.to_csv('cleaned_mental_health_trainset.csv')


# a heatmap to see the correlation of all the features and pick the features according to its correlation 

plt.figure(figsize=(16,11))

sns.heatmap(new_df.corr(), annot=True, vmin=-1, vmax=1, cbar =True, cmap='coolwarm')

plt.title("Heatmap for Mental Health Data",fontsize=40)

plt.show()

# plt.savefig('Mental Health Data Heatmap.jpg',format='jpg')


xy_df = new_df.copy()

xy_df.drop(columns=['Gender','City','Degree','Heritage Illness'], axis=1, inplace=True)

xy_df.head(5)


xy_df.duplicated().sum()


xy_df.drop_duplicates(inplace=True)


xy_df.duplicated().sum()


xy_df.shape


140700 - 139447


xy_df.info()


X = xy_df.iloc[:,:13]
X


y = xy_df['Depression']
y


# splitting the dataset into training set and test set

x_train , x_test, y_train, y_test = train_test_split(X,y, test_size=0.1, random_state=53)


# printing the dhape of the training and testing dataset

print(f"Training set : X->{x_train.shape}, y->{y_train.shape}")
print(f"Testing set : X->{x_test.shape}, y->{y_test.shape}")


print(len(x_train.columns))


# # my neural network 
# # sequential API

model = tf.keras.Sequential([
    tf.keras.Input(shape=(13,)),
    tf.keras.layers.Dense(128,activation='relu',activity_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.Dropout(rate=0.3),
    tf.keras.layers.Dense(64, activation='leaky_relu'),
    tf.keras.layers.Dense(32, activation='tanh'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dropout(rate=0.5),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(4, activation='relu'),
    tf.keras.layers.Dense(2, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])



model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy',  tf.keras.metrics.Precision(name='precision')
#              tf.keras.metrics.Recall(name='recall'),
#              tf.keras.metrics.AUC(name='auc')
])


model.summary()


model.fit(x_train,y_train,
         validation_split=0.2,
         epochs=50,
         batch_size=100,
         verbose=1)


# valuation of the model
final_accuracy = model.evaluate(x_test,y_test)

print(f"Testing accuracy : {final_accuracy} ")


# a custom neural network binary classifier model
# with the help of keras_tuner

def build_model(hp):
    model = tf.keras.Sequential()
    
    # First Dense Layer with tunable neurons
    model.add(tf.keras.Input(shape=(13,)))
    
    # Hidden Layer
    for i in range(hp.Int('No_of_Layers',min_value=1, max_value=6)):
        # tuning the layers in the model
        model.add(tf.keras.layers.Dense(units = hp.Int(f"units_{i}",
                                                       min_value=8,max_value=128,step=4),
                 #tuning the activation functions
                activation=hp.Choice(f"activation_{i}",values=['relu','leaky_relu','swish'])
                 ))
        
        # adding the dropout layer if incase it is overfitting
        if hp.Boolean(f"add_dropout_{i}"):
            model.add(tf.keras.layers.Dropout(rate=hp.Float(f"dropout_rate_{i}",
                                                            min_value=0.2,max_value=0.6,step=0.2)))
        
    
    # Output Layer
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    
    # Compile the Model
    model.compile(
        optimizer=hp.Choice('optimizer',values=['adam','sgd']),
        loss='binary_crossentropy',
        metrics=['accuracy','AUC']
    )
    return model



# initializing the tuner
tuner = kt.Hyperband(build_model,
                    objective = 'val_accuracy',
                    max_epochs = 66,
                    factor=3,
                    directory = 'mentalhealth_best_model',
                    project_name = 'mental_health_tuning_folder_best_params')


# adding callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                 patience=3,
                                                 restore_best_weights=True)


# training the model
tuner.search(
    x_train, 
    y_train,
    epochs=30,
    validation_split=0.2,
    verbose=1,
    callbacks=[early_stopping]  
)


# getting the best hyperparameters 
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(best_hps.values)



# building and training the best model 
best_model = tuner.hypermodel.build(best_hps)


# callbacks
early_stopping1 = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                 patience=5)


# fitting the model
best_model.fit(
    x_train, y_train,
    validation_split=0.2,
    epochs=75,
    batch_size=128,
    verbose = 1,
)



# valuation of the model
evaluation_02 = best_model.evaluate(x_test,y_test)

print(f"Testing accuracy : {evaluation_02} ")


y_predict = best_model.predict(x_test)


# converting the classes into categories of 0 and 1 from probabillities
y_predict_0_1 = (y_predict > 0.5).astype(int)


# printing the classification report
print(classification_report(y_test, y_predict_0_1))


# Assuming y_test is your true labels and y_predict_classes is your predicted labels
conf_matrix = confusion_matrix(y_test, y_predict_0_1)

# Plotting the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Depressed', 'NO Depression'], 
            yticklabels=['NO Depression', 'Depressed'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()



# valuation of the custom model
evaluation_03 = model.evaluate(x_test,y_test)

print(f"Testing accuracy : {evaluation_03} ")


y_predict_model = model.predict(x_test)


# converting the classes into categories of 0 and 1 from probabillities
y_predict_0_1_model = (y_predict_model > 0.5).astype(int)


# printing the classification report
print(classification_report(y_test, y_predict_0_1_model))


# Assuming y_test is your true labels and y_predict_classes is your predicted labels
conf_matrix = confusion_matrix(y_test, y_predict_0_1_model)

# Plotting the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Depressed', 'NO Depression'], 
            yticklabels=['NO Depression', 'Depressed'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()



# Saving the custom model in keras format
model.save('Mental_Health_custom_model.keras')


# saving the best hyperparameters model in keras format
best_model.save('Mental_Health_best_param_model.keras')

