# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sb

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


df = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')


df.info()


#1.is there any relationship between annual income and the ability to pay back loans?

sb.boxplot(data = df, x = 'loan_paid_back', y = 'annual_income');


pay_loan = df[df['loan_paid_back'] == 1]
gender_count = pay_loan['gender'].value_counts(normalize = True)


gender_index = gender_count.index.tolist()
gender_values = gender_count.values.tolist()


from matplotlib.patches import ConnectionPatch


# 2. Which gender is more likely to pay back their loan?

# make figure and assign axis objects
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
fig.subplots_adjust(wspace=0)


#pie parameters
angle = -180 * gender_count.reset_index()['proportion'][0]
explode = [0.1, 0, 0]
colors = sb.color_palette('dark')
wedges, *_ = ax1.pie(
                    gender_count.reset_index()['proportion'],
                    labels=gender_count.reset_index()['gender'],
                    autopct='%1.1f%%',
                    colors = colors,
                    startangle=angle,
                    explode = explode
                    )
ax1.set_title('Gender most likely to pay back loan');

## marital status of the gender most likely to pay back loan
#barchart
f_pay_loan = df[(df['loan_paid_back'] == 1) & (df['gender'] == 'Female')]
marital_count = f_pay_loan['marital_status'].value_counts(normalize = True)
marital_index = marital_count.index.tolist()
marital_values = marital_count.values.tolist()
bottom = 1
width = 0.2

for j, (height, label) in enumerate(reversed([*zip(marital_values, marital_index)])):
    bottom -= height
    bc = ax2.bar(0, height, width, bottom=bottom, color=colors, label=label,
                 alpha=0.1 + 0.25 * j)
    ax2.bar_label(bc, labels=[f"{height:.0%}"], label_type='center')

ax2.set_title('marital status of people most likely to pay loan')
ax2.legend()
ax2.axis('off')
ax2.set_xlim(-2.5 * width, 2.5 * width)


# use ConnectionPatch to draw lines between the two plots
theta1, theta2 = wedges[0].theta1, wedges[0].theta2 
center, r = wedges[0].center, wedges[0].r
bar_height = sum(marital_values)

# draw top connecting line
x = r * np.cos(np.pi / 180 * theta2) + center[0]
y = r * np.sin(np.pi / 180 * theta2) + center[1]
con = ConnectionPatch(xyA=(-width / 2, bar_height), coordsA=ax2.transData,
                      xyB=(x, y), coordsB=ax1.transData)
con.set_color([0, 0, 0])
con.set_linewidth(4)
ax2.add_artist(con)

# draw bottom connecting line
x = r * np.cos(np.pi / 180 * theta1) + center[0]
y = r * np.sin(np.pi / 180 * theta1) + center[1]
con = ConnectionPatch(xyA=(-width / 2, 0), coordsA=ax2.transData,
                      xyB=(x, y), coordsB=ax1.transData)
con.set_color([0, 0, 0])
ax2.add_artist(con)
con.set_linewidth(4)

plt.show()


#does the amount of loan collected affect the ability to pay back loan?
df[['loan_amount','loan_paid_back']].corr()


sb.boxplot(data = df, y = 'loan_amount', x = 'loan_paid_back');


df['loan_paid_back'].value_counts()


from sklearn.utils import resample


def resample_label(df):

    #divide minnority and majority class
    df_major = df[df['loan_paid_back'] == 1]
    df_minor = df[df['loan_paid_back'] == 0]

    # Upsampling minority class
    df_minor_sample = resample(df_minor,
                           
                           # Upsample with replacement
                           replace=True,    

                           # Number to match majority class
                           n_samples=295000,   #i chose a number somewhere in the middle  
                           random_state=42)

    # Upsampling minority class
    df_major_sample = resample(df_major,
                           
                           # downsample with replacement
                           replace=False,    

                           # Number to match majority class
                           n_samples=295000,   #i chose a number somewhere in the middle  
                           random_state=42)

    #join two dataframe
    df_resampled = pd.concat([df_major_sample,df_minor_sample])
    df_resampled = df_resampled.reset_index(drop=True)

    return df_resampled


df_resampled = resample_label(df)


df_resampled = df_resampled.drop(['id','grade_subgrade','loan_purpose'],axis = 1)


from sklearn.preprocessing import OneHotEncoder


def one_hot_encode(df):
    object_df = df.select_dtypes(include='object')
    encoders = {}
    encoded_parts = []

    for col in object_df.columns:
        # find most frequent category
        most_freq = df[col].value_counts().idxmax()

        # force category order so "most_freq" comes first
        cat_order = [most_freq] + [c for c in df[col].unique() if c != most_freq]

        encoder = OneHotEncoder(
            handle_unknown='ignore',
            drop='first',
            categories=[cat_order],
            sparse_output=False
        )

        arr = encoder.fit_transform(df[[col]])

        encoded_df = pd.DataFrame(
            arr,
            columns=encoder.get_feature_names_out([col]),
            index=df.index
        )

        encoded_parts.append(encoded_df)

    # drop original objects
    df_ = df.drop(object_df.columns, axis=1)

    # add encoded columns
    df_ = pd.concat([df_] + encoded_parts, axis=1)

    return df_


dummy_df = one_hot_encode(df_resampled)


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix,ConfusionMatrixDisplay


def split_data(df):
    X = df.drop('loan_paid_back', axis = 1)
    y = df['loan_paid_back']
    X_train, X_val, y_train, y_val = train_test_split(X,
                                                      y,
                                                      test_size = 0.2,
                                                      random_state = 42)
    return X, y, X_train, X_val, y_train, y_val
    


X, y, X_train, X_val, y_train, y_val = split_data(dummy_df)


def train_model(X_train, X_val, y_train, y_val,model):

    #fit model
    model.fit(X_train,y_train)

    #predict target column
    y_pred = model.predict(X_val)

    #check accuracy of model
    score = model.score(X_val, y_val)

    return model, y_pred, score


print('xg_accuracy :', train_model(X_train, X_val, y_train, y_val,model = XGBClassifier()))
print('lgbm_accuracy :', train_model(X_train, X_val, y_train, y_val,model = LGBMClassifier()))
print('dt_accuracy :', train_model(X_train, X_val, y_train, y_val,model = DecisionTreeClassifier()))


base_model, y_pred, base_score = train_model(X_train, 
                                             X_val, 
                                             y_train, 
                                             y_val, 
                                             model = DecisionTreeClassifier())


cm = confusion_matrix(y_val, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()


def get_importances(X,y):
    #create dict
    importances = {}

    for feature in X.columns:
        #create copy
        X_perm = X.copy()

        X_perm[feature] = np.random.permutation(X[feature])

        Xp_train, Xp_val, yp_train, yp_val = train_test_split(X_perm, 
                                                              y, test_size = 0.2, 
                                                              random_state = 42)

        #create perm score
        perm_score = base_model.score(Xp_val,yp_val)

        #calculate importance
        importance = base_score - perm_score

        #append to dict
        importances[feature] = importance
        
    #sort by values
    importance_scores = dict(sorted(importances.items(),key=lambda item: item[1]))

    features = list(importance_scores.keys())
    scores = list(importance_scores.values())
        
    return features, scores


features, scores =  get_importances(X,y)


plt.figure(figsize=(10,8))
plt.barh(features,scores);


#match importance to feature name

feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': base_model.feature_importances_
}).sort_values(by='Importance', ascending=True)

#plot
feature_importances.plot.barh(x='Feature',y='Importance',figsize=(10,10));


pip install eli5


import eli5


from eli5.sklearn import PermutationImportance


 # create permutation importance object using model
# and fit on test set
perm = PermutationImportance(base_model, random_state=1).fit(X_val, y_val)

# display weights using PermutationImportance object IN TEXT
print(eli5.format_as_text(eli5.explain_weights(perm, feature_names = X.columns.tolist())))


eli5.show_prediction(
                    base_model,
                    X_val.iloc[0],
                    feature_names = X.columns.tolist()
                    )


### CHECK CLASS IMBALANCE - DONE ALREADY


df.head()


int_df = df_resampled.select_dtypes(exclude = 'object')


plt.figure(figsize = (10,10))
sb.heatmap(int_df.corr(),annot = True);


df2 = dummy_df.copy()


df2 = df2.drop(['gender_Male',
          'gender_Other',
         'marital_status_Married',
          'marital_status_Widowed',
          'marital_status_Divorced',
         'education_level_High School',
          "education_level_Master's",
          'education_level_Other',
          'education_level_PhD'],axis=1)


#### AFFORDABILITY INDEX

df2['affordability_index'] = df2['annual_income'] / (df2['debt_to_income_ratio'] * 100)


### TOTAL INTEREST

df2['total_interest'] = df2['loan_amount'] * df2['interest_rate']


### LOAN_INCOME_RATIO

df2['loan_to_income'] = df2['loan_amount'] / df2['annual_income']


### MONTHLY INCOME

df2['monthly_income'] = df2['annual_income'] / 2


### DEBT BURDEN SCORE

df2['debt_burden'] = df2['debt_to_income_ratio'] * df2['loan_to_income']


### RISK FACTOR

df2['risk_factor'] = (1 / df2['credit_score']) * df2['loan_amount']


### Net Available Income

#DTI = total debt payments / income

df2['annual_debt_payment'] = df2['debt_to_income_ratio'] * df2['annual_income']
df2['net_available_income'] = df2['annual_income'] - df2['annual_debt_payment']


### AFFORDABILITY INDEX

df2['AI'] = 1 / df2['debt_to_income_ratio']  ###same as annual income / annual debt payment







