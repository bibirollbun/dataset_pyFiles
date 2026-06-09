import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow.keras as tf


train = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")

print(f"Train head:\n {train.head(5)}")
print(" ")
print(f"Test head:\n {test.head(5)}")


def check_table(df):
    print(f"Df info:\n {df.info()}")
    print(f"Df descriptive:\n {df.describe()}")
    print(f"Df shape:\n {df.shape}")
    print(f"column name:\n {df.columns}")
    print("Length of Value Count per col\n")
    for col in df.columns:
        vc = df[col].value_counts()
        print(f"{col}: {len(vc.index)}")
    print("Sum of Missing data per col\n")
    for col in df.columns:
        missing = df[col].isna().sum()
        print(f"{col} : {missing}")
    print(f"Sum of Duplication per col\n: {df.duplicated().sum()}")



check_table(train)


check_table(test)


#double check on missing data
import missingno as msno
msno.matrix(train)


msno.matrix(test)


uniq_col = [col for col in train.columns if len(train[col].unique()) < 15]
uniq_cat_col = {col : train[col].unique() for col in train[uniq_col]}
for col, value in uniq_cat_col.items():
    print(f"{col} : {[value]}")
print(f"Length of columns: {len(uniq_cat_col.keys())}")


def hist_box(df):
    num_col = df.select_dtypes(include=['number'])
    for col in num_col:

        fig, axes = plt.subplots(1,2, figsize=(7,5))
        axes[0].hist(df[col])
        axes[0].set_title(f"Histogram {col}")
        axes[0].set_xlabel(f"{col}")
        axes[0].set_ylabel(f"Frequency")
    
        axes[1].boxplot( df[col])
        axes[1].set_title(f"Boxplot {col}")
        axes[1].set_xlabel(f"{col}")
        axes[1].set_ylabel(f"Frequency")
    
        plt.tight_layout()
        plt.show()


hist_box(train)


more_cat_col = [col for col in train.columns if train[col].dtypes == 'object' and len(train[col].unique()) > 2]
two_cat_col = [col for col in train.columns if train[col].dtypes == 'object' and len(train[col].unique()) == 2]
numerical_col = [col for col in train.columns if np.issubdtype(train[col].dtype, np.number)]


print(more_cat_col)
print(two_cat_col)
print(numerical_col)


for col in two_cat_col:
    vc = train[col].value_counts()
    t_vc = test[col].value_counts()
    print("train", vc)
    print("test", t_vc)


# For the 'default' 
train['default'] = np.where(train['default'] == 'yes', 1, 0)
test['default'] = np.where(test['default'] == 'yes', 1, 0)

# For the 'housing' 
train['housing'] = np.where(train['housing'] == 'yes', 1, 0)
test['housing'] = np.where(test['housing'] == 'yes', 1, 0)

# For the 'loan' 
train['loan'] = np.where(train['loan'] == 'yes', 1, 0)
test['loan'] = np.where(test['loan'] == 'yes', 1, 0)


train


print(f"Train info: {train.info()}")
print(f"Test info: {test.info()}")


for col in train.columns:
    uniq = train[col].unique()
    if len(uniq) < 15:
        vc = train[col].value_counts().sort_index()
        print(f"{col} value counts: {vc}")


for col in train.columns:
    if train[col].nunique() > 1:
        print(f"Train {col} has more than one unique")
    else:
        print(f"Train {col} has less than one unique")


# few columns doesn't impact much, drop them now
train_1 = train.copy()
test_1 = test.copy()
#dropping col: id, day and month (unsure purpose)
train_1 = train_1.drop(columns = ['id', 'day','contact'])
test_1 = test_1.drop(columns = ['id', 'day','contact'])


#bin age
age_bins = [18, 28, 38, 48, 58, 68, 78, 88, 95]

age_labels = ['18-27', '28-37', '38-47', '48-57', '58-67', '68-77', '78-87', '88-95']

train_1['age_binned'] = pd.cut(train_1['age'], bins = age_bins, labels = age_labels, right=False)
test_1['age_binned'] = pd.cut(test_1['age'], bins=age_bins, labels = age_labels, right = False)


from sklearn.preprocessing import LabelEncoder

more_cat_col_1 = ['job', 'marital', 'education', 'poutcome', 'age_binned', 'month']

label_enc = {}
for col in more_cat_col_1:
    lb = LabelEncoder()
    train_1[col] = lb.fit_transform(train_1[col])
    test_1[col] = lb.transform(test_1[col])
    label_enc[col] = lb

for col, le in label_enc.items():
    mapping = dict(zip(le.classes_, range(len(le.classes_))))
    print(f"\n Mapping for {col}: \n")
    print(mapping)


train_1 = train_1.drop(columns=['age'])
test_1 = test_1.drop(columns=['age'])


#Robust Scaling 
from sklearn.preprocessing import RobustScaler

scale_col = ['balance', 'campaign', 'duration', 'previous', 'pdays']

rs = RobustScaler()
train_1[scale_col] = rs.fit_transform(train_1[scale_col])
test_1[scale_col] = rs.transform(test_1[scale_col])

train_1


#final check on shape to make sure no data leakage
print(f"Train shape: {train_1.shape}")
print(f"Test shape: {test_1.shape}")


corr = train_1.corr()
plt.figure(figsize=(10,10))
sns.heatmap(corr, 
            annot=True,
           fmt = '.2f', 
           cmap = 'inferno')


from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression


lr = LinearRegression()
rfe = RFE(lr, n_features_to_select=7)
rfe.fit(train_1.drop(columns=['y']), train_1['y'])

selected_features = train_1.drop(columns=['y']).columns[rfe.support_]
print("Selected Features:", selected_features)


#  use cross val score to measure performance 
from sklearn.model_selection import cross_val_score

tr_copy = train_1.copy()
x_h1 = tr_copy[['marital', 'education', 'housing', 'loan', 'duration', 'campaign','previous']]
x_h2 = tr_copy.drop(columns = ['y'])

y_h = tr_copy['y']

def eval_cvs(x, y, x_name):
    scores = cross_val_score(lr, x, y, scoring='neg_mean_squared_error', cv=5, n_jobs=1)
    print(f"{x_name} mean score : {-scores.mean():.5f} ; std score: {scores.std():.5f}")

eval_cvs(x_h1, y_h, "x_h1")
eval_cvs(x_h2, y_h, "x_h2")


# split
from sklearn.model_selection import train_test_split

X_train_1 = train_1[['marital', 'education', 'housing', 'loan', 'duration', 'campaign','previous']]
y_train_1 = train_1['y']
X_test = test_1[['marital', 'education', 'housing', 'loan', 'duration', 'campaign','previous']]

X_train, X_val, y_train, y_val = train_test_split(X_train_1, y_train_1, test_size = 0.2, random_state = 333)


# model 1 - tf bagging train on subset
from tensorflow.keras import layers

num_models = 5
tf_models = []

for i in range(num_models):
    #create random subset
    indices = np.random.choice(len(X_train), len(X_train), replace=True)
    X_subset, y_subset = X_train.iloc[indices], y_train.iloc[indices]

    #create model
    tfmodel = tf.Sequential([
        layers.Dense(128,input_shape = (X_train.shape[1],)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.2), 

        layers.Dense(64,input_shape = (X_train.shape[1],)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.2), 

        layers.Dense(1, activation = 'sigmoid')
    ])

    tfmodel.compile(optimizer= 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    print(f"First set: {i+1}")
    history = tfmodel.fit(X_subset, y_subset, epochs = 10, verbose=1)
    tf_models.append(tfmodel)


#mode 2 - catboost
from catboost import CatBoostClassifier

catmodel = CatBoostClassifier(iterations=100, verbose = 1)
catmodel.fit(X_train, y_train)  


#stack pred
#tf prediction
tf_pred_ls = [model.predict(X_val).flatten() for model in tf_models]
tf_bagged_preds = np.mean(tf_pred_ls, axis = 0)

#catboost prediction
cat_preds = catmodel.predict_proba(X_val)[:,1]

#stack
meta_feat = np.column_stack((tf_bagged_preds, cat_preds))



#train meat feat
from sklearn.linear_model import LogisticRegression

meta_mod = LogisticRegression()
meta_mod.fit(meta_feat, y_val)
val_pred = meta_mod.predict(meta_feat)


acc = 0
for true, pred in zip(y_val, val_pred):
    if true == pred:
        acc+=1
    else:
        continue

perc = acc / len(y_val) * 100
print(f"Accuracy perc: {perc}%")


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_val, val_pred)
sns.heatmap(cm, annot=True, fmt='.2f', linewidth = 1)


from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

precision = precision_score(y_val, val_pred, pos_label=1)
recall = recall_score(y_val, val_pred, pos_label=1)
f1 = f1_score(y_val, val_pred, pos_label=1)

print(f"Precision: {precision}")
print(f"Recall:{recall}")
print(f"f1 score: {f1}")


tf_pred_test_ls = [model.predict(X_test).flatten() for model in tf_models]
tf_bagged_test_preds = np.mean(tf_pred_test_ls, axis = 0)

#catboost prediction
cat_test_preds = catmodel.predict_proba(X_test)[:,1]

#stack
meta_feat_test = np.column_stack((tf_bagged_test_preds, cat_test_preds))


final_pred = meta_mod.predict_proba(meta_feat_test)


final_pred


positive_class_probabilities = final_pred[:, 1]

submission_df = pd.DataFrame({
    'id': test['id'],  
    'y': positive_class_probabilities
})


submission_df


submission_df.to_csv('/kaggle/working/submission.csv', index=False)




