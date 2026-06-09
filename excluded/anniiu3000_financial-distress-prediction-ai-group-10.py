#ignoring warnings to keep the code clean
import warnings
warnings.filterwarnings('ignore')


#importing dependencies
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib
import os
%matplotlib inline

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 150)
sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (10, 6)
matplotlib.rcParams['figure.facecolor'] = '#00000000'


train_df = pd.read_csv('/kaggle/input/GiveMeSomeCredit/cs-training.csv')
print(train_df.shape)
train_df.head()


test_df = pd.read_csv('/kaggle/input/GiveMeSomeCredit/cs-test.csv')
print(test_df.shape)
test_df.head()


submission =  pd.read_csv('/kaggle/input/GiveMeSomeCredit/sampleEntry.csv')
submission.head()


train_df['SeriousDlqin2yrs'].unique()


train_df.info()


test_df.info()


train_df.describe()


test_df.describe()


train_df['SeriousDlqin2yrs'].value_counts()


train0=train_df[train_df['SeriousDlqin2yrs']==0].sample(frac=0.06684)
train1=train_df[train_df['SeriousDlqin2yrs']==1].copy()
train_df=pd.concat([train0, train1], axis=0)
train_df['SeriousDlqin2yrs'].value_counts()


Atttributes= ['RevolvingUtilizationOfUnsecuredLines', 'age',
              'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio', 'MonthlyIncome',
              'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate',
              'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse',
              'NumberOfDependents']


for i in Atttributes:
    fig, axes = plt.subplots(1,2, figsize=(15, 5))
    axes[0].set_title(i+' (Train Data)')
    sns.distplot(train_df[i], ax=axes[0])
    axes[1].set_title(i+' (Test Data)')
    sns.distplot(test_df[i], ax=axes[1])


def normalizer(x,df):
    upper_boundary=df[x].mean()+2*df[x].std()
    lower_boundary=df[x].mean()-2*df[x].std()
    max_att=df[x].max()
    min_att=df[x].min()
    return {'Attribute':x, 'upper_boundary': upper_boundary, 'lower_boundary': lower_boundary, 
           'max_att':max_att, 'min_att':min_att }


train_limits = pd.DataFrame([normalizer(x, train_df) for x in Atttributes])
train_limits


test_limits = pd.DataFrame([normalizer(x, test_df) for x in Atttributes])
test_limits


def NormAtt(i, lim_df, df):
    Att=lim_df.iloc[i].Attribute
    UL=lim_df.iloc[i].upper_boundary
    LL=lim_df.iloc[i].lower_boundary
    fig, axes = plt.subplots(1,2, figsize=(15, 5))
    axes[0].set_title('Old Distribution of '+Att)
    sns.distplot(df[Att], ax=axes[0])
    df.loc[df[Att]<LL,Att]=LL
    df.loc[df[Att]>UL,Att]=UL
    axes[1].set_title('New Distribution of '+Att)
    sns.distplot(df[Att], ax=axes[1])


for i in range(0,10):
    NormAtt(i, train_limits, train_df)


for i in range(0,10):
    NormAtt(i, test_limits, test_df)


sns.heatmap(train_df.corr())
plt.title('Correlation Between Attributes');


sns.heatmap(test_df.corr())
plt.title('Correlation Between Attributes');


train_df.columns


input_cols = ['RevolvingUtilizationOfUnsecuredLines', 'age',
       'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio', 'MonthlyIncome',
       'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate',
       'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse',
       'NumberOfDependents']
target_col = 'SeriousDlqin2yrs'


inputs = train_df[input_cols].copy()
targets = train_df[target_col].copy()


test_inputs = test_df[input_cols].copy()


from sklearn.impute import SimpleImputer


imputer = SimpleImputer(strategy = 'median').fit(train_df[input_cols])


inputs[input_cols] = imputer.transform(inputs[input_cols])
test_inputs[input_cols] = imputer.transform(test_inputs[input_cols])


inputs[input_cols].isna().sum()


test_inputs[input_cols].isna().sum()


from sklearn.preprocessing import MinMaxScaler


scaler = MinMaxScaler().fit(train_df[input_cols])


inputs[input_cols] = scaler.transform(inputs[input_cols])
test_inputs[input_cols] = scaler.transform(test_inputs[input_cols])


inputs.describe().loc[['min', 'max']]


test_inputs.describe().loc[['min', 'max']]


inputs.head()


test_inputs.head()


from sklearn.model_selection import train_test_split
X_train, X_val, train_targets, val_targets = train_test_split(inputs, targets, test_size=0.25)


from sklearn.tree import DecisionTreeClassifier


model = DecisionTreeClassifier(random_state=42)


#fitting the model
model.fit(X_train, train_targets)


from sklearn.metrics import accuracy_score, confusion_matrix


train_preds = model.predict(X_train)


train_preds


pd.value_counts(train_preds)


#Probabilities for each prediction
train_probs = model.predict_proba(X_train)


train_probs


accuracy_score(train_targets, train_preds)


model.score(X_val, val_targets)


val_targets.value_counts() / len(val_targets)


#we will store this result as base_acc to use it later
base_acc = accuracy_score(train_targets, train_preds), model.score(X_val, val_targets)
base_acc


from sklearn.tree import plot_tree, export_text


plt.figure(figsize=(80,20))
plot_tree(model, feature_names=X_train.columns, max_depth=2, filled=True);


model.tree_.max_depth


tree_text = export_text(model, max_depth=10, feature_names=list(X_train.columns))
print(tree_text[:5000])


model.feature_importances_


importance_df = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)


importance_df.head(10)


plt.title('Feature Importance')
sns.barplot(data=importance_df.head(10), x='importance', y='feature');


def test_params(**params):
    model = DecisionTreeClassifier(random_state=42, **params)
    model.fit(X_train, train_targets)
    train_score = accuracy_score(model.predict(X_train), train_targets)
    val_score = accuracy_score(model.predict(X_val), val_targets)
    return train_score, val_score
def test_param_and_plot(param_name, param_values):
    train_acc, val_acc = [], [] 
    for value in param_values:
        params = {param_name: value}
        train_score, val_score = test_params(**params)
        train_acc.append(train_score)
        val_acc.append(val_score)
    plt.figure(figsize=(10,6))
    plt.title('Overfitting curve: ' + param_name)
    plt.plot(param_values, train_acc)
    plt.plot(param_values, val_acc)
    plt.xlabel(param_name)
    plt.ylabel('Accuracy')
    plt.legend(['Training', 'Validation'])
    print('Max Acc by:', val_acc.index(max(val_acc))+ int(min(param_values)))


model = DecisionTreeClassifier(criterion='entropy', random_state=42)
model.fit(X_train, train_targets)
print(base_acc)
model.score(X_train, train_targets), model.score(X_val, val_targets)


model = DecisionTreeClassifier(criterion='gini', random_state=42)
model.fit(X_train, train_targets)
print(base_acc)
model.score(X_train, train_targets), model.score(X_val, val_targets)


model = DecisionTreeClassifier(criterion='entropy', splitter='random', random_state=42)
model.fit(X_train, train_targets)
print(base_acc)
model.score(X_train, train_targets), model.score(X_val, val_targets)


model = DecisionTreeClassifier(criterion='entropy', splitter='best', random_state=42)
model.fit(X_train, train_targets)
print(base_acc)
model.score(X_train, train_targets), model.score(X_val, val_targets)


def test_params(**params):
    model = DecisionTreeClassifier(random_state=42,criterion='gini', **params)
    model.fit(X_train, train_targets)
    train_score = accuracy_score(model.predict(X_train), train_targets)
    val_score = accuracy_score(model.predict(X_val), val_targets)
    return train_score, val_score
max_depth=[i for i in range(1,32)]
test_param_and_plot('max_depth',max_depth)


model = DecisionTreeClassifier(criterion='entropy',max_depth=6, random_state=42)
model.fit(X_train, train_targets)
print(base_acc)
model.score(X_train, train_targets), model.score(X_val, val_targets)


def test_params(**params):
    model = DecisionTreeClassifier(criterion='entropy',max_depth=6, random_state=42, **params)
    model.fit(X_train, train_targets)
    train_score = accuracy_score(model.predict(X_train), train_targets)
    val_score = accuracy_score(model.predict(X_val), val_targets)
    return train_score, val_score
max_leaf_nodes=[i for i in range(2,150)]
test_param_and_plot('max_leaf_nodes',max_leaf_nodes)


model = DecisionTreeClassifier(criterion='entropy',max_depth=6, max_leaf_nodes=27,
                               random_state=42)
model.fit(X_train, train_targets)
print(base_acc)
model.score(X_train, train_targets), model.score(X_val, val_targets)


def test_params(**params):
    model = DecisionTreeClassifier(criterion='entropy',max_depth=6, max_leaf_nodes=27,
                                   random_state=42, **params)
    model.fit(X_train, train_targets)
    train_score = accuracy_score(model.predict(X_train), train_targets)
    val_score = accuracy_score(model.predict(X_val), val_targets)
    return train_score, val_score
min_samples_split=[i for i in range(2,300)]
test_param_and_plot('min_samples_split',min_samples_split)


model = DecisionTreeClassifier(criterion='entropy',max_depth=6, max_leaf_nodes=27,
                               min_samples_split=2, random_state=42)
model.fit(X_train, train_targets)
print(base_acc)
model.score(X_train, train_targets), model.score(X_val, val_targets)


def test_params(**params):
    model = DecisionTreeClassifier(criterion='entropy',max_depth=6, max_leaf_nodes=27,
                                   min_samples_split=2, random_state=42, **params)
    model.fit(X_train, train_targets)
    train_score = accuracy_score(model.predict(X_train), train_targets)
    val_score = accuracy_score(model.predict(X_val), val_targets)
    return train_score, val_score
min_samples_leaf=[i for i in range(1,150)]
test_param_and_plot('min_samples_leaf',min_samples_leaf)


model = DecisionTreeClassifier(criterion='entropy',max_depth=6, max_leaf_nodes=27,
                               min_samples_split=2, random_state=42,
                               min_samples_leaf=1)
model.fit(X_train, train_targets)
print(base_acc)
model.score(X_train, train_targets), model.score(X_val, val_targets)


def test_params(**params):
    model = DecisionTreeClassifier(criterion='entropy',max_depth=6, max_leaf_nodes=27,
                               min_samples_split=2, random_state=42,
                               min_samples_leaf=1, **params)
    model.fit(X_train, train_targets)
    train_score = accuracy_score(model.predict(X_train), train_targets)
    val_score = accuracy_score(model.predict(X_val), val_targets)
    return train_score, val_score
max_features=[i for i in range(1,11)]
test_param_and_plot('max_features',max_features)


model = DecisionTreeClassifier(criterion='entropy',max_depth=6, max_leaf_nodes=27,
                               min_samples_split=2, random_state=42,
                               min_samples_leaf=1,
                               max_features=10)
model.fit(X_train, train_targets)
print(base_acc)
model.score(X_train, train_targets), model.score(X_val, val_targets)


def test_params(**params):
    model = DecisionTreeClassifier(criterion='entropy',max_depth=6, max_leaf_nodes=27,
                               min_samples_split=2, random_state=42,
                               min_samples_leaf=1,
                               max_features=10, **params)
    model.fit(X_train, train_targets)
    train_score = accuracy_score(model.predict(X_train), train_targets)
    val_score = accuracy_score(model.predict(X_val), val_targets)
    return train_score, val_score
min_impurity_decrease=[1.0e-10,0, 1.0e-9, 1.0e-7, 1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0]
test_param_and_plot('min_impurity_decrease',min_impurity_decrease)


model = DecisionTreeClassifier(criterion='entropy',max_depth=6, max_leaf_nodes=27,
                               min_samples_split=2, random_state=42,
                               min_samples_leaf=1,
                               max_features=10,
                               min_impurity_decrease=0.0)
model.fit(X_train, train_targets)
print(base_acc)
model.score(X_train, train_targets), model.score(X_val, val_targets)


model = DecisionTreeClassifier(criterion='entropy',max_depth=6, max_leaf_nodes=27,
                               min_samples_split=2, random_state=42,
                               min_samples_leaf=1,
                               max_features=10,
                               min_impurity_decrease=0.0,class_weight={0:15.5, 1:1})
model.fit(X_train, train_targets)
print(base_acc)
model.score(X_train, train_targets), model.score(X_val, val_targets)


model = DecisionTreeClassifier(criterion='entropy',max_depth=6, max_leaf_nodes=27,
                               min_samples_split=2, random_state=42,
                               min_samples_leaf=1,
                               max_features=10,
                               min_impurity_decrease=0.0,class_weight={0:1.5, 1:1})
model.fit(X_train, train_targets)
print(base_acc)
model.score(X_train, train_targets), model.score(X_val, val_targets)


model = DecisionTreeClassifier(criterion='entropy',max_depth=6, max_leaf_nodes=27,
                               min_samples_split=2, random_state=42,
                               min_samples_leaf=1,
                               max_features=10,
                               min_impurity_decrease=0.0,class_weight={0:5, 1:1})
model.fit(X_train, train_targets)
print(base_acc)
model.score(X_train, train_targets), model.score(X_val, val_targets)


model = DecisionTreeClassifier(criterion='entropy',max_depth=6, max_leaf_nodes=27,
                               min_samples_split=2, random_state=42,
                               min_samples_leaf=1,
                               max_features=10,
                               min_impurity_decrease=0.0,class_weight={0:1, 1:0})
model.fit(X_train, train_targets)
print(base_acc)
model.score(X_train, train_targets), model.score(X_val, val_targets)


model = DecisionTreeClassifier(criterion='entropy',max_depth=6, max_leaf_nodes=27,
                               min_samples_split=2, random_state=42,
                               min_samples_leaf=1,
                               max_features=10,
                               min_impurity_decrease=0.0,class_weight={0:1, 1:0.5})
model.fit(X_train, train_targets)
print(base_acc)
model.score(X_train, train_targets), model.score(X_val, val_targets)


model = DecisionTreeClassifier(criterion='entropy',max_depth=6, max_leaf_nodes=27,
                               min_samples_split=2, random_state=42,
                               min_samples_leaf=1,
                               max_features=10,
                               min_impurity_decrease=0.0)
model.fit(X_train, train_targets)
model.score(X_train, train_targets), model.score(X_val, val_targets)


test_preds=model.predict(test_inputs)
test_preds


pd.value_counts(test_preds) / len(test_preds)


submission['Probability']=test_preds
submission.head()


#Saving Submissions as CSV File
submission.to_csv('submission.csv', index=None)


from sklearn.ensemble import RandomForestClassifier


model = RandomForestClassifier(n_jobs=-1, random_state=42)


model.fit(X_train, train_targets)


model.score(X_train, train_targets)


model.score(X_val, val_targets)


len(model.estimators_)


def estimator_acc(i):
    model.estimators_[i].fit(X_train, train_targets)
    return model.estimators_[i].score(X_val, val_targets)


accuracy=[]
for i in range(0,100):
    accuracy.append(estimator_acc(i))


max(accuracy)


train_probs = model.predict_proba(X_train)
train_probs


importance_df = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)


importance_df.head(10)


plt.title('Feature Importance')
sns.barplot(data=importance_df.head(10), x='importance', y='feature');


def test_params(**params):
    model = RandomForestClassifier(n_jobs=-1, random_state=42, **params)
    model.fit(X_train, train_targets)
    train_score = accuracy_score(model.predict(X_train), train_targets)
    val_score = accuracy_score(model.predict(X_val), val_targets)
    return train_score, val_score
def test_param_and_plot(param_name, param_values):
    train_acc, val_acc = [], [] 
    for value in param_values:
        params = {param_name: value}
        train_score, val_score = test_params(**params)
        train_acc.append(train_score)
        val_acc.append(val_score)
    plt.figure(figsize=(10,6))
    plt.title('Overfitting curve: ' + param_name)
    plt.plot(param_values, train_acc)
    plt.plot(param_values, val_acc)
    plt.xlabel(param_name)
    plt.ylabel('Accuracy')
    plt.legend(['Training', 'Validation'])
    print('Max Acc by:', val_acc.index(max(val_acc))+ int(min(param_values)))


base_model = RandomForestClassifier(random_state=42, n_jobs=-1).fit(X_train, train_targets)
base_train_acc = base_model.score(X_train, train_targets)
base_val_acc = base_model.score(X_val, val_targets)
base_accs = base_train_acc, base_val_acc
base_accs


def test_params(**params):
    model = RandomForestClassifier(random_state=42, n_jobs=-1, **params)
    model.fit(X_train, train_targets)
    train_score = accuracy_score(model.predict(X_train), train_targets)
    val_score = accuracy_score(model.predict(X_val), val_targets)
    return train_score, val_score
n_estimators=[100, 200, 300, 400, 500,600, 700, 800, 900, 1000]
test_param_and_plot('n_estimators',n_estimators)


model = RandomForestClassifier(random_state=42, n_jobs=-1, n_estimators=800)
model.fit(X_train, train_targets)
print(base_accs)
model.score(X_train, train_targets), model.score(X_val, val_targets)


def test_params(**params):
    model = RandomForestClassifier(random_state=42, n_jobs=-1,n_estimators=800, **params)
    model.fit(X_train, train_targets)
    train_score = accuracy_score(model.predict(X_train), train_targets)
    val_score = accuracy_score(model.predict(X_val), val_targets)
    return train_score, val_score
max_depth=[i for i in range(1,32)]
test_param_and_plot('max_depth',max_depth)


model = RandomForestClassifier(random_state=42, n_jobs=-1, n_estimators=800, max_depth=12)
model.fit(X_train, train_targets)
print(base_accs)
model.score(X_train, train_targets), model.score(X_val, val_targets)


def test_params(**params):
    model = RandomForestClassifier(random_state=42, n_jobs=-1,
                                   n_estimators=800, max_depth=12, **params)
    model.fit(X_train, train_targets)
    train_score = accuracy_score(model.predict(X_train), train_targets)
    val_score = accuracy_score(model.predict(X_val), val_targets)
    return train_score, val_score
max_leaf_nodes=[i for i in range(10,200)]
test_param_and_plot('max_leaf_nodes',max_leaf_nodes)


model = RandomForestClassifier(random_state=42, n_jobs=-1,
                               n_estimators=800, max_depth=12,
                               max_leaf_nodes=89)
model.fit(X_train, train_targets)
print(base_accs)
model.score(X_train, train_targets), model.score(X_val, val_targets)


def test_params(**params):
    model = RandomForestClassifier(random_state=42, n_jobs=-1,
                               n_estimators=800, max_depth=12,
                               max_leaf_nodes=89, **params)
    model.fit(X_train, train_targets)
    train_score = accuracy_score(model.predict(X_train), train_targets)
    val_score = accuracy_score(model.predict(X_val), val_targets)
    return train_score, val_score
max_features=[i for i in range(1,11)]
test_param_and_plot('max_features',max_features)


model = RandomForestClassifier(random_state=42, n_jobs=-1,
                               n_estimators=800, max_depth=12,
                               max_leaf_nodes=89, max_features=5)
model.fit(X_train, train_targets)
print(base_accs)
model.score(X_train, train_targets), model.score(X_val, val_targets)


def test_params(**params):
    model = RandomForestClassifier(random_state=42, n_jobs=-1,
                               n_estimators=800, max_depth=12,
                               max_leaf_nodes=89, max_features=5,
                                   **params)
    model.fit(X_train, train_targets)
    train_score = accuracy_score(model.predict(X_train), train_targets)
    val_score = accuracy_score(model.predict(X_val), val_targets)
    return train_score, val_score
min_samples_split=[i for i in range(10,40)]
test_param_and_plot('min_samples_split',min_samples_split)


model = RandomForestClassifier(random_state=42, n_jobs=-1,
                               n_estimators=800, max_depth=12,
                               max_leaf_nodes=89, max_features=5,
                               min_samples_split=11)
model.fit(X_train, train_targets)
print(base_accs)
model.score(X_train, train_targets), model.score(X_val, val_targets)


def test_params(**params):
    model = RandomForestClassifier(random_state=42, n_jobs=-1,
                               n_estimators=800, max_depth=12,
                               max_leaf_nodes=89, max_features=5,
                               min_samples_split=11,**params)
    model.fit(X_train, train_targets)
    train_score = accuracy_score(model.predict(X_train), train_targets)
    val_score = accuracy_score(model.predict(X_val), val_targets)
    return train_score, val_score
min_samples_leaf=[i for i in range(1,150)]
test_param_and_plot('min_samples_leaf',min_samples_leaf)


model = RandomForestClassifier(random_state=42, n_jobs=-1,
                               n_estimators=800, max_depth=12,
                               max_leaf_nodes=89, max_features=5,
                               min_samples_split=11,
                               min_samples_leaf=1)
model.fit(X_train, train_targets)
print(base_accs)
model.score(X_train, train_targets), model.score(X_val, val_targets)


def test_params(**params):
    model = RandomForestClassifier(random_state=42, n_jobs=-1,
                               n_estimators=800, max_depth=12,
                               max_leaf_nodes=89, max_features=5,
                               min_samples_split=11,
                               min_samples_leaf=1,**params)
    model.fit(X_train, train_targets)
    train_score = accuracy_score(model.predict(X_train), train_targets)
    val_score = accuracy_score(model.predict(X_val), val_targets)
    return train_score, val_score
min_impurity_decrease=[1.0e-10,0, 1.0e-9, 1.0e-7, 1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0]
test_param_and_plot('min_impurity_decrease',min_impurity_decrease)


model = RandomForestClassifier(random_state=42, n_jobs=-1,
                               n_estimators=800, max_depth=12,
                               max_leaf_nodes=89, max_features=5,
                               min_samples_split=11,
                               min_samples_leaf=1,
                               min_impurity_decrease=0.0)
model.fit(X_train, train_targets)
print(base_accs)
model.score(X_train, train_targets), model.score(X_val, val_targets)


model = RandomForestClassifier(random_state=42, n_jobs=-1,
                               n_estimators=800, max_depth=12,
                               max_leaf_nodes=89, max_features=5,
                               min_samples_split=11,
                               min_samples_leaf=1,
                               min_impurity_decrease=0.0, class_weight={0:4, 1:1})
model.fit(X_train, train_targets)
print(base_accs)
model.score(X_train, train_targets), model.score(X_val, val_targets)


model = RandomForestClassifier(random_state=42, n_jobs=-1,
                               n_estimators=800, max_depth=12,
                               max_leaf_nodes=89, max_features=5,
                               min_samples_split=11,
                               min_samples_leaf=1,
                               min_impurity_decrease=0.0, class_weight={0:1.5, 1:1})
model.fit(X_train, train_targets)
print(base_accs)
model.score(X_train, train_targets), model.score(X_val, val_targets)


model = RandomForestClassifier(random_state=42, n_jobs=-1,
                               n_estimators=800, max_depth=12,
                               max_leaf_nodes=89, max_features=5,
                               min_samples_split=11,
                               min_samples_leaf=1,
                               min_impurity_decrease=0.0, class_weight={0:1, 1:0})
model.fit(X_train, train_targets)
print(base_accs)
model.score(X_train, train_targets), model.score(X_val, val_targets)


model = RandomForestClassifier(random_state=42, n_jobs=-1,
                               n_estimators=800, max_depth=12,
                               max_leaf_nodes=89, max_features=5,
                               min_samples_split=11,
                               min_samples_leaf=1,
                               min_impurity_decrease=0.0)
model.fit(X_train, train_targets)
print(base_accs)
model.score(X_train, train_targets), model.score(X_val, val_targets)


test_preds=model.predict(test_inputs)
test_preds


pd.value_counts(test_preds) / len(test_preds)


submission['Probability']=test_preds
submission.head()


#Saving the submissions
submission.to_csv('submission1.csv', index=None)


import joblib


financial_distress = {
    'model': model,
    'imputer': imputer,
    'scaler': scaler,
    'input_cols': input_cols,
    'target_col': target_col
}
#all input calls are numeric cols


joblib.dump(financial_distress, 'financial_distress')


financial_distress2 = joblib.load('financial_distress')


val_preds2 = financial_distress2['model'].predict(X_val)
accuracy_score(val_targets, val_preds2)

