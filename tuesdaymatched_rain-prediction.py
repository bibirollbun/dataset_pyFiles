import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.model_selection import KFold, GridSearchCV, train_test_split
from sklearn.metrics import  confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, accuracy_score
from sklearn.pipeline import Pipeline
import lightgbm as lgb


# 1. Load data  --change the link UwU
df_train = pd.read_csv(r'/kaggle/input/playground-series-s5e3/train.csv')
df_test = pd.read_csv(r'/kaggle/input/playground-series-s5e3/test.csv')



# How many data that is missing
df_train.isnull().sum()


df_test.isnull().sum()


# in test data, there is 1 missing data in the winddirection column
# Therefor, we will fill the missing data with the median of the column
df_test['winddirection'].fillna(df_test['winddirection'].median(), inplace=True)
# Therefor,the train data has no missing data, but the test data has 1 missing data in the winddirection column


# Now shall we look for the dataset UwU
df_train.info()
df_train.describe(include='all').T


# All the data is in the right format(int and float), so there no need to label them

def group_day(day):
    if pd.isna(day):
        return 'Other'
    if 1 <= day <= 90:
        return 'Spring'
    elif 91 <= day <= 181:
        return 'Summer'
    elif 182 <= day <= 273:
        return 'Fall'
    elif 274 <= day <= 365:
        return 'Winter'
    else:
        return 'Other'

df_train['season'] = df_train['day'].apply(group_day)
df_test['season'] = df_test['day'].apply(group_day)

# Now we have a new column called season, which is the season of the day
df_train.drop(columns=['day'], inplace=True)
df_test.drop(columns=['day'], inplace=True)




def group_winddirection(degree):
    if (degree >= 315) or (degree < 45):
        return 'N'
    elif 45 <= degree < 135:
        return 'E'
    elif 135 <= degree < 225:
        return 'S'
    elif 225 <= degree < 315:
        return 'W'
    else:
        return 'Unknown'

df_train['wind_direction_label'] = df_train['winddirection'].apply(group_winddirection)
df_test['wind_direction_label'] = df_test['winddirection'].apply(group_winddirection)

df_train.drop(columns=['winddirection'], inplace=True)
df_test.drop(columns=['winddirection'], inplace=True)
# Now we have a new column called wind_direction_label, which is the direction of the wind


# Label Encoded
wind = LabelEncoder()
season = LabelEncoder()

df_train['season'] = season.fit_transform(df_train['season'])
df_test['season'] = season.transform(df_test['season'])

df_train['wind_direction_label'] = wind.fit_transform(df_train['wind_direction_label'])
df_test['wind_direction_label'] = wind.transform(df_test['wind_direction_label'])
wind_map = list(zip(wind.classes_, range(len(wind.classes_))))
season_map = list(zip(season.classes_, range(len(season.classes_))))

print("Wind:")
for original, encoded in wind_map:
    print(f"{original} -> {encoded}")
print('---------------------------------')
for original, encoded in season_map:
    print(f"{original} -> {encoded}")



# 2. Data visualization
hist_plot = df_train.drop(columns=['id','rainfall'])
fig, axes = plt.subplots(4, 3, figsize=(20, 10))
axes = axes.flatten()

for ax, i in zip(axes, hist_plot):
    sns.histplot(data = df_train,x = i,edgecolor = 'black',ax = ax, kde=True)
    ax.set_title(f'Histogram for {i}')
    for i in ax.get_xticklabels():
        i.set_rotation(45)
plt.tight_layout()
plt.show()


# Next let see the correlation between the data
X1 = df_train.drop(columns=['id'])
plt.figure(figsize=(10, 10))
sns.heatmap(X1.corr(), annot=True)
plt.title('Correlation between the data')
plt.show()

# These variences are all affect the rainfall outcomes, but it seem like cloud and humidity is the most effective variences



# Let see the p-value between variences to the target value
from scipy import stats
for i in  X1:
    coef, p_value = stats.pearsonr(df_train[i], df_train['rainfall'])
    print(f"{i}: coef: {coef} \t p_value: {p_value}")
# mintemp has the highest p-value(>0.05), so we can drop them



# 3. Data preprocessing
X2 = df_train.drop(columns=['id', 'mintemp', 'rainfall'])
Y2 = df_train['rainfall']


Input = [('scale', StandardScaler()),('classifier', LogisticRegression())]
pipe = Pipeline(Input)
models = [
    ('LR',LogisticRegression()),
    ('RFC',RandomForestClassifier()),
    ('SVC',SVC()),
    ('XGBoost',xgb.XGBClassifier()),
    ('AdaBoost',AdaBoostClassifier()),
    ('LGBM', lgb.LGBMClassifier())
]
param_grid = [
    {
        'classifier': [LogisticRegression()],
        'classifier__penalty': ['l1', 'l2'],
        'classifier__solver': ['liblinear', 'saga'],
        'classifier__random_state': [5, 25]
    },
    {
        'classifier': [RandomForestClassifier()],
        'classifier__n_estimators': [50, 100],
        'classifier__max_depth': [5, 7, 10],
        'classifier__random_state': [5, 25]
    },
    {
        'classifier': [SVC()],
        'classifier__C': [0.1, 1, 5, 10],
        'classifier__kernel': ['linear', 'rbf'],
        'classifier__random_state': [5, 25]
    },
    {
        'classifier': [xgb.XGBClassifier()],
        'classifier__n_estimators': [50, 100],
        'classifier__learning_rate': [0.01, 0.1, 0.2, 0.5],
        'classifier__max_depth': [3, 5, 10],
        'classifier__random_state': [5, 25]
    },
    {
        'classifier': [AdaBoostClassifier()],
        'classifier__n_estimators': [50, 100],
        'classifier__learning_rate': [0.01, 0.1, 0.2, 0.5],
        'classifier__algorithm': ['SAMME', 'SAMME.R'],
        'classifier__random_state': [5, 25]
    },
    {
        'classifier': [lgb.LGBMClassifier()],
        'classifier__n_estimators': [50, 100, 200],
        'classifier__learning_rate': [0.01, 0.1, 0.2, 0.5],
        'classifier__boosting_type': ['gbdt', 'dart', 'goss'],
        'classifier__random_state': [5, 25]
    }
]


k_cv = KFold(n_splits=5)
grid_search = GridSearchCV(pipe, param_grid, cv=k_cv, scoring='accuracy')
grid_search.fit(X2, Y2)


print(f"Best params: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_}")

# AdaBoostClassifier is the best model with the best score of 0.87



print('Confusion matrix for train subset:')
print(confusion_matrix(Y2, grid_search.predict(X2)))
print('-----------------------------------------------------')
print('Confusion matrix display:')
print(ConfusionMatrixDisplay(confusion_matrix(Y2, grid_search.predict(X2)), display_labels=['No Rain','Rain']).plot())
print('-----------------------------------------------------')
# Basically, the model is good at predicting with accuracy of train subset is 0.87 but it decraese approximately 0.01


# ROC
y_probs = grid_search.predict_proba(X2)[:, 1]  
fpr, tpr, _ = roc_curve(Y2, y_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='grey', linestyle='--') 
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()
# AUC is 0.90, which is good but decreased 0.03 compare to the last model



df_test['predicted'] = grid_search.predict(df_test.drop(columns=['id', 'mintemp']))
df_submission = pd.DataFrame(
        {
            "id": df_test['id'],
            "rainfall": df_test['predicted']
        }
    )
print(df_submission)
df_submission['rainfall'].value_counts()    

