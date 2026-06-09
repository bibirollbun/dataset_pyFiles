!pip install shap -q
!pip install plotly -q


import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style('darkgrid')
plt.rcParams["figure.figsize"] = (16, 6)

import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_contour, plot_slice
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import RobustScaler
import category_encoders as ce
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings("ignore")
import shap


train = pd.read_excel(r'/kaggle/input/tutors-lessons-prices-prediction/train.xlsx', index_col=0)
test = pd.read_excel(r'/kaggle/input/tutors-lessons-prices-prediction/test.xlsx', index_col=0)


train.head()


train.info()


train.describe()


train.columns


new_cols = {
    'предмет': 'subject',
    'ФИО': 'name',
    'Ученая степень 1': 'title_1',
    'Ученое звание 1': 'grade_1',
    'Ученая степень 2': 'title_2',
    'Ученое звание 2': 'grade_2'
}

train.rename(columns=new_cols, inplace=True)
test.rename(columns=new_cols, inplace=True)


print(f'Train shape: {train.shape}')
print(f'Test shape: {test.shape}')


def info(col):
    print('-' * 30, f'{col}', '-' * 30, sep='')
    
    print(f'Train dtype: {train[col].dtype}, Test dtype: {test[col].dtype}')
    
    print(f'NaN values: Train: {train[col].isna().sum()}, Test: {test[col].isna().sum()}')
    
    unique_values_train = train[col].unique()
    unique_values_test = test[col].unique()
    
    print(f'Nunique: Train: {len(unique_values_train)}, Test: {len(unique_values_test)}')
    if len(unique_values_train) < 20:
        print(f'Unique values: {", ".join(map(str, unique_values_train))}')
    
    print('-' * (60 + len(col)), end='\n'*2)


for col in train.columns[:-1]:
    info(col)


def parse_tags_categories(train, test, col):
    '''
    Processes a column containing lists of tags or categories in both train and test DataFrames. 
    It extracts unique tags, creates binary columns for each tag, and optionally removes the original column.
    '''
    unique = set(tag for tags in train[col].apply(eval) for tag in tags)
    
    for val in unique:
        train[val] = train[col].apply(lambda x: int(val in eval(x)))
        test[val] = test[col].apply(lambda x: int(val in eval(x)))

    # train[col] = train[col].apply(lambda x: '_'.join(eval(x)))
    # test[col] test[col].apply(lambda x: '_'.join(eval(x)))
    train.drop(columns=col, inplace=True)
    test.drop(columns=col, inplace=True)
    
    return train, test, unique


train, test, unique_tags = parse_tags_categories(train, test, col='tutor_head_tags')


train.tutor_rating.value_counts()


train.loc[train.tutor_rating.isna(), 'tutor_rating'] = train.tutor_rating.median()
test.loc[test.tutor_rating.isna(), 'tutor_rating'] = test.tutor_rating.median()


def pattern_fill(df, col, pattern):
    '''
    Processes a specified column in a DataFrame (df) and replaces values that match a given pattern with 'No'. 
    All other values are replaced with 'Yes'.
    '''
    df[col] = df[col].apply(
        lambda x: 'No' if str(x).strip().lower() == pattern else 'Yes'
    )
    return df


train.description.sample(5)


train = pattern_fill(train, 'description', 'репетитор не предоставил о себе дополнительных сведений')
test = pattern_fill(test, 'description', 'репетитор не предоставил о себе дополнительных сведений')


train.experience = train.experience.str\
.extract(r'(\d+)', expand=False)\
.fillna(0)\
.astype(int)

test.experience = test.experience.str\
.extract(r'(\d+)', expand=False)\
.fillna(0)\
.astype(int)


train.experience_desc.sample(5)


train = pattern_fill(train, 'experience_desc', 'репетитор не предоставил информацию об опыте работы')
test = pattern_fill(test, 'experience_desc', 'репетитор не предоставил информацию об опыте работы')


train.Education_1.value_counts()


for index in range(2, 4):
    train[f'Education_{index}'] = train[f'Education_{index}'].apply(lambda x: 'Yes' if x is not np.nan else 'No')
    test[f'Education_{index}'] = test[f'Education_{index}'].apply(lambda x: 'Yes' if x is not np.nan else 'No')


print(f'Desc 1: {train.Desc_Education_1[0]}')
print(f'Desc 2: {train.Desc_Education_2[2]}')


def extract_education_info(df, column, index):
    df[f'educ_end_{index}'] = df[column].str.extract(r'Год окончания: (\d{4})', expand=False).fillna(1900).astype(int)
    df[f'fac_{index}'] = df[column].str.extract(r'Факультет: (.*?),', expand=False).fillna('No')
    df[f'spec_{index}'] = df[column].str.extract(r'Специальность: (.*?),', expand=False).fillna('No')
    return df


for index in range(1, 4):
    train = extract_education_info(train, f'Desc_Education_{index}', index)
    test = extract_education_info(test, f'Desc_Education_{index}', index)


train.categories.sample(5)


train, test, unique_categories = parse_tags_categories(train, test, 'categories')


train.loc[train.status.isna(), 'status'] = train.status.value_counts().index.str.strip()[0]
test.loc[test.status.isna(), 'status'] = test.status.value_counts().index.str.strip()[0]


drop_col = [
    'name', 'Desc_Education_1', 'Desc_Education_2', 'Desc_Education_3', 'Desc_Education_4', 'Desc_Education_5',
    'Desc_Education_6', 'Education_4', 'Education_5', 'Education_6', 'grade_1', 'title_1', 'grade_2', 'title_2'
]
train.drop(columns=drop_col, inplace=True)
test.drop(columns=drop_col, inplace=True)


sum(train.isna().sum() != 0)


continuous = [
    'mean_price'
]
discrete = [
    'tutor_rating', 'tutor_reviews', 'experience',
    'educ_end_1', 'educ_end_2', 'educ_end_3'
]
cat_param = [
    'subject', 'description', 'experience_desc', 'Education_1', 'Education_2', 'Education_3', 'status',
    'fac_1', 'spec_1', 'fac_2', 'spec_2', 'fac_3', 'spec_3'
]


train['dataset'] = 'train'
test['dataset'] = 'test'

visualisation_df = pd.concat([train, test], axis=0)

train.drop(columns='dataset', inplace=True)
test.drop(columns='dataset', inplace=True)


for feature in discrete + cat_param[:-6]:
    if feature == 'Education_1':
        continue
    sns.countplot(data = visualisation_df, x=feature, hue='dataset', palette='summer')
    plt.xticks(rotation=45)
    plt.title(f'Distribution of {feature}')
    plt.show()


for unique in [unique_tags, unique_categories]:
    long_df = visualisation_df.melt(id_vars=['dataset'], value_vars=list(unique), 
                      var_name='Предмет', value_name='Встречается')
    
    long_df = long_df[long_df['Встречается'] == 1]
    sns.countplot(data=long_df, x='dataset', hue='Предмет', palette='viridis')
    plt.xlabel('Датасет')
    plt.ylabel('Количество')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()


for features in continuous+discrete:
    fig, ax = plt.subplots(nrows=1, ncols=2)
    sns.boxplot(data=visualisation_df, y='dataset', x=features, ax=ax[0], orient='h', palette='hot')
    sns.violinplot(data=visualisation_df, y='dataset', x=features, ax=ax[1], palette='summer')
    plt.show()


correlation_matrix = train.loc[:, continuous+discrete+cat_param].corr(numeric_only=True) 
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, annot = True, fmt = '.2f', cmap = 'coolwarm', mask=mask)
plt.title('Corr matrix')
plt.show()


X = train.drop(columns=['mean_price'])
target = train['mean_price']


TE = ce.TargetEncoder(smoothing=18, cols=cat_param)

train_encoded = TE.fit_transform(X[cat_param], target)

test_encoded = TE.transform(test[cat_param])

for col in cat_param:
    X[f'TE_{col}'] = train_encoded[col]
    test[f'TE_{col}'] = test_encoded[col]


for cat_col in cat_param:
    for num_col in discrete:
        intera_col = f'{cat_col}_x_{num_col}'
        X[intera_col] = X[cat_col].astype(str) + '_' + X[num_col].astype(str)
        test[intera_col] = test[cat_col].astype(str) + '_' + test[num_col].astype(str)
        X[intera_col] = X[intera_col].astype('category').cat.codes
        test[intera_col] = test[intera_col].astype('category').cat.codes
    X[cat_col] = X[cat_col].astype('category').cat.codes
    test[cat_col] = test[cat_col].astype('category').cat.codes


scaler = RobustScaler().set_output(transform="pandas")

scaler.fit(X)

X = scaler.transform(X)
test = scaler.transform(test)


def objective_xgb(trial):
    max_depth=trial.suggest_int("max_depth", 4, 6)
    learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True)
    min_child_weight=trial.suggest_int("min_child_weight", 1, 15)
    subsample=trial.suggest_float("subsample", 0.5, 1.0)
    colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0)
    n_estimators=trial.suggest_int("n_estimators", 500, 1000)
    reg_alpha=trial.suggest_float("reg_alpha", 0, 1)
    reg_lambda=trial.suggest_float("reg_lambda", 0.4, 1.5)

    model = XGBRegressor(
        tree_method="gpu_hist",
        random_state=42,
        max_depth=max_depth,
        learning_rate=learning_rate,
        min_child_weight=min_child_weight,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        n_estimators=n_estimators,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda
    )

    scores = cross_val_score(
        model, 
        X, 
        target, 
        cv=4,  
        scoring="neg_mean_squared_error"  
    )

    rmse_scores = -scores  
    
    return rmse_scores.mean()

study = optuna.create_study(direction="minimize")
study.optimize(objective_xgb, n_trials=300)


plot_optimization_history(study).show()
plot_param_importances(study).show()
plot_contour(study, params=["max_depth", "learning_rate"]).show()
plot_contour(study, params=["n_estimators", "learning_rate"]).show()
plot_contour(study, params=["n_estimators", "min_child_weight"]).show()
plot_contour(study, params=["n_estimators", "reg_lambda"]).show()
plot_slice(study, params=["max_depth", "learning_rate", "n_estimators"]).show()


model_xgb = XGBRegressor(
    tree_method="gpu_hist",
    enable_categorical=True,
    random_state=42,
    **study.best_trial.params
)
model_xgb.fit(
    X, 
    target,
    eval_metric="rmse",
    verbose=False
)


explainer = shap.TreeExplainer(model_xgb)
shap_values = explainer.shap_values(X)

shap.summary_plot(shap_values, X)


shap.initjs()
print('Correct answer:', target.iloc[2])
shap.force_plot(explainer.expected_value, shap_values[2,:], X.iloc[2,:])


pred = model_xgb.predict(test)


submission = pd.DataFrame({
    'index': test.index,
    'mean_price': pred
})
submission.to_csv('baseline.csv', index=False)


submission

