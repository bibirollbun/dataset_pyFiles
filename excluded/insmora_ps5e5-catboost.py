import pandas as pd
import numpy as np
import seaborn as sns
import time
import warnings
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import mean_squared_log_error, mean_squared_error
from catboost import Pool, CatBoostRegressor

warnings.filterwarnings("ignore")


train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv", index_col="id")
test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv", index_col="id")


train.head(5)


train.info()


# histograms for all numeric columns
train.hist(bins=30, figsize=(15, 10), layout=(len(train.columns) // 3 + 1, 3))
plt.tight_layout()
plt.show()



def add_features(df):
    df['Sex'] = df['Sex'].astype('category')
    df['Height'] = df['Height']/100
    df['BMI'] = df['Weight']/(df['Height']**2)
    bmi_bins = [0, 18.5, 25, 30, 35, 40, np.inf]
    bmi_labels = ['Underweight', 'Normal', 'Overweight', 'Obese_I', 'Obese_II', 'Obese_III']
    df['BMI_Category'] = pd.cut(df['BMI'], bins=bmi_bins, labels=bmi_labels, right=False)
    df['Intensity']=df['Heart_Rate']*df['Duration']
    df['HR_per_kg'] = df['Heart_Rate'] / df['Weight']
    bins_age = [0, 12, 18, 35, 50, 65, 100]
    labels_age = ['Child', 'Teen', 'Young_Adult', 'Adult', 'Middle_Aged', 'Senior']
    df['Age_Group'] = pd.cut(df['Age'], bins=bins_age, labels=labels_age, right=False)
    df['BMI*Duration'] = df['BMI']*df['Duration']
    df['BMI*HR']= df['BMI']*df['Heart_Rate']
    df['BMI*Intensity']= df['BMI']*df['Intensity']
    df['Temp*HR'] = df['Body_Temp'] * df['Heart_Rate']
    df['Body_Temp*Duration']=df['Body_Temp']*df['Duration']
    mean_height_by_sex = df.groupby('Sex')['Height'].transform('mean')
    df['Height_vs_SexMean'] = df['Height'] / mean_height_by_sex

    return df

train = add_features(train)
test = add_features(test)


# histograms for all numeric columns
train.hist(bins=30, figsize=(15, 10), layout=(len(train.columns) // 3 + 1, 3))
plt.tight_layout()
plt.show()


# Categorical columns
categoricals = train.select_dtypes(include='category').columns.tolist()


# One boxplot per categorical feature against Calories
for cat in categoricals:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=train[cat], y=train['Calories'])
    plt.title(f'Calories by {cat}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# Training and Test set

X = train.drop(columns='Calories')
y = np.log1p(train['Calories'])
X_test = test




# StratifiedKFold on Calorie bins
bins = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='quantile')
calorie_bins = bins.fit_transform(train[['Calories']]).astype(int).flatten()

cat_params = {
    'iterations': 2500,
    'learning_rate': 0.02,
    'depth': 10,
    'loss_function': 'RMSE',
    'l2_leaf_reg': 3,
    'random_seed': 42,
    'eval_metric': 'RMSE',
    'early_stopping_rounds': 200,
    'verbose': 0,
    'task_type': 'CPU'

}

cat_features = ['Age_Group', 'Sex', 'BMI_Category']
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

cat_oof = np.zeros(len(X))
cat_preds = np.zeros(len(X_test))
cat_scores = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, calorie_bins)):
    print(f"Fold {fold+1}")

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    train_pool = Pool(X_train, y_train, cat_features=cat_features)
    val_pool = Pool(X_val, y_val, cat_features=cat_features)

    model = CatBoostRegressor(**cat_params)
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)

    cat_oof[val_idx] = model.predict(val_pool)
    cat_preds += model.predict(X_test) / skf.n_splits

    # Calculate RMSLE
    fold_score = np.sqrt(mean_squared_log_error(
        np.expm1(y_val), np.expm1(cat_oof[val_idx])
    ))
    print(f"Fold {fold+1} - CatBoost RMSLE: {fold_score:.5f}")
    cat_scores.append(fold_score)

print(f"\nCatBoost Mean RMSLE: {np.mean(cat_scores):.5f}")





# convert predictions back from log1p
final_predictions = np.expm1(cat_preds)

# submission DataFrame
submission = pd.DataFrame({
    'id': test.index,
    'Calories': final_predictions
})

# save
submission.to_csv('submission.csv', index=False)

submission.head(10)



