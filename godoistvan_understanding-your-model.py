import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import roc_auc_score

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier,BaggingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
warnings.filterwarnings('ignore')



df_test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')
df_train = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
original = pd.read_csv('/kaggle/input/rainfall-prediction-using-machine-learning/Rainfall.csv')


df_train.head()


df_train.describe().T


df_train.info()


df_test.info()


original.info()


df_train.isnull().sum()


df_train.duplicated().sum()


df_train.shape


original.columns = original.columns.str.strip()


original['rainfall']=original['rainfall'].map({'Yes':1,'No':0})


df_test=df_test.drop(['id'],axis=1)
df_train=df_train.drop(['id'],axis=1)


print(f'Original shape{original.shape}')
print(f'Train shape{df_train.shape}')
print(f'Test shape{df_test.shape}')


training_color = "#1f77b4"
test_color = "#ff7f0e"
original_color = "#2ca02c"


for i in range(df_test.shape[1]):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    sns.histplot(df_train.iloc[:, i], kde=True, color=training_color, ax=axs[0])
    axs[0].set_title('Training Data')

    sns.histplot(df_test.iloc[:, i], kde=True, color=test_color, ax=axs[1])
    axs[1].set_title('Test Data')

    sns.histplot(original.iloc[:, i], kde=True, color=original_color, ax=axs[2])
    axs[2].set_title('Original Data')

    plt.tight_layout()
    plt.show()



import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

numerical_variables = ['winddirection', 'pressure', 'maxtemp', 'temparature',
                       'mintemp', 'dewpoint', 'humidity', 'cloud', 'sunshine', 'windspeed']
target_variable = 'rainfall'
categorical_variables = []

for var in numerical_variables:
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    sns.boxplot(x=df_train[var], color=training_color)
    plt.title(f'Training Data: {var}')

    plt.subplot(1, 3, 2)
    sns.boxplot(x=df_test[var], color=test_color)
    plt.title(f'Test Data: {var}')

    plt.subplot(1, 3, 3)
    sns.boxplot(x=original[var], color=original_color)
    plt.title(f'Original Data: {var}')

    plt.tight_layout()
    plt.show()



import matplotlib.pyplot as plt
for var1 in numerical_variables:
    for var2 in numerical_variables:
        if var1 != var2:
            plt.figure(figsize=(15, 5))

            plt.subplot(1, 3, 1)
            sns.scatterplot(x=df_train[var1], y=df_train[var2], hue=df_train[target_variable], palette="viridis", alpha=0.7)
            plt.title(f'Training Data: {var1} vs {var2}')

            plt.tight_layout()
            plt.show()



import matplotlib.pyplot as plt
import seaborn as sns

corr_train = df_train.corr()
corr_test = df_test.corr()
corr_original = original.corr()

fig, axs = plt.subplots(1, 3, figsize=(20, 6))

sns.heatmap(corr_train, annot=True, cmap="coolwarm", ax=axs[0])
axs[0].set_title("Correlation Heatmap: Training Data")

sns.heatmap(corr_test, annot=True, cmap="coolwarm", ax=axs[1])
axs[1].set_title("Correlation Heatmap: Test Data")

sns.heatmap(corr_original, annot=True, cmap="coolwarm", ax=axs[2])
axs[2].set_title("Correlation Heatmap: Original Data")

plt.tight_layout()
plt.show()





def compare_correlations(corr1, corr2, name1, name2):
    print(f"\nCorrelations for {name1}:")
    print(corr1)
    print(f"\nCorrelations for {name2}:")
    print(corr2)
    print(f"\nDelta between {name1} and {name2}:")
    print(corr1 - corr2)

compare_correlations(corr_train, corr_test, "Training Data", "Test Data")
compare_correlations(corr_train, corr_original, "Training Data", "Original Data")
compare_correlations(corr_test, corr_original, "Test Data", "Original Data")




df_train = pd.concat([df_train, original], ignore_index=True)
print(df_train.head())



df_train.dropna(subset=['rainfall'], inplace=True)


from imblearn.over_sampling import SMOTE

X = df_train.drop('rainfall', axis=1)
y = df_train['rainfall']

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X, y)




X=df_train.drop(['rainfall'],axis=1)
y=df_train['rainfall']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


from scipy.stats import skew
print("Skewness:", skew(y))
print("Variance:",y.var())
print("Standard Deviation",y.std())
print("Mean:",y.mean())
print("Min",y.min())
print("Max",y.max())


import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

mean = y.mean()
std = y.std()

x = np.linspace(0, 3, 1000)
y = stats.norm.pdf(x, loc=mean, scale=std)

plt.figure(figsize=(8, 5))
plt.plot(x, y, label='Normal Distribution', color='blue')
plt.title('Gaussian Distribution')
plt.xlabel('Value')
plt.ylabel('Probability Density')

plt.axvline(mean, color='red', linestyle='--', label=f'Mean = {mean:.2f}')

plt.fill_between(x, y, where=((x > mean - std) & (x < mean + std)),
                 color='orange', alpha=0.5, label='Â±1 Std Dev')

plt.legend()
plt.tight_layout()
plt.show()



print(X_train.shape)
print(X_train_smote.shape)


models = {
    'Logistic_Reg' : LogisticRegression(),
    'SVC' : LinearSVC(),
    'DT' : DecisionTreeClassifier(),
    'Ada' : AdaBoostClassifier(),
    'GB' : GradientBoostingClassifier(),
    'BG' :BaggingClassifier(),
    'RF' : RandomForestClassifier(),
    'XGB': XGBClassifier(),
    'Cat' : CatBoostClassifier(verbose=0),
    'LGB': LGBMClassifier(verbose=0),
}


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = {}
for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
    cv_scores[name] = scores
    print(f"{name}: Mean ROC-AUC = {np.mean(scores):.4f}, Std Dev = {np.std(scores):.4f}")

print("\nDetailed Scores:")
for name, scores in cv_scores.items():
    print(f"{name}: {scores}")


lgbmmodel = LGBMClassifier(
    lambda_l1=9.174588245873307e-07,
    lambda_l2=3.557675148244878e-07,
    num_leaves=165,
    feature_fraction=0.454197330888075,
    bagging_fraction=0.9684040684288668,
    bagging_freq=7,
    min_child_samples=15,
    learning_rate=0.016012411698713428,
    n_estimators=352,
    max_depth=11,
    min_split_gain=0.015419477187216356
)
lgbmmodel.fit(X_train_smote, y_train_smote)


catmodel = CatBoostClassifier(learning_rate=0.03352889020153416,
                           depth=9,
                           iterations=799,
                           l2_leaf_reg=0.006338341488887962,
                           bagging_temperature=0.9980912031384876,
                           border_count=239,
                           verbose=0)
catmodel.fit(X_train_smote, y_train_smote)


import pandas as pd
test_df = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')
test_ids = test_df['id']
X_test = test_df.drop(columns=['id'])
final_predictions = lgbmmodel.predict_proba(X_test)[:, 1]
cat_predictions=catmodel.predict_proba(X_test)[:,1]
submission = pd.DataFrame({
    'id': test_ids,
    'prediction': final_predictions
})
submission1 = pd.DataFrame({
    'id': test_ids,
    'prediction': final_predictions
})
submission.to_csv('lgbm.csv', index=False)
submission1.to_csv('cat.csv',index=False)


import numpy as np
import matplotlib.pyplot as plt

feature_importance_lgbm = lgbmmodel.feature_importances_
feature_names_lgbm = X_train_smote.columns
sorted_idx_lgbm = np.argsort(feature_importance_lgbm)[::-1]
sorted_importance_lgbm = feature_importance_lgbm[sorted_idx_lgbm]
sorted_names_lgbm = feature_names_lgbm[sorted_idx_lgbm]

feature_importance_cat = catmodel.feature_importances_
feature_names_cat = X_train_smote.columns
sorted_idx_cat = np.argsort(feature_importance_cat)[::-1]
sorted_importance_cat = feature_importance_cat[sorted_idx_cat]
sorted_names_cat = feature_names_cat[sorted_idx_cat]

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

axes[0].bar(sorted_names_lgbm, sorted_importance_lgbm)
axes[0].set_xticklabels(sorted_names_lgbm, rotation=90)
axes[0].set_title("LGBM Feature Importance")
axes[0].set_ylabel("Importance")

axes[1].bar(sorted_names_cat, sorted_importance_cat)
axes[1].set_xticklabels(sorted_names_cat, rotation=90)
axes[1].set_title("CatBoost Feature Importance")
axes[1].set_ylabel("Importance")

plt.tight_layout()
plt.show()




import shap
explainer = shap.Explainer(catmodel)
shap_values = explainer(X_train_smote)
shap.plots.waterfall(shap_values[0])
shap.summary_plot(shap_values, X_train_smote)
shap.dependence_plot("temparature", shap_values.values, X_train_smote)
shap.force_plot(explainer.expected_value, shap_values.values[0,:], X_train_smote.iloc[0,:])

