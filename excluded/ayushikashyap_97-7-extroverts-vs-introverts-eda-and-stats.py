import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import normaltest, skew, kurtosis
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.metrics import mean_absolute_error, accuracy_score

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


train = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv")


print(f"Number of rows in train dataset {len(train)} and in test dataset {len(test)}.")


train.head(2)


test.head(2)


train.info()


train.describe()


test.info()


test.describe()


value_count = train['Personality'].value_counts()
value_count


percentage_count = train['Personality'].value_counts(normalize = True) * 100
percentage_count


plt.figure(figsize = (6,4))
sns.countplot(data=train, x='Personality', palette='pastel')
plt.title('Personality Distribution')
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


if percentage_count.max() > 70:
    print("⚠️ Warning: The target variable appears to be imbalanced.")


categorical_cols = train.select_dtypes(include=['object']).columns.drop('Personality')

for col in categorical_cols:
    plt.figure(figsize=(7, 4))
    sns.countplot(data=train, x=col, hue='Personality', palette='Set2')
    plt.title(f'{col} vs Target')
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()


numeric_cols = train.select_dtypes(include=['float64', 'int64']).columns

for col in numeric_cols:
    if col != 'id':
        plt.figure(figsize=(7, 4))
        sns.boxplot(data=train, x='Personality', y=col, palette='coolwarm')
        plt.title(f'{col} distribution by Target Class')
        plt.tight_layout()
        plt.show()


from scipy.stats import chi2_contingency

for col in categorical_cols:
    contingency = pd.crosstab(train[col], train['Personality'])
    chi2, p, dof, expected = chi2_contingency(contingency)
    print(f"Chi-square test between '{col}' and target: p-value = {p:.4f}")


def extended_describe(df, columns):
    stats = []
    for col in columns:
        data = df[col].dropna() # Drop NAs from the dataset
        n = data.count() # Number of valid rows
        na = df[col].isna().sum() # Number of NA rows
        mean = data.mean() # Mean
        std = data.std() # Standard Deviation
        se_mean = std / np.sqrt(n) # Standard Error of Mean
        iqr = data.quantile(0.75) - data.quantile(0.25) # Inter Qunatile Range
        row = {
            'feature': col,
            'n': n,
            'na': na,
            'mean': mean,
            'sd': std,
            'se_mean': se_mean,
            'IQR': iqr,
            'skewness': skew(data), # Skewness
            'kurtosis': kurtosis(data), # Kutosis: Tail Heaviness
            'p01': data.quantile(0.01),
            'p05': data.quantile(0.05),
            'p10': data.quantile(0.10),
            'p25': data.quantile(0.25),
            'p50': data.quantile(0.50),
            'p75': data.quantile(0.75),
            'p90': data.quantile(0.90),
            'p95': data.quantile(0.95),
            'p99': data.quantile(0.99),
            'p100': data.max()
        }
        stats.append(row)
    return pd.DataFrame(stats)


numerical_cols = train.select_dtypes(include='number').columns
eda_table = extended_describe(train, numerical_cols)
print(eda_table)


def miss_data(data):
    miss_df = ((data == 0).sum().to_frame())
    miss_df = miss_df.rename(columns = {0:'Zeroes'})
    miss_df.index.name = 'Features'
    miss_df['NaN'] = (data.isnull()).sum()
    miss_df['None'] = (data == None).sum()
    miss_df['Total'] = miss_df['Zeroes'] + miss_df['NaN'] + miss_df['None']
    miss_df['Percent'] = 100 * miss_df['Total'] / len(data)
    miss_df['Type'] = [data[i].dtype for i in miss_df.index]
    return miss_df.sort_values(ascending = False, by = 'Percent')


miss_df = miss_data(data = train)
miss_df


train['Personality_encoded'] = train['Personality'].map({'Introvert': 0, 'Extrovert': 1})
corr_matrix = train.corr(numeric_only=True)

plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()


corr_target = corr_matrix['Personality_encoded'].sort_values(ascending=False)
print(corr_target)


numerical_columns_train = train.select_dtypes(include=['float64','int64']).columns
numerical_columns_test = test.select_dtypes(include=['float64','int64']).columns
train[numerical_columns_train] = train[numerical_columns_train].fillna(train[numerical_columns_train].median())
test[numerical_columns_test] = test[numerical_columns_test].fillna(test[numerical_columns_test].median())


category_columns = [col for col in train.select_dtypes(include='object').columns if col != 'Personality']


train[category_columns] = train[category_columns].fillna("Unknown")
train.replace("None","Unknown", inplace = True)
test[category_columns] = test[category_columns].fillna("Unknown")
test.replace("None","Unknown", inplace = True)


train['Social_activity_score'] = (
    train['Going_outside'] +
    train['Social_event_attendance'] +
    train['Post_frequency'] +
    train['Friends_circle_size']
)

test['Social_activity_score'] = (
    test['Going_outside'] +
    test['Social_event_attendance'] +
    test['Post_frequency'] +
    test['Friends_circle_size']
)


train.drop(columns = ['Going_outside','Social_event_attendance','Post_frequency','Friends_circle_size'])
test.drop(columns = ['Going_outside','Social_event_attendance','Post_frequency','Friends_circle_size'])


y = train['Personality']
x = train.drop(columns = ['Personality','Personality_encoded'], axis = 1)
features = x.columns
x_test = test[features]

x_train, x_valid, y_train, y_valid = train_test_split(x, y, train_size = 0.8, test_size = 0.2, random_state = 0)


model1 = CatBoostClassifier(iterations=300,learning_rate=0.05,depth=6,cat_features=['Drained_after_socializing','Stage_fear'],verbose=100,random_seed=42)
model2 = CatBoostClassifier(iterations=500,learning_rate=0.05,depth=6,cat_features=['Drained_after_socializing','Stage_fear'],verbose=100,random_seed=42)
model3 = CatBoostClassifier(iterations=1000,learning_rate=0.05,loss_function = 'MultiClass',depth=5,cat_features=['Drained_after_socializing','Stage_fear'],verbose=100,random_seed=42)
model4 = CatBoostClassifier(iterations=1500,learning_rate=0.01,depth=5,cat_features=['Drained_after_socializing','Stage_fear'],verbose=100,random_seed=42)
model5 = CatBoostClassifier(iterations=2000,learning_rate=0.01,loss_function = 'MultiClass',depth=8,cat_features=['Drained_after_socializing','Stage_fear'],verbose=100,random_seed=42)

models =[model1, model2, model3, model4, model5]


def score_model(model, x_t=x_train, x_v=x_valid, y_t=y_train, y_v=y_valid):
    model.fit(x_t, y_t)
    preds = model.predict(x_v)
    return accuracy_score(y_v, preds)

for i in range(0, len(models)):
    acc = score_model(models[i])
    print("Model %d Accuracy: %.2f%%" % (i+1, acc * 100))


final_model = CatBoostClassifier(iterations=2000,learning_rate=0.01,loss_function = 'MultiClass',depth=8,cat_features=['Drained_after_socializing','Stage_fear'],verbose=100,random_seed=42)
final_model.fit(x, y)
preds = final_model.predict(x_test)


preds = preds.ravel()
print(preds.shape)


submission = test.copy()
submission['Predicted_Personality'] = preds
submission[['id', 'Predicted_Personality']].head()
submission.rename(columns = {'Predicted_Personality':'Personality'}, inplace = True)


submission["Personality"].value_counts()


submission["Personality"].value_counts(normalize=True) * 100


submission[['id','Personality']].to_csv("submission.csv", index=False)




