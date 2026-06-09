import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


traindata = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv').drop(columns=['id'])
testdata = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')


traindata.head()


testdata.head()


traindata.isnull().sum()


testdata.isnull().sum()


traindata.info()


testdata.info()


traindata.duplicated().sum()


traindata = traindata.drop_duplicates()


testdata.duplicated().sum()


traindata.describe().T


testdata.describe().T


def features_engineering(df):
    df = df.copy()
    if df['Sex'].dtype == 'object':
        df['Sex'] = df['Sex'].replace({'male': 1, 'female': 0})
        
    df['BMI'] = np.round(df['Weight'] / ((df['Height']/100)**2), 2)
    # Mifflin-St Jeor(1990)
    # For men: BMR = 10W + 6.25H – 5A + 5
    # For women: BMR = 10W + 6.25H – 5A – 161
    # BMR Calculator: Learn Your Basal Metabolic Rate In 2025
    df['BMR'] = np.where(df['Sex'] == 1,
                         10*df['Weight']+6.25*df['Height']-5*df['Age']+5,
                         10*df['Weight']+6.25*df['Height']-5*df['Age']-161)

    # Dr. Tanaka(2001)
    # HRmax = 208 - ( 0.7 * A )
    df['HRmax'] = 208-(0.7*df['Age'])

    # Intensity = MET（Metabolic Equivalent） * Duration
    # How are METs calculated?
    # or replace MET with Heart_Rate
    # Intensity = Heart_Rate  * Duration
    df['Indensity_Duration'] = df['Heart_Rate']*df['Duration']
    
    # BodyTemp_Interaction = Body_Temp * Duration
    df['BodyXTemp_Duration'] = df['Body_Temp']*df['Duration']

    # Weight_Duration = Weight * Duration
    df['Weight_Duration'] = df['Weight']*df['Duration']

    # Age_HR = Age * Heart_Rate
    df['Age_HR'] = df['Age']*df['Heart_Rate']
    
    df = df.drop(columns=['Height', 'Weight', 'Age', 'Duration', 'Heart_Rate', 'Body_Temp'])

    return df


traindata = features_engineering(traindata)


traindata.head()


testdata = features_engineering(testdata)


testdata.head()


# plt.figure(figsize=(10, 8))
# sns.heatmap(data=traindata.drop(columns='Sex').corr(), fmt='.2f', annot=True)
# plt.show()


# sex_counts = traindata['Sex'].value_counts().reset_index()
# sex_counts.columns = ['sex', 'counts']

# plt.figure(figsize=(4, 3))
# sns.barplot(data=sex_counts, x='sex', y='counts')
# plt.title('traindata')


# tsetsex_counts = testdata['Sex'].value_counts().reset_index()
# tsetsex_counts.columns = ['sex', 'counts']

# plt.figure(figsize=(4, 3))
# sns.barplot(data=tsetsex_counts, x='sex', y='counts')
# plt.title('testdata')


# sns.histplot(data=traindata, x='Age', fill=False, kde=True, binwidth=1)


# sns.scatterplot(data=traindata, x='Height', y='Weight', hue='Sex')


# sns.scatterplot(data=traindata, x='BMI', y='Calories', hue='Sex')


# sns.scatterplot(data=traindata, x='Duration', y='Calories', hue='Sex', style='Sex')


# sns.scatterplot(data=traindata, x='Body_Temp', y='Calories', hue='Sex')


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


traindata['Sex'] = traindata['Sex'].replace({'male': 1, 'female': 0})


scaler = StandardScaler()
traindata_scaled = scaler.fit_transform(traindata.drop(columns='Calories'))
pca = PCA(n_components=2)
traindata_pca = pca.fit_transform(traindata_scaled)


df = pd.DataFrame(traindata_pca)
df.columns = ['PC1', 'PC2']
df['Sex'] = traindata['Sex'].values
df.head(3)


sns.scatterplot(data=df, x='PC1', y='PC2', hue='Sex')


from sklearn.metrics import mean_squared_log_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold


male_data = traindata[traindata['Sex'] == 1]
female_data = traindata[traindata['Sex'] == 0]

x_male = male_data.drop(columns=['Calories', 'Sex'])
y_male = male_data['Calories']
x_female = female_data.drop(columns=['Calories', 'Sex'])
y_female = female_data['Calories']

scaler_male = StandardScaler()
scaler_female = StandardScaler()

x_male_scaled = scaler_male.fit_transform(x_male)
x_female_scaled = scaler_female.fit_transform(x_female)

lr_male  = LinearRegression()
lr_female  = LinearRegression()

male_model = lr_male.fit(x_male_scaled, y_male)
female_model = lr_female.fit(x_female_scaled, y_female)

pred_male = male_model.predict(x_male_scaled)
pred_female = female_model.predict(x_female_scaled)

pred_male = np.maximum(pred_male, 0)
pred_female = np.maximum(pred_female, 0)

rmsle_male = np.sqrt(mean_squared_log_error(y_male, pred_male))
rmsle_female = np.sqrt(mean_squared_log_error(y_female, pred_female))

print(f"male RMSLE: {rmsle_male:.4f}")
print(f"female RMSLE: {rmsle_female:.4f}")


resi_male = male_model.predict(x_male_scaled)-y_male 
plt.figure(figsize=(8, 6))
sns.scatterplot(x=male_model.predict(x_male_scaled), y=resi_male)
plt.axhline(0, color='r', linestyle='--')
plt.xlabel('Predict Calories')
plt.ylabel('Residuals')
plt.title('Male')
plt.show()


resi_female = female_model.predict(x_female_scaled)-y_female 
plt.figure(figsize=(8, 6))
sns.scatterplot(x=female_model.predict(x_female_scaled), y=resi_female)
plt.axhline(0, color='r', linestyle='--')
plt.xlabel('Predict Calories')
plt.ylabel('Residuals')
plt.title('Female')
plt.show()


def cv_RMSLE(X, y, n_splits=10):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    RMSLE_scores = []

    for train_x, test_x in kf.split(X):
        X_train, X_test = X.iloc[train_x], X.iloc[test_x]
        y_train, y_test = y.iloc[train_x], y.iloc[test_x]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        lr = LinearRegression()
        model = lr.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_pred = np.maximum(y_pred, 0)

        score = np.sqrt(mean_squared_log_error(y_test, y_pred))
        RMSLE_scores.append(score)

    return np.mean(RMSLE_scores), RMSLE_scores



male_data = traindata[traindata['Sex'] == 1]
female_data = traindata[traindata['Sex'] == 0]

x_male = male_data.drop(columns='Calories')
y_male = male_data['Calories']
x_female = female_data.drop(columns='Calories')
y_female = female_data['Calories']

mean_rmsle_male, scores_male = cv_RMSLE(x_male, y_male)
mean_rmsle_female, scores_female = cv_RMSLE(x_female, y_female)

print(f"male 5-Fold RMSLE mean: {mean_rmsle_male:.4f}")
print(f"every fold score: {np.round(scores_male, 4)}\n")

print(f"female 5-Fold RMSLE mean: {mean_rmsle_female:.4f}")
print(f"every fold score: {np.round(scores_female, 4)}")



def precict_calories(testdata, scaler_male, scaler_female, male_model, female_model):
    df = testdata.copy()
    male_df = df[df['Sex']==1]
    female_df = df[df['Sex']==0]

    if not male_df.empty:
        X_male = male_df.drop(columns=['id', 'Sex'])
        X_male_scaled = scaler_male.transform(X_male)
        y_male_pred = male_model.predict(X_male_scaled)
        df.loc[male_df.index, 'Predicted_Calories'] = np.maximum(y_male_pred, 0)
    if not female_df.empty:
        X_female = female_df.drop(columns=['id', 'Sex'])
        X_female_scaled = scaler_female.transform(X_female)
        y_female_pred = female_model.predict(X_female_scaled)
        df.loc[female_df.index, 'Predicted_Calories'] = np.maximum(y_female_pred, 0)
    return df[['id', 'Predicted_Calories']].rename(columns={'Predicted_Calories': 'Calories'})


results = precict_calories(testdata, scaler_male, scaler_female, male_model, female_model)


results.head()


results.shape


results.to_csv('submission.csv', index=False)

