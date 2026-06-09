import warnings; warnings.simplefilter('ignore')
import numpy as np, pandas as pd


original = pd.read_csv('/kaggle/input/calories-burnt-prediction/calories.csv')    
train    = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')       
test     = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')  
original = original.rename(columns={'User_ID': 'id', 'Gender': 'Sex'})

original['source'] = 'original'
train['source'] = 'train_gen'
test['source'] = 'test_gen'

test['Calories'] = pd.NA
df_all = pd.concat([original, train, test], ignore_index=True)
df_all['Sex'] = df_all['Sex'].astype('category')
df_all['Calories'] = pd.to_numeric(df_all['Calories'], errors='coerce')


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.kdeplot(data=df_all[df_all['source']=='original'], x='Calories', label='original')
sns.kdeplot(data=df_all[df_all['source']=='train_gen'], x='Calories', label='train_gen')
plt.title("Calories Distribution by Source")
plt.legend()
plt.show()


for col in ['Age','Height','Weight','Duration','Heart_Rate','Body_Temp']:
    plt.figure(figsize=(8,4))
    sns.kdeplot(data=df_all[df_all['source']=='original'], x=col, label='original')
    sns.kdeplot(data=df_all[df_all['source']=='train_gen'], x=col, label='train_gen')
    plt.title(f"{col} Distribution by Source")
    plt.legend()
    plt.show()


plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=df_all[df_all['source'] == 'train_gen'],
    x='Duration', y='Calories',
    alpha=0.1, s=5
)
sns.scatterplot(
    data=df_all[df_all['source'] == 'original'],
    x='Duration', y='Calories',
    alpha=0.1, s=5, color='red'
)
plt.title("Calories vs Duration (orange=train_gen, red=original)")
plt.show()


from sklearn.linear_model import LinearRegression

df_encoded = pd.get_dummies(df_all, columns=['Sex'], drop_first=True)  # → 'Sex_male'

feature_cols_encoded = ['Sex_male', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
orig_df = df_encoded[df_encoded['source'] == 'original'].dropna()
gen_df  = df_encoded[df_encoded['source'] == 'train_gen'].dropna()

X_orig = orig_df[feature_cols_encoded]
y_orig = orig_df['Calories']
X_gen  = gen_df[feature_cols_encoded]
y_gen  = gen_df['Calories']

model = LinearRegression().fit(X_orig, y_orig)

gen_pred = model.predict(X_gen)
residuals = y_gen - gen_pred

plt.figure(figsize=(8,5))
sns.histplot(residuals, bins=100, kde=True)
plt.title("Residuals (train_gen - LinearReg(original))")
plt.xlabel("Residual (Observed - Predicted)")
plt.show()




