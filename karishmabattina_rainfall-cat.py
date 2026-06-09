import numpy as np
import pandas as pd


train_data = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
test_data = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")


train_data.shape


test_data.shape


train_data.info()


train_data.head()


test_data.head()


train_data.isnull().sum()


test_data.isnull().sum()


test_data['winddirection'].fillna(test_data['winddirection'].median(), inplace=True)


test_data.isnull().sum()


numerical_variables = ['winddirection', 'pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint', 'humidity', 'cloud', 'sunshine', 'windspeed']
target_variable = 'rainfall' 
categorical_variables = []


train_data[numerical_variables].hist(figsize=(12, 10))


import matplotlib.pyplot as plt
import seaborn as sns

# Correlation heatmap
def plot_correlation_heatmap(data, title, annot_size=12):
    plt.figure(figsize=(12, 8))
    corr_matrix = data.corr()
    sns.heatmap(corr_matrix, annot=True, annot_kws={"size": annot_size},cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title(f'Correlation Heatmap - {title}', fontsize=16)
    plt.show()

plot_correlation_heatmap(train_data, "Train Data")


import matplotlib.pyplot as plt
import seaborn as sns

sns.countplot(x='rainfall', data=train_data)


sns.scatterplot(x=train_data['cloud'], y=train_data['temparature'], hue=train_data['rainfall'])


sns.scatterplot(x=train_data['sunshine'], y=train_data['temparature'], hue=train_data['rainfall'])


sns.scatterplot(x=train_data['pressure'], y=train_data['temparature'], hue=train_data['rainfall'])


sns.scatterplot(x=train_data['humidity'], y=train_data['temparature'], hue=train_data['rainfall'])


def engineer_features(df):
    df['winddirection_sin'] = np.sin(2 * np.pi * df['winddirection'] / 360)
    df['winddirection_cos'] = np.cos(2 * np.pi * df['winddirection'] / 360)
    df['dew_depression'] = df['temparature'] - df['dewpoint']
    df.drop('winddirection', axis=1, inplace=True)
    return df


train_data = engineer_features(train_data)
test_data = engineer_features(test_data)


from sklearn.model_selection import train_test_split

# Split data
X = train_data.drop('rainfall', axis=1)
y = train_data['rainfall']
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)


features = ['day', 'pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint', 'humidity', 'cloud', 'sunshine', 'windspeed', 'winddirection_sin', 'winddirection_cos', 'dew_depression']


x_train = x_train[features]
x_test = x_test[features]


from catboost import CatBoostClassifier

# Model training
model = CatBoostClassifier(bagging_temperature=0.7,
    depth=12,
    iterations=100,
    l2_leaf_reg=8,
    learning_rate=0.03,
    random_strength=4,  
    eval_metric="AUC",
    verbose=0,
    random_seed=42,
    auto_class_weights="Balanced")

model.fit(X[features], y)

#model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50)


preds = model.predict(test_data)


submission = pd.DataFrame({
    'id': test_data['id'],
    'rainfall': preds
})
submission.to_csv('submission.csv', index=False)

