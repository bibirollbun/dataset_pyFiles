import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns, plotly.express as px

import warnings
warnings.filterwarnings('ignore')


train_df = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')


target_column = (set(train_df.columns) - set(test_df.columns)).pop()

print(f"Target column: {target_column}\nData type: {train_df[target_column].dtype}")


print(f"The Rows Train dataset contains : {train_df.shape[0]}\nThe Columns dataset contains : {train_df.shape[1]}")
print("-"*50)
print(f"The Rows Test dataset contains : {test_df.shape[0]}\nThe Columns dataset contains : {test_df.shape[1]}")


print(train_df.columns)
print("-"*50)
print(test_df.columns)


def generate_null_analysis(df):
    count = df.isnull().sum()
    percen = count / len(df) * 100
    
    df_null = pd.DataFrame({
        'column name': df.columns,
        'total count': count,
        'percentage': percen
    })
    
    df_null.reset_index(drop = True, inplace = True)
    df_null_sorted = df_null.sort_values(by = 'percentage', ascending = False)
    df_filtered = df_null_sorted[df_null_sorted['percentage'] > 0]
    df_filtered.reset_index(drop = True, inplace = True)
    
    return df_filtered

df_filtered_train = generate_null_analysis(train_df)
df_filtered_test = generate_null_analysis(test_df)

def style_null_analysis(df):
    return df.style.background_gradient(cmap = 'YlOrRd', subset = ['percentage', 'total count'])

df_filtered_train_styled = style_null_analysis(df_filtered_train)
df_filtered_test_styled = style_null_analysis(df_filtered_test)


display(df_filtered_train_styled)


display(df_filtered_test_styled)


display(train_df.head(1))
display(test_df.head(1))


test_df['winddirection'].fillna(test_df['winddirection'].mean(), inplace=True)  


for col in train_df : 
    fig, axes = plt.subplots(1, 2, figsize = (14, 4), gridspec_kw={'width_ratios': [1, 2]})

    # Box plot on the left
    sns.boxplot(y=train_df[col],  ax = axes[0], color = 'skyblue')
    axes[0].set(title='Box Plot of {col}', xlabel='', ylabel = col)

    # Histogram with KDE on the right
    sns.histplot(train_df[col],  kde = True, bins = 50, ax=axes[1], color='green')
    axes[1].set(title = f'Histogram and KDE of {col}', xlabel = f'{col}', ylabel = 'Frequency')

    # Adjust layout
    plt.tight_layout()
    plt.show()


plt.figure(figsize=(15,12))
sns.heatmap(train_df.corr().round(2), annot=True, mask=np.triu(train_df.corr()))
plt.show()


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, KFold  


X = train_df.iloc[:, :-1]  
y = train_df.iloc[:, -1]  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  


logistic_model = LogisticRegression()  
logistic_model.fit(X_train, y_train)  
y_pred = logistic_model.predict(X_test)  

accuracy = accuracy_score(y_test, y_pred)  
f1 = f1_score(y_test, y_pred)  
conf_matrix = confusion_matrix(y_test, y_pred)  

print("Logistic Regression Results:")  
print(f"Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")  
print("Confusion Matrix:")  
print(conf_matrix)  


rf_model = RandomForestClassifier(random_state=42) 

param_dist_rf = {  
    'n_estimators': [100, 200, 300],  
    'max_depth': [10, 20, 30],  
    'min_samples_split': [2, 5, 10]  
}  

random_search_rf = RandomizedSearchCV(rf_model, param_distributions=param_dist_rf, n_iter=10, cv=5, n_jobs=-1, random_state=42)  
random_search_rf.fit(X_train, y_train)  

best_rf_model = random_search_rf.best_estimator_  
y_pred_rf = best_rf_model.predict(X_test)  

accuracy_rf = accuracy_score(y_test, y_pred_rf)  
f1_rf = f1_score(y_test, y_pred_rf)  
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)  

print("Random Forest Results (with Tuning):")  
print(f"Best Params: {random_search_rf.best_params_}")  
print(f"Accuracy: {accuracy_rf:.4f}, F1 Score: {f1_rf:.4f}")  
print("Confusion Matrix:")  
print(conf_matrix_rf)  


knn_model = KNeighborsClassifier()
param_dist_knn = {  
    'n_neighbors': [3, 5, 7, 9, 11],  
    'weights': ['uniform', 'distance'],  
    'p': [1, 2]  
}  

random_search_knn = RandomizedSearchCV(knn_model, param_distributions=param_dist_knn, n_iter=10, cv=5, n_jobs=-1, random_state=42)  
random_search_knn.fit(X_train, y_train)  
best_knn_model = random_search_knn.best_estimator_  
y_pred_knn = best_knn_model.predict(X_test)  

accuracy_knn = accuracy_score(y_test, y_pred_knn)  
f1_knn = f1_score(y_test, y_pred_knn)  
conf_matrix_knn = confusion_matrix(y_test, y_pred_knn)  

print("K-Nearest Neighbors Results (with Tuning):")  
print(f"Best Params: {random_search_knn.best_params_}")  
print(f"Accuracy: {accuracy_knn:.4f}, F1 Score: {f1_knn:.4f}")  
print("Confusion Matrix:")  
print(conf_matrix_knn)  


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping


X = train_df.iloc[:, : -1]
y = train_df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


def create_model(neurons=64, activation='relu', dropout_rate=0.2):  
    model = Sequential()  
    model.add(Dense(neurons, input_dim=X_train.shape[1], activation=activation))  
    model.add(Dropout(dropout_rate))  
    model.add(Dense(neurons, activation=activation))  
    model.add(Dropout(dropout_rate))  
    model.add(Dense(1, activation='sigmoid'))  # Use sigmoid activation for binary classification  
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  # Change to binary_crossentropy  
    
    return model  

early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1)  
model = create_model(neurons=64, activation='relu', dropout_rate=0.2)  

history = model.fit(X_train, y_train, epochs=10, batch_size=16,   
                    validation_data=(X_test, y_test),   
                    callbacks=[early_stopping], verbose=1)  

y_pred = model.predict(X_test)  
y_pred_binary = (y_pred > 0.5).astype(int)  # Convert probabilities to binary output  

# Calculate and print classification metrics  
accuracy = accuracy_score(y_test, y_pred_binary)  
f1 = f1_score(y_test, y_pred_binary)  
conf_matrix = confusion_matrix(y_test, y_pred_binary)  

print(f"Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")  
print("Confusion Matrix:")  
print(conf_matrix)  


test_predict = best_knn_model.predict(test_df)
test_df['rainfall'] = test_predict
test_df['rainfall'].unique()


submission_df = test_df[['id', 'rainfall']]
submission_df.to_csv('submission.csv', index = False)







