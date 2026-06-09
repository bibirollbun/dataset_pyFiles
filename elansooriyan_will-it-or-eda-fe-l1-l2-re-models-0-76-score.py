#importing necessy libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')


data.head(10)


data.isnull().sum()


data.dtypes


data.shape


data['day'].nunique()
data['day'].value_counts().sort_index()


data['rainfall'].value_counts().plot.pie(autopct='%1.1f%%')


con_num_features = data.select_dtypes(include='float64')


con_num_features.columns


con_num_features.describe(percentiles=[0.682,0.957,0.99]).T


temp_features = [feature for feature in data if 'temp' in feature]


temp_features


temp_var = data['maxtemp'] - data['mintemp']


#finding day's average temperature
data["avg_temp"] = (data['maxtemp'] + data['mintemp']) / 2


data["avg_temp"] = (data['maxtemp'] + data['mintemp']) / 2
data["temp_kind"] = data.apply(
    lambda x : "above_avg" if (x['temparature'] < x['avg_temp']) else("avg" if x['temparature'] == x['avg_temp'] else 'below_avg'),axis = 1)
data.drop('avg_temp',axis = 1)


data[data['rainfall'] == 1]["temp_kind"].value_counts().plot.pie(autopct='%1.1f%%')


wind_features = [f for f in data if 'wind' in f]
wind_features


data['winddirection'].value_counts().sort_index().plot.bar(edgecolor = 'black')


data['wind_x']= np.cos(np.radians(data['winddirection']))
data['wind_y']= np.sin(np.radians(data['winddirection']))



def categorize_wind_direction(angle):
    if (angle >= 0 and angle < 22.5) or angle == 360:
        return "N"
    elif angle < 45:
        return "NNE"
    elif angle < 67.5:
        return "NE"
    elif angle < 90:
        return "ENE"
    elif angle < 112.5:
        return "E"
    elif angle < 135:
        return "ESE"
    elif angle < 157.5:
        return "SE"
    elif angle < 180:
        return "SSE"
    elif angle < 202.5:
        return "S"
    elif angle < 225:
        return "SSW"
    elif angle < 247.5:
        return "SW"
    elif angle < 270:
        return "WSW"
    elif angle < 292.5:
        return "W"
    elif angle < 315:
        return "WNW"
    elif angle < 337.5:
        return "NW"
    elif angle < 360:
        return "NNW"


data['wind_category'] = data['winddirection'].apply(categorize_wind_direction)


#Filter the raining data
wind_counts = data[data['rainfall'] == 1]['wind_category'].value_counts()

# Wind direction angles for alignment
directions = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
angles = np.linspace(0, 2 * np.pi, len(directions), endpoint=False).tolist()

# Align data to angles
values = [wind_counts.get(direction, 0) for direction in directions]

# Plotting
fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
ax.set_theta_zero_location("N")  # North on top
ax.set_theta_direction(-1)  # Clockwise

# Create the pie chart
ax.bar(angles, values, width=0.3, bottom=0.2, color=plt.cm.viridis(np.linspace(0, 1, len(values))), edgecolor="k")

# Set labels
ax.set_xticks(angles)
ax.set_xticklabels(directions)

plt.title("Wind Direction Distribution")
plt.show()


data['windspeed'].hist(edgecolor='black',bins=20)


pivot_data = data.groupby(['wind_category', 'rainfall'])['windspeed'].median().unstack()

# Plotting
plt.figure(figsize=(8, 6))
sns.heatmap(pivot_data, annot=True, cmap="coolwarm")
plt.title("Median Windspeed by Wind Category and Rainfall")
plt.xlabel("Rainfall (0/1)")
plt.ylabel("Wind Category")
plt.show()


data[['sunshine','cloud','humidity']].describe()


data['sunshine_category'] = pd.cut(
    data['sunshine'],
    bins=[-1, 4, 8, 12.1],
    labels=['Mostly Cloudy', 'Partly Sunny', 'Sunny']
)

# Categorizing Cloud Cover
data['cloud_category'] = pd.cut(
    data['cloud'],
    bins=[-1, 30, 70, 100],
    labels=['Clear Sky', 'Partly Cloudy', 'Overcast']
)

# Categorizing Humidity
data['humidity_category'] = pd.cut(
    data['humidity'],
    bins=[-1, 60, 80, 100],
    labels=['Dry', 'Comfortable', 'Humid']
)


data[data['rainfall'] == 1]['sunshine_category'].value_counts().plot.bar()


data[data['rainfall'] == 1]['cloud_category'].value_counts().plot.bar()


data[data['rainfall'] == 1]['humidity_category'].value_counts().plot.bar()


# Importing test data set along
test_data = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')
test_id = test_data['id']
test_data['winddirection'] = test_data['winddirection'].fillna(test_data['winddirection'].median())
test_data.isnull().sum()


#classify features as left,right and normal distribution
left_skewed = []
right_skewed = []
normal = []
for i in con_num_features:
    if i not in left_skewed and data[i].skew() < -0.5:
        left_skewed.append(i)
    elif i not in right_skewed and data[i].skew() > 0.5:
        right_skewed.append(i)
    else:
        if i not in normal:
            normal.append(i)



#z_score method
def zscore_boundary(i):
    left_boundary = data[i].mean() - (3 * data[i].std())
    right_boundary = data[i].mean() + (3 * data[i].std())
    return left_boundary,right_boundary


for i in normal:
    lb,ub = zscore_boundary(i)
    outliers = data[(data[i] < lb) | (data[i] > ub)][i]    
    print(f'No of outliers for {i} : {len(outliers)}')


#IQR method
def IQR_boundary(i):
    IQR = data[i].quantile(0.75) - data[i].quantile(0.25)    
    left_boundary = data[i].quantile(0.25) - (1.5 * IQR)
    right_boundary = data[i].quantile(0.75) + (1.5 * IQR)
    return left_boundary,right_boundary


for i in left_skewed+right_skewed:
    lb,ub = IQR_boundary(i)
    outliers = data[(data[i] < lb) | (data[i] > ub)][i]
    print(f'No of outliers for {i} : {len(outliers)}')


copydata = data.copy()
for i in left_skewed:
    plt.figure(figsize=(10, 4))

    # Plot before transformation
    plt.subplot(1, 2, 1)
    plt.hist(copydata[i], bins=20, color='skyblue')
    plt.title(f'{i} - Before Transformation')

    # Perform transformation
    copydata[i] = 1 / (copydata[i] + 1)

    # Plot after transformation
    plt.subplot(1, 2, 2)
    plt.hist(copydata[i], bins=20, color='salmon')
    plt.title(f'{i} - After Transformation')

    plt.tight_layout()



copydata = data.copy()
for i in left_skewed:
    plt.figure(figsize=(10, 4))

    # Plot before transformation
    plt.subplot(1, 2, 1)
    plt.hist(copydata[i], bins=20, color='skyblue')
    plt.title(f'{i} - Before Transformation')

    # Perform transformation
    copydata[i] = np.power(copydata[i],2)

    # Plot after transformation
    plt.subplot(1, 2, 2)
    plt.hist(copydata[i], bins=20, color='salmon',edgecolor='white')
    plt.title(f'{i} - After Transformation')

    plt.tight_layout()






copydata = data.copy()
for i in right_skewed:
    plt.figure(figsize=(10, 4))

    # Plot before transformation
    plt.subplot(1, 2, 1)
    plt.hist(copydata[i], bins=20, color='skyblue',edgecolor='white')
    plt.title(f'{i} - Before Transformation')

    # Perform transformation
    copydata[i] = np.log1p(copydata[i])

    # Plot after transformation
    plt.subplot(1, 2, 2)
    plt.hist(copydata[i], bins=20, color='salmon',edgecolor='white')
    plt.title(f'{i} - After Transformation')

    plt.tight_layout()



for i in left_skewed:
    data[i] = np.square(data[i])
    test_data[i] = np.square(test_data[i])

for i in right_skewed:
    data[i] = np.log1p(data[i])
    test_data[i] = np.log1p(test_data[i])
normal


from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()

scaled_data = pd.DataFrame(scalar.fit_transform(data[normal]),columns=normal)
test_scaled_data = pd.DataFrame(scalar.fit_transform(test_data[normal]),columns=normal)


from sklearn.preprocessing import RobustScaler
mmscalar = RobustScaler()
scaled_data2 = pd.DataFrame(mmscalar.fit_transform(data[left_skewed+right_skewed+['day']]),columns=left_skewed+right_skewed+['day'])
test_scaled_data2 = pd.DataFrame(mmscalar.fit_transform(test_data[left_skewed+right_skewed+['day']]),columns=left_skewed+right_skewed+['day'])


X = pd.concat([scaled_data2,scaled_data],axis=1)
Xtest = pd.concat([test_scaled_data2,test_scaled_data],axis=1)
Y = data['rainfall']


Xtest


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score,classification_report,confusion_matrix,roc_auc_score



model = LogisticRegression()
model.fit(X,Y)


Yp = model.predict(X)
print('roc_auc score : '+str(roc_auc_score(Y,Yp)))
print(confusion_matrix(Y,Yp))
print(classification_report(Y,Yp))


ridgemodel = LogisticRegression(penalty='l2', C=1.0, solver='liblinear')
ridgemodel.fit(X,Y)


Yp = ridgemodel.predict(X)
print('roc_auc score : '+str(roc_auc_score(Y,Yp)))
print(confusion_matrix(Y,Yp))
print(classification_report(Y,Yp))


lassomodel = LogisticRegression(penalty='l1', C=1.0, solver='liblinear')
lassomodel.fit(X,Y)


Yp = lassomodel.predict(X)
print('roc_auc score : '+str(roc_auc_score(Y,Yp)))
print(confusion_matrix(Y,Yp))
print(classification_report(Y,Yp))


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


params = {
    'n_estimators': [50, 100, 150],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf, params, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)


best_rf = grid_search.best_estimator_
print("Best Hyperparameters:", grid_search.best_params_)


y_pred = best_rf.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


predictData = lassomodel.predict(Xtest)


submission_vote = pd.DataFrame({
    "id": test_id,
    "rainfall": predictData
})

submission_vote.to_csv("submission.csv", index=False)

