import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder ,LabelEncoder ,StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV



data=pd.read_csv("/kaggle/input/playground-series-s4e2/train.csv")


data


new_column_names = {
    "FAVC": "HighCalorieFoodFreq",
    "FCVC": "VegetableConsumptionFreq",
    "NCP": "MainMealsPerDay",
    "CAEC": "SnackingFrequency",
    "SMOKE": "SmokingHabit",
    "CH2O": "DailyWaterIntake",
    "SCC": "CaloricMonitoring",
    "FAF": "PhysicalActivityPerWeek",
    "TUE": "ScreenTimePerDay",
    "CALC": "AlcoholConsumption",
    "MTRANS": "ModeOfTransportation"
}

# Rename columns
data.rename(columns=new_column_names, inplace=True)


data.head()


data.info()


data.describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]).T.style.background_gradient(cmap='viridis')


data.describe(include=object).T.style.background_gradient(cmap='greens')


# X=data.drop("NObeyesdad",axis=1)
# y=data.NObeyesdad


# x_train,x_valid,y_train,y_valid=train_test_split(X,y,test_size=0.3,random_state=123)


numerical = data.select_dtypes(include=["number"])
numerical.columns


categorical = data.select_dtypes(include=["object"])
categorical.columns


numerical_stats = numerical[1:].agg([min, max, 'mean',"median"]).T.style.background_gradient(cmap="Greens")  
numerical_stats



numerical["VegetableConsumptionFreq"].value_counts()


numerical["MainMealsPerDay"].value_counts()


numerical["DailyWaterIntake"].value_counts()


numerical["PhysicalActivityPerWeek"].value_counts()


numerical["ScreenTimePerDay"].value_counts()


for col in list(data.describe(include="object")):
    print(f"Column: {col}'s count values:\n")

    # Create a dictionary to store value counts
    value_count_dict = {
        'Value': data[col].value_counts().index.tolist(),
        'Count': data[col].value_counts().values.tolist()
    }

    # Convert dictionary to DataFrame
    value_count_df = pd.DataFrame(value_count_dict)
    display(value_count_df)
    
    print("\n" + "-"*40 + "\n")


data.isnull().sum()


data.duplicated().sum()


number_of_outliers = [None] * len(data.select_dtypes(include=["number"]).columns)
q95 = [None] * len(data.select_dtypes(include=["number"]).columns)
q5 = [None] * len(data.select_dtypes(include=["number"]).columns)
total_rows = len(data)
for i, p in enumerate(data.select_dtypes(include=["number"]).columns):
    q95[i], q5[i] = np.percentile(data[p], [95, 5])
    outliers = (data[p] > q95[i]) | (data[p] < q5[i])
    number_of_outliers[i] = outliers.sum()
    print(f'Outliers in {p} = {number_of_outliers[i]}')
    print("*" * 40)


data[["Age","Height","Weight","VegetableConsumptionFreq","MainMealsPerDay","PhysicalActivityPerWeek"]].describe(percentiles=[0.01,0.05,0.95,0.99])[3:]


numerical.hist(figsize=(12, 10), bins=20, color='#4caba4', grid=False)


def showplot(columnname):
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax = ax.flatten()
    value_counts = categorical[columnname].value_counts()
    labels = value_counts.index.tolist()
    colors =["#4caba4", "#d68c78",'#a3a2a2','#ab90a0', '#e6daa3', '#6782a8', '#8ea677']
    
    # Donut Chart
    wedges, texts, autotexts = ax[0].pie(
        value_counts, autopct='%1.1f%%',textprops={'size': 9, 'color': 'white','fontweight':'bold' }, colors=colors,
        wedgeprops=dict(width=0.35),  startangle=80,   pctdistance=0.85  )
    # circle
    centre_circle = plt.Circle((0, 0), 0.6, fc='white')
    ax[0].add_artist(centre_circle)
    
    # Count Plot
    sns.countplot(data=categorical, y=columnname, ax=ax[1], palette=colors, order=labels)
    for i, v in enumerate(value_counts):
        ax[1].text(v + 1, i, str(v), color='black',fontsize=10, va='center')
    sns.despine(left=True, bottom=True)
    plt.yticks(fontsize=9,color='black')
    ax[1].set_ylabel(None)
    plt.xlabel("")
    plt.xticks([])
    fig.suptitle(columnname, fontsize=15, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.show()


for column in categorical.columns:
    print(f"Plotting: {column}")
    try:
        showplot(column)
    except Exception as e:
        print(f"Error plotting {column}: {e}")


data.groupby(['NObeyesdad',"family_history_with_overweight"]).size().reset_index(name='count')


# Get counts of each group
count_data = data.groupby(['NObeyesdad', 'family_history_with_overweight']).size().reset_index(name='count')

# Create bar chart of counts
plt.figure(figsize=(12, 8))
sns.barplot(x='NObeyesdad', y='count', hue='family_history_with_overweight', data=count_data)
plt.title('Count of Cases by Obesity Level and Family History')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


grouped_data=data.groupby(['NObeyesdad',"SnackingFrequency"]).size().reset_index(name='count')


# Store the grouped data
grouped_data = data.groupby(['NObeyesdad', 'SnackingFrequency']).size().reset_index(name='count')

# Create a grouped bar chart
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 8))
sns.barplot(x='NObeyesdad', y='count', hue='SnackingFrequency', data=grouped_data)
plt.title('Distribution of Snacking Frequency by Obesity Level', fontsize=16)
plt.xlabel('Obesity Level', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.xticks(rotation=45)
plt.legend(title='Snacking Frequency')
plt.tight_layout()
plt.show()


# Create a pivot table for plotting
pivot_data = grouped_data.pivot(index='NObeyesdad', columns='SnackingFrequency', values='count')

# Create a stacked bar chart
pivot_data.plot(kind='bar', stacked=True, figsize=(12, 8))
plt.title('Distribution of Snacking Frequency by Obesity Level', fontsize=16)
plt.xlabel('Obesity Level', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.xticks(rotation=45)
plt.legend(title='Snacking Frequency')
plt.tight_layout()
plt.show()


# Store the grouped data
grouped_data = data.groupby(['NObeyesdad', 'ModeOfTransportation']).size().reset_index(name='count')



plt.figure(figsize=(12, 8))
sns.barplot(x='NObeyesdad', y='count', hue='ModeOfTransportation', data=grouped_data)
plt.title('Distribution of Snacking Frequency by Obesity Level', fontsize=16)
plt.xlabel('Obesity Level', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.xticks(rotation=45)
plt.legend(title='Snacking Frequency')
plt.tight_layout()
plt.show()


grouped_alcohol = data.groupby(['NObeyesdad', 'AlcoholConsumption']).size().reset_index(name='count')


plt.figure(figsize=(12, 8))
sns.barplot(x='NObeyesdad', y='count', hue='AlcoholConsumption', data=grouped_alcohol)
plt.title('Distribution of Alcohol Consumption by Obesity Level', fontsize=16)
plt.xlabel('Obesity Level', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.xticks(rotation=45)
plt.legend(title='Alcohol Consumption')
plt.tight_layout()
plt.show()


cross_tab = pd.crosstab(data['NObeyesdad'], data['AlcoholConsumption'])
plt.figure(figsize=(10, 5))
sns.heatmap(cross_tab, annot=True, cmap='pink_r', fmt='d', cbar=False)
plt.title(' Alcohol')
plt.xlabel('')
plt.ylabel('')
plt.show()


cross_tab = pd.crosstab(data['NObeyesdad'], data['ModeOfTransportation'])
plt.figure(figsize=(10, 5))
sns.heatmap(cross_tab, annot=True, cmap='Blues', fmt='d', cbar=False)
plt.title(' NObeyesdad and MTRANS')
plt.xlabel('')
plt.ylabel('')
plt.show()


cross_tab = pd.crosstab(data['NObeyesdad'], data['HighCalorieFoodFreq'])
plt.figure(figsize=(10, 5))
sns.heatmap(cross_tab, annot=True, cmap='Blues', fmt='d', cbar=False)
plt.title(' High Calorie Food Frequency')
plt.xlabel('')
plt.ylabel('')
plt.show()


colors = ['#1f77b4', '#fc6c44', '#2b8a2b', '#fc7c7c', '#9467bd', '#4ba4ad', '#c7ad18', '#7f7f7f', '#69d108']


plt.figure(figsize=(15, 6))
ax = sns.countplot(x='Gender', hue='NObeyesdad', data=data, palette=colors, dodge=True)
plt.title('Distribution of NObeyesdad across Gender')
sns.despine(left=True, bottom=False)
plt.xlabel('')
plt.ylabel('')
plt.yticks([])
for p in ax.patches:
    height = p.get_height()
    ax.annotate(f'{round(height)}', (p.get_x() + p.get_width() / 2., height),
                ha='center', va='center', xytext=(0, 8), textcoords='offset points')
plt.show()


data.groupby(['NObeyesdad']).sum().sort_values("VegetableConsumptionFreq" , ascending= False)[["VegetableConsumptionFreq"]]


import matplotlib.pyplot as plt

grouped_data = data.groupby('NObeyesdad').sum().sort_values("VegetableConsumptionFreq", ascending=False)

vcf_values = grouped_data["VegetableConsumptionFreq"]

plt.figure(figsize=(10, 5))  
plt.plot(vcf_values.index, vcf_values.values, marker='o', linestyle='-', color='green')

plt.xlabel("Obesity Levels")
plt.ylabel("Total Vegetable Consumption Frequency")
plt.title("Vegetable Consumption Frequency by Obesity Level")
plt.xticks(rotation=45)  

plt.show()



sns.displot(data=data, x="PhysicalActivityPerWeek", col="NObeyesdad" , kind = "kde")


plt.figure(figsize=(15, 10))  # Create a larger figure first
sns.displot(data=data, x="PhysicalActivityPerWeek", col="NObeyesdad", kind="kde", 
            height=4, aspect=1.2, col_wrap=3, facet_kws={'sharex': False, 'sharey': False})
plt.tight_layout()
plt.show()


sns.displot(data=data, x="MainMealsPerDay", col="NObeyesdad" )


plt.figure(figsize=(12, 8))
sns.heatmap(data.select_dtypes(include=['int64', 'float64']).corr(),annot = True , cmap = "Greens")


data.columns


plt.figure(figsize=(14, 8))
sns.boxplot(data=data, x='ModeOfTransportation', y='PhysicalActivityPerWeek')
plt.title('Physical Activity Distribution by Transportation Mode', fontsize=16)
plt.xlabel('Mode of Transportation', fontsize=14)
plt.ylabel('Physical Activity Per Week', fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


import plotly.express as px

fig = px.scatter(data, x="Age", y="ModeOfTransportation", color="PhysicalActivityPerWeek",
                 title="Age vs Physical Activity Per Week",
                 labels={"Age": "Age", "PhysicalActivityPerWeek": "Physical Activity (Per Week)"},
                 )  # Optional: Show mode of transportation on hover
fig.show()



fig = px.box(data, x="CaloricMonitoring", y="SnackingFrequency",
             title="Snacking Frequency by Caloric Monitoring",
             labels={"CaloricMonitoring": "Caloric Monitoring", "SnackingFrequency": "Snacking Frequency"})
fig.show()



import plotly.express as px

fig = px.box(data, x="CaloricMonitoring", y="PhysicalActivityPerWeek", 
                 title="Caloric Monitoring vs Physical Activity Per Week",
                 labels={"CaloricMonitoring": "Caloric Monitoring", "PhysicalActivityPerWeek": "Physical Activity (Per Week)"},
                 hover_data=["Age", "ModeOfTransportation"])  # Optional extra details
fig.show()



import plotly.express as px

fig = px.scatter(data, x="VegetableConsumptionFreq", y="DailyWaterIntake",color="HighCalorieFoodFreq",
                 title="Vegetable Consumption vs High Calorie Food Consumption",
                 labels={"VegetableConsumptionFreq": "Vegetable Consumption Frequency",
                         "HighCalorieFoodFreq": "High Calorie Food Frequency"},
                 )  # Add extra details on hover
fig.show()




import plotly.express as px

fig = px.scatter(data, x="VegetableConsumptionFreq", y="MainMealsPerDay", color="CaloricMonitoring",
                    title="3D Relationship: Vegetable Consumption, Caloric Monitoring & Main Meals",
                    labels={"VegetableConsumptionFreq": "Vegetable Consumption",
                            "MainMealsPerDay": "Main Meals Per Day",
                            "CaloricMonitoring": "Caloric Monitoring"})
fig.show()



fig = px.scatter(data, x="MainMealsPerDay", y="Weight", 
                title="Weight Distribution Based on Number of Meals Per Day",
                labels={"MainMealsPerDay": "Main Meals Per Day", "Weight": "Weight (kg)"})
fig.show()



data.isnull().sum()


X=data.drop("NObeyesdad",axis=1)
y=data["NObeyesdad"]


x_train,x_valid,y_train,y_valid=train_test_split(X,y,test_size=0.2,random_state=123)


x_train.isnull().sum()


data.columns


x_train.drop(columns=["id","SmokingHabit"],inplace=True)
x_valid.drop(columns=["id","SmokingHabit"],inplace=True)


data_vif = data.select_dtypes(include=[np.number])


vif_data = pd.DataFrame()
vif_data["Feature"] = data_vif.columns
vif_data["VIF"] = [variance_inflation_factor(data_vif.values, i) for i in range(data_vif.shape[1])]

# Sort by VIF in descending order
vif_data = vif_data.sort_values(by="VIF", ascending=False)
print(vif_data)


categorical.columns


nominal_cols = ["Gender", "family_history_with_overweight", "CaloricMonitoring", "ModeOfTransportation","HighCalorieFoodFreq"]
ordinal_cols = ["SnackingFrequency", "AlcoholConsumption"]


encoder = OrdinalEncoder()
x_train[ordinal_cols] = encoder.fit_transform(x_train[ordinal_cols])
x_valid[ordinal_cols] = encoder.transform(x_valid[ordinal_cols])


HotEncoder = OneHotEncoder(sparse_output=False,drop="first")


from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

# Ordinal Encoding
ordinal_encoder = OrdinalEncoder()
x_train[ordinal_cols] = ordinal_encoder.fit_transform(x_train[ordinal_cols])
x_valid[ordinal_cols] = ordinal_encoder.transform(x_valid[ordinal_cols])

# One-Hot Encoding
onehot_encoder = OneHotEncoder(sparse_output=False, drop='first')
x_train_onehot = onehot_encoder.fit_transform(x_train[nominal_cols])
onehot_columns = onehot_encoder.get_feature_names_out(nominal_cols)
onehot_df = pd.DataFrame(x_train_onehot, columns=onehot_columns, index=x_train.index)

# Prepare final training dataset
x_train = x_train.drop(columns=nominal_cols)
x_train = pd.concat([x_train, onehot_df], axis=1)


x_train


x_valid_onehot = onehot_encoder.transform(x_valid[nominal_cols])
valid_onehot_df = pd.DataFrame(x_valid_onehot, columns=onehot_columns, index=x_valid.index)

# Prepare final validation dataset
x_valid = x_valid.drop(columns=nominal_cols)
x_valid = pd.concat([x_valid, valid_onehot_df], axis=1)


x_valid


data["NObeyesdad"]


lb = LabelEncoder()
y_train= lb.fit_transform(y_train) 
y_valid= lb.transform(y_valid)


scaler = StandardScaler() 
x_train[numerical.columns[1:]]= scaler.fit_transform(x_train[numerical.columns[1:]]) 
x_valid[numerical.columns[1:]]= scaler.transform(x_valid[numerical.columns[1:]]) 


LG=LogisticRegression(penalty='l1',class_weight="balanced",solver="saga")


LG.fit(x_train,y_train)


LG.score(x_train,y_train)


LG.score(x_valid,y_valid)


y_pred=LG.predict(x_valid)


from sklearn.metrics import classification_report


print(classification_report(y_valid, y_pred))


lG_2=LogisticRegression()


lG_2.fit(x_train,y_train)


lG_2.score(x_train,y_train)


lG_2.score(x_valid,y_valid)


y_pred=lG_2.predict(x_valid)


print(classification_report(y_valid, y_pred))


from  sklearn.tree import DecisionTreeClassifier


Tree=DecisionTreeClassifier()


Tree.fit(x_train,y_train)


Tree.score(x_train,y_train)


Tree.score(x_valid,y_valid)


Tree.get_depth()


Tree.feature_names_in_


Tree.feature_importances_


Tree.tree_.node_count


Tree_based=DecisionTreeClassifier(criterion="entropy",splitter="best",max_depth=9)


Tree_based.fit(x_train,y_train)


Tree_based.score(x_train,y_train)


Tree_based.score(x_valid,y_valid)


Tree_based.get_n_leaves()


Tree_based.max_leaf_nodes


Tree_based.min_samples_leaf


Tree_based.min_samples_split


from sklearn import tree


tree.plot_tree(Tree_based)
plt.figure(figsize=(12,8))


tree.export_graphviz(Tree_based)


param_grid = {
    'criterion': ['gini', 'entropy'],  
    'splitter': ['best', 'random'],  
    'max_depth': [9, 10, 8],  
    'min_samples_split': [200, 100, 50], 
    'min_samples_leaf': [100, 200, 300], 
    'max_features': [None, 'sqrt', 'log2']  
}


dt = DecisionTreeClassifier(random_state=42)
grid_search = GridSearchCV(dt, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(x_train, y_train)


# Print best parameters and best score
print("Best Parameters:", grid_search.best_params_)
print("Best Accuracy:", grid_search.best_score_)

# Evaluate the best model on the test set
best_model = grid_search.best_estimator_
test_accuracy = best_model.score(x_valid, y_valid)
print("Test Accuracy:", test_accuracy)





# pip install xgboost



from sklearn.ensemble import GradientBoostingClassifier


gb_clf = GradientBoostingClassifier()


gb_clf.fit(x_train,y_train)


gb_clf.score(x_train,y_train)


gb_clf.score(x_valid,y_valid)


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Generate predictions
y_pred = gb_clf.predict(x_valid)

# Compute confusion matrix
cm = confusion_matrix(y_valid, y_pred)

# Display the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues")  # Use a blue color map for better visibility
plt.title("Confusion Matrix")
plt.show()



from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

# Binarize the labels for multi-class ROC
n_classes = len(set(y_valid))  # Get the number of unique classes
y_valid_bin = label_binarize(y_valid, classes=range(n_classes))

# Get class probabilities
y_probs = gb_clf.predict_proba(x_valid)  # Predict probability for all classes

# Plot ROC curve for each class
plt.figure(figsize=(8, 6))
for i in range(n_classes):
    fpr, tpr, _ = roc_curve(y_valid_bin[:, i], y_probs[:, i])  # Compute ROC curve
    roc_auc = auc(fpr, tpr)  # Compute AUC
    plt.plot(fpr, tpr, lw=2, label=f"Class {i} (AUC = {roc_auc:.2f})")

# Plot the diagonal line (random guessing)
plt.plot([0, 1], [0, 1], color="gray", linestyle="--")

# Labels and Title
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Multi-Class ROC Curve")
plt.legend(loc="lower right")
plt.show()



y_pred=gb_clf.predict(x_valid)


print(classification_report(y_valid, y_pred))


from xgboost import XGBClassifier


xgb = XGBClassifier(use_label_encoder=False,eval_metric='mlogloss')


xgb.fit(x_train,y_train)


xgb.score(x_train,y_train)


xgb.score(x_valid,y_valid)

