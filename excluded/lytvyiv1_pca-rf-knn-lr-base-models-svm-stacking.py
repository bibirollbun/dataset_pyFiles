# Data handling
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Patch
from matplotlib import gridspec
import seaborn as sns

# Scikit-learn: preprocessing, models, metrics, utilities
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_curve, roc_auc_score, confusion_matrix, auc
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

# Warnings
import warnings
warnings.filterwarnings("ignore")


df = pd.read_csv("/kaggle/input/playground-series-s5e7/train.csv", delimiter = ',')


df.head()


df.info()


# 1. Remove the non-informative ID column
df = df.drop("id", axis = 1)


import warnings
# 2. Truncate decimal values in selected columns
to_cut = ['Social_event_attendance', 'Going_outside','Friends_circle_size', 'Post_frequency']
df[to_cut] = np.trunc(df[to_cut])
# 3. Encode 'Personality' as binary: Extrovert â†’ 0, Introvert â†’ 1
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=FutureWarning)
    df['Personality'] = df['Personality'].replace('Extrovert', 0)
    df['Personality'] = df['Personality'].replace('Introvert', 1)


# 4. Replace 'Yes'/'No' with binary values and infer object types
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=FutureWarning)
    df = df.replace('Yes', 1.0)
    df = df.replace('No', 0.0).infer_objects(copy=False)



count_nan_rows = df.isna().any(axis=1).sum()

print(f'Number of rows with at least one NaN value: {count_nan_rows}')


df.info()
def show_null(df):
    null_stats = pd.DataFrame({
    '%NaN': df.isna().mean() * 100 
    })
    print(null_stats)
print("-------------------------------------------------------")
show_null(df)


# Separate features and target variable
Xdata = df.drop('Personality', axis=1)
Ydata = df['Personality']

# Set random seed for reproducibility
random_seed = 50

# Split into training (50%) and temp (50%)
Xtrain, Xval, Ytrain, Yval = train_test_split(Xdata, Ydata, test_size=0.5, random_state=random_seed)

# Split temp into validation (25%) and test (25%)
Xval, Xtest, Yval, Ytest = train_test_split(Xval, Yval, test_size=0.5, random_state=random_seed)

# Print shapes of the splits
print(f"Train shape, X: {Xtrain.shape}, y: {Ytrain.shape}")
print(f"Validation shape, X: {Xval.shape}, y: {Yval.shape}")
print(f"Test shape, X: {Xtest.shape}, y: {Ytest.shape}")


Xtrain_clean = Xtrain.dropna()


fig = plt.figure(figsize=(15, 13), constrained_layout=True)
spec = gridspec.GridSpec(ncols=2, nrows=2, figure=fig)

categories = [
    'Social_event_attendance', 
    'Going_outside', 
    'Friends_circle_size', 
    'Post_frequency'
]

for i, category in enumerate(categories):
    ax = fig.add_subplot(spec[i // 2, i % 2])
    data = Xtrain_clean[category]
    
    mode_val = data.mode()[0]  


    bars = sns.countplot(x=data, ax=ax, color='skyblue', edgecolor='black')


    for patch in ax.patches:
        x_val = patch.get_x() + patch.get_width()/2  

        bin_val = round(x_val)
        if bin_val == mode_val:
            patch.set_facecolor('darkorchid')
        else:
            patch.set_facecolor('cornflowerblue') 


    heights = {round(p.get_x() + p.get_width()/2): p.get_height() for p in ax.patches}
    mode_height = heights.get(mode_val, 0)
    ax.text(mode_val, mode_height + 1, f'{mode_val}', 
            ha='center', va='bottom', color='purple',
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    ax.set_title(f'Distribution of {category}')
    ax.set_xlabel('Value')
    ax.set_ylabel('Count')
    mode_patch = mpatches.Patch(color='darkorchid', label='Mode')
    other_patch = mpatches.Patch(color='cornflowerblue', label='Other values')
    ax.legend(handles=[mode_patch, other_patch])
    ax.grid(True)

plt.show()


plt.figure(figsize=(8, 5))
data = Xtrain_clean['Time_spent_Alone']
mean_val = data.mean()
sns.histplot(Xtrain_clean['Time_spent_Alone'], kde=True, label='Histogram', bins=14, color = "darkorange")

plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, label='Mean')
plt.text(mean_val, plt.ylim()[1]*(-0.1), f'{mean_val:.2f}', 
         ha='center', va='bottom', color='red',
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

plt.title('Distribution of Time_spent_Alone')
plt.xlabel('Value')
plt.ylabel('Density')

plt.grid(True)
plt.legend()
plt.show()


categories = [
    'Drained_after_socializing', 
    'Stage_fear'
]
fig = plt.figure(figsize=(15, 15), constrained_layout=True)
spec = gridspec.GridSpec(ncols=2, nrows=1, figure=fig)

labels = ['Yes', 'No']

for i, category in enumerate(categories):
    ax = fig.add_subplot(spec[0, i])
    data = Xtrain_clean[category]

    value_counts = pd.Series(data).value_counts()
    sizes = value_counts.values.tolist()
    ax.pie(sizes, labels=labels, colors = ['indianred', 'firebrick'], autopct='%.0f%%')
    
    ax.set_title(f'Pieplot for {category}')
    ax.legend()
    ax.grid(True)

plt.show()



Ytrain_df = pd.DataFrame(Ytrain)
fig = plt.figure(figsize=(12, 5), constrained_layout=True)
spec = gridspec.GridSpec(ncols=2, nrows=1, figure=fig)

classes = Ytrain_df['Personality'].unique()
palette = dict(zip(classes, sns.color_palette("viridis", len(classes))))
    
ax1 = fig.add_subplot(spec[0, 0])
ax2 = fig.add_subplot(spec[0, 1])

sns.countplot(x='Personality', data = Ytrain_df, palette=palette,ax = ax1)
ax1.set_title('Personality Distribution')

legend_handles = [Patch(color=palette[cls], label=cls) for cls in classes]
ax1.legend(handles=legend_handles, title='Personality')

for label in ax1.containers:
  ax1.bar_label(label)
    
value_counts = Ytrain_df['Personality'].value_counts()
counts = value_counts.values.tolist()

wedges, texts, autotexts = ax2.pie(
    counts,
    labels=classes,
    autopct='%.1f%%',
    startangle=90,
    colors=[palette[cls] for cls in classes],
    wedgeprops={'linewidth': 1, 'edgecolor': 'white'},
    textprops={'fontsize': 12}
)


centre_circle = plt.Circle((0, 0), 0.6, color='white')
ax2.add_artist(centre_circle)
ax2.set_title('Class Distribution (%)', fontsize=14)

plt.show()


# Calculate imputation values
time_median = Xtrain_clean['Time_spent_Alone'].median()
stage_mode =Xtrain_clean['Stage_fear'].mode()[0]
social_mode = Xtrain_clean['Social_event_attendance'].mode()[0]
outside_mode = Xtrain_clean['Going_outside'].mode()[0]
drained_mode = Xtrain_clean['Drained_after_socializing'].mode()[0]
friends_mode = Xtrain_clean['Friends_circle_size'].mode()[0]
post_mode = Xtrain_clean['Post_frequency'].mode()[0]


# Define a dictionary of fill values
fill_values = {
    'Time_spent_Alone': time_median,
    'Stage_fear': stage_mode,
    'Social_event_attendance': social_mode,
    'Going_outside': outside_mode,
    'Drained_after_socializing': drained_mode,
    'Friends_circle_size': friends_mode,
    'Post_frequency': post_mode                        
}
# Apply the fill to all data splits
Xtrain.fillna(value = fill_values, inplace = True)
Xval.fillna(value = fill_values, inplace = True)
Xtest.fillna(value = fill_values, inplace = True)


fill_values


# Check if any missing values remain in the training set
Xtrain.isna().sum().sum()


data_numeric = ['Time_spent_Alone',
            'Social_event_attendance',
            'Going_outside',
            'Friends_circle_size',
            'Post_frequency']
plt.figure(figsize=(10, 8))
sns.heatmap(Xtrain[data_numeric].corr(), annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Matrix of correlations of features")
plt.show()


def create_features(Xdata):
    Xdata['social_balance'] = Xdata['Going_outside'] - Xdata['Time_spent_Alone']
    Xdata['social_activity_index'] = Xdata['Social_event_attendance'] + Xdata['Going_outside'] + Xdata['Post_frequency']
    
    Xdata['drain_ratio'] = (Xdata['Time_spent_Alone'] + 1) / (1 + Xdata['Going_outside'])
    Xdata['drained_score'] = Xdata['Drained_after_socializing'] * Xdata['drain_ratio']

    Xdata['posts_x_friends'] = np.log1p(Xdata['Post_frequency'] * Xdata['Friends_circle_size']) 
    Xdata['fear_x_event'] = Xdata['Stage_fear'] * Xdata['Social_event_attendance'] 


create_features(Xtrain)
create_features(Xval)
create_features(Xtest)


new_data = ['social_balance', 
                    'social_activity_index', 
                    'drain_ratio', 
                    'drained_score', 
                    'posts_x_friends',
                   'fear_x_event',
           ]
data_numeric = np.concatenate((data_numeric, new_data))


plt.figure(figsize=(10, 8))
sns.heatmap(Xtrain[data_numeric].corr(), annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Matrix of correlations of features")
plt.show()


correlations = Xtrain[data_numeric].corrwith(Ytrain)
print(correlations.sort_values(ascending=False))


Xtrain.shape[1]


def tune_pca_with_model(model, param_grid, scaler, Xtrain, Ytrain, Xval, Yval):
    pipe = Pipeline([
        ('scaler', scaler),
        ('pca', PCA()),
        ('model', model)
    ])
    
    grid = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
    grid.fit(Xtrain, Ytrain)
    
    print("Best params:", grid.best_params_)
    print("Train accuracy:", accuracy_score(Ytrain, grid.predict(Xtrain)))
    print("Validation accuracy:", accuracy_score(Yval, grid.predict(Xval)))


param_grid = {
    'pca__n_components': [8, 9, 0.9],
    'model__max_depth': np.arange(2, 14, 2).tolist(),
    'model__n_estimators': np.arange(40, 130, 10).tolist(),
    'model__min_samples_leaf': [2, 5, 10],
    'model__max_features': ['sqrt', 'log2', 0.8]
}
warnings.filterwarnings("ignore", category=RuntimeWarning)
tune_pca_with_model(RandomForestClassifier(random_state = 42), param_grid, None, Xtrain, Ytrain, Xval, Yval)



param_grid = {
    'pca__n_components': [0.95, 0.9, 7, 8, 9],
    'model__n_neighbors': np.arange(2, 62, 1).tolist(),
    'model__weights': ['uniform', 'distance'],
    'model__metric': ['euclidean', 'manhattan']
}


warnings.filterwarnings("ignore", category=RuntimeWarning)
tune_pca_with_model(KNeighborsClassifier(), param_grid, None, Xtrain, Ytrain, Xval, Yval)
tune_pca_with_model(KNeighborsClassifier(), param_grid, StandardScaler(), Xtrain, Ytrain, Xval, Yval)
tune_pca_with_model(KNeighborsClassifier(), param_grid, MinMaxScaler(), Xtrain, Ytrain, Xval, Yval)


Cs1 = np.linspace(1e-6, 1, 40)
Cs2 = np.linspace(1, 1000, 30)
Cs = np.concatenate((Cs1, Cs2))

param_grid = {
    'pca__n_components': [0.95, 0.9, 7, 8, 9, 10, 11, 12],
    'model__C': Cs,
    'model__class_weight': [None, 'balanced', 
                            {0: 0.568, 1: 0.432}], # introvert/extrovert distribution in the real world
    'model__penalty': ['l1', 'l2']
}

tune_pca_with_model(LogisticRegression(solver = 'liblinear'), param_grid, None, Xtrain, Ytrain, Xval, Yval)
tune_pca_with_model(LogisticRegression(solver = 'liblinear'), param_grid, StandardScaler(), Xtrain, Ytrain, Xval, Yval)
tune_pca_with_model(LogisticRegression(solver = 'liblinear'), param_grid, MinMaxScaler(), Xtrain, Ytrain, Xval, Yval)


scaler = StandardScaler()
Xtrain_scaled = pd.DataFrame(
    scaler.fit_transform(Xtrain),
    columns=Xtrain.columns,
    index=Xtrain.index)

Xval_scaled = pd.DataFrame(
    scaler.fit_transform(Xval),
    columns=Xval.columns,
    index=Xval.index)

Xtest_scaled = pd.DataFrame(
    scaler.fit_transform(Xtest),
    columns=Xtest.columns,
    index=Xtest.index)


# Initialize PCA for Random Forest and KNN models to reduce to 8 components
pca_rf_knn = PCA(n_components = 8)
# Initialize PCA for Logistic Regression model to reduce to 11 components
pca_lr = PCA(n_components = 11)
# Fit PCA on the original training data for RF and KNN pipelines
pca_rf_knn.fit(Xtrain)
# Fit PCA on the scaled training data for Logistic Regression pipeline
pca_lr.fit(Xtrain_scaled)


# Transform the training data using the fitted PCA for RF and KNN models
Xtrain_pca_rf_knn = pca_rf_knn.transform(Xtrain)
# Transform the scaled training data using the fitted PCA for Logistic Regression model
Xtrain_pca_lr = pca_lr.transform(Xtrain_scaled)

#------------------------------------------------RF------------------------------------------------
# Initialize Random Forest classifier with specified hyperparameters
meta_rf = RandomForestClassifier(random_state = 42, max_depth = 10, max_features = 'log2', min_samples_leaf = 2, 
                              n_estimators = 90)
# Fit Random Forest model on PCA-transformed training data
meta_rf.fit(Xtrain_pca_rf_knn, Ytrain)


#------------------------------------------------kNN-----------------------------------------------
# Initialize K-Nearest Neighbors classifier with specified parameters
meta_knn = KNeighborsClassifier(metric = 'manhattan', n_neighbors = 7, weights = 'uniform')
# Fit KNN model on PCA-transformed training data
meta_knn.fit(Xtrain_pca_rf_knn, Ytrain)


#------------------------------------------------LR------------------------------------------------
# Initialize Logistic Regression model with given hyperparameters
meta_lr = LogisticRegression(solver = 'liblinear', 
                             C = 0.0512,
                            class_weight = None, penalty = 'l2')
# Fit Logistic Regression model on PCA-transformed scaled training data
meta_lr.fit(Xtrain_pca_lr, Ytrain)


models = {
    "Random Forest": (meta_rf, Xtrain_pca_rf_knn),
    "kNN": (meta_knn, Xtrain_pca_rf_knn),
    "Logistic Regression": (meta_lr, Xtrain_pca_lr)
}


fig = plt.figure(figsize=(15, 13), constrained_layout=True)
spec = gridspec.GridSpec(ncols=3, nrows=2, figure=fig)
# Plot ROC curves on the first row
for i, (name, (model, X)) in enumerate(models.items()):
    y_pred = model.predict(X)
    f1 = f1_score(Ytrain, y_pred)
    print(f"{name} F1-score: {f1:.3f}")

    y_proba = model.predict_proba(X)[:, 1]
    fpr, tpr, _ = roc_curve(Ytrain, y_proba)
    roc_auc = auc(fpr, tpr)

    ax = fig.add_subplot(spec[0, i])
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_title(f"{name} ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    ax.grid(True)

    # Confusion matrix below ROC curve
    cm = confusion_matrix(Ytrain, y_pred)
    tn, fp, fn, tp = cm.ravel()

    ax_cm = fig.add_subplot(spec[1, i])
    im = ax_cm.matshow(cm, cmap=plt.cm.Blues)
    for (j, k), val in np.ndenumerate(cm):
        ax_cm.text(k, j, val, ha='center', va='center', color='red', fontsize=16)

    ax_cm.set_title(f"{name} Confusion Matrix")
    ax_cm.set_xlabel("Predicted")
    ax_cm.set_ylabel("True")

plt.show()


def prepare_data_for_meta_model(Xdata, models):
    meta_df = pd.DataFrame()
    for (name, model), X in zip(models, Xdata):
        probs = model.predict_proba(X)[:, 1]
        meta_df[name] = probs
    return meta_df

def add_features(df, features_to_add):
    for name, feature in features_to_add:
        df[name] = feature.values


meta_models = [('rf', meta_rf),
              ('knn', meta_knn),
              ('lr', meta_lr)]

Xtrain_data = [Xtrain_pca_rf_knn, Xtrain_pca_rf_knn, Xtrain_pca_lr]


meta_rf_no_pca = RandomForestClassifier(random_state = 42, max_depth = 6, max_features = 'log2', min_samples_leaf = 2, 
                              n_estimators = 120)
meta_rf_no_pca.fit(Xtrain, Ytrain)


Xtrain_pca_rf_knn_df = pd.DataFrame(Xtrain_pca_rf_knn)
importances = meta_rf_no_pca.feature_importances_
feature_names = Xtrain.columns

feat_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)
plt.figure(figsize=(8, 4))
plt.barh(feat_df['Feature'].head(10), feat_df['Importance'].head(10))
plt.gca().invert_yaxis()
plt.title('Top Feature Importances')
plt.show()


meta_Xtrain = prepare_data_for_meta_model(Xtrain_data, meta_models)
features_to_add = [('social_activity_index', Xtrain['social_activity_index']), 
                   ('social_balance', Xtrain['social_balance']), 
                   ('Drained_after_socializing', Xtrain['Drained_after_socializing'])]
add_features(meta_Xtrain, features_to_add)


meta_Xtrain.head()


plt.figure(figsize=(10, 8))
sns.heatmap(meta_Xtrain.corr(), annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Matrix of correlations of features")
plt.show()


#--------------------------------------VALIDATION--------------------------------------
Xval_pca_rf_knn = pca_rf_knn.transform(Xval)
Xval_pca_lr = pca_lr.transform(Xval_scaled)

Xval_data = [Xval_pca_rf_knn, Xval_pca_rf_knn, Xval_pca_lr]
meta_Xval = prepare_data_for_meta_model(Xval_data, meta_models)

#-----------------------------------------TEST-----------------------------------------

Xtest_pca_rf_knn = pca_rf_knn.transform(Xtest)
Xtest_pca_lr = pca_lr.transform(Xtest_scaled)
Xtest_data = [Xtest_pca_rf_knn, Xtest_pca_rf_knn, Xtest_pca_lr]

meta_Xtest = prepare_data_for_meta_model(Xtest_data, meta_models)


#--------------------------------------VALIDATION--------------------------------------
features_to_add = [('social_activity_index', Xval['social_activity_index']), 
                   ('social_balance', Xval['social_balance']), 
                   ('Drained_after_socializing', Xval['Drained_after_socializing'])]

add_features(meta_Xval,features_to_add)

#-----------------------------------------TEST-----------------------------------------

features_to_add = [('social_activity_index', Xtest['social_activity_index']), 
                   ('social_balance', Xtest['social_balance']), 
                   ('Drained_after_socializing', Xtest['Drained_after_socializing'])]

add_features(meta_Xtest,features_to_add)


def train_svm(scaler, Xtrain, Xval, Ytrain, Yval):
    Cs1 = np.linspace(1e-6, 1, 30)
    Cs2 = np.linspace(1, 1000, 20)
    Cs = np.concatenate((Cs1, Cs2))
    Xtrain_scaled = Xtrain
    Xval_scaled = Xval
    
    best_accuracy = 0
    best_C = 0
    best_kernel = None 
    if scaler is not None:
        Xtrain_scaled = scaler.fit_transform(Xtrain)
        Xval_scaled = scaler.transform(Xval)
    for kernel in ['rbf', 'poly']:    
        for C in Cs:
            model = SVC(kernel = kernel, C = C)
            model.fit(Xtrain_scaled, Ytrain)
            current_accuracy = accuracy_score(Yval, model.predict(Xval_scaled))
            if current_accuracy > best_accuracy:
                best_accuracy = current_accuracy
                best_C = C
                best_kernel = kernel
    print(f"Best C: {best_C}, best kernel: {best_kernel}")
    final_model = SVC(C= best_C, kernel = best_kernel)
    final_model.fit(Xtrain_scaled, Ytrain)
    print("Train accuracy:", accuracy_score(Ytrain, final_model.predict(Xtrain_scaled)))
    print("Validation accuracy:", accuracy_score(Yval, final_model.predict(Xval_scaled)))
    return final_model


print("-----------------------------------NO SCALER-----------------------------------")
final_model_ns = train_svm(None, meta_Xtrain, meta_Xval, Ytrain, Yval)
print("--------------------------------STANDARD SCALER--------------------------------")
final_model_ss = train_svm(StandardScaler(), meta_Xtrain, meta_Xval, Ytrain, Yval)
print("---------------------------------MINMAX SCALER---------------------------------")
final_model_mm = train_svm(MinMaxScaler(), meta_Xtrain, meta_Xval, Ytrain, Yval)


scaler = StandardScaler()


meta_Xtrain_scaled = pd.DataFrame(
    scaler.fit_transform(meta_Xtrain),
    columns=meta_Xtrain.columns,
    index=meta_Xtrain.index
)

meta_Xtest_scaled = pd.DataFrame(
    scaler.transform(meta_Xtest),
    columns=meta_Xtest.columns,
    index=meta_Xtest.index
)


final_model = SVC(C = 0.0344, kernel = 'rbf')
final_model.fit(meta_Xtrain_scaled, Ytrain)
accuracy_score(Ytest, final_model.predict(meta_Xtest_scaled))


test_data = pd.read_csv("/kaggle/input/playground-series-s5e7/test.csv", delimiter = ',')
test_data_id = test_data["id"]
test_data = test_data.drop("id", axis=1)

# Truncate decimal values in selected columns
to_cut = ['Social_event_attendance', 'Going_outside','Friends_circle_size', 'Post_frequency']
test_data[to_cut] = np.trunc(test_data[to_cut])

# Replace 'Yes'/'No' with binary values and infer object types
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=FutureWarning)
    test_data = test_data.replace('Yes', 1.0)
    test_data = test_data.replace('No', 0.0).infer_objects(copy=False)
# Fill any missing values in the dataset using predefined fill_values dictionary
test_data.fillna(value = fill_values, inplace = True)
# Generate additional features based on the test data (custom function)
create_features(test_data)

# Apply dimensionality reduction for different models
test_data_rf_knn = pca_rf_knn.transform(test_data) # For Random Forest and KNN models
test_data_lr = pca_lr.transform(test_data) # For Logistic Regression model

# Prepare the transformed data as input for the meta-model ensemble
X_test_data = [test_data_rf_knn, test_data_rf_knn, test_data_lr]
meta_Xtest_data = prepare_data_for_meta_model(X_test_data, meta_models)

# Add selected features back to the meta-model input dataset
features_to_add = [('social_activity_index', test_data['social_activity_index']), 
                   ('social_balance', test_data['social_balance']), 
                   ('Drained_after_socializing', test_data['Drained_after_socializing'])]

add_features(meta_Xtest_data,features_to_add)
# Scale the final meta-model input features using a pre-fitted scaler
meta_Xtest_data_scaled = pd.DataFrame(
    scaler.transform(meta_Xtest_data),
    columns=meta_Xtest_data.columns,
    index=meta_Xtest_data.index
)


meta_Xtest_data_scaled.head()


meta_Xtest_data_scaled = pd.DataFrame(
    scaler.transform(meta_Xtest_data),
    columns=meta_Xtest_data.columns,
    index=meta_Xtest_data.index
)
preds = final_model.predict(meta_Xtest_data_scaled)
preds


result_df = pd.DataFrame()
result_df['id'] = test_data_id
result_df['Personality'] = preds
result_df['Personality'] = result_df['Personality'].replace(0, 'Extrovert')
result_df['Personality'] = result_df['Personality'].replace(1, 'Introvert')


result_df.head()


result_df.to_csv('submission.csv', index=False, sep=',')

