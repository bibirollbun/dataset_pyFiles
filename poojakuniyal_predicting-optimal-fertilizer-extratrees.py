import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 15
matplotlib.rcParams['figure.figsize'] = (12,6)
import warnings
warnings.filterwarnings('ignore')


sample_df = pd.read_csv('/kaggle/input/playground-series-s5e6/sample_submission.csv')
sample_df.sample(5)


train_df = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
train_df.head()


train_df.info()


train_df.describe()


train_df.describe(include=object)


train_df['Fertilizer Name'].value_counts()


train_df['Soil Type'].unique()


train_df['Crop Type'].unique()


test_df = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
test_df.head()


test_id = test_df['id']


print('Training data size',train_df.shape)
print('Testing data size', test_df.shape)


train_df.drop(columns='id', axis=1, inplace=True)
test_df.drop(columns='id',axis=1, inplace=True) 


categorical_features  = train_df.select_dtypes(include=['object']).columns
categorical_features = [i for i in categorical_features if i not in 'Fertilizer Name']

numerical_features = train_df.select_dtypes(exclude='object').columns
print('categorical_features are: ',categorical_features)
print('numerical_features are: ',numerical_features)


# Create subplots
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))  
axes = axes.flatten()  # Convert 2D array of subplots into a flat list for easy iteration

# Loop through each feature
for i, feature in enumerate(numerical_features ):
    axes[i].hist(train_df[feature], bins=15, color='orange', edgecolor='red') 
    axes[i].set_title(f'Histogram of {feature}')
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel('Frequency')

plt.tight_layout() 
plt.show()


train_df_grouped = train_df.groupby('Fertilizer Name', as_index=False).sum()
fig, ax = plt.subplots(figsize=(10, 6))

x = range(len(train_df_grouped['Fertilizer Name']))
width = 0.3  # Width of the bars

ax.bar(x, train_df_grouped['Nitrogen'], width, color='#FF69B4', label='Nitrogen')
ax.bar([p + width for p in x], train_df_grouped['Potassium'], width, color='#FFD700', label='Potassium')
ax.bar([p + width*2 for p in x], train_df_grouped['Phosphorous'], width, color='#87CEFA', label='Phosphorous')

ax.set_xticks([p + width for p in x])
ax.set_xticklabels(train_df_grouped['Fertilizer Name'])
ax.set_xlabel("Fertilizer Name")
ax.set_title("Distribution of Nutrient Amounts in Fertilzer")
ax.legend()

plt.xticks(rotation=45)
plt.show()


# Group and calculate mean temperature levels
soil_temperature_avg = train_df.groupby('Soil Type')['Temparature'].mean()

# Plot as a pie chart
plt.figure(figsize=(5, 5))
soil_temperature_avg.plot(kind='pie', autopct='%1.1f%%', colors=['lightblue', 'salmon', 'purple', 'yellow'])

# Title
plt.title('Average Temperature by Soil Type')

plt.ylabel('')
plt.show()



# Group and calculate mean nitrogen levels
soil_humidity_avg = train_df.groupby('Soil Type')['Humidity'].mean()

# Plot as a pie chart
plt.figure(figsize=(5, 5))
soil_humidity_avg.plot(kind='pie', autopct='%1.1f%%', colors=['coral', 'lightgreen', 'pink', 'gold'])

# Title
plt.title('Mean Humidity for different Soil Type')

plt.ylabel('')  # Hide ylabel since pie charts don't need one
plt.show()


# Group and calculate mean moisture levels
soil_moisture_avg = train_df.groupby('Soil Type')['Moisture'].mean()

# Plot as a pie chart
plt.figure(figsize=(5, 5))
soil_moisture_avg.plot(kind='pie', autopct='%1.1f%%', colors=['teal', 'orange', 'cyan', 'lime'])

# Title
plt.title('Average Moisture Level by Soil Type')

plt.ylabel('')
plt.show() 


import matplotlib.pyplot as plt
from statsmodels.graphics.mosaicplot import mosaic
import matplotlib.cm as cm

# Get unique values
crop_types = ['Sugarcane', 'Millets', 'Barley', 'Paddy', 'Pulses', 'Tobacco',
              'Ground Nuts', 'Maize', 'Cotton', 'Wheat', 'Oil seeds']
soil_types = ['Clayey', 'Sandy', 'Red', 'Loamy', 'Black']

# Generate a colormap
cmap = cm.get_cmap('Set2', len(crop_types) * len(soil_types))

# Create a dictionary mapping each (Soil Type, Crop Type) to a unique color
custom_colors = {
    (soil, crop): cmap(i / (len(crop_types) * len(soil_types)))  # Normalize index for colormap
    for i, (soil, crop) in enumerate([(s, c) for s in soil_types for c in crop_types])
}

# Create the mosaic plot with Soil Type on X-axis and Crop Type on Y-axis
plt.figure(figsize=(12, 6))  # Adjust figure size
mosaic(train_df, ['Soil Type', 'Crop Type'], properties=lambda key: {'color': custom_colors.get(key, 'gray')})

# Rotate X-axis labels for better readability
plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')
plt.title('Crop Distribution by Soil Type: A Mosaic Visualization')
plt.tight_layout()  # Adjust layout
plt.show();



# Compute the correlation matrix
corr_matrix = train_df[numerical_features].corr()

# Create the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)

# Title and adjustments
plt.title("Correlation Heatmap of Numerical Features")
plt.xticks(rotation=45)
plt.yticks(rotation=0)

plt.show()


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder


x = train_df.drop(columns='Fertilizer Name', axis=1)
y = train_df['Fertilizer Name']
X_train,X_test, y_train,y_test = train_test_split(x,y, test_size=0.2, random_state=87)


X_train.shape, X_test.shape, y_train.shape, y_test.shape


from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


numerical_transformer = Pipeline(steps=[
    ('scaling', StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ('encode', OneHotEncoder())
])

# Combining transformers in a ColumnTransformer
preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_transformer, numerical_features),
    ('cat', categorical_transformer, categorical_features)
]) 


# Pre-process the data
X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)
test_df_transformed = preprocessor.transform(test_df)


# Get categorical feature names after transformation
cat_transformer = preprocessor.named_transformers_['cat']  # Extract the categorical transformer
cat_feature_names = cat_transformer.get_feature_names_out(categorical_features)  # Get the new feature names

# Convert to a list for easier use
cat_feature_names = list(cat_feature_names)
print(cat_feature_names)


feature_names = list(numerical_features) + cat_feature_names


# Recreate DataFrame with transformed data and original column names
X_train = pd.DataFrame(X_train_transformed, columns=feature_names)
X_test = pd.DataFrame(X_test_transformed, columns=feature_names)
test_df = pd.DataFrame(test_df_transformed, columns=feature_names)


X_train.head(3)


def map_at_k(y_true, y_pred, k=3):
    map_score = 0.0
    num_samples = len(y_true)

    for i in range(num_samples):
        true_labels = set(y_true[i])  # correct labels (usually 1 item)
        predicted_labels = y_pred[i][:k]  # top-k predicted

        precision_at_k = 0.0
        correct_count = 0

        for j, pred in enumerate(predicted_labels):
            if pred in true_labels:
                correct_count += 1
                precision_at_k += correct_count / (j + 1)
                true_labels.remove(pred)  # avoid duplicate credit

        if correct_count > 0:
            precision_at_k /= min(len(y_true[i]), k)

        map_score += precision_at_k

    return map_score / num_samples if num_samples > 0 else 0.0



def get_top_k_predictions(probabilities, classes, k=3):
    return [list(classes[np.argsort(p)[::-1][:k]]) for p in probabilities]



from sklearn.ensemble import ExtraTreesClassifier


# et = ExtraTreesClassifier(**{'n_estimators': 300, 'min_samples_split': 15,
#                              'min_samples_leaf': 5, 'max_features': 35, 'max_depth': 70,
#                              'class_weight': 'balanced', 'bootstrap': True})   # score 0.3020


# et = ExtraTreesClassifier(**{'n_estimators': 700, 'min_samples_split': 15,
#                              'min_samples_leaf': 10, 'max_features': 35, 'max_depth': 70,
#                              'class_weight': 'balanced', 'bootstrap': True})     # score 0.3041 


et = ExtraTreesClassifier(**{'n_estimators': 1000, 'min_samples_split': 25,
                             'min_samples_leaf': 15, 'max_features': 55, 'max_depth': 70,
                             'class_weight': 'balanced', 'bootstrap': False})


et.fit(X_train,y_train)



# 1. Predict top-3 fertilizers
def get_top_k_predictions(probabilities, classes, k=3):
    predictions = []
    for probs in probabilities:
        top_indices = np.argsort(probs)[::-1][:k]
        top_classes = [classes[i] for i in top_indices]
        predictions.append(" ".join(top_classes))  # Space-separated
    return predictions

# 2. Run prediction
probs = et.predict_proba(X_test)
classes = et.classes_
top3_preds = get_top_k_predictions(probs, classes, k=3)




# 2. Run prediction
probs = et.predict_proba(X_test)
classes = et.classes_
top3_preds = get_top_k_predictions(probs, classes, k=3)



y_true = [[label] for label in y_test]
top3_preds_split = [pred.split(" ") for pred in top3_preds]
map3_score = map_at_k(y_true, top3_preds_split, k=3)
print(f"MAP@3 Score: {map3_score:.4f}")



top3_preds[:10] 


# Predict probabilities for final test data


probs = et.predict_proba(test_df)
classes = et.classes_
top3_preds = get_top_k_predictions(probs, classes, k=3)




# Create a submission DataFrame
submission_df = pd.DataFrame({
    'id': test_id,  # Ensure these are your real test IDs in correct order
    'Fertilizer Name': top3_preds
})
submission_df.head() 


# Save to CSV
submission_df.to_csv('submission.csv', index=False)
print("âœ… Submission file 'submission.csv' generated successfully.")





