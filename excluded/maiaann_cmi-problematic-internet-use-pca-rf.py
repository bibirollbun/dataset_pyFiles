import pandas as pd # for data manipulation/analysis tabularly 
import numpy as np # for calculations and working with arrays
import matplotlib.pyplot as plt # for visuals
import seaborn as sns # for graphics (easier)
from sklearn.model_selection import train_test_split # splits dataset into training and testing sets
from sklearn.ensemble import RandomForestClassifier # builds models
from sklearn.metrics import classification_report, confusion_matrix # evaluates performance of model
from sklearn.preprocessing import StandardScaler # standardize data
from sklearn.decomposition import PCA # dimensionality reduction = visualizing and improving model performance


# to load csv data
trainSet = pd.read_csv('/kaggle/input/child-mind-institute-problematic-internet-use/train.csv')
testSet = pd.read_csv('/kaggle/input/child-mind-institute-problematic-internet-use/test.csv')
trainSet[:15] # to check the training data


# to get stat details on the training se (variable features - distribution, scale, etc.)
trainSet.describe().transpose()


trainSet.info() # tells us about the dataset structure (3960 entries, 82 columns)
# quite a few columns like Physical-Weight, Physical-HeartRate, etc., have NaN values - cleaning needed


trainSet['sii'].value_counts() # breakdown of sii rankings across the values (ranking ranges from 0 to 3)
# background: sii = severity impairment index (standard measure of problematic internet use): 0 for None, 1 for Mild, 2 for Moderate, and 3 for Severe


# many columns have NaN values, so we set a threshold for cleaning to be 50%
# if column has more than 50% non-NaN values, we keep them selected (bc they are more reliable data)
# otherwise, we replace NaN values with zero in columns with less than 50% non-NaN values (helps mainutain consistency and avoid issues during modeling)

threshold = 0.5 * len(trainSet) # take half the length of the training set
reliableCols = trainSet.columns[trainSet.isnull().sum() < threshold] # pick cols from training set where missing values < 50%
trainSet = trainSet[reliableCols] # the data we train model with will be just reliableCols

trainSet = trainSet.fillna(0) # replace any NaN values in the training set with zeroes


targetCol = 'sii' # target = what we want to predict or analyze
trainSetCleaned = trainSet.dropna(subset=[targetCol]) # so, get rid of rows that have missing sii values (this way, we only deal with complete data for target variable)

trainSetCleaned[:10] # check that this works and there is a cleaned set
trainSetCleaned.info() # details about the cleaned training set


# categorical (object) columns besides 'id'
categoryCols = ['Basic_Demos-Enroll_Season', 'CGAS-Season', 'Physical-Season','FGC-Season', 'BIA-Season', 'PCIAT-Season', 'SDS-Season', 'PreInt_EduHx-Season'] 

# need boxplots btwn target (sii) and the categorical columns (excluding 'id')
plt.figure(figsize = (16, 24))

for i, col in enumerate(categoryCols, 1): # i for index, enumerate iterates over categorical columns
    plt.subplot(4, 2, i)  # 4 rows, 2 columns, plot i
    sns.boxplot(x = col, y = 'sii', data = trainSetCleaned)
    plt.xticks(rotation = 45)
    plt.title(f"'sii' vs {col}")

plt.tight_layout()
plt.show()

# how do these categories (season of enrollment/participation for bio-electric impedance analysis, parent-child internet addiction test, sleep disturbance scale, 
# children's global assessment scale, physical measures, fitnessgram, internet use) affect the sii rankings?


numericCols = trainSetCleaned.select_dtypes(include=['float64', 'int64']).columns # now consider the numerical columns that have floats and integers as values

# to define plots per row 
plotsPerRow = 5
numRows = (len(numericCols) + plotsPerRow - 1) // plotsPerRow

# compare sii rankings against numerical columns
plt.figure(figsize = (20, 4 * numRows))

for i, col in enumerate(numericCols): # i is index, enumerate iterates over (in this case) the numerical columns
    plt.subplot(numRows, plotsPerRow, i + 1)
    sns.boxplot(x = 'sii', y = col, data = trainSetCleaned)
    plt.title(col)
    plt.tight_layout()

plt.show()

# similar thought: how do the numerical columns affect the sii rankings?


# categorical columns for seasons
seasonCols = ['Basic_Demos-Enroll_Season', 'CGAS-Season', 'Physical-Season', 'FGC-Season', 'BIA-Season', 'PCIAT-Season', 'SDS-Season', 'PreInt_EduHx-Season']
seasonMap = {'Spring': 0,'Summer': 1,'Fall': 2,'Winter': 3} # to encode these, we create a mapping that assigns numbers to each season (easier for algorithms to work with)

# loop through each season column to replace them with integer numbers (bc most ml models need numerical input)
for col in seasonCols:
    if col in trainSetCleaned.columns: # if we get a column that is part of the training set's included columns...
        trainSetCleaned[col] = trainSetCleaned[col].replace(seasonMap).infer_objects(copy=False).astype(int) # replace the values in the column with the number mapping
        # also make sure that values will be stored as integers + helps in cases where values were previously objects (strings) to make them integers


trainSetNoID = trainSetCleaned.drop(columns = ['id'], errors = 'ignore') # ignore the ID column bc it is unneeded for analysis
correlMatrix = trainSetNoID.corr() # to get correlation matrix, which quantifies how strongly pairs of features are related to each other (on scale of -1 to +1)
# positive value = direct relationship, negative value = inverse one

# now a heatmap to tie it all together
# colors = correlation strength (visually helps us pinpoint which features are most closely correlated)
plt.figure(figsize = (30, 30))
sns.heatmap(correlMatrix, annot = True, fmt = '.1f', cmap = 'coolwarm', square = True) # bluer
plt.title('Correlation Heatmap')
plt.show()


colsInCommon = trainSetCleaned.columns.intersection(testSet.columns) # check cleaned training set and test set to see what columns are shared in common

# Prepare the feature matrix X and target vector y
X = trainSetCleaned[colsInCommon].drop(columns=['id']) # select the shared columns from the cleaned training set, but ignore the ID column (which is useless for modeling)
y = trainSetCleaned['sii'] # the sii column in the cleaned training set bc it is our target variable that we want to predict (helps prep for ml model)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 2) # to split matrix X and target vector y into training and testing sets
# train_test_split is a scikit-learn function; test_size = 0.2 means it will set aside 20% of the data for testing (to help evaluate model performance later on)


# to standardize the features in X to have mean of 0 and stdev of 1 (helps improve ml algorithm performances)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) # fits scaler (gets mean and stdev) on X_train 
X_test_scaled = scaler.transform(X_test) # uses the same mean and stdev to transform X_test

# remember: 
# X = the columns shared in common between the cleaned training set and test set
# y = the sii rankings column from cleaned training set


# pca for dimensionality reduction of the testing and training sets
pca = PCA(n_components = 0.95)  # we keep 95% of the variance 
X_train_pca = pca.fit_transform(X_train_scaled) # fits pca on standardized X_train_scaled to make it lower-dimension
X_test_pca = pca.transform(X_test_scaled) # uses same pca transformation on X_test_scaled to keep consistent with training data

# X_train_pca has less features, but the most important patterns in the data are represented


# to create the random forest classifier (improves prediction accuracy and controls overfitting)
rfModel = RandomForestClassifier(n_estimators = 100, random_state = 42, max_depth = 10,  min_samples_split = 10, min_samples_leaf=4)
# we use 100 decision trees, limit each tree's depth to 10, and set the min and max samples for splitting/leaf nodes to make sure the model generalizes adequately


# to fit random forest model to pca-transformed data
rfModel.fit(X_train_pca, y_train) 
# use random forest for the pca-transformed/reduced X features and target vector y from training set


# to make predictions on the test set, which has data the model hasn't seen yet
y_pred = rfModel.predict(X_test_pca) 
# make predictions about the sii rankings (y) based on the pca-reduced X_test data

print('Classification Report: ') # for making classification report
print(classification_report(y_test, y_pred)) # print the actual sii rankings versus model's predicted ones

print('\nExplanation of the Classification Report:')
print('precision = how many predicted positives were actual positives (minimizes false POSITIVES).')
print('recall = how many actual positives were correctly identified (minimizes false NEGATIVES).')
print('f1-score = harmonic mean of precision and recall.')
print('support = number of actual occurrences of the class in the dataset.\n')

print('Confusion Matrix: ')
print(confusion_matrix(y_test, y_pred)) # get confusion matrix to evaluate the model's prediction accuracy

print('\nExplanation of the Confusion Matrix:')
print('each row = actual class; each column = predicted class.')
print('for example, entry at (i, j) = number of times class i was predicted as class j.')
print('diagonal entries = number of correct predictions for each class.')
print('off-diagonal entries = misclassifications.\n')

accuracy = rfModel.score(X_test_pca, y_test) 
# how do the actual (test) sii rankings compare to what model predicted from pca-reduced training set X features?

print(f'Model accuracy is {accuracy}.')


# define all columns for SEASONS (not looking strictly at categorical or numerical this time!!)
seasonColDef = ['Basic_Demos-Enroll_Season', 'CGAS-Season', 'Physical-Season', 'FGC-Season', 'BIA-Season', 'PCIAT-Season', 'SDS-Season', 'PreInt_EduHx-Season']
seasonMapper = {'Spring': 0, 'Summer': 1, 'Fall': 2, 'Winter': 3} # create a season mapper again

for col in seasonColDef: # replace season values in test data using the mapping
    if col in testSet.columns:  # Check if the column exists in testSet
        testSet[col] = testSet[col].map(seasonMapper) 
        # if column does exist, replace season name values with number mapping


testSet.fillna(0, inplace=True) # replace any NaN values with zeroes in the testing set
colsInCommon = trainSetCleaned.columns.intersection(testSet.columns) # check cleaned training set and test set to see what columns are shared in common

# to prep test data using columns shared in common
X_test_data = testSet[colsInCommon].drop(columns = ['id']) # consider all test data columns except ID one
X_test_scaled = scaler.transform(X_test_data) # same scaler used on training data gets used on test data

# for applying pca to test data
X_test_pca = pca.transform(X_test_scaled) #pca-reduced test X features are just the scaled test X features with pca transformations made
predictor = rfModel.predict(X_test_pca) # now make predictions based on the pca-reduced test data X features

finish = pd.DataFrame({'id': testSet['id'], 'sii': predictor}) # use the ID from testing set + the model's predicted sii rankings
finish.to_csv('submission.csv', index = False) # to save final dataframe as csv file
print(finish[:30]) # check that final dataframe is good to go

