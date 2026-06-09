import numpy
import pandas
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import seaborn as sns


# Definition of a class that contains all the methods that will be used for the competition
class HeartDisease():
    datasetPath = "/kaggle/input/k-means-clustering-for-heart-disease-analysis/heart_disease.csv"
    samplePath = "/kaggle/input/k-means-clustering-for-heart-disease-analysis/sample.csv"

    def loadDataset(self):
        return pandas.read_csv(self.datasetPath, sep = ',')

    def substValWithNan(self, set, colname, val):
        zerosIndx = set[set[colname] == val].index
        set.loc[zerosIndx, colname] = numpy.nan
        return set

    def substNanWithAverage(self, set, colname):
        le = LabelEncoder()
        set[colname] = le.fit_transform(set[colname])

        # find nan in column and remove corresponding rows
        nanIndxs = set[set[colname] == (len(le.classes_) - 1)].index
        tmp = set.drop(nanIndxs)

        # compute mean value (with nan excluded)
        meanValue = numpy.mean(tmp[colname])

        # substitute nan with mean value
        if set[colname].dtype == numpy.int64:
            set.loc[nanIndxs, colname] = numpy.int64(meanValue)
        elif set[colname].dtype == numpy.float64:
            set.loc[nanIndxs, colname] = numpy.float64(meanValue)
        return set

    def transformToNumeric(self, set, colname):
        le = LabelEncoder()
        set[colname] = le.fit_transform(set[colname])
        return set

    def normalizeData(self, set):
        normalizedSet = (set - set.min()) / (set.max() - set.min())
        return normalizedSet

    def PCA(self, set, n):
        pca = PCA(n_components = n)
        pca.fit(set)
        print('Explained variance ratios:')
        print(pca.explained_variance_ratio_)
        print('Percentage of represented data:')
        print(sum(pca.explained_variance_ratio_) * 100)
        reducedTrainset = pca.transform(set)
        return pandas.DataFrame(reducedTrainset)

    def computeDistances(self, kmeans, set):
        indxs = []
        for i in range(0, kmeans.n_clusters):
            indxs.append([])
            indxs[i] = numpy.where(kmeans.labels_ == i)[0]
        
        max_distances = numpy.zeros((kmeans.n_clusters, kmeans.n_clusters))
        min_distances = numpy.zeros((kmeans.n_clusters, kmeans.n_clusters)) \
            + 1000
        avg_distances = numpy.zeros((kmeans.n_clusters, kmeans.n_clusters))
        for i in range(0, kmeans.n_clusters - 1):
            for j in range(i + 1, kmeans.n_clusters):
                first_set = set.loc[indxs[i], ]
                second_set = set.loc[indxs[j], ]
                for k in range(0, len(first_set.index)):
                    for h in range(0, len(second_set.index)):
                        dist = float(numpy.linalg.norm( \
                            first_set.loc[indxs[i][k], :] - \
                            second_set.loc[indxs[j][h], :]))
                        if dist > max_distances[i, j]:
                            max_distances[i, j] = dist
                        if dist < min_distances[i, j]:
                            min_distances[i, j] = dist
                        avg_distances[i, j] = \
                            avg_distances[i, j] + dist
                avg_distances[i, j] = 1 / (len(first_set.index) * \
                    len(second_set.index)) * avg_distances[i, j]
        min_distances[min_distances == 1000] = 0
        print('Distances matrix (Complete linkage)')
        print(max_distances)
        print('Distances matrix (Single linkage)')
        print(min_distances)
        print('Distances matrix (Complete Average)')
        print(avg_distances)

    def generateSubmission(self, kmeans, saveName):
        sample_submission = pandas.read_csv(self.samplePath, sep = ',')
        sample_submission['cluster'] = \
            kmeans.labels_[sample_submission.id.to_list()]
        sample_submission.to_csv(saveName, index=False)


# load dataset
hd = HeartDisease()
trainset = hd.loadDataset()


# id column is useless
trainset = trainset.drop(['id'], axis = 1)


# chol and trestbps columns have zeros in their columns... nonsense, substitute
# with nan
trainset = hd.substValWithNan(trainset, 'chol', 0)
trainset = hd.substValWithNan(trainset, 'trestbps', 0)


# substitute NaN with average column value
featureList = ['thal', 'slope', 'chol', 'fbs', 'oldpeak', 'trestbps', \
    'thalch', 'exang', 'restecg', 'ca']
for i in range(0, len(featureList)):
    trainset = hd.substNanWithAverage(trainset, featureList[i])


# transform string columns to numeric
featureList = ['sex', 'dataset', 'cp']
for i in range(0, len(featureList)):
    trainset = hd.transformToNumeric(trainset, featureList[i])

trainset.isnull().sum().sort_values(ascending = False)


# normalize data and apply PCA (10 components)
normalizedTrainset = hd.normalizeData(trainset)
reducedTrainset = hd.PCA(normalizedTrainset, 10)


# kmeans with n_clusters = 4, 3, 2 and comparing distances matrix
set = reducedTrainset
from sklearn.cluster import KMeans
print('-------------- 4 clusters:')
kmeans = KMeans(n_clusters = 4, random_state = 0, n_init = "auto").fit(set)
hd.computeDistances(kmeans, set)
print('-------------- 3 clusters:')
kmeans = KMeans(n_clusters = 3, random_state = 0, n_init = "auto").fit(set)
hd.computeDistances(kmeans, set)
print('-------------- 2 clusters:')
kmeans = KMeans(n_clusters = 2, random_state = 0, n_init = "auto").fit(set)
hd.computeDistances(kmeans, set)

# By looking at Single linkage distances matrices, clusters become rather close
# when n_clusters > 2. Therefore, n_clusters = 2 seems a good choice


hd.generateSubmission(kmeans, '/kaggle/working/heart_disease_submission.csv')

