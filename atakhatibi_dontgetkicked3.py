import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


train_FE_scaled = pd.read_csv('/kaggle/input/dontgetkicked/Carvana_train_FE.csv')
train_FE_scaled = train_FE_scaled.rename(columns={'Unnamed: 0': 'Id'})
train_FE_scaled.set_index('Id', inplace=True)


y_train = train_FE_scaled.IsBadBuy
x_train = train_FE_scaled.drop('IsBadBuy', axis=1)
x_train.info()


categorical = ['IsOnlineSale_z_score', 'Auction_MANHEIM_z_score', 'Auction_OTHER_z_score',
               'Make_CHRYSLER_z_score', 'Make_DODGE_z_score', 'Make_FORD_z_score', 'Make_HYUNDAI_z_score', 
               'Make_JEEP_z_score', 'Make_KIA_z_score', 'Make_MAZDA_z_score', 'Make_MERCURY_z_score', 'Make_MITSUBISHI_z_score', 
               'Make_NISSAN_z_score', 'Make_OTHER_z_score', 'Make_PONTIAC_z_score', 'Make_SATURN_z_score', 'Make_SUZUKI_z_score', 
               'Make_TOYOTA_z_score', 'Color_BLACK_z_score', 'Color_BLUE_z_score', 'Color_GOLD_z_score', 'Color_GREEN_z_score', 
               'Color_GREY_z_score', 'Color_MAROON_z_score', 'Color_OTHER_z_score', 'Color_RED_z_score', 'Color_SILVER_z_score', 
               'Color_WHITE_z_score', 'Transmission_MANUAL_z_score', 'WheelType_Covers_z_score', 'WheelType_Special_z_score', 
               'Nationality_OTHER_z_score', 'Nationality_OTHER ASIAN_z_score', 'Nationality_TOP LINE ASIAN_z_score', 'Size_CROSSOVER_z_score', 
               'Size_LARGE_z_score', 'Size_LARGE SUV_z_score', 'Size_LARGE TRUCK_z_score', 'Size_MEDIUM_z_score', 'Size_MEDIUM SUV_z_score', 
               'Size_SMALL SUV_z_score', 'Size_SMALL TRUCK_z_score', 'Size_SPECIALTY_z_score', 'Size_SPORTS_z_score', 'Size_VAN_z_score', 
               'TopThreeAmericanName_FORD_z_score', 'TopThreeAmericanName_GM_z_score', 'TopThreeAmericanName_OTHER_z_score',
               'PRIMEUNIT_YES_z_score','PRIMEUNIT_unknown_z_score','AUCGUART_RED_z_score','AUCGUART_unknown_z_score']

continuous = x_train.drop(categorical, axis=1).columns.tolist()


len(categorical), len(continuous)


correlation_matrix = x_train[continuous].corr()

# Visualize correlation matrix using a heatmap
plt.figure(figsize=(10, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", annot_kws={"size": 7})
plt.title('Pearson Correlation Heatmap')
plt.show()


from sklearn.decomposition import PCA

# PCA
pca = PCA(n_components= 2, random_state=717)
pca.fit(x_train[continuous])

pc_name = pd.DataFrame([f'pc_{i+1}' for i in range(pca.n_components_)], columns=['name'])
variance = pd.DataFrame(pca.explained_variance_, columns=['variance'])
variance_ratio = pd.DataFrame(pca.explained_variance_ratio_, columns=['variance_natio'])
total_explained_variance = variance_ratio.sum()
component_weights = pd.DataFrame(pca.components_, columns=x_train[continuous].columns)

pca_report = pd.concat((pc_name, variance, variance_ratio, component_weights), axis=1).set_index('name')


pca_x_train = pca.transform(x_train[continuous])
pca_x_train = pd.DataFrame(pca_x_train, columns = pc_name['name'].tolist())

k = 2
pca_train = pd.concat((y_train.reset_index(drop=True), pca_x_train.iloc[:, 0:k]), axis=1)


# Set up pairplot with hue
sns.pairplot(pca_train, hue='IsBadBuy')
# Show the pairplot
plt.show()


train_pca_fe = pd.concat((pca_train, x_train[categorical]), axis=1)
train_pca_fe.to_csv('/kaggle/working/train_pca_fe.csv')


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Step 1: Fit the LDA model for full n_components
lda = LinearDiscriminantAnalysis(n_components=None)



lda.fit(x_train[continuous], y_train)


plt.plot(lda.explained_variance_ratio_)

plt.title("LDA component and their variance ratio")
plt.xlabel("nth LDA component")
plt.ylabel("variance ratio")
plt.show()


# Step 2: Fit and Transform the original features into the reduced-dimensional space
lda = LinearDiscriminantAnalysis(n_components=1)
lda_x_train = lda.fit_transform(x_train[continuous], y_train)

columns_name = [f'lda_{i+1}' for i in range(lda_x_train.shape[1])]
lda_x_train = pd.DataFrame(lda_x_train, columns = columns_name)


lda_train = pd.concat((y_train.reset_index(drop=True), lda_x_train), axis=1)

# Set up pairplot with hue
sns.pairplot(lda_train, hue='IsBadBuy')
# Show the pairplot
plt.show()


train_lda_fe = pd.concat((lda_train, x_train[categorical]), axis=1)
train_lda_fe.to_csv('/kaggle/working/train_lda_fe.csv')

