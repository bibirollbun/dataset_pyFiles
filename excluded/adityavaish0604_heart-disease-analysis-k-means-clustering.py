from pathlib import Path
import polars as pl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer

sns.set_style("whitegrid")


# Path to the heart disease dataset file
HEART_DISEASE_FILE_PATH = Path(
    "/kaggle/input/k-means-clustering-for-heart-disease-analysis/heart_disease.csv"
)
HEART_DISEASE_SAMPLE = Path(
    "/kaggle/input/k-means-clustering-for-heart-disease-analysis/sample.csv"
)
# Dictionary to rename columns to improve readability
COLUMNS_DICTIONARY = {
    "sex": "sex",
    "cp": "chest_pain_type",
    "trestbps": "resting_blood_pressure",
    "chol": "cholesterol",
    "fbs": "fasting_blood_sugar",
    "restecg": "resting_electrocardiogram",
    "thalch": "maximum_heart_rate",
    "exang": "exercise_angina",
    "oldpeak": "exercise_st",
    "slope": "slope_st",
    "ca": "n_major_vessels",
    "thal": "thalassemia",
}
CATEGORICAL_COLUMNS = [
    "sex",
    "chest_pain_type",
    "resting_electrocardiogram",
    "slope_st",
    "thalassemia",
]
RANDOM_STATE = 1506
CMAP = sns.diverging_palette(230, 20, n=4)

sns.set_palette(CMAP)


# Dataframe with heart disease raw data
df_heart = (
    pl.scan_csv(HEART_DISEASE_FILE_PATH)
    .drop(["id", "dataset"])
    .rename(COLUMNS_DICTIONARY)
    .collect()
)


df_heart.shape


list_columns = df_heart.columns
dtypes = df_heart.dtypes
null_count = df_heart.null_count()
nulls = (
    null_count.transpose()
    .with_columns(
        [
            pl.Series(values=list_columns, name="column_name"),
            pl.Series(values=dtypes, name="dtype"),
        ]
    )
    .rename({"column_0": "null_count"})
    .select(["column_name", "dtype", "null_count"])
    .filter(pl.col("null_count") > 0)
    .with_columns()
    .sort(by=pl.col("null_count"), descending=True)
)
# Get the columns with null values
null_columns = nulls["column_name"].to_list()
nulls


fig, ax = plt.subplots(ncols=2, nrows=5, figsize=(10, 20))

pd_heart = df_heart.to_pandas()
for i, name_col in enumerate(null_columns):
    n_col = i % 2
    n_row = i // 2

    counts = pd_heart[name_col].value_counts().reset_index()

    ax[n_row, n_col].set_title(name_col)
    if (
        name_col in CATEGORICAL_COLUMNS
        or name_col == "fasting_blood_sugar"
        or name_col == "exercise_angina"
    ):
        sns.barplot(counts, x=name_col, y="count", ax=ax[n_row, n_col])
    else:
        sns.histplot(counts, x=name_col, ax=ax[n_row, n_col], element="step")
    # ax[n_row, n_col].tick_params(axis='x', rotation=45)
    ax[n_row, n_col].set_ylabel("Count")
    ax[n_row, n_col].set_xlabel("")


plt.show()


df_heart = (
    df_heart
    .with_columns(
        [
            pl.col("n_major_vessels").cast(pl.Int64),
            pl.col(pl.Boolean).cast(pl.Int64)
        ]
    )
    .with_columns(
        [
            pl.col("n_major_vessels").fill_null(0),
            pl.col(pl.Float64).fill_null(strategy="mean"),
        ]
        +
        [
            # Get the mode without the nulls
            pl.col(col).fill_null(pl.col(col).drop_nulls().mode())
            for col in ["thalassemia", "slope_st", "fasting_blood_sugar", "exercise_angina", "resting_electrocardiogram"]
        ]
    )
)
null_count = df_heart.null_count()
null_count


class HeartDiseaseDataset:
    """
    A class to load and preprocess the heart disease dataset.
    """

    def __init__(
        self,
        data_path: Path = HEART_DISEASE_FILE_PATH,
        ids: list[int] = [],
        classes: dict[dict[int, str]] = {},
    ):
        self.data_path = data_path
        self.classes = classes
        self.ids = ids

        self.dataframe = self._get_dataframe()

    def _get_dataframe(self) -> pl.DataFrame:
        """
        Load the heart disease dataset from the given path.

        Parameters:
        data_path (Path): The path to the dataset.
        type (str): The type of the dataset to load. Can be "polars" or "numpy".

        Returns:
        pl.DataFrame: The loaded dataset.
        """
        dataframe = pl.scan_csv(HEART_DISEASE_FILE_PATH)

        # Filters the dataframe by the ids if provided
        if self.ids:
            dataframe = dataframe.filter(pl.col("id").is_in(self.ids))

        dataframe = (
            dataframe
            # drop the id and dataset columns
            .drop(["id", "dataset"])
            .rename(COLUMNS_DICTIONARY)
            # Cast the `n_major_vessels` column to int
            .with_columns(
                [pl.col("n_major_vessels").cast(pl.Int64)]
                + [pl.col(pl.Boolean).cast(pl.Int64)]
            )
            .with_columns(
                # Fill the float columns with the mean
                [
                    pl.col("n_major_vessels").fill_null(0),
                    pl.col(pl.Float64).fill_null(strategy="mean"),
                ]
                +
                # Fill the categorical columns with the mode without the nulls
                [
                    # Get the mode without the nulls
                    pl.col(col).fill_null(pl.col(col).drop_nulls().mode())
                    for col in [
                        "thalassemia",
                        "slope_st",
                        "fasting_blood_sugar",
                        "exercise_angina",
                        "resting_electrocardiogram",
                    ]
                ]
            )
        )

        # Map the categorical columns to their corresponding labels
        # if classes are provided, else map them using the LabelEncoder
        if self.classes:
            dataframe = dataframe.with_columns(
                pl.col(col).map_elements(
                    lambda x: self.classes[col][x]
                )
                for col in CATEGORICAL_COLUMNS
            )
        else:
            dataframe = dataframe.with_columns(
                pl.col(col).map_batches(lambda col: self._encode_categories(col))
                for col in CATEGORICAL_COLUMNS
            )

        # Cast all the columns to float64
        dataframe = dataframe.collect()
        return dataframe

    def get_data(self, type: str = "numpy") -> np.ndarray | pl.DataFrame | pd.DataFrame:
        """
        Get the dataset as a numpy array, polars dataframe, or pandas dataframe.

        Parameters:
        type (str): The type of the dataset to return. Can be "numpy", "polars", or "pandas".

        Returns:
        np.ndarray | pl.DataFrame | pd.DataFrame: The dataset as the specified type.
        """
        if type == "numpy":
            return self.dataframe.to_numpy()
        elif type == "polars":
            return self.dataframe
        elif type == "pandas":
            return self.dataframe.to_pandas()
        else:
            raise ValueError("Invalid type. Must be 'polars', 'numpy', or 'pandas'.")

    def _encode_categories(self, column: pl.Series) -> pl.Series:
        """
        Encode the categorical columns in the dataset.

        Parameters:
        column (pl.Series): The column to encode.

        Returns:
        pl.Series: The encoded column.
        """
        encoder = LabelEncoder()
        new_col = encoder.fit_transform(column.to_numpy())
        self.classes |= {
            column.name: {label: i for i, label in enumerate(encoder.classes_)}
        }
        return pl.Series(new_col, dtype=pl.Int64).alias(column.name)

    def get_ndim_data(self) -> np.ndarray:
        """
        Reduce the dimensionality of the dataset using PCA.

        Parameters:
        n_components (int): The number of components to keep.

        Returns:
        np.ndarray: The reduced dataset.
        """
        data = self.get_data()
        pca = PCA(n_components=2)
        return pca.fit_transform(data)

    def visualize_data(self):
        ndim_data = self.get_ndim_data()
        plt.figure(figsize=(10, 10))
        sns.scatterplot(x=ndim_data[:, 0], y=ndim_data[:, 1], palette=CMAP)
        plt.xlabel("First Principal Component")
        plt.ylabel("Second Principal Component")
        plt.show()


heart_dataset = HeartDiseaseDataset()
heart_dataset.visualize_data()


# Get the data as numpy array
np_heart_dataset = heart_dataset.get_data()



model = KMeans(random_state=RANDOM_STATE)
k_elbow = KElbowVisualizer(model, k=(1, 15))
k_elbow.fit(np_heart_dataset)
k_elbow.show()


kmeans_2 = KMeans(n_clusters=2, random_state=RANDOM_STATE)
silhouette_2 = SilhouetteVisualizer(kmeans_2, colors="yellowbrick")
silhouette_2.fit(np_heart_dataset)
score = silhouette_2.silhouette_score_
print(f"Silhouette Score: {score:.4f}")
silhouette_2.show()


kmeans_3 = KMeans(n_clusters=3, random_state=RANDOM_STATE) 
silhouette_3 = SilhouetteVisualizer(kmeans_3, colors="yellowbrick")
silhouette_3.fit(np_heart_dataset)
score = silhouette_3.silhouette_score_
print(f"Silhouette Score: {score:.4f}")
silhouette_3.show()


# Train the model using 3 clusters
model = KMeans(max_iter=1000, n_clusters=2, random_state=RANDOM_STATE)
model.fit(np_heart_dataset)


# Make predictions using the model 
preds = model.predict(np_heart_dataset)
heart_dataset_2d = heart_dataset.get_ndim_data()
predictions = np.append(heart_dataset_2d, preds[:, None], axis=1) 

# Visualize the predictions
fig = plt.figure(figsize=(10, 10))
sns.scatterplot(x=predictions[:, 0], y=predictions[:, 1], hue=predictions[:, 2])
plt.xlabel("First Principal Component")
plt.ylabel("Second Principal Component")
plt.show()


sample_ids = pl.read_csv(HEART_DISEASE_SAMPLE).get_column("id").to_numpy()
sample_preds = [preds[i] for i in sample_ids]
sample_2d_data = heart_dataset_2d[sample_ids]

plt.figure(figsize=(10, 10))
sns.scatterplot(x=sample_2d_data[:, 0], y=sample_2d_data[:, 1], hue=sample_preds)
plt.xlabel("First Principal Component")
plt.ylabel("Second Principal Component")
plt.show()




pl.DataFrame({"id": sample_ids, "cluster": sample_preds}).write_csv("submission.csv")

