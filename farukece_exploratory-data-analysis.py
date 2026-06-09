from IPython.display import Image
Image(url='https://www.renemagritte.org/assets/img/paintings/the-lovers-1.jpg')


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid", palette="muted", font_scale=1.1)


import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="seaborn")


train = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')


train.sample(5)


test.sample(5)


def perform_eda(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """
    Perform Exploratory Data Analysis (EDA) on train and test datasets.
    
    
    Args:
    train (pd.DataFrame): Training dataset containing numeric music features.
    test (pd.DataFrame): Testing dataset containing numeric music features.
    
    
    The function:
    - Cleans inf/-inf values
    - Shows basic info and descriptive statistics
    - Plots distributions of features
    - Displays correlation heatmap
    - Compares train vs test feature distributions
    """
    
    
    def clean_data(df: pd.DataFrame) -> pd.DataFrame:
        """Replace inf/-inf with NaN and drop duplicates."""
        return df.replace([np.inf, -np.inf], np.nan).drop_duplicates()
    
    
    # Clean data
    train = clean_data(train.drop(columns=["id"], errors="ignore"))
    test = clean_data(test.drop(columns=["id"], errors="ignore"))
    
    
    print("===== TRAIN INFO =====")
    print(train.info())
    print("\n===== TEST INFO =====")
    print(test.info())
    
    
    print("\n===== TRAIN DESCRIPTIVE STATISTICS =====")
    print(train.describe().T)
    
    
    # Histograms of all numeric columns
    train.hist(bins=30, figsize=(16, 12), color="#1f77b4", edgecolor="black")
    plt.suptitle("Train Data Distributions", fontsize=18, y=1.02)
    plt.tight_layout()
    plt.show()
    
    
    # Pairplot (sampled for speed)
    sns.pairplot(train.sample(min(2000, len(train))), diag_kind="kde")
    plt.suptitle("Pairplot of Train Features (Sampled)", fontsize=18, y=1.02)
    plt.show()
    
    
    # Correlation heatmap
    plt.figure(figsize=(10, 8))
    corr = train.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
    plt.title("Feature Correlation Heatmap", fontsize=16)
    plt.show()
    
    
    # Compare train vs test distributions (only first 6 columns for readability)
    for col in train.columns[:6]:
        plt.figure(figsize=(8, 4))
        sns.kdeplot(train[col], label="Train", fill=True, alpha=0.5)
        sns.kdeplot(test[col], label="Test", fill=True, alpha=0.5)
        plt.title(f"Train vs Test Distribution: {col}")
        plt.legend()
        plt.show()


perform_eda(train, test)




