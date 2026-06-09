# Library Imports

# Data Analysis
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Data Processing
from sklearn.model_selection import train_test_split
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.preprocessing import RobustScaler

# Machine Learning Algorithm
from xgboost import XGBClassifier

# Model Tuning and Selection
from sklearn.model_selection import GridSearchCV, KFold


import warnings
warnings.simplefilter("ignore", FutureWarning)


# Import data

train = pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')


train.head(5)

train.info()


class DataExplorer:
    """
    Class for exploring and analyzing a DataFrame.
    """
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df

    def get_info(self) -> None:
        """
        Displays general dataset information.
        """
        return self.df.info()

    def get_dimensions(self) -> tuple:
        """
        Returns the DataFrame's shape (rows, columns).
        """
        return self.df.shape

    def get_statistics(self) -> pd.DataFrame:
        """
        Returns descriptive statistics.
        """
        return self.df.describe()

    def get_missing_values(self) -> pd.Series:
        """
        Returns the number of missing values per column.
        """
        return self.df.isnull().sum()

    def get_duplicates_count(self) -> int:
        """
        Returns the number of duplicate rows.
        """
        return self.df.duplicated().sum()

    def plot_boxplots(self) -> None:
        """
        Displays boxplots for numerical variables.
        """
        try:
            num_cols = self.df.select_dtypes(include=['number']).columns
            n_cols = 5
            n_rows = (len(num_cols) + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
            axes = axes.flatten()

            for i, col in enumerate(num_cols):
                sns.boxplot(y=self.df[col], ax=axes[i])
                axes[i].set_title(col)

            for j in range(i + 1, len(axes)):
                fig.delaxes(axes[j])

            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Error generating boxplots: {e}")

    def plot_distributions(self) -> None:
        """
        Displays distributions for numerical variables.
        """
        try:
            num_cols = self.df.select_dtypes(include=['number']).columns
            n_cols = 5
            n_rows = (len(num_cols) + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
            axes = axes.flatten()

            for i, col in enumerate(num_cols):
                sns.histplot(self.df[col], kde=True, ax=axes[i])
                axes[i].set_title(col)

            for j in range(i + 1, len(axes)):
                fig.delaxes(axes[j])

            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Error generating distribution plots: {e}")

    def plot_correlation_heatmap(self) -> None:
        """
        Displays a correlation heatmap.
        """
        try:
            plt.figure(figsize=(10, 8))
            sns.heatmap(self.df.corr(), annot=True, fmt=".2f", cmap='coolwarm')
            plt.title("Correlation Heatmap")
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Error generating correlation heatmap: {e}")



explorer = DataExplorer(train)

print("Dataset dimensions:", explorer.get_dimensions())
print("\nGeneral statistics:\n", explorer.get_statistics())
print("\nMissing values:\n", explorer.get_missing_values())
print("\nNumber of duplicate rows:", explorer.get_duplicates_count())


explorer.plot_boxplots()


explorer.plot_distributions()


explorer.plot_correlation_heatmap()


import numpy as np

corr_matrix = train.corr().abs()

upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.90)]

train = train.drop(columns=to_drop)
test = test.drop(columns=to_drop)


class DataPreprocessing:
    """
    Handles data preprocessing, including scaling and missing value imputation.
    """
    def __init__(self, df_train: pd.DataFrame, df_test: pd.DataFrame, target_variable: str) -> None:
        """
        Initializes the class with training and test DataFrames.
        """
        self.df_train = df_train
        self.df_test = df_test
        self.target_variable = target_variable
        
        self.x_train = None
        self.y_train = None
        self.x_test = None
        
        self.columns = None
        self.scaler = None  
        self.columns_drop = None
        self.medians = None  # Store medians for imputation

    def preprocess_train(self):
        """
        Preprocesses training data: imputes missing values, removes highly correlated features,
        and scales the features.
        """
        try:
            # Remove 'id' column if present
            if "id" in self.df_train.columns:
                self.df_train = self.df_train.drop("id", axis=1)

            # Impute missing values with median
            self.medians = self.df_train.median(numeric_only=True)
            self.df_train = self.df_train.fillna(self.medians)

            # Identify and remove highly correlated features (>0.90)
            corr_matrix = self.df_train.corr().abs()
            upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            self.columns_drop = [col for col in upper_triangle.columns if any(upper_triangle[col] > 0.90)]
            self.df_train = self.df_train.drop(columns=self.columns_drop)

            # Split features (X) and target (y)
            X = self.df_train.drop(self.target_variable, axis=1)
            y = self.df_train[self.target_variable]

            # Store column names
            self.columns = X.columns

            # Scale features
            self.scaler = RobustScaler()
            self.x_train = self.scaler.fit_transform(X)
            self.y_train = y

        except Exception as e:
            print(f"Error preprocessing training data: {e}")

    def preprocess_test(self):
        """
        Preprocesses test data using the same scaling and imputation as training.
        """
        try:
            # Impute missing values using training set medians
            self.df_test = self.df_test.fillna(self.medians)

            # Remove highly correlated columns (same as train)
            self.df_test = self.df_test.drop(columns=self.columns_drop)

            # Extract and remove 'id' column
            if "id" in self.df_test.columns:
                self.test_ids = self.df_test["id"].values
                self.df_test = self.df_test.drop("id", axis=1)
            else:
                self.test_ids = np.arange(len(self.df_test))

            # Apply the trained scaler
            self.x_test = self.scaler.transform(self.df_test) if self.scaler else self.df_test.values

        except Exception as e:
            print(f"Error preprocessing test data: {e}")

    def preprocess_data(self):
        """
        Runs preprocessing for both train and test data.
        """
        try:
            self.preprocess_train()
            self.preprocess_test()
            return self.x_train, self.y_train, self.x_test, self.columns
        except Exception as e:
            print(f"Preprocessing failed: {e}")
            return None, None, None, None



preprocessor = DataPreprocessing(train, test, 'rainfall')

x_train, y_train, x_test, columns = preprocessor.preprocess_data()

print("Preprocessing completed:")
print("x_train shape:", x_train.shape)
print("y_train distribution:\n", pd.Series(y_train).value_counts())
print("x_test shape:", x_test.shape)


class ModelTrainer:
    """
    Class for training, tuning, and evaluating the model.
    """
    def __init__(self, x_train, y_train, x_test, df_test, param_grid, model: str) -> None:
        """
        Initializes the class with training and test sets, and the hyperparameter grid.
        """
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test  
        self.df_test = df_test  

        self.param_grid = param_grid
        self.model = model

        self.best_model = None
        self.grid_search = None
        self.best_score = None

    def train_model(self):
        """
        Trains the model using GridSearchCV with K-Fold and optimizes based on ROC-AUC.
        """
        try:

            if self.model == 'xgboost':
                model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')

            # Configure cross-validation with K-Fold
            kfold = KFold(n_splits=5, random_state=42, shuffle=True)

            # Search for the best model using ROC-AUC
            self.grid_search = GridSearchCV(
                estimator=model,
                param_grid=self.param_grid,
                scoring='roc_auc',
                cv=kfold,
                n_jobs=-1
            )

            # Train the model
            self.grid_search.fit(self.x_train, self.y_train)

            # Save the best model and its score
            self.best_model = self.grid_search.best_estimator_
            self.best_score = self.grid_search.best_score_

            print(f"Best validation ROC-AUC: {self.best_score:.4f}")
            print(f"Best hyperparameters explored: {self.grid_search.best_params_}")

            return self.best_model, self.best_score

        except Exception as e:
            print(f"Error during training: {e}")
            return None

    def predict_test(self):
        """
        Predicts the probability of rain on the test set and associates each result with its ID.
        """
        try:
            # Verify that the model has been trained
            if self.best_model is None:
                print("The model has not been trained yet.")
                return None

            if isinstance(self.df_test, pd.DataFrame):
                if "id" in self.df_test.columns:
                    test_ids = self.df_test["id"].values  # Extract IDs
                else:
                    print("Warning: Column 'id' not found in df_test. Automatic IDs will be assigned.")
                    test_ids = np.arange(len(self.x_test))  # Generate automatic IDs
            else:
                print("Error: df_test is not a valid DataFrame.")
                return None

            # Verify that `x_test` and `test_ids` have the same length
            if len(test_ids) != len(self.x_test):
                print(f"Error: df_test has {len(test_ids)} IDs, but x_test has {len(self.x_test)} samples.")
                return None

            # Predict probabilities on the entire test set
            prob_test = self.best_model.predict_proba(self.x_test)[:, 1]  # Probability of rain (class 1)

            # Create DataFrame with ID and rain probability
            results_df = pd.DataFrame({"id": test_ids, "rainfall": prob_test})

            results_df.to_csv('sample_submission.csv', index=False)

            print("Predictions completed. Results:")
            print(results_df.head(10))  # Display the first rows in the console

            return results_df

        except Exception as e:
            print(f"Error predicting on the test set: {e}")
            return None

    def plot_feature_importances(self, columns):
        """
        Plots feature importances from the trained model using matplotlib.
        """
        try:
            # Verify that the model has been trained
            if self.best_model is None:
                print("The model has not been trained yet.")
                return

            importances = self.best_model.feature_importances_
            feature_names = columns

            # Create a DataFrame for sorting and plotting
            df_importances = pd.DataFrame({
                "Feature": feature_names,
                "Importance": importances
            }).sort_values(by="Importance", ascending=False).reset_index(drop=True)

            plt.figure(figsize=(10, 6))
            plt.barh(df_importances["Feature"], df_importances["Importance"])
            plt.xlabel("Importance")
            plt.title("Feature Importances from XGBoost Model")
            plt.gca().invert_yaxis()  # Show most important features at the top
            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"Error generating feature importances plot: {e}")



# --- XGBoost ----- #

num_neg = (y_train == 0).sum()  
num_pos = (y_train == 1).sum() 

scale_pos_weight = (num_neg / num_pos)
print(f"scale_pos_weight: {scale_pos_weight:.2f}")

param_grid = {
    'n_estimators': [10, 50, 90, 100, 110],
    'max_depth': [5, 8, 9, 10, 11, 20],
    'learning_rate': [0.1, 0.2, 0.3],
    'colsample_bytree': [0.5, 0.7, 0.9, 1.0],
    'reg_lambda': [0.01, 0.1, 1, 10],
    'scale_pos_weight': [scale_pos_weight],
}


test.info()


trainer = ModelTrainer(x_train, y_train, x_test, test, param_grid, model='xgboost')
best_model, best_score = trainer.train_model()

print(f"Best validation ROC-AUC: {best_score:.4f}")

df_results = trainer.predict_test()


casi_1 = (df_results["rainfall"] > 0.95).sum()

casi_0 = (df_results["rainfall"] < 0.05).sum()

print(df_results.shape)

print(f"Number of predictions close to 1: {casi_1}")
print(f"Number of predictions close to 0: {casi_0}")
print(f"Percentage of extreme predictions: {((casi_1 + casi_0) / len(df_results)) * 100:.2f}%")


varianza_probs = df_results["rainfall"].var()

print(f"Variance of predicted probabilities on the test set: {varianza_probs:.4f}")


trainer.plot_feature_importances(columns)


plt.figure(figsize=(8, 5))
sns.histplot(df_results["rainfall"], bins=30, kde=True, color="royalblue")
plt.xlabel("Rain Probability (`rainfall`)")
plt.ylabel("Frequency")
plt.title("Distribution of Rain Probabilities in the Test Set")
plt.show()

