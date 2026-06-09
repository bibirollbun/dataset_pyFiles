import pandas as pd


df = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
df.head()


df.info()


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap


sns.set_style("whitegrid")
sns.set_context("notebook", font_scale=1.2)

columns_to_plot = [col for col in df.columns if col != "id"]

# Create custom color palettes
blues = LinearSegmentedColormap.from_list("blues", ["#D4E6F1", "#2874A6", "#1A5276"])
greens = LinearSegmentedColormap.from_list("greens", ["#D5F5E3", "#2ECC71", "#1D8348"])
reds = LinearSegmentedColormap.from_list("reds", ["#FADBD8", "#E74C3C", "#943126"])
purples = LinearSegmentedColormap.from_list(
    "purples", ["#EBDEF0", "#8E44AD", "#4A235A"]
)

# Define color scheme for each variable
color_schemes = {
    "Sex": "#3498DB",
    "Age": blues,
    "Height": greens,
    "Weight": reds,
    "Duration": purples,
    "Heart_Rate": "#E67E22",
    "Body_Temp": "#F1C40F",
    "Calories": "#2ECC71",
}

# Set up the plot
fig = plt.figure(figsize=(20, 16))
fig.suptitle(
    "Distribution of Variables for Caloric Expenditure Prediction", fontsize=22, y=0.98
)

# Create subplot grid - 3 rows, 3 columns (with the last spot empty)
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Counter for subplot position
plot_idx = 0

for i, col in enumerate(columns_to_plot):
    row = plot_idx // 3
    col_idx = plot_idx % 3

    ax = fig.add_subplot(gs[row, col_idx])

    # Different plot types based on variable type
    if col == "Sex":
        # Countplot for categorical data
        sns.countplot(x=col, data=df, palette=[color_schemes[col]], ax=ax)
        ax.set_title(f"Distribution of {col}", fontsize=18)
        for p in ax.patches:
            ax.annotate(
                f"{int(p.get_height())}",
                (p.get_x() + p.get_width() / 2.0, p.get_height()),
                ha="center",
                va="bottom",
                fontsize=14,
            )
    else:
        # Histograms with KDE for numerical data
        if isinstance(color_schemes[col], str):
            color = color_schemes[col]
            sns.histplot(df[col], kde=True, color=color, ax=ax, alpha=0.7)
        else:
            sns.histplot(
                df[col], kde=True, color=color_schemes[col](0.6), ax=ax, alpha=0.7
            )

        ax.set_title(f"Distribution of {col}", fontsize=18)

        # Add statistical info
        stats_text = f"Mean: {df[col].mean():.2f}\nStd: {df[col].std():.2f}\nMin: {df[col].min():.2f}\nMax: {df[col].max():.2f}"
        props = dict(boxstyle="round", facecolor="white", alpha=0.7)
        ax.text(
            0.05,
            0.95,
            stats_text,
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox=props,
        )

    ax.set_xlabel(col, fontsize=14)
    if col != "Sex":
        ax.set_ylabel("Frequency", fontsize=14)

    plot_idx += 1

# Add a text box with information about the dataset
info_ax = fig.add_subplot(gs[2, 2])
info_ax.axis("off")
info_text = (
    "Dataset Information:\n\n"
    f"Number of rows: {len(df)}\n\n"
    "Notes:\n"
    "- The units of measure are derived from the numebrs\n"
    "- Sex is categorical\n"
    "- Age is in years\n"
    "- Height is in cm\n"
    "- Weight is in kg\n"
    "- Duration is in minutes\n"
    "- Heart Rate is in bpm\n"
    "- Body Temperature is in °C"
)
info_ax.text(
    0.1,
    0.5,
    info_text,
    fontsize=14,
    va="center",
    bbox=dict(boxstyle="round", facecolor="#F8F9F9", alpha=0.8),
)

plt.tight_layout()
plt.subplots_adjust(top=0.92)
plt.show()


df["Sex"] = df["Sex"].map({"male": 0, "female": 1})


plt.figure(figsize=(8, 6))
sns.heatmap(
    df[columns_to_plot].corr("spearman"), annot=True, cmap="coolwarm", fmt=".2f"
)
plt.title("Spearman's Rank correlation")
plt.show()


fig = plt.figure(figsize=(20, 16))
fig.suptitle("Correlation of Variables with Calories", fontsize=22, y=0.98)

# Create subplot grid - 2 rows, 4 columns (with the last spot empty)
gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

# Counter for subplot position
plot_idx = 0

for i, col in enumerate(columns_to_plot[1:-1]):
    row = plot_idx // 3
    col_idx = plot_idx % 3

    ax = fig.add_subplot(gs[row, col_idx])
    sns.scatterplot(x=col, y="Calories", data=df)
    ax.set_title(f"Calories VS {col}", fontsize=18)
    for p in ax.patches:
        ax.annotate(
            f"{int(p.get_height())}",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="bottom",
            fontsize=14,
        )

    ax.set_xlabel(col, fontsize=14)
    plot_idx += 1

plt.subplots_adjust(top=0.92)
plt.show()


df_features = df.copy(deep=True).drop(columns="id")
df_features = df_features.drop_duplicates(keep="first")
print(
    f"{df_features.shape[0]} ({df_features.shape[0]/df.shape[0]:.1%}) data points retained after dropping duplicates."
)

df_test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
df_test["Sex"] = df_test["Sex"].map({"male": 0, "female": 1})

for data in [df_features, df_test]:

    # Derived Physiological Measures
    data["age_adjusted_hr"] = data["Heart_Rate"] / (220 - data["Age"])
    data["body_surface_area"] = (
        0.007184 * (data["Height"] ** 0.725) * (data["Weight"] ** 0.425)
    )  # DuBois formula
    data["bmi"] = data["Weight"] / (data["Height"] / 100) ** 2

    # Basic Interaction Features
    data["hr_temp"] = data["Body_Temp"] * data["Heart_Rate"]
    data["rate_per_min"] = data["Heart_Rate"] / data["Duration"]
    data["bmi_hr"] = data["bmi"] * data["Heart_Rate"]
    data["temp_sq"] = data["Body_Temp"] ** 2

    # Basal Metabolic Rate (BMR)
    data["bmr"] = 0.0
    is_female = data["Sex"] == 0
    is_male = data["Sex"] == 1
    height_m = data["Height"] / 100
    data.loc[is_female, "bmr"] = (
        data.loc[is_female, "Weight"] * 10
        + height_m[is_female] * 6.25
        - data.loc[is_female, "Age"] * 5
        - 161
    )
    data.loc[is_male, "bmr"] = (
        data.loc[is_male, "Weight"] * 10
        + height_m[is_male] * 6.25
        - data.loc[is_male, "Age"] * 5
        + 5
    )


from sklearn.model_selection import (
    KFold,
    train_test_split,
    StratifiedKFold,
    RandomizedSearchCV,
)
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor, plot_importance


y = np.log1p(df_features["Calories"])
X = df_features.drop(columns=["Calories"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)


from skopt.space import Integer, Real

settings = {
    "hyperparams": {
        "task": "train",
        "boosting_type": "gbdt",
        "num_iterations": 500,
        "device_type": "cpu",
        "random_state": 42,
        "n_jobs": -1,
        "objective": "huber",
    },
    "bayesian": {
        "alpha": Real(0.2, 2.0, prior="uniform"),
        "n_estimators": Integer(100, 400),
        "learning_rate": Real(0.001, 0.25, prior="log-uniform"),
        "max_depth": Integer(3, 11),
        "num_leaves": Integer(30, 70),
        "bagging_fraction": Real(0.4, 1.0),
        "bagging_freq": Integer(0, 4),
        "feature_fraction": Real(0.7, 1.0),
        "min_child_samples": Integer(10, 50),
        "subsample": Real(0.15, 0.5),
        "reg_alpha": Real(0.001, 0.1, prior="log-uniform"),
        "reg_lambda": Real(0.001, 0.1, prior="log-uniform"),
    },
}


import statsmodels.api as sm
from scipy.stats import kstest


class ResidualAnalyses:
    """
    Class to perform:
        - Kolmogorov-Smirnov test for the normality of the residuals
        - Breusch-Pagan test of heteroschedasticity
        - Residual plots
        - Identify larger residuals
    """

    def __init__(
        self,
        y_pred: np.ndarray,
        y_true: np.ndarray,
        X: np.ndarray,
        alpha: float = 0.05,
    ):
        self.y_pred = y_pred
        self.y_true = y_true
        self.alpha = alpha
        self.X = X

        self.results = None
        self.residuals = None

    def calculate(self) -> None:
        """Run Kolmogorov-Smirnov and Breusch-Pagan tests"""
        self.residuals = self.y_pred - self.y_true

        ks_stats, ks_p_val = kstest(self.residuals, "norm")
        bp_result = sm.stats.het_breuschpagan(self.residuals, sm.add_constant(self.X))
        bp_stat, bp_p_val = bp_result[0], bp_result[1]  # noqa

        results_dict = {
            "Test Statistics": [round(ks_stats, 4), round(bp_stat, 4)],
            "p-value": [round(ks_p_val, 4), round(bp_p_val, 4)],
        }
        self.results = pd.DataFrame.from_dict(
            results_dict,
            orient="index",
            columns=["Kolmogorov-Smirnov", "Breusch-Pagan"],
        )

    def plot_residuals(self) -> None:
        residual_df = pd.DataFrame(
            [self.y_true.values, self.y_pred, self.residuals],
            index=["True values", "Predictcions", "Residuals"],
        ).T
        fig, axs = plt.subplots(1, 3, figsize=(16, 5))

        sns.scatterplot(data=residual_df, x="True values", y="Predictcions", ax=axs[0])
        axs[0].set_title("Actual VS Predicted", fontsize=18)
        axs[0].plot(
            self.y_pred, self.y_pred, color="black", linestyle="--", label="45° Line"
        )
        axs[0].legend()

        sns.histplot(self.residuals, kde=True, color=color, ax=axs[1], alpha=0.7)
        axs[1].set_title("Residuals Distribution", fontsize=18)

        sns.scatterplot(data=residual_df, x="Predictcions", y="Residuals", ax=axs[2])
        axs[2].set_title("Residuals Volatility", fontsize=18)
        axs[2].hlines(
            y=self.residuals.mean(),
            xmin=self.y_pred.min(),
            xmax=self.y_pred.max(),
            color="red",
            linestyle="--",
            label="Average",
        )
        axs[2].legend()

        plt.show()

    def get_top_n_residuals(self, n: int = 10) -> np.ndarray:
        return self.residuals.abs().nlargest(n).index


from sklearn.base import BaseEstimator
from typing import Tuple, Union
from skopt import BayesSearchCV


class ModelTrainer:
    def __init__(
        self,
        model_object: BaseEstimator,
        df_train: pd.DataFrame,
        ls_features: list,
        col_target: str,
        n_folds: int,
        grid: dict,
        refit: Union[str, bool],
        n_jobs: int = -1,
        random_state: int = 42,
        sample_weight: np.ndarray = None,
    ):
        self.model_object = model_object
        self.df_train = df_train
        self.ls_features = ls_features
        self.col_target = col_target
        self.n_folds = n_folds
        self.grid = grid
        self.refit = refit
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.sample_weight = sample_weight

    def _split_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, list]:
        """
        Split the data into features and target; split folds for CV
        :return: the feature df, the target df, list of tuples containing indices for CV (in order train, val)
        """
        # Shuffle the training data and drop the index
        self.df_train = self.df_train.sample(
            frac=1, random_state=self.random_state
        ).reset_index(drop=True)
        # Split the features and the target
        X_train, y_train = (
            self.df_train[self.ls_features],
            self.df_train[self.col_target],
        )
        # Recast the target as a dataframe
        y_train = pd.DataFrame(y_train)

        # Use Sturge's Rule to determine number of bins; cap at 10
        num_bins = int(np.floor(1 + np.log2(len(X_train))))
        if num_bins > 10:
            num_bins = 10

        ls_indices = list()

        X_train.loc[:, "target"] = y_train[self.col_target].values
        X_train["target_bins"] = pd.cut(
            X_train.loc[:, "target"].ravel(), bins=num_bins, labels=False
        )
        X_train.drop(columns=["target"], inplace=True)
        # Do StratifiedKFold on the binned target; Insert folds and return train and test indices
        kf = StratifiedKFold(
            n_splits=self.n_folds, shuffle=True, random_state=self.random_state
        )
        for fold, (train_idx, val_idx) in enumerate(
            kf.split(X=X_train, y=X_train["target_bins"].values)
        ):
            X_train.loc[val_idx, "fold"] = fold
            y_train.loc[val_idx, "fold"] = fold

            ls_indices.append((train_idx, val_idx))

        X_train = X_train.drop("target_bins", axis=1)

        return X_train, y_train, ls_indices

    def grid_search(
        self, metric: str = "neg_root_mean_squared_error", n_iter=10
    ) -> Tuple[dict, BayesSearchCV]:
        """
        Do a full, randomized, or Bayesian grid search
        :param metric: the scoring metric for the grid search
        :param n_iter: number of iterations for the randomized or Bayesian grid search; not used for the full grid search
        :return: the best parameters based on the search and the fitted grid object
        """

        X_train, y_train, ls_indices = self._split_data()
        grid_object = BayesSearchCV(
            estimator=self.model_object,
            search_spaces=self.grid,
            cv=ls_indices,
            n_jobs=self.n_jobs,
            refit=self.refit,
            scoring=metric,
            n_iter=n_iter,
            verbose=10,
            random_state=self.random_state,
        )
        grid_object.fit(
            X_train[self.ls_features],
            y_train[self.col_target].values,
            sample_weight=self.sample_weight,
        )

        print(f"Best parameters found by grid search are: {grid_object.best_params_}")
        best_params = grid_object.best_estimator_.get_params()

        return best_params, grid_object


model = LGBMRegressor(**settings["hyperparams"])

mt = ModelTrainer(
    model_object=model,
    df_train=X_train.join(y_train),
    ls_features=X_train.columns,
    col_target="Calories",
    n_folds=5,
    grid=settings["bayesian"],
    refit=True,
)
best_params, best_grid = mt.grid_search()


fitted_model = best_grid.best_estimator_
y_pred = fitted_model.predict(X_test)

print(f"RMSLE on Unseen data = {np.sqrt(mean_squared_error(y_test, y_pred))}")


best_grid.best_score_


fitted_model = best_grid.best_estimator_
y_pred = fitted_model.predict(X_test)

print(f"RMSLE on Unseen data = {np.sqrt(mean_squared_error(y_test, y_pred))}")


analyser = ResidualAnalyses(y_true=y_test, y_pred=y_pred, X=X_test.values)
analyser.calculate()
analyser.results


analyser.plot_residuals()


larger_residuals = analyser.get_top_n_residuals(n=15)


fig, axs = plt.subplots(2, 2, figsize=(16, 12), sharey=True)
large_errors = np.where(
    X_test.index.isin(larger_residuals), "Top 15 residuals", "Smaller residuals"
)

sns.scatterplot(x=X_test.Duration, y=y_test, ax=axs[0, 0], hue=large_errors)
sns.scatterplot(x=X_test.Body_Temp, y=y_test, ax=axs[0, 1], hue=large_errors)
sns.scatterplot(x=X_test.Heart_Rate, y=y_test, ax=axs[1, 0], hue=large_errors)
sns.scatterplot(x=X_test.rate_per_min, y=y_test, ax=axs[1, 1], hue=large_errors)
plt.suptitle(
    "The main drivers of the larger model errors are outliers in Duration and Body_Temp"
)
plt.show()


feature_importance = plot_importance(
    fitted_model,
    max_num_features=15,
    importance_type="auto",
    title="Feature Importance",
)




