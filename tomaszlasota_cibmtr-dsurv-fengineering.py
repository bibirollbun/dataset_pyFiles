import pandas as pd
import numpy as np
import optuna, json
from typing import Tuple
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
from optuna.integration import TFKerasPruningCallback
import plotly.express as px
import plotly.graph_objects as go
from tabulate import tabulate 
from lifelines import KaplanMeierFitter
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler, StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from sksurv.metrics import concordance_index_censored
pd.options.display.max_columns = None
import plotly.io as pio

pio.renderers.default = "notebook"
pd.options.display.max_columns = None
optuna.logging.set_verbosity(optuna.logging.ERROR)
optuna.logging.set_verbosity(optuna.logging.CRITICAL)


tf.keras.utils.set_random_seed(42)


class ConfigSettings:

    train = pd.read_csv("/kaggle/input/cibmtr/train.csv")
    test = pd.read_csv("/kaggle/input/cibmtr/test.csv")

    colorscale = "YlOrRd"
    color = "#EADDCA"


class EDAplot:

    def __init__(self, colorscale:str, color:str, df:pd.DataFrame=None):
        self.colorscale = colorscale
        self.color = color
        self.df = df
    
    def _get_df(self, df:pd.DataFrame):
        return df if df is not None else self.df
    
    def _prepare_data(self, df:pd.DataFrame = None, x=None, y=None):
        """
        Handles data input from DataFrame or x, y lists/arrays.
        """
        if x is not None and y is not None:
            return pd.DataFrame({'x': x, 'y': y})
        elif df is not None:
            return df
        elif self.df is not None:
            return self.df
        else:
            raise ValueError("No valid data source provided. Provide a DataFrame or x and y values.")

    
    def figure_template(self, fig, title):

        fig.update_layout(
            #template="plotly_dark",
            title=title,
            font=dict(color=self.color, family="Segoe UI", size=16),
            plot_bgcolor='rgba(40, 40, 43, 1)',  
            paper_bgcolor='rgba(40, 40, 43, 1)',
            legend=dict(font=dict(color=self.color)),
            margin=dict(l=20, r=20, t=40, b=20),
        )
        fig.show()
        return fig
    
    def plot_distribution(self, column, title, xaxis_name, df:pd.DataFrame=None, nbins=100, hue=None):

        df = self._get_df(df)

        if df is None:
            raise ValueError("A DataFrame must be provided for plot_distribution.")

        if hue is not None:
            color_sequence = px.colors.sequential.YlOrRd
        else:
            color_sequence = [self.color]

        fig = px.histogram(
            df,
            x=column,
            nbins = nbins,
            color=hue,
            color_discrete_sequence=color_sequence
        )
        fig.update_layout(
            xaxis_title=xaxis_name,
            yaxis_title='Density',
            xaxis=dict(gridcolor="grey"),
            yaxis=dict(gridcolor="grey", zerolinecolor="grey"),
            bargap=0.2
        )
        fig.update_traces(hovertemplate='Value: %{x:.2f}<br>Density: %{y:,}')

        fig = self.figure_template(fig, f"{title}")
    
    def count_chart(self, column, title, yaxis_name, xaxis_name, df:pd.DataFrame=None):

        df = self._get_df(df)

        if df is None:
            raise ValueError("A DataFrame must be provided for plot_distribution.")

        value_counts = df[column].value_counts().reset_index()
        value_counts.columns = [column, 'count']

        fig = px.bar(
            value_counts,
            x = column,
            y = 'count',
            color = "count",
            color_continuous_scale=self.colorscale,
            labels={column: column.capitalize(), 'count': 'Count'},
        )
        fig.update_layout(
            xaxis_title=xaxis_name,
            yaxis_title=yaxis_name,
            xaxis=dict(gridcolor="grey"),
            yaxis=dict(gridcolor="grey", zerolinecolor="grey"),
            margin=dict(l=20, r=20, t=60, b=20),
            bargap=0.2
        )
        fig.update_traces(
            hovertemplate=(
                '<b>Value:</b> %{x}<br>'
                '<b>Count:</b> %{y:,}<br>'
            ),
            marker=dict(line=dict(width=1, color='DarkSlateGrey'))
        )
        fig = self.figure_template(fig, f"{title}")
    
    def pie_chart(self, column, title, df:pd.DataFrame=None):

        df = self._get_df(df)

        if df is None:
            raise ValueError("A DataFrame must be provided for plot_distribution.")
        
        value_counts = df[column].value_counts().reset_index()
        
        fig = px.pie(
            value_counts,
            values='count',
            names=column,
            color_discrete_sequence=px.colors.sequential.YlOrRd,
            hole=0.4
            )
        
        fig.update_traces(
            textinfo='percent+label',
            hovertemplate=(
                '<b>Value:</b> %{label}<br>'
                '<b>Count:</b> %{value:,}<br>'
                '<b>Percentage:</b> %{percent}<br>'
            ),
            hoverlabel=dict(
                font=dict(color=self.color),
                bgcolor='rgba(40, 40, 43, 1)'
            )
        )
        
        fig = self.figure_template(fig, f'{title}')
    
    def bubble_chart(self, column, title, yaxis_name, xaxis_name, df:pd.DataFrame=None):

        df = self._get_df(df)

        if df is None:
            raise ValueError("A DataFrame must be provided for plot_distribution.")
        
        value_counts = df[column].value_counts().reset_index()
        
        fig = px.scatter(
            value_counts,
            x=column,
            y='count',
            size='count',
            color='count',
            size_max=180,
            color_continuous_scale=self.colorscale,
            labels={column: column.capitalize(), 'count': 'Count'},
            hover_name=column,
            hover_data={'count': True}
        )
        
        fig.update_layout(
            title_text=f'{title}',
            title_x=0.5,
            font=dict(color=self.color, family="Segoe UI", size=16),
            margin=dict(l=20, r=20, t=50, b=20),
            xaxis_title=xaxis_name,
            yaxis_title=yaxis_name,
        )
        
        fig.update_traces(
            marker=dict(line=dict(width=1, color='DarkSlateGrey')),
            hovertemplate=(
                '<b>Value:</b> %{x}<br>'
                '<b>Count:</b> %{y:,}<br>'
            ),
            hoverlabel=dict(
                font=dict(color=self.color), 
                bgcolor='rgba(40, 40, 43, 1)'
            )
        )
        
        fig = self.figure_template(fig, f'{title}')
    
    def heatmap(self, title, matrix):

        fig = go.Figure(data=go.Heatmap(
            z=matrix.values,
            x=matrix.columns,
            y=matrix.index,
            colorscale=self.colorscale,
            colorbar=dict(title="Coefficient"),
        ))

        # Update the layout for the heatmap
        fig.update_layout(
            title=title,
            title_x=0.5,
            xaxis_title='Features',
            yaxis_title='Features',
            font=dict(color=self.color, family="Segoe UI", size=16),
            height = 600
        )

        fig = self.figure_template(fig, f'{title}')
    
    def scatter_plot(self, x, y, title, xaxis_name, yaxis_name, color=None, size=None, df:pd.DataFrame=None):
        """
        Create a scatter plot for two numeric features.
        """
        df = self._get_df(df)

        if df is None:
            raise ValueError("A DataFrame must be provided for plot_distribution.")
        
        fig = px.scatter(
            df,
            x=x,
            y=y,
            color=color,
            size=size,
            title=title,
            color_continuous_scale=self.colorscale
        )
        fig.update_layout(
            xaxis_title=xaxis_name,
            yaxis_title=yaxis_name,
            xaxis=dict(gridcolor="grey"),
            yaxis=dict(gridcolor="grey", zerolinecolor="grey"),
            margin=dict(l=20, r=20, t=60, b=20)
        )
        fig.update_traces(
            marker=dict(line=dict(width=1, color='DarkSlateGrey')),
            hovertemplate=(
                '<b>X:</b> %{x}<br>'
                '<b>Y:</b> %{y}<br>'
            )
        )
        fig = self.figure_template(fig, title)
    
    def plot_bar_chart(self, x=None, y=None, title=None, xaxis_name=None, yaxis_name=None, df: pd.DataFrame = None, 
                       x_column=None, y_column=None):
        
        df = self._prepare_data(df, x, y)
        
        if x is not None and y is not None:
            fig = px.bar(df, x='x', y='y', color_discrete_sequence=[self.color])
        else:
            if x_column is None or y_column is None:
                raise ValueError("x_column and y_column must be specified when using a DataFrame.")
            fig = px.bar(df, x=x_column, y=y_column, color_discrete_sequence=[self.color])
        
        fig.update_layout(
            xaxis_title=xaxis_name,
            yaxis_title=yaxis_name,
            xaxis=dict(gridcolor="grey"),
            yaxis=dict(gridcolor="grey", zerolinecolor="grey")
        )
        fig = self.figure_template(fig, title)
    

    def plot_line_chart(self, x=None, y=None, title=None, xaxis_name=None, yaxis_name=None, 
                        df: pd.DataFrame = None, x_column=None, y_column=None):
        
        df = self._prepare_data(df, x, y)

        if x is not None and y is not None:
            fig = px.line(df, x='x', y='y', color_discrete_sequence=[self.color], markers=True)
        else:
            if x_column is None or y_column is None:
                raise ValueError("x_column and y_column must be specified when using a DataFrame.")
            fig = px.line(df, x=x_column, y=y_column, color_discrete_sequence=[self.color])
        
        fig.update_layout(
            xaxis_title=xaxis_name,
            yaxis_title=yaxis_name,
            xaxis=dict(gridcolor="grey"),
            yaxis=dict(gridcolor="grey", zerolinecolor="grey")
        )
        fig = self.figure_template(fig, title)
    

    def pie_chart_with_hue(self, column, title, df: pd.DataFrame = None, hue: str = None):
        df = self._get_df(df)

        if df is None:
            raise ValueError("A DataFrame must be provided for pie_chart.")

        if hue is not None:
            # Group by both column and hue
            value_counts = df.groupby([column, hue]).size().reset_index(name='count')
        else:
            # Default behavior: Aggregate by column
            value_counts = df[column].value_counts().reset_index(name='count')
            value_counts.rename(columns={'index': column}, inplace=True)

        # Choose the chart type based on hue
        if hue:
            # Sunburst Chart for hue support
            fig = px.sunburst(
                value_counts,
                path=[hue, column],  # Define hierarchy: hue -> column
                values='count',
                color=hue,
                color_discrete_sequence=px.colors.qualitative.Set1,
            )
            # Adjust textinfo specifically for Sunburst
            fig.update_traces(
                textinfo='label+percent parent',  # Shows label and % of parent
                hovertemplate=(
                    '<b>Category:</b> %{label}<br>'
                    '<b>Count:</b> %{value:,}<br>'
                    '<b>Percentage of Parent:</b> %{percentParent}<br>'
                )
            )
        else:
            # Standard Pie Chart
            fig = px.pie(
                value_counts,
                values='count',
                names=column,
                color_discrete_sequence=px.colors.sequential.YlOrRd,
                hole=0.4,
            )
            # Adjust textinfo for Pie
            fig.update_traces(
                textinfo='percent+label',
                hovertemplate=(
                    '<b>Value:</b> %{label}<br>'
                    '<b>Count:</b> %{value:,}<br>'
                    '<b>Percentage:</b> %{percent}<br>'
                )
            )

        fig.update_layout(
            title_text=f'{title}',
            title_x=0.5,
            font=dict(color=self.color, family="Segoe UI", size=16),
            margin=dict(l=20, r=20, t=50, b=20),
        )

        fig = self.figure_template(fig, f'{title}')



class EDADescriptive:
    def __init__(self, df:pd.DataFrame = None):
        self.df = df
    
    def _get_df(self, df):
        return df if df is not None else self.df

    def missing_values_summary(self, df:pd.DataFrame = None):
        df = self._get_df(df)
        missing_values = df.isnull().sum()
        missing_percentage = (missing_values / len(df)) * 100
        missing_data = pd.DataFrame({
            'Missing Values': missing_values,
            'Percentage': missing_percentage
        })

        # Filter columns with missing values
        missing_data = missing_data[missing_data['Missing Values'] > 0]
        # Sort by the number of missing values for better readability
        missing_data = missing_data.sort_values(by='Missing Values', ascending=False)
        # Adjust display options for better visibility
        pd.set_option('display.float_format', '{:.2f}'.format)
        pd.set_option('display.max_columns', None)  # Show all columns
        pd.set_option('display.width', 1000)       # Increase width for wide tables
        # Display the missing data summary in a professional format using tabulate
        print("ğŸ”� Missing Values Summary:")
        print(tabulate(missing_data, headers='keys', tablefmt='pretty'))

    def get_vif(self, num_cols:list,df:pd.DataFrame = None):
        df = self._get_df(df)
        df_constant = add_constant(df[num_cols])
        vif = pd.DataFrame()
        vif["Features"] = df_constant.columns
        vif["VIF Factor"] = [variance_inflation_factor(df_constant.values, i) for i in range(df_constant.shape[1])]
        vif = vif.sort_values(by="VIF Factor", ascending=False)
        print(tabulate(vif, headers="keys", tablefmt="psql"))


class DataPreprocessor:
    def __init__(self, X_train:pd.DataFrame = None, X_test: pd.DataFrame = None, X_val: pd.DataFrame = None):
        self.X_test = X_test
        self.X_train = X_train
        self.X_val = X_val
        self.scalers = {}
        self.imputers = {}
        self.encoders = {}

    
    def min_max_scale(self, columns:list) -> tuple:
        """
        Min-max scales numerical columns for X_train, X_test and X_val.
        """
        missing_cols = [col for col in columns if col not in self.X_train.columns 
                        or col not in self.X_test]
        if missing_cols:
            raise ValueError(f"The following columns are not in the DataFrame: {missing_cols}")
        
        scaler = MinMaxScaler()
        self.X_train[columns] = scaler.fit_transform(self.X_train[columns])
        if self.X_test is not None:
            self.X_test[columns] = scaler.transform(self.X_test[columns])
        if self.X_val is not None:
            self.X_val[columns] = scaler.transform(self.X_val[columns])
        self.scalers["min_max"] = scaler
        return self.X_train, self.X_test, self.X_val
    
    
    def simple_impute(self, strategy:str, columns:list) -> tuple:
        """
        Simple imputation for numerical or categorical columns in X_train, X_test and X_val.
        """
        
        # Validate if all columns exist in the DataFrame
        missing_cols = [col for col in columns if col not in self.X_train.columns 
                        or col not in self.X_test]
        if missing_cols:
            raise ValueError(f"The following columns are not in the DataFrame: {missing_cols}")
        
        simple_imputer = SimpleImputer(strategy = strategy)
        self.X_train[columns] = simple_imputer.fit_transform(self.X_train[columns])
        if self.X_test is not None:
            self.X_test[columns] = simple_imputer.transform(self.X_test[columns])
        if self.X_val is not None:
            self.X_val[columns] = simple_imputer.transform(self.X_val[columns])
        self.imputers["simple"] = simple_imputer
        return self.X_train, self.X_test, self.X_val
    
    
    def knn_impute(self, n_neighbors:int, columns:list, scaler:MinMaxScaler=None)-> tuple:
        """
        Perform KNN imputation on specified numerical columns in X_train and X_test.
        Optionally, rescale the imputed values to the original scale using a fitted scaler.
        
        Args:
            n_neighbors (int): Number of neighbors for KNN imputation.
            columns (list): List of numerical columns to impute.
            scaler (MinMaxScaler, optional): Fitted scaler to apply inverse transformation after imputation.

        Returns:
            tuple: Updated X_train, X_test and X_val DataFrames with imputed and optionally rescaled values.
        """
        
        # Validate if all columns exist in the DataFrame
        missing_cols = [col for col in columns if col not in self.X_train.columns 
                        or col not in self.X_test]
        if missing_cols:
            raise ValueError(f"The following columns are not in the DataFrame: {missing_cols}")

        knn_imputer = KNNImputer(n_neighbors=n_neighbors)

        self.X_train[columns] = knn_imputer.fit_transform(self.X_train[columns])

        if self.X_test is not None:
            self.X_test[columns] = knn_imputer.transform(self.X_test[columns])
        if self.X_val is not None:
            self.X_val[columns] = knn_imputer.transform(self.X_val[columns])

        self.imputers["knn"] = knn_imputer

        # If a scaler is provided, apply inverse scaling to return to the original scale
        if scaler:
            if not hasattr(scaler, 'scale_'):
                raise ValueError("The scaler provided must be fitted before calling this function.")
            self.X_train[columns] = scaler.inverse_transform(self.X_train[columns])
            if self.X_test is not None:
                self.X_test[columns] = scaler.inverse_transform(self.X_test[columns])
            if self.X_val is not None:
                self.X_val[columns] = scaler.inverse_transform(self.X_val[columns])
        return self.X_train, self.X_test, self.X_val
    

    def fillna_with_mapping(self,col_fill:str,col_map:str,mapping:dict,na_value:str)-> tuple:
        """
        Fills missing values in col_fill using mapping from col_map for X_train and X_test.
        """
        self.X_train.loc[self.X_train[col_fill].isna(), col_fill] = (
            self.X_train[col_map].map(mapping).fillna(na_value)
        )
        if self.X_test is not None:
            self.X_test.loc[self.X_test[col_fill].isna(), col_fill] = (
                self.X_test[col_map].map(mapping).fillna(na_value)
            )
        
        if self.X_val is not None:
            self.X_val.loc[self.X_val[col_fill].isna(), col_fill] = (
                self.X_val[col_map].map(mapping).fillna(na_value)
            )
        return self.X_train, self.X_test, self.X_val
    

    def frequency_encoder(self, cat_features:str) -> Tuple[pd.DataFrame,pd.DataFrame]:
        """
        Frequency encodes specified categorical columns in the training and testing DataFrames.

        Parameters:
            cat_features (list): List of categorical column names to be frequency encoded.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing:
                - `X_train` with the specified columns frequency encoded and original categorical columns removed.
                - `X_test` with the specified columns frequency encoded and original categorical columns removed.
        """
        
        for feature in cat_features:
            freq_map = self.X_train[feature].value_counts().to_dict()
            self.X_train[feature] = self.X_train[feature].map(freq_map)
            self.X_test[feature] = self.X_test[feature].map(freq_map).fillna(0)

        return self.X_train, self.X_test
    
    def one_hot_encode(self, columns:list) -> tuple:

        """
        One-hot encodes specified categorical columns in the training, test, and validation DataFrames.

        Parameters:
            columns (list): 
                List of column names to be one-hot encoded. This must include all categorical columns
                you wish to keep in the dataset. All other columns are excluded from the resulting DataFrames.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: 
                A tuple containing the updated X_train, X_test, and X_val DataFrames,
                with specified columns replaced by their one-hot-encoded counterparts.

        Raises:
            ValueError: 
                - If any column in `columns` is not found in either X_train or X_test.
                - If the `columns` parameter is missing or set to None.

        Notes:
            - The function ensures only the specified columns are retained in the dataset. 
            All other columns are removed.
            - If `columns` includes both categorical and non-categorical columns, only the 
            categorical columns are one-hot encoded; non-categorical columns are retained as is.
            - OneHotEncoder is used with `handle_unknown='ignore'` to handle categories 
            that may appear in one dataset but not the other.
            - The function transforms X_train, X_test, and X_val consistently using the encoder 
            fitted on X_train.
        """

        # Validate if all columns exist in the DataFrame
        missing_cols = [col for col in columns if col not in self.X_train.columns 
                        or col not in self.X_test]
        if missing_cols:
            raise ValueError(f"The following columns are not in the DataFrame: {missing_cols}")
        
        # Identify all categorical/object columns if columns are not specified
        if columns is None:
            raise ValueError("Column parameter is missing!!!")

        self.X_train = self.X_train[columns]
        self.X_test = self.X_test[columns]
        self.X_val = self.X_val[columns]

        categorical_cols = self.X_train.select_dtypes(include=["object", "category"]).columns
        
         # Initialize OneHotEncoder
        ohe = OneHotEncoder(drop=None, sparse_output=False, handle_unknown='ignore')
        
        # Fit the encoder on X_train and transform all datasets
        ohe.fit(self.X_train[categorical_cols])
        datasets_encoded = []
        
        for dataset in [self.X_train, self.X_test, self.X_val]:
            if dataset is not None:
                encoded = pd.DataFrame(
                    ohe.transform(dataset[categorical_cols]),
                    columns=ohe.get_feature_names_out(categorical_cols),
                    index=dataset.index,
                )
                # Drop original categorical columns and concatenate encoded features
                dataset = pd.concat([dataset.drop(categorical_cols, axis=1), encoded], axis=1)
            datasets_encoded.append(dataset)

        self.X_train, self.X_test, self.X_val = datasets_encoded

        return self.X_train, self.X_test, self.X_val
    
    def ordinal_encode(self, columns:list) -> tuple:

        """
        Applies ordinal encoding to specified categorical columns in a DataFrame.

        Parameters:
        columns : list
            List of column names to encode.

        Returns:
        pd.DataFrames
            The DataFrame with specified columns ordinally encoded.

        Raises:
        ValueError
            If any column in `columns` is not found in the DataFrame.
        """

        # Validate if all columns exist in the DataFrame
        missing_cols = [col for col in columns if col not in self.X_train.columns 
                        or col not in self.X_test]
        if missing_cols:
            raise ValueError(f"The following columns are not in the DataFrame: {missing_cols}")
        
        oridinal_encoder = OrdinalEncoder()
        self.X_train[columns] = oridinal_encoder.fit_transform(self.X_train[columns])
        if self.X_test is not None:
            self.X_test[columns] = oridinal_encoder.transform(self.X_test[columns])
        if self.X_val is not None:
            self.X_val[columns] = oridinal_encoder.transform(self.X_val[columns])
        
        self.encoders["oridinal"] = oridinal_encoder

        return self.X_train, self.X_test, self.X_val

    def label_encode(self, columns:list) -> Tuple[pd.DataFrame, pd.DataFrame]:

        """
        Encodes specified categorical columns in the training and test DataFrames using label encoding.

        Parameters:
        columns : list
            List of column names to encode.

        Returns:
        Tuple[pd.DataFrame, pd.DataFrame]
            A tuple containing the updated training and test DataFrames with the specified columns 
            label-encoded as new columns.

        Raises:
        ValueError
            If any column in `columns` is not found in either the training or test DataFrame.
        """
        # Validate if all columns exist in the DataFrame
        missing_cols = [col for col in columns if col not in self.X_train.columns 
                        or col not in self.X_test]
        if missing_cols:
            raise ValueError(f"The following columns are not in the DataFrame: {missing_cols}")
        
        label_encoder = LabelEncoder()
        for col in columns:
            self.X_train[f"{col}_encoded"] = label_encoder.fit_transform(self.X_train[col])
            if self.X_test is not None:
                self.X_test[f"{col}_encoded"] = label_encoder.transform(self.X_test[col])
        
        return self.X_train, self.X_test


    
    def imputing_pipeline(self, numerical_columns: list, categorical_columns: list,
                               tce_columns: list, mappings: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """

        Full preprocessing pipeline combining scaling, imputing, and mapping.

        """
        # Step 1: MinMax Scaling for numerical columns
        print("ğŸ“Š Step 1: MinMax Scaling Numerical Columns")
        self.X_train, self.X_test, self.X_val = self.min_max_scale(columns=numerical_columns)
        
        # Step 2: KNN Imputation for numerical columns
        print("ğŸ› ï¸� Step 2: KNN Imputation for Numerical Columns")
        scaler = self.scalers["min_max"]
        self.X_train, self.X_test, self.X_val = self.knn_impute(n_neighbors=3, columns=numerical_columns, scaler=scaler)
        self.X_train[numerical_columns] = np.round(self.X_train[numerical_columns])
        self.X_test[numerical_columns] = np.round(self.X_test[numerical_columns])
        self.X_val[numerical_columns] = np.round(self.X_val[numerical_columns])

        # Step 3: Fillna with mapping for TCE columns
        print("ğŸ› ï¸� Step 3: Simple Imputation for Categorical Columns")
        self.X_train, self.X_test, self.X_val = self.simple_impute(columns=categorical_columns, strategy="most_frequent")
        
        # Step 4: Simple Imputation for categorical columns
        print("ğŸ”„ Step 4: TCE Mapping Imputation")
        self.X_train, self.X_test, self.X_val = self.fillna_with_mapping(
            col_fill=tce_columns[1], col_map=tce_columns[2],
            mapping=mappings['mapping_a'], na_value="Permissive mismatched"
        )
        self.X_train, self.X_test, self.X_val = self.fillna_with_mapping(
            col_fill=tce_columns[0], col_map=tce_columns[1],
            mapping=mappings['mapping_b'], na_value="Permissive"
        )
        
        # Step 5: Final Simple Imputation for Remaining NaNs
        print("âœ… Step 5: Final Simple Imputation")
        # Identify columns with NaNs in X_train, X_test, and X_val
        nan_columns_train = self.X_train.columns[self.X_train.isna().any()].tolist()
        nan_columns_test = self.X_test.columns[self.X_test.isna().any()].tolist() if self.X_test is not None else []
        nan_columns_val = self.X_val.columns[self.X_val.isna().any()].tolist() if self.X_val is not None else []

        # Union of columns with NaNs across all datasets
        nan_columns = list(set(nan_columns_train).union(nan_columns_test, nan_columns_val))

        self.X_train, self.X_test, self.X_val = self.simple_impute(columns=nan_columns, strategy="most_frequent")
        
        print("ğŸš€ Preprocessing Complete!")
        return self.X_train, self.X_test, self.X_val
    
    


class FeatureEngineering:
    def __init__(self, df:pd.DataFrame = None):
        self.df = df
    
    def _get_df(self, df):
        return df if df is not None else self.df
    
    def get_KMF(self, event_name:str, event_time:str,df:pd.DataFrame = None) -> pd.DataFrame:
        
        df = self._get_df(df)
        kmf = KaplanMeierFitter()
        kmf.fit(df[event_time], df[event_name])

        # Interpolate survival probabilities for all data points and assign to the 'target' column
        df["KMF"] = kmf.survival_function_at_times(df[event_time]).values

        return df
    
    def pca(self, df:pd.DataFrame = None, n_components:int = None, num_cols:list = None) -> tuple:

        df = self._get_df(df)

        scaler = StandardScaler()
        scaled_df = scaler.fit_transform(df[num_cols])

        pca = PCA(n_components=n_components)
        pca.fit(scaled_df)
        explained_variance = pca.explained_variance_ratio_

        # Loadings and Feature Importance
        loadings = pd.DataFrame(pca.components_, columns=df[num_cols].columns)
        loadings.index = [f'PC{i+1}' for i in range(n_components)]
        feature_importance = loadings.abs()
        total_contribution = feature_importance.sum(axis=0).sort_values(ascending=False)

        return explained_variance, total_contribution
    
    
    def get_features_RFE(self, X_train:pd.DataFrame, X_test, y_train, y_test, label:str, num_features:list, steps:int = 10):
        
        result = []
        estimator = RandomForestRegressor(n_jobs= -1,random_state=42) # wllows parallel processing

        for n_features in num_features:

            selector = RFE(estimator, n_features_to_select=n_features, step=steps)
            selector.fit(X_train, y_train[label])

            # get selected features
            selected_features = X_train.columns[selector.support_]

            # transform data to only include the seleceted features
            X_train_selected = X_train[selected_features]
            X_test_selected = X_test[selected_features]

            # train model on selected features
            estimator.fit(X_train_selected, y_train[label])
            y_pred = estimator.predict(X_test_selected)
            mse = mean_squared_error(y_test[label], y_pred)

            result.append({
                "n_features": n_features,
                "mse": mse,
                "features": list(selected_features)
            })

            final_df = pd.DataFrame(result)

            final_df.to_json("/kaggle/working/features.json", orient="records")

        return final_df
    



class DeepSurv:
    
    def __init__(self, X_train:pd.DataFrame, y_train:pd.DataFrame, X_test:pd.DataFrame, y_test:pd.DataFrame):
        self.model = None
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.best_params = None

        # Split the data while preserving the original distribution
        X_train_sp, X_val, y_train_sp, y_val = train_test_split(
            self.X_train, self.y_train, test_size=0.2, random_state=42, stratify=self.y_train["efs"]
        )

        # Get the event times
        self.event_time_train = y_train_sp["efs_time"].values
        self.event_time_test = self.y_test["efs_time"].values
        self.event_time_val = y_val["efs_time"].values

        # Get the events
        self.indicator_train = y_train_sp["efs"].values
        self.indicator_val = y_val["efs"].values
        self.indicator_test = y_test["efs"].values

        self.X_train_sp = X_train_sp
        self.X_val = X_val

    @staticmethod
    def c_index(event_time, event, estimate):
        c_index = concordance_index_censored(
            event.astype(bool),
            event_time,
            estimate,
        )[0]
        return c_index
    
    
    @staticmethod
    def c_index_loss(y_true, y_pred):
        # Extract event_time and event
        event_time = y_true[:, 0]
        event = y_true[:, 1]

        # Use TensorFlow operations to compute the concordance index
        order = tf.argsort(event_time, direction="ASCENDING")
        sorted_event_time = tf.gather(event_time, order)
        sorted_event = tf.gather(event, order)
        sorted_pred = tf.gather(y_pred, order)

        concordant = tf.constant(0, dtype=tf.float32)
        permissible = tf.constant(0, dtype=tf.float32)

        for i in tf.range(tf.shape(sorted_event_time)[0] - 1):
            for j in tf.range(i + 1, tf.shape(sorted_event_time)[0]):
                if sorted_event_time[i] < sorted_event_time[j]:  # Permissible pair
                    permissible += 1.0
                    concordant += tf.cast(
                        (sorted_pred[i] < sorted_pred[j] and sorted_event[i] == 1) or
                        (sorted_pred[i] > sorted_pred[j] and sorted_event[j] == 1),
                        tf.float32,
                    )

        concordance_index = concordant / permissible
        return concordance_index  # Negative for minimization


        
    def objective(self, trial):
        # Suggest hyperparameters
        num_dense_layers = trial.suggest_int("num_dense_layers", 1, 5)  # Number of Dense layers
        neurons = trial.suggest_int("neurons", 32, 256, step=32)       # Neurons per layer
        dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)   # Dropout rate
        activation = trial.suggest_categorical("activation", ["relu", "tanh", "sigmoid"])
        optimizer_name = trial.suggest_categorical("optimizer", ["adam", "sgd", "rmsprop"])
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        
        model = Sequential()
        model.add(Dense(neurons, activation=activation, input_dim=self.X_train_sp.shape[1]))
        model.add(Dropout(dropout_rate))
        for _ in range(num_dense_layers - 1):
            model.add(Dense(neurons, activation=activation))
            model.add(Dropout(dropout_rate))
        model.add(Dense(1, activation=None))  # Output layer

        # Compile the model
        if optimizer_name == "adam":
            optimizer = Adam(learning_rate=learning_rate)
        elif optimizer_name == "sgd":
            optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
        elif optimizer_name == "rmsprop":
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)

        
        def wrapped_loss(y_true, y_pred):
            # Ensure y_pred is 1D
            y_pred = tf.squeeze(y_pred)

            # Compute concordance index loss
            return DeepSurv.c_index_loss(y_true, y_pred)

        model.compile(optimizer=optimizer, loss=wrapped_loss)

        # Prepare training and validation labels
        y_train = np.stack([self.event_time_train, self.indicator_train], axis=-1)
        y_val = np.stack([self.event_time_val, self.indicator_val], axis=-1)

        # Create TensorFlow datasets with batching
        train_dataset = tf.data.Dataset.from_tensor_slices((self.X_train_sp, y_train))
        train_dataset = train_dataset.batch(32, drop_remainder=True)

        val_dataset = tf.data.Dataset.from_tensor_slices((self.X_val, y_val))
        val_dataset = val_dataset.batch(32, drop_remainder=True)

        # Early stopping callback to prevent overfitting
        early_stopping = EarlyStopping(
            monitor="val_loss",  # Monitor validation loss
            patience=5,          # Stop after 5 epochs of no improvement
            restore_best_weights=True  # Restore the best model weights
        )
        pruning_callback = TFKerasPruningCallback(trial, monitor="val_loss")

        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=20,
            verbose=1,
            callbacks=[pruning_callback],
        )

        predictions = model.predict(self.X_val, verbose=0).flatten()
        final_c_index = DeepSurv.c_index(self.event_time_val, self.indicator_val, predictions)

        return final_c_index

    
    def get_best_param(self, n_trials:int = 50, save_path: str = "/kaggle/input/cibmrt_dsurv_model_1/tensorflow2/default/1/best_params.json"):
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, n_trials=n_trials)
        self.best_params = study.best_params
        print("Best hyperparameters:", self.best_params)

        # save the paramters into a file
        with open(save_path, "w") as file:
            json.dump(self.best_params, file)

        return self.best_params
    
    def load_best_params(self, load_path: str = "/kaggle/input/cibmrt_dsurv_model_1/tensorflow2/default/1/best_params.json"):

        with open(load_path, "r") as file:
            self.best_params = json.load(file)

        print(f"Best hyperparamters loaded from {load_path}")

        return self.best_params
    
    def build_best_model(self):

        if not self.best_params:
            raise ValueError("Best parameters not found. Load or generate them first.")
        
        best_params = self.best_params
        final_model = Sequential()
        final_model.add(Dense(best_params["neurons"], activation=best_params["activation"], input_dim=self.X_train.shape[1]))
        final_model.add(Dropout(best_params["dropout_rate"]))
        for _ in range(best_params["num_dense_layers"] - 1):
            final_model.add(Dense(best_params["neurons"], activation=best_params["activation"]))
            final_model.add(Dropout(best_params["dropout_rate"]))
        final_model.add(Dense(1, activation=None))

        # Compile the final model
        if best_params["optimizer"] == "adam":
            optimizer = Adam(learning_rate=best_params["learning_rate"])
        elif best_params["optimizer"] == "sgd":
            optimizer = tf.keras.optimizers.SGD(learning_rate=best_params["learning_rate"])
        elif best_params["optimizer"] == "rmsprop":
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=best_params["learning_rate"])

        final_model.compile(optimizer=optimizer, loss=DeepSurv.c_index_loss)

        self.model = final_model

        return self.model
        
    
    def evaluate_model(self):
        if not self.model:
            raise ValueError("Model not trained. Run 'build_best_model' first.")
        metrics = self.model.evaluate(self.X_test, self.indicator_test, verbose=1)
        print("Evaluation Metrics:", metrics)
        return metrics
    
    


cfs = ConfigSettings()
df_train = cfs.train.drop("ID", axis=1)
df_test = cfs.test.drop("ID", axis=1)


# get names of categorical and numerical features
cat_train_columns = df_train.select_dtypes(include=["object", "category"]).columns
num_train_columns = df_train.select_dtypes(include=["number"]).columns
edad = EDADescriptive(df_train)


# Quick summary statistics of all the numerical features
df_train.describe()


num_col = num_train_columns


df_train[num_col].head()


# Check for correlation
df_train[num_col].corr()


# Check for covariance
df_train[num_col].cov()


# Check for duplicated rows
df_train.duplicated().sum()


# Check for null values (displayes as count and %)
edad.missing_values_summary()


# NOTE: Split the dataset into train and test sets and then perform preprocessing and feature selection to avoid data leakage

# STEP 1. Perform training and test split
fe = FeatureEngineering(df=df_train)
df_train = fe.get_KMF(event_name="efs", event_time="efs_time")
X = df_train.drop(columns=["efs", "efs_time", "KMF"], axis=1)
y = df_train[["efs", "efs_time", "KMF"]]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
race_groups = X_train["race_group"]

# STEP 2. Define Columns and Mappings
na_columns = df_train.select_dtypes(include=["number"]).columns[
    df_train.select_dtypes(include=["number"]).isna().any()
].tolist()

categorical_columns = ["tce_imm_match"]
tce_columns = ["tce_match", "tce_div_match", "tce_imm_match"]

mappings = {
    'mapping_a': {
        'P/P': 'Permissive mismatched', 
        'G/G': 'GvH non-permissive', 
        'H/H': 'HvG non-permissive'
    },
    'mapping_b': {
        "Permissive mismached": "Permissive",
        "GvH non-permissive": "GvH non-permissive",
        "HvG non-permissive": "HvG non-permissive",
    }
}

# STEP 3. Run Preprocessing Pipeline on X_test and X_train datasets
preprocessor = DataPreprocessor(X_train=X_train, X_test=X_test, X_val=df_test)
X_train, X_test, X_val = preprocessor.imputing_pipeline(
    numerical_columns=na_columns,
    categorical_columns=categorical_columns,
    tce_columns=tce_columns,
    mappings=mappings
)


# STEP 5. Check if preprocessing was successful
edad_train = EDADescriptive(X_train)
edad_train.missing_values_summary()


edad_test = EDADescriptive(X_test)
edad_test.missing_values_summary()


edad_val = EDADescriptive(X_val)
edad_test.missing_values_summary()


# STEP 6. Get Variance Inflation Factor (VIF) for feature selection
num_cols = X_train.select_dtypes(include=["number"]).columns
edad_train.get_vif(num_cols=num_cols)


# STEP 7. Perfrom PCA for feature selection
fe_train = FeatureEngineering(df=X_train)
explained_variance, total_contribution = fe_train.pca(n_components= len(num_cols), num_cols=num_cols)


eda_pca = EDAplot(colorscale=cfs.colorscale, color=cfs.color)



eda_pca.plot_bar_chart(x=total_contribution.head(20).index, y=total_contribution.head(20).values,
                       title="Top Contributing Features", 
                       xaxis_name="Feature", yaxis_name="Contribution")


x = range(1, len(explained_variance)+1)
z = y=np.cumsum(explained_variance)
eda_pca.plot_line_chart(x = x, y = z, title = "Explained Variance by Principal Components", 
                        xaxis_name = "Principal Component", yaxis_name = "Cumulative Explained Variance")


# STEP 7. Normalize numerical data and encode ordinal features.
# Numerical features will be normalized using Min-Max scaling.
# Features with an ordinal scale will be encoded using ordinal encoding to preserve their inherent order.

cont_col = ["age_at_hct", "donor_age"]
ordinal_col = num_cols.drop(cont_col)
X_train, X_test, X_val = preprocessor.min_max_scale(columns = cont_col)
X_train, X_test, X_val = preprocessor.ordinal_encode(columns=ordinal_col)


# STEP 8. Convert categorical features into numeric values using frequency encoding.
# This step should utilize a separate preprocessing object to ensure the original dataset remains unaltered.
fq_X_train = X_train.copy()
fq_X_test = X_test.copy()
freq_preprocessor = DataPreprocessor(X_train=fq_X_train, X_test=fq_X_test)
fq_X_train, fq_X_test = freq_preprocessor.frequency_encoder(cat_features=cat_train_columns)


# STEP 9. Check feature importance using Recursive Feature Elimination
# Tip: Save the RFE results to a file to avoid re-running the feature selection process, 
# saving time for future executions.

eda_features = EDAplot(colorscale=cfs.colorscale, color=cfs.color)
feature_selection = FeatureEngineering()

features = feature_selection.get_features_RFE(fq_X_train, fq_X_test, y_train, y_test, label="KMF", 
                                              num_features=[50, 45, 40, 35, 30, 25, 20], steps=10)
eda_features.plot_line_chart(x = features["n_features"], y = features["mse"], title = "", 
                        xaxis_name = "Number of selected features [KMF]", yaxis_name = "Mean Sq Error")



# STEP 10. Load the selected features and apply one-hot encoding.
# NOTE: Perform this step on the original X_train and X_test datasets that were preprocessed 
# with Min-Max scaling and ordinal encoding.

# Load the selected features from a JSON file or directly access the features DataFrame generated by RFE.
features = pd.read_json("/kaggle/working/features.json")
chosen_cols = features[features["n_features"] == 40]["features"].explode().values.tolist()

X_train, X_test, X_val = preprocessor.one_hot_encode(columns=chosen_cols)


eda_train = EDAplot(colorscale=cfs.colorscale, color=cfs.color, df=df_train)


eda_train.plot_distribution("efs_time", title="Distribution of time-to-event-free survival", xaxis_name="Months", 
                            hue="efs")


eda_train.plot_distribution("efs_time", title="Distribution of time-to-event-free survival", xaxis_name="Months",
                            hue="race_group")


eda_train.plot_distribution("age_at_hct", title="Distribution of patient age", xaxis_name="Years", hue="graft_type")


eda_train.plot_distribution("age_at_hct", title="Distribution of patient age", xaxis_name="Years", hue="race_group")


eda_train.plot_distribution("donor_age", title="Distribution of donor age", xaxis_name="Years", hue="donor_related")


eda_train.plot_distribution("KMF", title="Distribution of Kaplan-Meier Survival", xaxis_name="Survival Probability", nbins=10)


eda_train.plot_distribution("KMF", title="Distribution of Nelson-Aalen Survival", xaxis_name="Survival Probability",
                            hue="graft_type", nbins=10)


eda_train.plot_distribution("KMF", title="Distribution of Nelson-Aalen Survival", xaxis_name="Survival Probability",
                            hue="race_group", nbins=10)


eda_train.plot_distribution("KMF", title="Distribution of Nelson-Aalen Survival", xaxis_name="Survival Probability",
                            hue="donor_related", nbins=10)


eda_train.count_chart("efs", title="Event-free Survival", xaxis_name="0: Censoring, 1: Event", yaxis_name="Count")


eda_train.count_chart("ethnicity", title="Ethnicity", xaxis_name="", yaxis_name="Count")


eda_train.count_chart("race_group", title="Race", xaxis_name="", yaxis_name="Count")


eda_train.count_chart("arrhythmia", title="Arrhythmia", xaxis_name="", yaxis_name="Count")


eda_train.count_chart("comorbidity_score", title="Comorbidity Score", xaxis_name="SORRO score", yaxis_name="Count")


eda_train.count_chart("donor_related", title="Types of Donor-Recipient Relationships", xaxis_name="", yaxis_name="Count")


eda_train.pie_chart("graft_type", title="Graft Type")


eda_train.pie_chart_with_hue("graft_type", title="Graft Type", hue="race_group")


eda_train.bubble_chart("cardiac", title="Cardiac Issues", yaxis_name="Count", xaxis_name="")


eda_train.bubble_chart("obesity", title="Obesity", yaxis_name="Count", xaxis_name="")


eda_train.bubble_chart("diabetes", title="Diabetes", yaxis_name="Count", xaxis_name="")


eda_train.heatmap(title= "Correlation", matrix=df_train[num_col].corr())


eda_train.heatmap(title= "Covariance", matrix=df_train[num_col].cov())


# STEP 12. Initialize the model processor object for optimizing and building the model
model_processor = DeepSurv(X_train, y_train, X_test, y_test)


# # STEP 13. Perform Optuna hyperparameter optimization for the model.
# # Tip: Save the optimized hyperparameters as a JSON file to avoid rerunning the optimization process 
# # in future sessions.
# parameters = model_processor.get_best_param(n_trials=50)


# Load best_params from best_params.json
best_params = model_processor.load_best_params()


# # save model
# # save model
# model.save("/kaggle/working/DeepSurv_1.h5")



# load model 
loaded_model = load_model("/kaggle/input/cibmrt_dsurv_model_1/tensorflow2/default/1/DeepSurv_2.h5", custom_objects={"c_index_loss": DeepSurv.c_index_loss})


val_pred = loaded_model.predict(X_val)


predictions = val_pred.flatten()
rounded_predictions = [round(pred, 1) for pred in predictions]


submission = pd.DataFrame(columns=["ID", "prediction"])
submission["ID"] = cfs.test["ID"]
submission["prediction"] = rounded_predictions


submission.to_csv("/kaggle/working/submission.csv", index=False)

