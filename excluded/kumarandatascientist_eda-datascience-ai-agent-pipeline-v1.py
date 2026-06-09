import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm  # Import tqdm for progress bars

# Optional: if you want an automated EDA report, uncomment the following line.
from pandas_profiling import ProfileReport
from pandas.plotting import parallel_coordinates

class EDADatascienceAIAgent:
    def __init__(self, filepath):
        """
        Initialize the agent with the path to the dataset.
        """
        self.filepath = filepath
        self.data = None

    def load_data(self):
        """
        Loads the data from a CSV file and checks if it is empty.
        """
        try:
            self.data = pd.read_csv(self.filepath)
            if self.data.empty:
                print("Warning: Loaded data is empty!")
            else:
                print(f"Data loaded successfully. Shape: {self.data.shape}")
        except Exception as e:
            print(f"Error loading data: {e}")

    def clean_data(self):
        """
        Cleans the dataset by:
          - Dropping duplicate rows
          - Filling missing values: numeric columns are filled with their mean,
            while categorical columns are filled with their mode.
        Uses tqdm to show progress for each column processed.
        """
        if self.data is None:
            print("Data not loaded yet. Call load_data() first.")
            return

        # Drop duplicate rows
        self.data.drop_duplicates(inplace=True)
        print("Duplicates removed.")

        # Fill missing values for each column with a progress bar
        for col in tqdm(self.data.columns, desc="Cleaning Columns"):
            try:
                if self.data[col].dtype in [np.float64, np.int64]:
                    if self.data[col].isnull().sum() > 0:
                        mean_val = self.data[col].mean()
                        self.data[col].fillna(mean_val, inplace=True)
                        print(f"\nFilled missing numeric values in '{col}' with mean: {mean_val:.2f}")
                else:
                    if self.data[col].isnull().sum() > 0:
                        mode_val = self.data[col].mode()[0]
                        self.data[col].fillna(mode_val, inplace=True)
                        print(f"\nFilled missing categorical values in '{col}' with mode: {mode_val}")
            except Exception as e:
                print(f"Error cleaning column {col}: {e}")
        print("Data cleaning complete.")

    def summarize_data(self):
        """
        Provides data statistics and a summary:
          - Data dimensions
          - Count of missing values per column
          - Statistical summary for all columns (including non-numeric)
        """
        if self.data is None:
            print("Data not loaded yet. Call load_data() first.")
            return

        print("\n=== Data Summary ===")
        print(f"Data Dimensions: {self.data.shape}")
        print("\n=== Missing Values Count per Column ===")
        print(self.data.isnull().sum())
        print("\n=== Statistical Summary ===")
        print(self.data.describe(include='all'))

    def perform_eda(self):
        """
        Performs exploratory data analysis by:
          - Displaying data summary and statistics
          - Visualizing missing data as a heatmap
          - Creating a correlation heatmap for numeric columns
        """
        if self.data is None:
            print("Data not loaded yet. Call load_data() first.")
            return

        self.summarize_data()
        
        try:
            # Visualize missing values
            plt.figure(figsize=(8, 4))
            sns.heatmap(self.data.isnull(), cbar=False, cmap='viridis')
            plt.title("Missing Values Heatmap")
            plt.show()
        except Exception as e:
            print(f"Error generating missing values heatmap: {e}")

        try:
            # Correlation heatmap for numeric columns
            numeric_data = self.data.select_dtypes(include=[np.number])
            if not numeric_data.empty:
                plt.figure(figsize=(10, 8))
                corr = numeric_data.corr()
                sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
                plt.title("Correlation Heatmap")
                plt.show()
            else:
                print("No numeric columns available for a correlation heatmap.")
        except Exception as e:
            print(f"Error generating correlation heatmap: {e}")

    def visualize_data(self):
        """
        Generates basic visualizations:
          - Histograms with KDE for numeric features
          - Box plots for numeric features
          - Pair plots for relationships between numeric features (if dataset is small)
        Uses tqdm to provide progress updates.
        """
        if self.data is None:
            print("Data not loaded yet. Call load_data() first.")
            return

        numeric_columns = self.data.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_columns:
            print("No numeric columns available for visualization.")
            return

        # Histograms with KDE
        for col in tqdm(numeric_columns, desc="Generating Histograms"):
            try:
                plt.figure(figsize=(6, 4))
                sns.histplot(self.data[col], kde=True, color='skyblue')
                plt.title(f"Distribution of {col}")
                plt.xlabel(col)
                plt.ylabel("Frequency")
                plt.show()
            except Exception as e:
                print(f"Error generating histogram for {col}: {e}")

        # Box plots to identify outliers
        for col in tqdm(numeric_columns, desc="Generating Box Plots"):
            try:
                plt.figure(figsize=(6, 4))
                sns.boxplot(x=self.data[col], color='lightgreen')
                plt.title(f"Box Plot of {col}")
                plt.xlabel(col)
                plt.show()
            except Exception as e:
                print(f"Error generating box plot for {col}: {e}")

        # Pair plot for relationships (if there are not too many numeric features)
        if len(numeric_columns) <= 10:
            try:
                sns.pairplot(self.data[numeric_columns])
                plt.suptitle("Pair Plot of Numeric Features", y=1.02)
                plt.show()
            except Exception as e:
                print(f"Error generating pair plot: {e}")
        else:
            print("Too many numeric features for a pair plot.")

    def additional_visualizations(self):
        """
        Generates over 15 additional EDA visualizations.
        Each visualization block is wrapped in a try/except block to skip errors while sharing details.
        """
        if self.data is None:
            print("Data not loaded yet. Call load_data() first.")
            return

        # Identify numeric and categorical columns
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = self.data.select_dtypes(exclude=[np.number]).columns.tolist()

        # 1. Violin Plots for numeric columns
        for col in tqdm(numeric_cols, desc="Generating Violin Plots"):
            try:
                plt.figure(figsize=(6, 4))
                sns.violinplot(y=self.data[col], color='lightblue')
                plt.title(f"Violin Plot of {col}")
                plt.xlabel(col)
                plt.show()
            except Exception as e:
                print(f"Error generating violin plot for {col}: {e}")

        # 2. KDE Plots for numeric columns
        for col in tqdm(numeric_cols, desc="Generating KDE Plots"):
            try:
                plt.figure(figsize=(6, 4))
                sns.kdeplot(self.data[col], shade=True, color='purple')
                plt.title(f"KDE Plot of {col}")
                plt.xlabel(col)
                plt.show()
            except Exception as e:
                print(f"Error generating KDE plot for {col}: {e}")

        # 3. Joint Plot for the first two numeric columns (if available)
        if len(numeric_cols) >= 2:
            try:
                sns.jointplot(data=self.data, x=numeric_cols[0], y=numeric_cols[1], kind='scatter')
                plt.suptitle(f"Joint Plot: {numeric_cols[0]} vs {numeric_cols[1]}", y=1.02)
                plt.show()
            except Exception as e:
                print(f"Error generating joint plot: {e}")

        # 4. Regression Plot for the first two numeric columns (if available)
        if len(numeric_cols) >= 2:
            try:
                plt.figure(figsize=(6, 4))
                sns.regplot(data=self.data, x=numeric_cols[0], y=numeric_cols[1], color='red')
                plt.title(f"Regression Plot: {numeric_cols[0]} vs {numeric_cols[1]}")
                plt.show()
            except Exception as e:
                print(f"Error generating regression plot: {e}")

        # 5. Rug Plots for numeric columns
        for col in tqdm(numeric_cols, desc="Generating Rug Plots"):
            try:
                plt.figure(figsize=(6, 2))
                sns.rugplot(self.data[col], color='black')
                plt.title(f"Rug Plot of {col}")
                plt.xlabel(col)
                plt.show()
            except Exception as e:
                print(f"Error generating rug plot for {col}: {e}")

        # 6. Count Plots for categorical columns
        for col in tqdm(cat_cols, desc="Generating Count Plots"):
            try:
                plt.figure(figsize=(8, 4))
                sns.countplot(y=self.data[col], palette='viridis')
                plt.title(f"Count Plot of {col}")
                plt.xlabel("Count")
                plt.ylabel(col)
                plt.show()
            except Exception as e:
                print(f"Error generating count plot for {col}: {e}")

        # 7. Pie Charts for categorical columns with < 10 unique values
        for col in tqdm(cat_cols, desc="Generating Pie Charts"):
            try:
                if self.data[col].nunique() < 10:
                    plt.figure(figsize=(6, 6))
                    self.data[col].value_counts().plot.pie(autopct='%1.1f%%', colors=sns.color_palette('pastel'))
                    plt.title(f"Pie Chart of {col}")
                    plt.ylabel("")
                    plt.show()
            except Exception as e:
                print(f"Error generating pie chart for {col}: {e}")

        # 8. Bar Chart for missing values per column
        try:
            missing_counts = self.data.isnull().sum()
            missing = missing_counts[missing_counts > 0]
            if not missing.empty:
                plt.figure(figsize=(10, 4))
                missing.plot.bar(color='orange')
                plt.title("Missing Values per Column")
                plt.xlabel("Column")
                plt.show()
            else:
                print("No missing values to plot in bar chart.")
        except Exception as e:
            print(f"Error generating missing values bar chart: {e}")

        # 9. Clustermap of numeric columns' correlation
        try:
            if numeric_cols:
                corr = self.data[numeric_cols].corr()
                sns.clustermap(corr, annot=True, cmap='coolwarm')
                plt.suptitle("Clustermap of Numeric Correlation", y=1.05)
                plt.show()
        except Exception as e:
            print(f"Error generating clustermap: {e}")

        # 10. Swarm Plots for categorical vs. numeric (using the first numeric column)
        if cat_cols and numeric_cols:
            for col in tqdm(cat_cols, desc="Generating Swarm Plots"):
                try:
                    plt.figure(figsize=(8, 4))
                    sns.swarmplot(x=self.data[col], y=self.data[numeric_cols[0]], palette='Set2')
                    plt.title(f"Swarm Plot: {numeric_cols[0]} vs {col}")
                    plt.xlabel(col)
                    plt.ylabel(numeric_cols[0])
                    plt.show()
                except Exception as e:
                    print(f"Error generating swarm plot for {col}: {e}")

        # 11. Scatter Plot for the first two numeric columns (if available)
        if len(numeric_cols) >= 2:
            try:
                plt.figure(figsize=(6, 4))
                plt.scatter(self.data[numeric_cols[0]], self.data[numeric_cols[1]], color='green', alpha=0.5)
                plt.title(f"Scatter Plot: {numeric_cols[0]} vs {numeric_cols[1]}")
                plt.xlabel(numeric_cols[0])
                plt.ylabel(numeric_cols[1])
                plt.show()
            except Exception as e:
                print(f"Error generating scatter plot: {e}")

        # 12. Line Plot for time series (if a datetime column exists)
        datetime_cols = [col for col in self.data.columns if 'date' in col.lower() or 'time' in col.lower()]
        if datetime_cols and numeric_cols:
            try:
                self.data[datetime_cols[0]] = pd.to_datetime(self.data[datetime_cols[0]])
                data_sorted = self.data.sort_values(datetime_cols[0])
                plt.figure(figsize=(10, 4))
                plt.plot(data_sorted[datetime_cols[0]], data_sorted[numeric_cols[0]], marker='o', linestyle='-')
                plt.title(f"Line Plot: {numeric_cols[0]} over {datetime_cols[0]}")
                plt.xlabel(datetime_cols[0])
                plt.ylabel(numeric_cols[0])
                plt.xticks(rotation=45)
                plt.show()
            except Exception as e:
                print(f"Error generating line plot: {e}")

        # 13. Area Plot for time series (if a datetime column exists)
        if datetime_cols and numeric_cols:
            try:
                self.data[datetime_cols[0]] = pd.to_datetime(self.data[datetime_cols[0]])
                data_sorted = self.data.sort_values(datetime_cols[0])
                plt.figure(figsize=(10, 4))
                plt.fill_between(data_sorted[datetime_cols[0]], data_sorted[numeric_cols[0]], color="skyblue", alpha=0.4)
                plt.plot(data_sorted[datetime_cols[0]], data_sorted[numeric_cols[0]], color="Slateblue", alpha=0.6)
                plt.title(f"Area Plot: {numeric_cols[0]} over {datetime_cols[0]}")
                plt.xlabel(datetime_cols[0])
                plt.ylabel(numeric_cols[0])
                plt.xticks(rotation=45)
                plt.show()
            except Exception as e:
                print(f"Error generating area plot: {e}")

        # 14. Parallel Coordinates Plot for numeric columns grouped by a categorical column (if available)
        if numeric_cols and cat_cols:
            try:
                plt.figure(figsize=(10, 6))
                parallel_coordinates(self.data[[cat_cols[0]] + numeric_cols], class_column=cat_cols[0], colormap='viridis')
                plt.title("Parallel Coordinates Plot")
                plt.xticks(rotation=45)
                plt.show()
            except Exception as e:
                print(f"Error generating parallel coordinates plot: {e}")

        # 15. Cumulative Sum Plots for numeric columns
        for col in tqdm(numeric_cols, desc="Generating Cumulative Sum Plots"):
            try:
                plt.figure(figsize=(6, 4))
                plt.plot(self.data[col].cumsum(), color='brown')
                plt.title(f"Cumulative Sum of {col}")
                plt.xlabel("Index")
                plt.ylabel("Cumulative Sum")
                plt.show()
            except Exception as e:
                print(f"Error generating cumulative sum plot for {col}: {e}")

        # 16. PairGrid of numeric features
        if len(numeric_cols) > 1:
            try:
                g = sns.PairGrid(self.data[numeric_cols])
                g.map_diag(sns.histplot)
                g.map_offdiag(sns.scatterplot)
                plt.suptitle("PairGrid of Numeric Features", y=1.02)
                plt.show()
            except Exception as e:
                print(f"Error generating PairGrid: {e}")

        # 17. Overlay Distribution Plots for numeric columns (Histogram with KDE overlay)
        for col in tqdm(numeric_cols, desc="Generating Overlay Distribution Plots"):
            try:
                plt.figure(figsize=(6, 4))
                sns.histplot(self.data[col], kde=True, color='teal', stat="density")
                plt.title(f"Overlay Distribution of {col}")
                plt.xlabel(col)
                plt.ylabel("Density")
                plt.show()
            except Exception as e:
                print(f"Error generating overlay distribution plot for {col}: {e}")




if __name__ == "__main__":
    # Specify the path to your dataset (CSV format)
    filepath = "/kaggle/input/home-data-for-ml-course/train.csv"  # Replace with your dataset path
    
    # Instantiate the AI agent
    agent = EDADatascienceAIAgent(filepath)
    
    # Run the agent's methods
    agent.load_data()
    agent.clean_data()
    agent.perform_eda()
    agent.visualize_data()
    #agent.additional_visualizations()



if __name__ == "__main__":
    # Specify the path to your dataset (CSV format)
    filepath = "/kaggle/input/playground-series-s5e2/train.csv"  # Replace with your dataset path
    
    # Instantiate the AI agent
    agent = EDADatascienceAIAgent(filepath)
    
    # Run the agent's methods
    agent.load_data()
    agent.clean_data()
    agent.perform_eda()
    agent.visualize_data()
    #agent.additional_visualizations()



if __name__ == "__main__":
    # Specify the path to your dataset (CSV format)
    filepath = "/kaggle/input/playground-series-s5e1/train.csv"  # Replace with your dataset path
    
    # Instantiate the AI agent
    agent = EDADatascienceAIAgent(filepath)
    
    # Run the agent's methods
    agent.load_data()
    agent.clean_data()
    agent.perform_eda()
    agent.visualize_data()
    #agent.additional_visualizations()







