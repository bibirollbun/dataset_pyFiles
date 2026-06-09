!pip install imbalanced-learn


import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, ks_2samp, mannwhitneyu
from matplotlib.gridspec import GridSpec

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class EDAConfig:
    """Configuration for EDA analysis"""

    # Paths
    TRAIN_PATH: str = '/kaggle/input/playground-series-s5e12/train.csv'
    TEST_PATH: str = '/kaggle/input/playground-series-s5e12/test.csv'
    OUTPUT_DIR: str = 'eda_outputs'

    # Target and ID
    TARGET: str = 'diagnosed_diabetes'
    ID_COL: str = 'id'

    # Visualization settings
    FIGURE_SIZE: Tuple[int, int] = (15, 10)
    DPI: int = 100
    COLOR_PALETTE: str = 'Set2'
    STYLE: str = 'whitegrid'

    # Statistical settings
    ALPHA: float = 0.05  # Significance level
    CORRELATION_THRESHOLD: float = 0.3

    # Medical thresholds (based on clinical guidelines)
    MEDICAL_THRESHOLDS: Dict = None

    def __post_init__(self):
        self.MEDICAL_THRESHOLDS = {
            'bmi': {
                'underweight': 18.5,
                'normal': 25,
                'overweight': 30,
                'obese': 35
            },
            'cholesterol_total': {
                'desirable': 200,
                'borderline': 240,
                'high': 240
            },
            'systolic_bp': {
                'normal': 120,
                'elevated': 130,
                'hypertension_stage1': 140,
                'hypertension_stage2': 180
            },
            'glucose_equivalent': {
                'normal': 100,
                'prediabetes': 126,
                'diabetes': 126
            }
        }


class StatisticalProfiler:
    """
    Advanced statistical profiling with probabilistic analysis.
    """

    def __init__(self, config: EDAConfig):
        self.config = config

    def generate_comprehensive_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate comprehensive statistical summary including:
        - Central tendency (mean, median, mode)
        - Dispersion (std, IQR, range)
        - Distribution shape (skewness, kurtosis)
        - Missing values
        - Unique values
        - Outlier detection
        """
        print("\n" + "="*80)
        print("COMPREHENSIVE STATISTICAL SUMMARY")
        print("="*80)

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if self.config.ID_COL in numeric_cols:
            numeric_cols.remove(self.config.ID_COL)
        if self.config.TARGET in numeric_cols:
            numeric_cols.remove(self.config.TARGET)

        summary_stats = []

        for col in numeric_cols:
            data = df[col].dropna()

            # Central Tendency
            mean_val = data.mean()
            median_val = data.median()
            mode_val = data.mode()[0] if len(data.mode()) > 0 else np.nan

            # Dispersion
            std_val = data.std()
            iqr_val = data.quantile(0.75) - data.quantile(0.25)
            range_val = data.max() - data.min()
            cv = (std_val / mean_val * 100) if mean_val != 0 else np.nan  # Coefficient of variation

            # Distribution Shape
            skewness = stats.skew(data)
            kurtosis_val = stats.kurtosis(data)

            # Missing & Unique
            missing_count = df[col].isnull().sum()
            missing_pct = (missing_count / len(df)) * 100
            unique_count = data.nunique()
            unique_pct = (unique_count / len(data)) * 100

            # Outlier Detection (IQR method)
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))).sum()
            outlier_pct = (outliers / len(data)) * 100

            # Normality Test (Shapiro-Wilk for sample, Anderson-Darling for larger)
            if len(data) < 5000:
                _, p_value_normality = stats.shapiro(data[:5000])
            else:
                result = stats.anderson(data)
                p_value_normality = 1.0 if result.statistic < result.critical_values[2] else 0.0

            summary_stats.append({
                'Feature': col,
                'Count': len(data),
                'Missing': missing_count,
                'Missing_%': f"{missing_pct:.2f}%",
                'Unique': unique_count,
                'Unique_%': f"{unique_pct:.2f}%",
                'Mean': mean_val,
                'Median': median_val,
                'Mode': mode_val,
                'Std': std_val,
                'CV_%': f"{cv:.2f}%" if not np.isnan(cv) else "N/A",
                'IQR': iqr_val,
                'Range': range_val,
                'Skewness': skewness,
                'Kurtosis': kurtosis_val,
                'Outliers': outliers,
                'Outliers_%': f"{outlier_pct:.2f}%",
                'Normal?': 'Yes' if p_value_normality > 0.05 else 'No',
                'p_value': p_value_normality
            })

        summary_df = pd.DataFrame(summary_stats)

        print("\nStatistical Summary Generated:")
        print(f"  → {len(numeric_cols)} numeric features analyzed")
        print(f"  → {summary_df['Missing'].sum()} total missing values")
        print(f"  → {summary_df['Outliers'].sum()} total outliers detected")

        return summary_df

    def analyze_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze categorical features with frequency and entropy"""
        print("\n" + "="*80)
        print("CATEGORICAL FEATURES ANALYSIS")
        print("="*80)

        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        if not categorical_cols:
            print("  → No categorical features found")
            return pd.DataFrame()

        cat_summary = []

        for col in categorical_cols:
            data = df[col].dropna()

            # Frequency analysis
            value_counts = data.value_counts()
            n_unique = len(value_counts)

            # Most frequent
            most_frequent = value_counts.index[0]
            most_frequent_count = value_counts.iloc[0]
            most_frequent_pct = (most_frequent_count / len(data)) * 100

            # Entropy (measure of uncertainty/information)
            probs = value_counts / len(data)
            entropy = -np.sum(probs * np.log2(probs + 1e-10))

            # Cardinality
            cardinality = 'Low' if n_unique < 10 else 'Medium' if n_unique < 50 else 'High'

            cat_summary.append({
                'Feature': col,
                'Unique_Values': n_unique,
                'Most_Frequent': most_frequent,
                'Frequency': most_frequent_count,
                'Frequency_%': f"{most_frequent_pct:.2f}%",
                'Entropy': entropy,
                'Cardinality': cardinality,
                'Missing': df[col].isnull().sum(),
                'Missing_%': f"{(df[col].isnull().sum() / len(df)) * 100:.2f}%"
            })

        cat_df = pd.DataFrame(cat_summary)

        print(f"\n  → {len(categorical_cols)} categorical features analyzed")
        return cat_df


class ProbabilisticAnalyzer:
    """
    Comprehensive Bayesian and probabilistic analysis for diabetes prediction.

    Implements:
    - Conditional probabilities P(Diabetes | Feature)
    - Bayesian risk scoring
    - Posterior probability distributions
    - Bayesian credible intervals
    - Likelihood ratios
    - Prior and posterior belief updates
    """

    def __init__(self, config: EDAConfig):
        self.config = config
        self.prior_prob = None
        self.likelihood_ratios = {}
        self.posterior_distributions = {}

    def calculate_conditional_probabilities(self, df: pd.DataFrame) -> Dict:
        """Calculate P(Diabetes | Feature) for various feature values."""
        print("\n" + "="*80)
        print("CONDITIONAL PROBABILITY ANALYSIS")
        print("="*80)

        results = {}

        # Overall diabetes rate (prior probability)
        prior_prob = df[self.config.TARGET].mean()
        print(f"\n  Prior P(Diabetes) = {prior_prob:.4f} ({prior_prob*100:.2f}%)")

        # Analyze numeric features by bins
        numeric_features = {
            'age': [(0, 30), (30, 40), (40, 50), (50, 60), (60, 100)],
            'bmi': [(0, 18.5), (18.5, 25), (25, 30), (30, 35), (35, 100)],
            'cholesterol_total': [(0, 200), (200, 240), (240, 1000)],
            'systolic_bp': [(0, 120), (120, 130), (130, 140), (140, 300)]
        }

        for feature, bins in numeric_features.items():
            if feature not in df.columns:
                continue

            print(f"\n  {feature.upper()}:")
            feature_results = []

            for low, high in bins:
                mask = (df[feature] >= low) & (df[feature] < high)
                if mask.sum() == 0:
                    continue

                diabetes_rate = df[mask][self.config.TARGET].mean()
                count = mask.sum()
                lift = diabetes_rate / prior_prob if prior_prob > 0 else 0

                feature_results.append({
                    'Range': f"{low}-{high}",
                    'Count': count,
                    'Diabetes_Rate': diabetes_rate,
                    'Diabetes_%': f"{diabetes_rate*100:.2f}%",
                    'Lift': lift
                })

                print(f"    [{low:6.1f} - {high:6.1f}]: P(D|X) = {diabetes_rate:.4f} "
                      f"(Lift: {lift:.2f}x, n={count})")

            results[feature] = feature_results

        # Analyze categorical features
        categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()

        for feature in categorical_features:
            if feature == self.config.ID_COL:
                continue

            print(f"\n  {feature.upper()}:")
            feature_results = []

            for category in df[feature].unique()[:10]:
                mask = df[feature] == category
                if mask.sum() == 0:
                    continue

                diabetes_rate = df[mask][self.config.TARGET].mean()
                count = mask.sum()
                lift = diabetes_rate / prior_prob if prior_prob > 0 else 0

                feature_results.append({
                    'Category': category,
                    'Count': count,
                    'Diabetes_Rate': diabetes_rate,
                    'Diabetes_%': f"{diabetes_rate*100:.2f}%",
                    'Lift': lift
                })

                print(f"    {category:20s}: P(D|X) = {diabetes_rate:.4f} "
                      f"(Lift: {lift:.2f}x, n={count})")

            results[feature] = feature_results

        return results

    def calculate_posterior_distributions(self, df: pd.DataFrame, features: Optional[List[str]] = None,
                                      ci_level: float = 0.95) -> Dict:

        print("\n" + "="*80)
        print("POSTERIOR PROBABILITY DISTRIBUTIONS")
        print("="*80)
    
        posteriors = {}
    
        # Default features if none provided
        if features is None:
            default_features = ['age', 'bmi', 'cholesterol_total', 'systolic_bp']
            features = [f for f in default_features if f in df.columns]
        else:
            # Validate that all features exist and are numeric
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            features = [f for f in features if f in df.columns and f in numeric_cols]
            if not features:
                print("  ⚠ Warning: No valid numeric features found!")
                return {}
    
        # Calculate quantiles for CI
        lower_quantile = (1 - ci_level) / 2
        upper_quantile = 1 - lower_quantile
    
        for feature in features:
            print(f"\n{feature.upper()}:")
    
            # Split by diabetes status
            has_diabetes = df[df[self.config.TARGET] == 1][feature].dropna()
            no_diabetes = df[df[self.config.TARGET] == 0][feature].dropna()
    
            if len(has_diabetes) == 0 or len(no_diabetes) == 0:
                print(f"  ⚠ Warning: Insufficient data for {feature}")
                continue
    
            # Calculate statistics for each group
            diabetes_mean = has_diabetes.mean()
            diabetes_std = has_diabetes.std()
            diabetes_median = has_diabetes.median()
            diabetes_ci_lower = has_diabetes.quantile(lower_quantile)
            diabetes_ci_upper = has_diabetes.quantile(upper_quantile)
    
            no_diabetes_mean = no_diabetes.mean()
            no_diabetes_std = no_diabetes.std()
            no_diabetes_median = no_diabetes.median()
            no_diabetes_ci_lower = no_diabetes.quantile(lower_quantile)
            no_diabetes_ci_upper = no_diabetes.quantile(upper_quantile)
    
            print(f"  Diabetes = 1:")
            print(f"    Mean: {diabetes_mean:.2f} ({ci_level*100:.0f}% CI: [{diabetes_ci_lower:.2f}, {diabetes_ci_upper:.2f}])")
            print(f"    Median: {diabetes_median:.2f}, Std: {diabetes_std:.2f}")
    
            print(f"  Diabetes = 0:")
            print(f"    Mean: {no_diabetes_mean:.2f} ({ci_level*100:.0f}% CI: [{no_diabetes_ci_lower:.2f}, {no_diabetes_ci_upper:.2f}])")
            print(f"    Median: {no_diabetes_median:.2f}, Std: {no_diabetes_std:.2f}")
    
            # Effect size
            pooled_std = np.sqrt((diabetes_std**2 + no_diabetes_std**2) / 2)
            cohen_d = (diabetes_mean - no_diabetes_mean) / pooled_std if pooled_std != 0 else 0
            print(f"  Cohen's d effect size: {cohen_d:.3f}")
    
            posteriors[feature] = {
                'diabetes_yes': {
                    'mean': diabetes_mean,
                    'std': diabetes_std,
                    'median': diabetes_median,
                    'ci_lower': diabetes_ci_lower,
                    'ci_upper': diabetes_ci_upper,
                    'data': has_diabetes.values
                },
                'diabetes_no': {
                    'mean': no_diabetes_mean,
                    'std': no_diabetes_std,
                    'median': no_diabetes_median,
                    'ci_lower': no_diabetes_ci_lower,
                    'ci_upper': no_diabetes_ci_upper,
                    'data': no_diabetes.values
                },
                'cohen_d': cohen_d
            }
    
        self.posterior_distributions = posteriors
        return posteriors

    def bayesian_risk_scoring(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Bayesian risk scores for each patient."""
        print("\n" + "="*80)
        print("BAYESIAN RISK SCORING")
        print("="*80)

        df = df.copy()
        df['risk_score'] = 0.0

        # Age risk
        if 'age' in df.columns:
            df['risk_score'] += np.where(df['age'] >= 60, 2.0,
                                np.where(df['age'] >= 50, 1.5,
                                np.where(df['age'] >= 40, 1.0, 0.5)))

        # BMI risk
        if 'bmi' in df.columns:
            df['risk_score'] += np.where(df['bmi'] >= 35, 3.0,
                                np.where(df['bmi'] >= 30, 2.0,
                                np.where(df['bmi'] >= 25, 1.0, 0.0)))

        # Blood pressure risk
        if 'systolic_bp' in df.columns:
            df['risk_score'] += np.where(df['systolic_bp'] >= 140, 2.0,
                                np.where(df['systolic_bp'] >= 130, 1.5,
                                np.where(df['systolic_bp'] >= 120, 1.0, 0.0)))

        # Cholesterol risk
        if 'cholesterol_total' in df.columns:
            df['risk_score'] += np.where(df['cholesterol_total'] >= 240, 2.0,
                                np.where(df['cholesterol_total'] >= 200, 1.0, 0.0))

        # Family history
        if 'family_history_diabetes' in df.columns:
            df['risk_score'] += df['family_history_diabetes'] * 3.0

        # Hypertension history
        if 'hypertension_history' in df.columns:
            df['risk_score'] += df['hypertension_history'] * 2.0

        # Cardiovascular history
        if 'cardiovascular_history' in df.columns:
            df['risk_score'] += df['cardiovascular_history'] * 2.5

        # Normalize to 0-100 scale
        max_score = df['risk_score'].max()
        if max_score > 0:
            df['risk_score_normalized'] = (df['risk_score'] / max_score) * 100
        else:
            df['risk_score_normalized'] = 0

        # Risk categories
        df['risk_category'] = pd.cut(df['risk_score_normalized'],
                                     bins=[0, 25, 50, 75, 100],
                                     labels=['Low', 'Moderate', 'High', 'Very High'])

        print(f"\n  Risk Score Statistics:")
        print(f"    Mean: {df['risk_score_normalized'].mean():.2f}")
        print(f"    Median: {df['risk_score_normalized'].median():.2f}")
        print(f"    Std: {df['risk_score_normalized'].std():.2f}")

        print(f"\n  Risk Category Distribution:")
        print(df['risk_category'].value_counts().to_string())

        if self.config.TARGET in df.columns:
            correlation = df['risk_score_normalized'].corr(df[self.config.TARGET])
            print(f"\n  Correlation with Diabetes: {correlation:.4f}")

        return df[['risk_score', 'risk_score_normalized', 'risk_category']]

    def calculate_likelihood_ratios(self, df: pd.DataFrame) -> Dict:
        """
        Calculate Likelihood Ratios for each feature.

        LR+ = P(Feature+ | Disease+) / P(Feature+ | Disease-)
        LR- = P(Feature- | Disease+) / P(Feature- | Disease-)

        LR > 1: Evidence for disease
        LR = 1: No diagnostic value
        LR < 1: Evidence against disease
        """
        print("\n" + "="*80)
        print("LIKELIHOOD RATIO ANALYSIS")
        print("="*80)

        self.prior_prob = df[self.config.TARGET].mean()
        prior_odds = self.prior_prob / (1 - self.prior_prob)

        print(f"\nPrior Probability: {self.prior_prob:.4f}")
        print(f"Prior Odds: {prior_odds:.4f}")

        likelihood_ratios = {}

        # Analyze key risk factors
        risk_factors = {
            'bmi': [(0, 25, 'Normal'), (25, 30, 'Overweight'), (30, 100, 'Obese')],
            'age': [(0, 40, 'Young'), (40, 60, 'Middle'), (60, 100, 'Senior')],
            'systolic_bp': [(0, 120, 'Normal'), (120, 140, 'Elevated'), (140, 300, 'Hypertension')],
            'family_history_diabetes': [(0, 0.5, 'No History'), (0.5, 2, 'Positive History')]
        }

        for feature, ranges in risk_factors.items():
            if feature not in df.columns:
                continue

            print(f"\n{feature.upper()}:")
            print(f"{'Category':<20} {'LR+':<10} {'LR-':<10} {'Post_Prob':<12} {'Interpretation'}")
            print("-" * 70)

            feature_lrs = []

            for low, high, label in ranges:
                # Feature present
                mask_feature_pos = (df[feature] >= low) & (df[feature] < high)

                if mask_feature_pos.sum() == 0:
                    continue

                # P(Feature+ | Disease+)
                sensitivity = df[mask_feature_pos & (df[self.config.TARGET] == 1)].shape[0] / df[df[self.config.TARGET] == 1].shape[0]

                # P(Feature+ | Disease-)
                false_pos_rate = df[mask_feature_pos & (df[self.config.TARGET] == 0)].shape[0] / df[df[self.config.TARGET] == 0].shape[0]

                # Calculate LR+
                lr_positive = sensitivity / false_pos_rate if false_pos_rate > 0 else np.inf

                # Calculate posterior probability using Bayes theorem
                posterior_odds = prior_odds * lr_positive
                posterior_prob = posterior_odds / (1 + posterior_odds)

                # Interpretation
                if lr_positive > 10:
                    interpretation = "Strong evidence FOR"
                elif lr_positive > 5:
                    interpretation = "Moderate evidence FOR"
                elif lr_positive > 2:
                    interpretation = "Weak evidence FOR"
                elif lr_positive > 0.5:
                    interpretation = "Minimal change"
                elif lr_positive > 0.2:
                    interpretation = "Weak evidence AGAINST"
                else:
                    interpretation = "Moderate evidence AGAINST"

                print(f"{label:<20} {lr_positive:<10.2f} {'N/A':<10} {posterior_prob:<12.4f} {interpretation}")

                feature_lrs.append({
                    'category': label,
                    'lr_positive': lr_positive,
                    'posterior_prob': posterior_prob,
                    'sensitivity': sensitivity,
                    'false_pos_rate': false_pos_rate
                })

            likelihood_ratios[feature] = feature_lrs

        self.likelihood_ratios = likelihood_ratios
        return likelihood_ratios

    def bayesian_belief_update(self, df: pd.DataFrame, feature_evidence: Dict) -> pd.DataFrame:
        """
        Perform Bayesian belief updating given evidence.

        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        feature_evidence : Dict
            Dictionary of feature values as evidence
            Example: {'bmi': 32, 'age': 55, 'family_history_diabetes': 1}

        Returns:
        --------
        pd.DataFrame : Updated probabilities
        """
        print("\n" + "="*80)
        print("BAYESIAN BELIEF UPDATE")
        print("="*80)

        prior_prob = df[self.config.TARGET].mean()
        print(f"\nPrior P(Diabetes) = {prior_prob:.4f}")

        # Start with prior
        current_prob = prior_prob
        current_odds = current_prob / (1 - current_prob)

        print(f"\nUpdating beliefs with evidence:")
        print("-" * 60)

        for feature, value in feature_evidence.items():
            if feature not in df.columns:
                continue

            # Calculate likelihood ratio for this evidence
            if isinstance(value, (int, float)):
                # Continuous feature - find similar values
                tolerance = df[feature].std() * 0.5
                mask = (df[feature] >= value - tolerance) & (df[feature] <= value + tolerance)
            else:
                # Categorical feature
                mask = df[feature] == value

            if mask.sum() < 10:
                print(f"  {feature} = {value}: Insufficient data")
                continue

            # Calculate likelihood
            p_evidence_given_disease = df[mask & (df[self.config.TARGET] == 1)].shape[0] / df[df[self.config.TARGET] == 1].shape[0]
            p_evidence_given_no_disease = df[mask & (df[self.config.TARGET] == 0)].shape[0] / df[df[self.config.TARGET] == 0].shape[0]

            if p_evidence_given_no_disease == 0:
                p_evidence_given_no_disease = 1e-10

            lr = p_evidence_given_disease / p_evidence_given_no_disease

            # Update odds
            current_odds = current_odds * lr
            current_prob = current_odds / (1 + current_odds)

            print(f"  {feature} = {value}:")
            print(f"    Likelihood Ratio: {lr:.4f}")
            print(f"    Updated P(Diabetes): {current_prob:.4f}")

        print(f"\n{'='*60}")
        print(f"FINAL POSTERIOR P(Diabetes) = {current_prob:.4f}")
        print(f"Change from prior: {(current_prob - prior_prob):.4f} ({((current_prob - prior_prob)/prior_prob)*100:+.1f}%)")
        print(f"{'='*60}")

        return pd.DataFrame({
            'prior_probability': [prior_prob],
            'posterior_probability': [current_prob],
            'probability_change': [current_prob - prior_prob],
            'evidence': [str(feature_evidence)]
        })


class AdvancedVisualizer:
    """
    Create interactive visualizations optimized for Jupyter notebooks.
    """

    def __init__(self, config: EDAConfig):
        self.config = config
        sns.set_style(config.STYLE)
        sns.set_palette(config.COLOR_PALETTE)

        # Enable inline plotting for matplotlib
        try:
            from IPython import get_ipython
            get_ipython().run_line_magic('matplotlib', 'inline')
        except:
            pass

    def plot_target_distribution(self, df: pd.DataFrame):
        """Visualize target variable distribution"""
        print("\n  → Creating target distribution plot...")

        value_counts = df[self.config.TARGET].value_counts()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Bar plot
        colors = ['#2ecc71', '#e74c3c']
        bars = ax1.bar(['No Diabetes', 'Diabetes'], value_counts.values, color=colors)
        ax1.set_ylabel('Count')
        ax1.set_title('Target Distribution', fontsize=12, fontweight='bold')

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom')

        # Pie chart
        wedges, texts, autotexts = ax2.pie(value_counts.values,
                                             labels=['No Diabetes', 'Diabetes'],
                                             colors=colors,
                                             autopct='%1.1f%%',
                                             startangle=90,
                                             wedgeprops=dict(width=0.3))
        ax2.set_title('Target Proportion', fontsize=12, fontweight='bold')

        # Make percentage text bold
        for autotext in autotexts:
            autotext.set_color('black')
            autotext.set_fontweight('bold')

        plt.suptitle('Diabetes Diagnosis Distribution', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.show()

    def plot_probabilistic_distributions(self, df: pd.DataFrame):
        """Create violin plots showing probability density functions."""
        print("\n  → Creating probabilistic distribution plots...")

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c not in [self.config.ID_COL, self.config.TARGET]]

        n_features = min(12, len(numeric_cols))
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows*4))
        axes = axes.flatten() if n_features > 1 else [axes]

        for idx, col in enumerate(numeric_cols[:n_features]):
            ax = axes[idx]

            parts = ax.violinplot(
                [df[df[self.config.TARGET] == 0][col].dropna(),
                 df[df[self.config.TARGET] == 1][col].dropna()],
                positions=[0, 1],
                showmeans=True,
                showmedians=True
            )

            colors = ['#2ecc71', '#e74c3c']
            for pc, color in zip(parts['bodies'], colors):
                pc.set_facecolor(color)
                pc.set_alpha(0.7)

            ax.set_xticks([0, 1])
            ax.set_xticklabels(['No Diabetes', 'Diabetes'])
            ax.set_ylabel('Value')
            ax.set_title(f'{col}', fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.3)

        for idx in range(n_features, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        plt.show()

    def plot_correlation_heatmap(self, df: pd.DataFrame):
        """Create correlation heatmap"""
        print("\n  → Creating correlation heatmap...")

        numeric_df = df.select_dtypes(include=[np.number])
        if self.config.ID_COL in numeric_df.columns:
            numeric_df = numeric_df.drop(columns=[self.config.ID_COL])

        corr_matrix = numeric_df.corr()

        fig, ax = plt.subplots(figsize=(16, 14))

        # Create heatmap
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r',
                   center=0, square=True, linewidths=0.5,
                   cbar_kws={"shrink": 0.8, "label": "Correlation"},
                   ax=ax, annot_kws={'size': 8})

        ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold', pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()

    def plot_feature_importance_analysis(self, df: pd.DataFrame):
        """Analyze and visualize feature importance using multiple methods."""
        print("\n  → Calculating feature importance...")

        from sklearn.feature_selection import mutual_info_classif

        numeric_df = df.select_dtypes(include=[np.number])
        if self.config.ID_COL in numeric_df.columns:
            numeric_df = numeric_df.drop(columns=[self.config.ID_COL])

        X = numeric_df.drop(columns=[self.config.TARGET])
        y = numeric_df[self.config.TARGET]

        # Mutual Information
        mi_scores = mutual_info_classif(X, y, random_state=2025)
        mi_scores = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)

        # Statistical tests
        p_values = []
        effect_sizes = []

        for col in X.columns:
            group0 = df[df[self.config.TARGET] == 0][col].dropna()
            group1 = df[df[self.config.TARGET] == 1][col].dropna()

            _, p_val = mannwhitneyu(group0, group1, alternative='two-sided')
            p_values.append(p_val)

            # Cohen's d (effect size)
            mean_diff = group1.mean() - group0.mean()
            pooled_std = np.sqrt((group0.var() + group1.var()) / 2)
            cohens_d = mean_diff / pooled_std if pooled_std != 0 else 0
            effect_sizes.append(abs(cohens_d))

        importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Mutual_Information': mi_scores.values,
            'P_Value': p_values,
            'Effect_Size': effect_sizes,
            'Significant': ['Yes' if p < 0.05 else 'No' for p in p_values]
        }).sort_values('Mutual_Information', ascending=False)

        # Plot
        top_n = 15
        fig, axes = plt.subplots(1, 3, figsize=(20, 8))

        # Mutual Information
        top_mi = importance_df.head(top_n)
        axes[0].barh(range(len(top_mi)), top_mi['Mutual_Information'], color='#3498db')
        axes[0].set_yticks(range(len(top_mi)))
        axes[0].set_yticklabels(top_mi['Feature'])
        axes[0].set_xlabel('Mutual Information')
        axes[0].set_title('Mutual Information', fontsize=12, fontweight='bold')
        axes[0].invert_yaxis()
        axes[0].grid(axis='x', alpha=0.3)

        # Statistical Significance
        top_pval = importance_df.nsmallest(top_n, 'P_Value')
        neg_log_pval = -np.log10(top_pval['P_Value'] + 1e-10)
        axes[1].barh(range(len(top_pval)), neg_log_pval, color='#e74c3c')
        axes[1].set_yticks(range(len(top_pval)))
        axes[1].set_yticklabels(top_pval['Feature'])
        axes[1].set_xlabel('-log10(p-value)')
        axes[1].set_title('Statistical Significance (-log10 p-value)', fontsize=12, fontweight='bold')
        axes[1].invert_yaxis()
        axes[1].grid(axis='x', alpha=0.3)

        # Effect Size
        top_effect = importance_df.nlargest(top_n, 'Effect_Size')
        axes[2].barh(range(len(top_effect)), top_effect['Effect_Size'], color='#2ecc71')
        axes[2].set_yticks(range(len(top_effect)))
        axes[2].set_yticklabels(top_effect['Feature'])
        axes[2].set_xlabel("Effect Size (Cohen's d)")
        axes[2].set_title("Effect Size (Cohen's d)", fontsize=12, fontweight='bold')
        axes[2].invert_yaxis()
        axes[2].grid(axis='x', alpha=0.3)

        plt.suptitle('Feature Importance Analysis (Top 15 Features)', fontsize=14, fontweight='bold', y=1.0)
        plt.tight_layout()
        plt.show()

        return importance_df

    def plot_conditional_probability_curves(self, df: pd.DataFrame, features: Optional[List[str]] = None,
                                        n_cols: int = 2, window_size: Optional[float] = None):
        """
        Plot P(Diabetes | Feature Value) curves for any features.
    
        """
        print("\n  → Creating conditional probability curves...")
    
        # Default features if none provided
        if features is None:
            default_features = ['age', 'bmi', 'cholesterol_total', 'systolic_bp']
            features = [f for f in default_features if f in df.columns]
        else:
            # Validate that all features exist and are numeric
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            features = [f for f in features if f in df.columns and f in numeric_cols]
            if not features:
                print("  ⚠ Warning: No valid numeric features found!")
                return
    
        n_features = len(features)
        n_rows = (n_features + n_cols - 1) // n_cols
    
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(8*n_cols, 6*n_rows))
        if n_features == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if n_features > 1 else [axes]
    
        prior_prob = df[self.config.TARGET].mean()
    
        for idx, feature in enumerate(features):
            ax = axes[idx]
    
            # Determine window size if not provided
            if window_size is None:
                window = df[feature].std() * 0.1  # 10% of std deviation
            else:
                window = window_size
    
            percentiles = np.linspace(0, 100, 50)
            feature_values = [df[feature].quantile(p/100) for p in percentiles]
            probabilities = []
    
            for val in feature_values:
                mask = (df[feature] >= val - window) & (df[feature] <= val + window)
                if mask.sum() > 10:
                    prob = df[mask][self.config.TARGET].mean()
                    probabilities.append(prob)
                else:
                    probabilities.append(np.nan)
    
            # Plot line
            ax.plot(feature_values, probabilities, 'o-', linewidth=3, markersize=6, color='#3498db')
    
            # Add baseline
            ax.axhline(y=prior_prob, color='red', linestyle='--', linewidth=2,
                      label=f'Baseline: {prior_prob:.2%}')
    
            ax.set_xlabel(feature.replace('_', ' ').title(), fontsize=11)
            ax.set_ylabel('P(Diabetes)', fontsize=11)
            ax.set_title(feature.replace('_', ' ').title(), fontsize=12, fontweight='bold')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
    
        # Hide unused subplots
        for idx in range(n_features, len(axes)):
            axes[idx].axis('off')
    
        plt.suptitle('Conditional Probability: P(Diabetes | Feature Value)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()

    def plot_risk_factor_combinations(self, df: pd.DataFrame):
        """Analyze and visualize combinations of risk factors."""
        print("\n  → Creating risk factor combination analysis...")

        df_risk = df.copy()

        if 'bmi' in df.columns:
            df_risk['high_bmi'] = (df_risk['bmi'] >= 30).astype(int)
        if 'systolic_bp' in df.columns:
            df_risk['high_bp'] = (df_risk['systolic_bp'] >= 140).astype(int)
        if 'cholesterol_total' in df.columns:
            df_risk['high_chol'] = (df_risk['cholesterol_total'] >= 240).astype(int)
        if 'age' in df.columns:
            df_risk['senior'] = (df_risk['age'] >= 60).astype(int)

        risk_cols = ['high_bmi', 'high_bp', 'high_chol', 'senior']
        risk_cols = [c for c in risk_cols if c in df_risk.columns]

        df_risk['num_risk_factors'] = df_risk[risk_cols].sum(axis=1)

        risk_analysis = df_risk.groupby('num_risk_factors').agg({
            self.config.TARGET: ['mean', 'count']
        }).reset_index()

        risk_analysis.columns = ['Num_Risk_Factors', 'Diabetes_Rate', 'Count']

        fig, ax1 = plt.subplots(figsize=(12, 6))

        # Bar plot for diabetes rate
        bars = ax1.bar(risk_analysis['Num_Risk_Factors'], risk_analysis['Diabetes_Rate'],
                      color='#e74c3c', alpha=0.7, label='Diabetes Rate')
        ax1.set_xlabel('Number of Risk Factors Present', fontsize=12)
        ax1.set_ylabel('Diabetes Rate', fontsize=12, color='#e74c3c')
        ax1.tick_params(axis='y', labelcolor='#e74c3c')

        # Add percentage labels on bars
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1%}',
                    ha='center', va='bottom', fontweight='bold')

        # Line plot for sample size
        ax2 = ax1.twinx()
        normalized_count = risk_analysis['Count'] / risk_analysis['Count'].max()
        ax2.plot(risk_analysis['Num_Risk_Factors'], normalized_count,
                'o-', color='#3498db', linewidth=3, markersize=8,
                label='Sample Size (normalized)')
        ax2.set_ylabel('Relative Sample Size', fontsize=12, color='#3498db')
        ax2.tick_params(axis='y', labelcolor='#3498db')

        plt.title('Diabetes Risk vs Number of Risk Factors', fontsize=14, fontweight='bold')

        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

        ax1.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_medical_threshold_analysis(self, df: pd.DataFrame):
        """Analyze how medical thresholds align with diabetes risk."""
        print("\n  → Creating medical threshold analysis...")

        features_to_analyze = ['bmi', 'cholesterol_total', 'systolic_bp']
        features_to_analyze = [f for f in features_to_analyze if f in df.columns]

        fig, axes = plt.subplots(1, len(features_to_analyze), figsize=(20, 6))
        if len(features_to_analyze) == 1:
            axes = [axes]

        for idx, feature in enumerate(features_to_analyze):
            ax = axes[idx]
            no_diabetes = df[df[self.config.TARGET] == 0][feature].dropna()
            has_diabetes = df[df[self.config.TARGET] == 1][feature].dropna()

            # Histogram
            ax.hist(no_diabetes, bins=30, alpha=0.6, color='#2ecc71', label='No Diabetes')
            ax.hist(has_diabetes, bins=30, alpha=0.6, color='#e74c3c', label='Diabetes')

            # Add medical thresholds
            if feature in self.config.MEDICAL_THRESHOLDS:
                thresholds = self.config.MEDICAL_THRESHOLDS[feature]
                colors = ['yellow', 'orange', 'red']

                for (name, value), color in zip(list(thresholds.items())[1:], colors):
                    ax.axvline(x=value, color=color, linestyle='--', linewidth=2)
                    ax.text(value, ax.get_ylim()[1]*0.95, name.replace('_', ' ').title(),
                           rotation=90, verticalalignment='top', fontsize=9)

            ax.set_xlabel(feature.replace('_', ' ').title(), fontsize=11)
            ax.set_ylabel('Frequency', fontsize=11)
            ax.set_title(feature.replace('_', ' ').title(), fontsize=12, fontweight='bold')
            if idx == 0:
                ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)

        plt.suptitle('Distribution Analysis with Medical Thresholds', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()

    def plot_bayesian_posterior_distributions(self, posteriors: Dict, features: Optional[List[str]] = None,
                                          n_cols: int = 2, bins: int = 30):
        """
        Visualize posterior probability distributions with credible intervals for any features.
    
        """
        print("\n  → Creating Bayesian posterior distributions...")
    
        # Select features to plot
        if features is None:
            features = list(posteriors.keys())
        else:
            # Only use features that exist in posteriors
            features = [f for f in features if f in posteriors]
            if not features:
                print("  ⚠ Warning: No valid features found in posteriors!")
                return
    
        n_features = len(features)
        n_rows = (n_features + n_cols - 1) // n_cols
    
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(9*n_cols, 6*n_rows))
        if n_features == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if n_features > 1 else [axes]
    
        for idx, feature in enumerate(features):
            ax = axes[idx]
            post_data = posteriors[feature]
    
            diabetes_data = post_data['diabetes_yes']['data']
            no_diabetes_data = post_data['diabetes_no']['data']
    
            # Histogram for both groups
            ax.hist(diabetes_data, bins=bins, alpha=0.5, color='#e74c3c',
                   label='Diabetes', density=True)
            ax.hist(no_diabetes_data, bins=bins, alpha=0.5, color='#2ecc71',
                   label='No Diabetes', density=True)
    
            # Add mean lines with credible intervals
            # Diabetes = 1
            mean_diab = post_data['diabetes_yes']['mean']
            ci_lower_diab = post_data['diabetes_yes']['ci_lower']
            ci_upper_diab = post_data['diabetes_yes']['ci_upper']
    
            ax.axvline(x=mean_diab, color='#c0392b', linestyle='-', linewidth=2,
                      label=f'Mean (Diabetes): {mean_diab:.1f}')
            ax.axvspan(ci_lower_diab, ci_upper_diab, alpha=0.2, color='#e74c3c')
    
            # Diabetes = 0
            mean_no_diab = post_data['diabetes_no']['mean']
            ci_lower_no = post_data['diabetes_no']['ci_lower']
            ci_upper_no = post_data['diabetes_no']['ci_upper']
    
            ax.axvline(x=mean_no_diab, color='#27ae60', linestyle='-', linewidth=2,
                      label=f'Mean (No Diabetes): {mean_no_diab:.1f}')
            ax.axvspan(ci_lower_no, ci_upper_no, alpha=0.2, color='#2ecc71')
    
            ax.set_xlabel(feature.replace('_', ' ').title(), fontsize=11)
            ax.set_ylabel('Probability Density', fontsize=11)
            ax.set_title(feature.replace('_', ' ').title(), fontsize=12, fontweight='bold')
            ax.legend(loc='best', fontsize=8)
            ax.grid(True, alpha=0.3)
    
        # Hide unused subplots
        for idx in range(n_features, len(axes)):
            axes[idx].axis('off')
    
        plt.suptitle('Bayesian Posterior Distributions with Credible Intervals',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def plot_likelihood_ratio_nomogram(self, likelihood_ratios: Dict):
        """
        Create a nomogram visualization for likelihood ratios.
        """
        print("\n  → Creating likelihood ratio nomogram...")

        # Collect all likelihood ratios
        all_lrs = []
        for feature, lrs in likelihood_ratios.items():
            for lr_data in lrs:
                all_lrs.append({
                    'feature': feature,
                    'category': lr_data['category'],
                    'lr': lr_data['lr_positive'],
                    'post_prob': lr_data['posterior_prob']
                })

        lr_df = pd.DataFrame(all_lrs)

        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

        # LR plot (bar chart with log scale)
        features = lr_df['feature'].unique()
        width = 0.8 / len(features)
        x = np.arange(len(lr_df['category'].unique()))

        colors = plt.cm.Set3(np.linspace(0, 1, len(features)))

        for idx, feature in enumerate(features):
            feat_data = lr_df[lr_df['feature'] == feature]
            offset = width * (idx - len(features) / 2)
            bars = ax1.bar([i + offset for i in range(len(feat_data))],
                          feat_data['lr'].values,
                          width, label=feature.replace('_', ' ').title(),
                          color=colors[idx])

            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}',
                        ha='center', va='bottom', fontsize=8)

        # Add reference lines
        ax1.axhline(y=1, color='gray', linestyle='--', linewidth=2, label='No diagnostic value')
        ax1.axhline(y=2, color='orange', linestyle=':', linewidth=2, label='Weak evidence')
        ax1.axhline(y=5, color='red', linestyle=':', linewidth=2, label='Moderate evidence')

        ax1.set_xlabel('Category', fontsize=12)
        ax1.set_ylabel('Likelihood Ratio (LR+)', fontsize=12)
        ax1.set_title('Likelihood Ratios by Feature', fontsize=12, fontweight='bold')
        ax1.set_yscale('log')
        ax1.legend(loc='upper left', fontsize=9)
        ax1.grid(True, alpha=0.3)

        # Posterior probability plot
        for feature in features:
            feat_data = lr_df[lr_df['feature'] == feature]
            ax2.plot(range(len(feat_data)), feat_data['post_prob'].values,
                    'o-', linewidth=2, markersize=8,
                    label=feature.replace('_', ' ').title())

        ax2.set_xlabel('Category', fontsize=12)
        ax2.set_ylabel('Posterior Probability', fontsize=12)
        ax2.set_title('Posterior Probability', fontsize=12, fontweight='bold')
        ax2.set_xticks(range(len(lr_df['category'].unique())))
        ax2.set_xticklabels(lr_df['category'].unique(), rotation=45, ha='right')
        ax2.legend(loc='best', fontsize=9)
        ax2.grid(True, alpha=0.3)

        plt.suptitle('Likelihood Ratio Analysis & Posterior Probabilities',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()

    def plot_bayesian_belief_update_waterfall(self, df: pd.DataFrame, example_patient: Dict):
        """
        Create a waterfall chart showing how evidence updates belief.

        Parameters:
        -----------
        df : pd.DataFrame
            Training data
        example_patient : Dict
            Example patient features, e.g., {'bmi': 32, 'age': 55, 'family_history_diabetes': 1}
        """
        print("\n  → Creating Bayesian belief update waterfall...")

        prior_prob = df[self.config.TARGET].mean()
        current_prob = prior_prob

        probabilities = [prior_prob]
        labels = ['Prior']
        changes = [0]

        for feature, value in example_patient.items():
            if feature not in df.columns:
                continue

            # Calculate likelihood ratio
            if isinstance(value, (int, float)):
                tolerance = df[feature].std() * 0.5
                mask = (df[feature] >= value - tolerance) & (df[feature] <= value + tolerance)
            else:
                mask = df[feature] == value

            if mask.sum() < 10:
                continue

            p_evidence_given_disease = df[mask & (df[self.config.TARGET] == 1)].shape[0] / df[df[self.config.TARGET] == 1].shape[0]
            p_evidence_given_no_disease = df[mask & (df[self.config.TARGET] == 0)].shape[0] / df[df[self.config.TARGET] == 0].shape[0]

            if p_evidence_given_no_disease == 0:
                p_evidence_given_no_disease = 1e-10

            lr = p_evidence_given_disease / p_evidence_given_no_disease

            # Update probability
            current_odds = (current_prob / (1 - current_prob)) * lr
            new_prob = current_odds / (1 + current_odds)

            change = new_prob - current_prob

            probabilities.append(new_prob)
            labels.append(f"{feature}={value}")
            changes.append(change)

            current_prob = new_prob

        # Create waterfall chart using matplotlib
        fig, ax = plt.subplots(figsize=(14, 7))

        # Calculate cumulative values for waterfall
        cumulative = [0]
        for change in changes[1:]:
            cumulative.append(cumulative[-1] + change)

        # Plot bars
        for i in range(len(labels)):
            if i == 0:
                # Prior - starts at 0
                ax.bar(i, probabilities[i], color='#3498db', alpha=0.7)
            elif i == len(labels) - 1:
                # Final posterior
                ax.bar(i, probabilities[i], color='#3498db', alpha=0.7)
            else:
                # Changes
                color = '#e74c3c' if changes[i] > 0 else '#2ecc71'
                ax.bar(i, changes[i], bottom=cumulative[i-1] + probabilities[0],
                      color=color, alpha=0.7)

                # Draw connector line
                if i > 0:
                    ax.plot([i-1, i], [cumulative[i-1] + probabilities[0],
                           cumulative[i-1] + probabilities[0]],
                           'k--', linewidth=1, alpha=0.5)

        # Add probability labels on top of bars
        for i, prob in enumerate(probabilities):
            ax.text(i, prob + 0.01, f'{prob:.2%}',
                   ha='center', va='bottom', fontweight='bold', fontsize=10)

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel('Probability', fontsize=12)
        ax.set_xlabel('Evidence', fontsize=12)
        ax.set_title('Bayesian Belief Update: Patient Example', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.show()

        return current_prob

    def plot_prior_posterior_comparison(self, posteriors: Dict, features: Optional[List[str]] = None,
                                    kde_bw_method: str = 'scott'):
        """
        Create ridgeline plot comparing prior (overall) vs posterior distributions for any features.
    
        """
        print("\n  → Creating prior vs posterior comparison...")
    
        # Select features to plot
        if features is None:
            features = list(posteriors.keys())
        else:
            # Only use features that exist in posteriors
            features = [f for f in features if f in posteriors]
            if not features:
                print("  ⚠ Warning: No valid features found in posteriors!")
                return
    
        n_features = len(features)
        fig, axes = plt.subplots(n_features, 1, figsize=(14, 3*n_features))
    
        if n_features == 1:
            axes = [axes]
    
        for idx, feature in enumerate(features):
            ax = axes[idx]
    
            post_data = posteriors[feature]
    
            # Diabetes = 1 distribution
            diabetes_data = post_data['diabetes_yes']['data']
            no_diabetes_data = post_data['diabetes_no']['data']
    
            # Plot KDE
            from scipy.stats import gaussian_kde
    
            # Diabetes
            if len(diabetes_data) > 1:
                try:
                    kde_diabetes = gaussian_kde(diabetes_data, bw_method=kde_bw_method)
                    x_range = np.linspace(diabetes_data.min(), diabetes_data.max(), 200)
                    ax.fill_between(x_range, kde_diabetes(x_range), alpha=0.5, color='#e74c3c', label='P(X | Diabetes=1)')
                except Exception as e:
                    print(f"  ⚠ Warning: Could not create KDE for {feature} (Diabetes=1): {e}")
    
            # No Diabetes
            if len(no_diabetes_data) > 1:
                try:
                    kde_no_diabetes = gaussian_kde(no_diabetes_data, bw_method=kde_bw_method)
                    x_range_no = np.linspace(no_diabetes_data.min(), no_diabetes_data.max(), 200)
                    ax.fill_between(x_range_no, kde_no_diabetes(x_range_no), alpha=0.5, color='#2ecc71', label='P(X | Diabetes=0)')
                except Exception as e:
                    print(f"  ⚠ Warning: Could not create KDE for {feature} (Diabetes=0): {e}")
    
            # Add means
            ax.axvline(post_data['diabetes_yes']['mean'], color='#c0392b', linestyle='--', linewidth=2, label='Mean (Diabetes=1)')
            ax.axvline(post_data['diabetes_no']['mean'], color='#27ae60', linestyle='--', linewidth=2, label='Mean (Diabetes=0)')
    
            ax.set_ylabel('Density', fontsize=10)
            ax.set_xlabel(feature.replace('_', ' ').title(), fontsize=10)
            ax.set_title(f'{feature.replace("_", " ").title()}', fontsize=12, fontweight='bold')
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)
    
        plt.suptitle('Prior vs Posterior Distribution Comparison', fontsize=14, fontweight='bold', y=1.0)
        plt.tight_layout()
        plt.show()


# Initialize configuration
config = EDAConfig()

train_df = pd.read_csv(config.TRAIN_PATH)
test_df = pd.read_csv(config.TEST_PATH)

print(f"\nTrain Dataset:")
print(f"  → Shape: {train_df.shape}")
print(f"  → Features: {train_df.shape[1]}")
print(f"  → Samples: {train_df.shape[0]:,}")

print(f"\nTest Dataset:")
print(f"  → Shape: {test_df.shape}")
print(f"  → Samples: {test_df.shape[0]:,}")

train_df.columns


profiler = StatisticalProfiler(config)

# Numeric features summary
numeric_summary = profiler.generate_comprehensive_summary(train_df)

# Categorical features summary
categorical_summary = profiler.analyze_categorical_features(train_df)


# Display top insights
# Most skewed features
print("\nMost Skewed Features (Top 5):")
top_skewed = numeric_summary.nlargest(5, 'Skewness')[['Feature', 'Skewness']]
for _, row in top_skewed.iterrows():
    print(f"  {row['Feature']:30s}: {row['Skewness']:8.3f}")

# Features with most outliers
print("\nFeatures with Most Outliers (Top 5):")
numeric_summary['Outliers_num'] = numeric_summary['Outliers_%'].str.replace('%', '').astype(float)
top_outliers = numeric_summary.nlargest(5, 'Outliers_num')[['Feature', 'Outliers_%']]
for _, row in top_outliers.iterrows():
    print(f"  {row['Feature']:30s}: {row['Outliers_%']:>8s}")


prob_analyzer = ProbabilisticAnalyzer(config)

# Conditional probabilities
conditional_probs = prob_analyzer.calculate_conditional_probabilities(train_df)

# Bayesian risk scoring
risk_scores = prob_analyzer.bayesian_risk_scoring(train_df)

# Likelihood ratio analysis
likelihood_ratios = prob_analyzer.calculate_likelihood_ratios(train_df)


visualizer = AdvancedVisualizer(config)

# Target distribution
visualizer.plot_target_distribution(train_df)


# Probabilistic distributions
visualizer.plot_probabilistic_distributions(train_df)


# Correlation heatmap
visualizer.plot_correlation_heatmap(train_df)


# Feature importance
importance_df = visualizer.plot_feature_importance_analysis(train_df)


# Analyze ALL numeric features at once
numeric_features = train_df.select_dtypes(include=[np.number]).columns.tolist()
numeric_features.remove('id')
numeric_features.remove('diagnosed_diabetes')
visualizer.plot_conditional_probability_curves(
    train_df,
    features=numeric_features,  # First 8 features
    n_cols=3
)


# Bayesian posterior distribution plots
features = [ # lipid
            'cholesterol_total', 'hdl_cholesterol', 'ldl_cholesterol','triglycerides']
posteriors = prob_analyzer.calculate_posterior_distributions(
    train_df,
    features=features
)
visualizer.plot_bayesian_posterior_distributions(
    posteriors,
    n_cols=2,
    bins=40
)


# Bayesian belief update example (waterfall chart)
example_patient = {
    'bmi': 32,
    'age': 55,
    'family_history_diabetes': 1,
    'systolic_bp': 145
}
print(f"\n  Demonstrating Bayesian belief update with example patient:")
print(f"  Features: {example_patient}")
final_prob = visualizer.plot_bayesian_belief_update_waterfall(train_df, example_patient)


# Interactive belief update
belief_update_result = prob_analyzer.bayesian_belief_update(train_df, example_patient)


# Medical threshold analysis
visualizer.plot_medical_threshold_analysis(train_df)


import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

import os
import pickle
from typing import Dict, List, Tuple
from dataclasses import dataclass
from pathlib import Path

# Data Processing
# Data Processing
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (roc_auc_score, log_loss, classification_report,
                             confusion_matrix, accuracy_score, f1_score,
                             precision_score, recall_score, roc_curve,
                             precision_recall_curve)
from sklearn.calibration import CalibratedClassifierCV

# Machine Learning Models
import xgboost as xgb
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

# Hyperparameter Optimization
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner

# Feature Engineering & Selection
from sklearn.feature_selection import mutual_info_classif
from itertools import combinations
import shap

from imblearn.over_sampling import (SMOTE, ADASYN, BorderlineSMOTE,
                                        SVMSMOTE, KMeansSMOTE)
from imblearn.under_sampling import (RandomUnderSampler, TomekLinks,
                                     EditedNearestNeighbours, NeighbourhoodCleaningRule)
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.ensemble import (BalancedBaggingClassifier, BalancedRandomForestClassifier,
                                RUSBoostClassifier, EasyEnsembleClassifier)
from imblearn.pipeline import Pipeline as ImbPipeline
IMBLEARN_AVAILABLE = True


# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ImprovedConfig:
    """Enhanced configuration based on 2024-2025 research"""

    # File Paths
    TRAIN_PATH: str = '/kaggle/input/playground-series-s5e12/train.csv'
    TEST_PATH: str = '/kaggle/input/playground-series-s5e12/test.csv'
    SAMPLE_SUBMISSION_PATH: str = '/kaggle/input/playground-series-s5e12/sample_submission.csv'
    SUBMISSION_PATH: str = 'submission.csv'
    MODELS_DIR: str = 'models_advanced'

    # Column Names
    TARGET: str = 'diagnosed_diabetes'
    ID_COL: str = 'id'

    # Categorical Features (will be target-encoded)
    CATEGORICAL_FEATURES: List[str] = None

    def __post_init__(self):
        self.CATEGORICAL_FEATURES = [
            'gender', 'ethnicity', 'education_level', 'income_level',
            'smoking_status', 'employment_status'
        ]

    # Random Seed
    SEED: int = 42

    # Cross-Validation (FIXED: Proper stratified K-fold)
    N_FOLDS: int = 5
    N_REPEATS: int = 1  # Can increase for more robust validation

    # Feature Engineering
    USE_DOMAIN_FEATURES: bool = True
    USE_POLYNOMIAL_FEATURES: bool = True
    USE_STATISTICAL_FEATURES: bool = True
    USE_INTERACTION_FEATURES: bool = True

    # Feature Selection (BorutaShap-inspired)
    USE_FEATURE_SELECTION: bool = True
    FEATURE_SELECTION_THRESHOLD: float = 0.001
    MAX_FEATURES: int = 50  # Will test multiple sizes

    # Hyperparameter Optimization
    USE_OPTUNA: bool = True
    N_TRIALS: int = 5  # Increased for better optimization
    OPTUNA_TIMEOUT: int = 3600  # 1 hour per model

    # Ensemble Configuration
    USE_STACKING: bool = True
    USE_CALIBRATION: bool = True
    CALIBRATION_METHOD: str = 'isotonic'  # Better with enough data

    # Class Imbalance Handling - ADVANCED MULTI-STRATEGY APPROACH (2024-2025 SOTA)
    USE_CLASS_WEIGHTS: bool = True  # Better than SMOTE per 2024 research
    CLASS_WEIGHT_METHOD: str = 'dynamic'  # 'effective_num', 'sqrt', 'balanced', 'dynamic'
    CLASS_WEIGHT_BETA: float = 0.9999  # For 'effective_num' method (0.99-0.9999)

    # CRITICAL: Bayesian Optimization for Class Weights (PROVEN - Drug Discovery Study)
    # Based on CILBO pipeline (Nature Scientific Reports 2022): ROC-AUC 0.896 → 0.917 (+2.3%)
    USE_CLASS_WEIGHT_OPTIMIZATION: str = 'bayesian'  # 'bayesian', 'grid', 'none'
    # Bayesian: Proven best (Nature 2022) - optimizes class_weight WITH other hyperparameters
    # Grid: Fast 3-fold search around optimal ratio
    # None: Use calculated weights only

    # Sampling Strategies (Best practice: combine multiple approaches)
    USE_SAMPLING: bool = True  # Enable resampling techniques
    SAMPLING_STRATEGY: str = 'hybrid'  # 'none', 'oversample', 'undersample', 'hybrid', 'ensemble'
    SAMPLING_METHOD: str = 'borderline_smote'  # For oversample: 'smote', 'borderline_smote', 'adasyn', 'svmsmote'
    HYBRID_METHOD: str = 'smote_tomek'  # For hybrid: 'smote_tomek', 'smote_enn'

    # Threshold Optimization (CRITICAL - Most effective per 2024 research)
    USE_THRESHOLD_TUNING: bool = True  # Find optimal decision threshold
    THRESHOLD_METRIC: str = 'f1'  # 'f1', 'gmean', 'youden', 'pr_optimal'

    # Ensemble Imbalanced Learning
    USE_IMBALANCED_ENSEMBLE: bool = False  # Use specialized ensemble methods
    IMBALANCED_ENSEMBLE_METHOD: str = 'rusboost'  # 'rusboost', 'easy_ensemble', 'balanced_bagging'

    # Focal Loss (for compatible models)
    USE_FOCAL_LOSS: bool = False  # Experimental: focal loss for XGBoost/LightGBM
    FOCAL_ALPHA: float = 0.25  # Weight for positive class
    FOCAL_GAMMA: float = 2.0  # Focusing parameter

    # Model Selection
    MODELS_TO_USE: List[str] = None

    def __post_init__(self):
        if self.CATEGORICAL_FEATURES is None:
            self.CATEGORICAL_FEATURES = [
                'gender', 'ethnicity', 'education_level', 'income_level',
                'smoking_status', 'employment_status'
            ]
        if self.MODELS_TO_USE is None:
            self.MODELS_TO_USE = ['catboost','lightgbm', 'xgboost'] #'extratrees', 'catboost', 


class MedicalFeatureEngineer:
    """Domain-knowledge based feature engineering for diabetes prediction"""

    def __init__(self, config: ImprovedConfig):
        self.config = config

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create medical domain-specific features"""
        df = df.copy()

        print("  → Creating medical domain features...")

        # ═══════════════════════════════════════════════════════════════
        # CHOLESTEROL & LIPID RATIOS (Critical for diabetes/CVD)
        # ═══════════════════════════════════════════════════════════════
        if all(col in df.columns for col in ['cholesterol_total', 'hdl_cholesterol']):
            df['chol_hdl_ratio'] = df['cholesterol_total'] / (df['hdl_cholesterol'] + 1e-5)
            df['atherogenic_index'] = (df['cholesterol_total'] - df['hdl_cholesterol']) / (df['hdl_cholesterol'] + 1e-5)

        if all(col in df.columns for col in ['ldl_cholesterol', 'hdl_cholesterol']):
            df['ldl_hdl_ratio'] = df['ldl_cholesterol'] / (df['hdl_cholesterol'] + 1e-5)

        if all(col in df.columns for col in ['triglycerides', 'hdl_cholesterol']):
            df['trig_hdl_ratio'] = df['triglycerides'] / (df['hdl_cholesterol'] + 1e-5)
            # Atherogenic dyslipidemia indicator
            df['atherogenic_dyslipidemia'] = ((df['triglycerides'] > 150) & (df['hdl_cholesterol'] < 40)).astype(int)

        if all(col in df.columns for col in ['cholesterol_total', 'hdl_cholesterol', 'ldl_cholesterol']):
            df['non_hdl_chol'] = df['cholesterol_total'] - df['hdl_cholesterol']
            df['cholesterol_balance'] = (df['hdl_cholesterol'] - df['ldl_cholesterol']) / (df['cholesterol_total'] + 1e-5)

        # ═══════════════════════════════════════════════════════════════
        # BLOOD PRESSURE & CARDIOVASCULAR METRICS
        # ═══════════════════════════════════════════════════════════════
        if all(col in df.columns for col in ['systolic_bp', 'diastolic_bp']):
            df['pulse_pressure'] = df['systolic_bp'] - df['diastolic_bp']
            df['mean_arterial_pressure'] = df['diastolic_bp'] + (df['pulse_pressure'] / 3)
            df['bp_product'] = df['systolic_bp'] * df['diastolic_bp']
            df['bp_ratio'] = df['systolic_bp'] / (df['diastolic_bp'] + 1e-5)

            # Hypertension categories
            df['hypertension_stage'] = 0
            df.loc[(df['systolic_bp'] >= 120) & (df['systolic_bp'] < 130), 'hypertension_stage'] = 1
            df.loc[(df['systolic_bp'] >= 130) & (df['systolic_bp'] < 140), 'hypertension_stage'] = 2
            df.loc[df['systolic_bp'] >= 140, 'hypertension_stage'] = 3

        # ═══════════════════════════════════════════════════════════════
        # BMI & BODY COMPOSITION
        # ═══════════════════════════════════════════════════════════════
        if 'bmi' in df.columns:
            # BMI categories (WHO classification)
            df['bmi_category'] = pd.cut(df['bmi'],
                                       bins=[0, 18.5, 25, 30, 35, 40, 100],
                                       labels=[0, 1, 2, 3, 4, 5]).astype(int)
            df['is_obese'] = (df['bmi'] >= 30).astype(int)
            df['is_severely_obese'] = (df['bmi'] >= 35).astype(int)

            # BMI squared (non-linear relationship)
            df['bmi_squared'] = df['bmi'] ** 2
            df['bmi_cubed'] = df['bmi'] ** 3

            if 'age' in df.columns:
                df['bmi_age_interaction'] = df['bmi'] * df['age']
                df['obesity_years'] = df['is_obese'] * df['age']

        if 'waist_to_hip_ratio' in df.columns:
            df['central_obesity'] = (df['waist_to_hip_ratio'] > 0.90).astype(int)  # For men, 0.85 for women

            if 'bmi' in df.columns:
                df['bmi_whr_product'] = df['bmi'] * df['waist_to_hip_ratio']
                df['metabolic_risk_index'] = df['bmi'] * df['waist_to_hip_ratio'] * 10

        # ═══════════════════════════════════════════════════════════════
        # METABOLIC SYNDROME SCORE (Crucial for diabetes)
        # ═══════════════════════════════════════════════════════════════
        metabolic_score = 0
        if 'bmi' in df.columns:
            metabolic_score += (df['bmi'] >= 30).astype(int)
        if 'systolic_bp' in df.columns:
            metabolic_score += (df['systolic_bp'] >= 130).astype(int)
        if 'triglycerides' in df.columns:
            metabolic_score += (df['triglycerides'] >= 150).astype(int)
        if 'hdl_cholesterol' in df.columns:
            metabolic_score += (df['hdl_cholesterol'] < 40).astype(int)
        if 'waist_to_hip_ratio' in df.columns:
            metabolic_score += (df['waist_to_hip_ratio'] > 0.90).astype(int)

        df['metabolic_syndrome_score'] = metabolic_score
        df['has_metabolic_syndrome'] = (metabolic_score >= 3).astype(int)

        # ═══════════════════════════════════════════════════════════════
        # LIFESTYLE FACTORS
        # ═══════════════════════════════════════════════════════════════
        if 'physical_activity_minutes_per_week' in df.columns:
            df['daily_activity_minutes'] = df['physical_activity_minutes_per_week'] / 7
            df['is_sedentary'] = (df['physical_activity_minutes_per_week'] < 150).astype(int)
            df['is_active'] = (df['physical_activity_minutes_per_week'] >= 300).astype(int)

            if 'bmi' in df.columns:
                df['activity_bmi_ratio'] = df['physical_activity_minutes_per_week'] / (df['bmi'] + 1)
                df['sedentary_obese'] = df['is_sedentary'] * df['is_obese']

        if 'diet_score' in df.columns:
            df['poor_diet'] = (df['diet_score'] < 5).astype(int)
            df['good_diet'] = (df['diet_score'] >= 7).astype(int)

            if 'bmi' in df.columns:
                df['diet_bmi_interaction'] = df['diet_score'] * df['bmi']

        if 'sleep_hours_per_day' in df.columns:
            df['sleep_deprived'] = (df['sleep_hours_per_day'] < 6).astype(int)
            df['sleep_excess'] = (df['sleep_hours_per_day'] > 9).astype(int)
            df['sleep_abnormal'] = df['sleep_deprived'] + df['sleep_excess']

            if 'screen_time_hours_per_day' in df.columns:
                df['sleep_screen_ratio'] = df['sleep_hours_per_day'] / (df['screen_time_hours_per_day'] + 1e-5)
                df['rest_quality_score'] = df['sleep_hours_per_day'] - df['screen_time_hours_per_day']

        if 'alcohol_consumption_per_week' in df.columns:
            df['heavy_drinker'] = (df['alcohol_consumption_per_week'] > 7).astype(int)  # >7 drinks/week

        # ═══════════════════════════════════════════════════════════════
        # RISK FACTORS & FAMILY HISTORY
        # ═══════════════════════════════════════════════════════════════
        risk_cols = ['family_history_diabetes', 'hypertension_history', 'cardiovascular_history']
        available_risk_cols = [col for col in risk_cols if col in df.columns]

        if available_risk_cols:
            df['total_risk_factors'] = df[available_risk_cols].sum(axis=1)
            df['has_family_history'] = (df['total_risk_factors'] > 0).astype(int)
            df['multiple_risk_factors'] = (df['total_risk_factors'] >= 2).astype(int)

            if 'age' in df.columns:
                df['age_risk_interaction'] = df['age'] * df['total_risk_factors']

        # ═══════════════════════════════════════════════════════════════
        # AGE-RELATED FEATURES
        # ═══════════════════════════════════════════════════════════════
        if 'age' in df.columns:
            df['age_group'] = pd.cut(df['age'],
                                    bins=[0, 30, 40, 50, 60, 100],
                                    labels=[0, 1, 2, 3, 4]).astype(int)
            df['is_senior'] = (df['age'] >= 60).astype(int)
            df['is_middle_aged'] = ((df['age'] >= 40) & (df['age'] < 60)).astype(int)
            df['age_squared'] = df['age'] ** 2

            # Age interactions with biomarkers
            if 'cholesterol_total' in df.columns:
                df['age_cholesterol'] = df['age'] * df['cholesterol_total']
            if 'systolic_bp' in df.columns:
                df['age_bp'] = df['age'] * df['systolic_bp']

        # ═══════════════════════════════════════════════════════════════
        # COMPOSITE RISK SCORES
        # ═══════════════════════════════════════════════════════════════
        # Simplified diabetes risk score
        diabetes_risk = 0
        if 'age' in df.columns:
            diabetes_risk += (df['age'] >= 45).astype(int) * 2
        if 'bmi' in df.columns:
            diabetes_risk += (df['bmi'] >= 30).astype(int) * 2
        if 'family_history_diabetes' in df.columns:
            diabetes_risk += df['family_history_diabetes'] * 3
        if 'hypertension_history' in df.columns:
            diabetes_risk += df['hypertension_history']
        if 'physical_activity_minutes_per_week' in df.columns:
            diabetes_risk += (df['physical_activity_minutes_per_week'] < 150).astype(int)

        df['diabetes_risk_score'] = diabetes_risk
        df['high_risk'] = (diabetes_risk >= 5).astype(int)

        # ═══════════════════════════════════════════════════════════════
        # POLYNOMIAL & LOG TRANSFORMS for key biomarkers
        # ═══════════════════════════════════════════════════════════════
        important_numeric_cols = [
            'cholesterol_total', 'ldl_cholesterol', 'triglycerides',
            'systolic_bp', 'diastolic_bp', 'heart_rate'
        ]

        for col in important_numeric_cols:
            if col in df.columns:
                df[f'{col}_squared'] = df[col] ** 2
                df[f'{col}_log'] = np.log1p(df[col].clip(lower=0))
                df[f'{col}_sqrt'] = np.sqrt(df[col].clip(lower=0))

        # ═══════════════════════════════════════════════════════════════
        # STATISTICAL AGGREGATIONS
        # ═══════════════════════════════════════════════════════════════
        # Cholesterol profile summary
        chol_cols = [col for col in df.columns if 'cholesterol' in col.lower()
                    and col in df.select_dtypes(include=[np.number]).columns]
        if len(chol_cols) >= 2:
            df['cholesterol_avg'] = df[chol_cols].mean(axis=1)
            df['cholesterol_std'] = df[chol_cols].std(axis=1)
            df['cholesterol_range'] = df[chol_cols].max(axis=1) - df[chol_cols].min(axis=1)

        # Lifestyle summary
        lifestyle_keywords = ['sleep', 'screen', 'physical', 'diet', 'alcohol']
        lifestyle_cols = []
        for keyword in lifestyle_keywords:
            lifestyle_cols.extend([col for col in df.columns if keyword in col.lower()
                                  and col in df.select_dtypes(include=[np.number]).columns])
        lifestyle_cols = list(set(lifestyle_cols))

        if len(lifestyle_cols) >= 2:
            df['lifestyle_avg'] = df[lifestyle_cols].mean(axis=1)
            df['lifestyle_std'] = df[lifestyle_cols].std(axis=1)

        print(f"    ✓ Created {len([c for c in df.columns]) - len(df.columns)} medical features")

        return df


class KFoldTargetEncoder:
    """
    K-Fold target encoding to prevent data leakage.
    Based on 2024 research: prevents overfitting better than label encoding.
    """

    def __init__(self, categorical_features: List[str], n_splits: int = 5,
                 smoothing: float = 1.0, seed: int = 42):
        self.categorical_features = categorical_features
        self.n_splits = n_splits
        self.smoothing = smoothing
        self.seed = seed
        self.global_means = {}
        self.encodings = {}

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Fit and transform training data using K-Fold"""
        X_encoded = X.copy()

        for col in self.categorical_features:
            if col not in X.columns:
                continue

            # Calculate global mean for this feature
            self.global_means[col] = y.mean()

            # Create K-Fold encoding - initialize with zeros
            target_enc_values = np.zeros(len(X))

            kf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)

            for train_idx, val_idx in kf.split(X, y):
                # Calculate means on training fold
                train_stats = y.iloc[train_idx].groupby(X[col].iloc[train_idx]).agg(['mean', 'count'])

                # Apply smoothing
                smoothed_means = (
                    (train_stats['mean'] * train_stats['count'] +
                     self.global_means[col] * self.smoothing) /
                    (train_stats['count'] + self.smoothing)
                )

                # Encode validation fold
                target_enc_values[val_idx] = (
                    X[col].iloc[val_idx].map(smoothed_means).fillna(self.global_means[col]).values
                )

            # Add encoded column
            X_encoded[f'{col}_target_enc'] = target_enc_values

            # Store final encodings for transform
            final_stats = y.groupby(X[col]).agg(['mean', 'count'])
            self.encodings[col] = (
                (final_stats['mean'] * final_stats['count'] +
                 self.global_means[col] * self.smoothing) /
                (final_stats['count'] + self.smoothing)
            )

        return X_encoded

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform test data using fitted encodings"""
        X_encoded = X.copy()

        for col in self.categorical_features:
            if col not in X.columns or col not in self.encodings:
                continue

            X_encoded[f'{col}_target_enc'] = (
                X[col].map(self.encodings[col]).fillna(self.global_means[col])
            )

        return X_encoded


class ShapFeatureSelector:
    """
    SHAP-based feature selection inspired by BorutaShap.
    Based on 2024 research showing superior performance.
    """

    def __init__(self, n_estimators: int = 216, threshold: float = 0.001, max_features: int = 50):
        self.n_estimators = n_estimators
        self.threshold = threshold
        self.max_features = max_features
        self.selected_features = None
        self.feature_importances = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fit selector using SHAP values"""
        print("\n  → Running SHAP-based feature selection...")

        # Train a fast model for SHAP
        model = LGBMClassifier(
            n_estimators=self.n_estimators,
            learning_rate=0.05,
            max_depth=5,
            random_state=42,
            verbose=-1,
            n_jobs=-1
        )
        model.fit(X, y)

        # Calculate SHAP values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        # Get mean absolute SHAP values
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # For binary classification

        mean_abs_shap = np.abs(shap_values).mean(axis=0)

        # Create feature importance dataframe
        self.feature_importances = pd.DataFrame({
            'feature': X.columns,
            'shap_importance': mean_abs_shap,
            'tree_importance': model.feature_importances_
        })

        # Combined importance score
        self.feature_importances['combined_importance'] = (
            0.6 * self.feature_importances['shap_importance'] / self.feature_importances['shap_importance'].max() +
            0.4 * self.feature_importances['tree_importance'] / self.feature_importances['tree_importance'].max()
        )

        self.feature_importances = self.feature_importances.sort_values(
            'combined_importance', ascending=False
        ).reset_index(drop=True)

        # Select features above threshold
        selected = self.feature_importances[
            self.feature_importances['combined_importance'] >= self.threshold
        ]['feature'].tolist()

        # Limit to max_features
        self.selected_features = selected[:self.max_features]

        print(f"    ✓ Selected {len(self.selected_features)} features out of {len(X.columns)}")
        print(f"\n    Top 15 features:")
        for idx, row in self.feature_importances.head(15).iterrows():
            print(f"      {idx+1:2d}. {row['feature']:40s} | Score: {row['combined_importance']:.4f}")

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data to selected features"""
        if self.selected_features is None:
            raise ValueError("Must fit selector before transform")

        return X[self.selected_features]

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Fit and transform in one step"""
        self.fit(X, y)
        return self.transform(X)


class ImbalancedLearningHandler:
    """
    State-of-the-art imbalanced learning handler implementing multiple strategies.

    Based on 2024-2025 research showing that hybrid approaches combining
    sampling + threshold tuning + ensemble methods achieve best results.

    References:
        - Borderline-SMOTE: 97% accuracy in ANN studies (2023)
        - Threshold Calibration: Most effective in 40% of datasets (2024)
        - Hybrid methods (CSBBoost, HD-Ensemble): Superior to single techniques
    """

    def __init__(self, config: ImprovedConfig):
        self.config = config
        self.sampler = None
        self.optimal_threshold = 0.5

    def get_resampler(self, strategy: str = None):
        """
        Get the appropriate resampler based on configuration.

        Returns a sampling object compatible with scikit-learn Pipeline.
        """
        if not IMBLEARN_AVAILABLE:
            print("⚠️  imbalanced-learn not available. Skipping resampling.")
            return None

        strategy = strategy or self.config.SAMPLING_STRATEGY

        if strategy == 'none':
            return None

        elif strategy == 'oversample':
            method = self.config.SAMPLING_METHOD
            if method == 'smote':
                return SMOTE(random_state=self.config.SEED, k_neighbors=5)
            elif method == 'borderline_smote':
                # BEST performer per 2023 studies: 97% accuracy
                return BorderlineSMOTE(random_state=self.config.SEED, k_neighbors=5, kind='borderline-1')
            elif method == 'adasyn':
                return ADASYN(random_state=self.config.SEED, n_neighbors=5)
            elif method == 'svmsmote':
                return SVMSMOTE(random_state=self.config.SEED, k_neighbors=5)
            else:
                return SMOTE(random_state=self.config.SEED)

        elif strategy == 'undersample':
            return RandomUnderSampler(random_state=self.config.SEED)

        elif strategy == 'hybrid':
            # Hybrid methods combine over and under sampling
            method = self.config.HYBRID_METHOD
            if method == 'smote_tomek':
                return SMOTETomek(random_state=self.config.SEED)
            elif method == 'smote_enn':
                return SMOTEENN(random_state=self.config.SEED)
            else:
                return SMOTETomek(random_state=self.config.SEED)

        return None

    def apply_sampling(self, X: pd.DataFrame, y: pd.Series):
        """
        Apply the configured sampling strategy.

        Returns resampled X, y
        """
        if not self.config.USE_SAMPLING or not IMBLEARN_AVAILABLE:
            return X, y

        sampler = self.get_resampler()
        if sampler is None:
            return X, y

        print(f"\n  → Applying {self.config.SAMPLING_STRATEGY} sampling...")
        print(f"     Method: {self.config.SAMPLING_METHOD if self.config.SAMPLING_STRATEGY == 'oversample' else self.config.HYBRID_METHOD}")
        print(f"     Original distribution: {y.value_counts().to_dict()}")

        X_resampled, y_resampled = sampler.fit_resample(X, y)

        print(f"     Resampled distribution: {pd.Series(y_resampled).value_counts().to_dict()}")

        return pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled)

    def find_optimal_threshold(self, y_true, y_pred_proba, metric='f1'):
        """
        Find optimal classification threshold using specified metric.

        CRITICAL: Research shows threshold tuning is most consistently effective
        technique for imbalanced data (40% of datasets, 2024 study).

        Metrics:
            - 'f1': Maximize F1-score
            - 'gmean': Geometric mean of sensitivity and specificity
            - 'youden': Youden's J statistic (sensitivity + specificity - 1)
            - 'pr_optimal': Optimal point on Precision-Recall curve
        """
        thresholds = np.arange(0.1, 0.9, 0.01)
        best_score = 0
        best_threshold = 0.5

        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)

            if metric == 'f1':
                score = f1_score(y_true, y_pred)
            elif metric == 'gmean':
                # Geometric mean: sqrt(sensitivity * specificity)
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                score = np.sqrt(sensitivity * specificity)
            elif metric == 'youden':
                # Youden's J = sensitivity + specificity - 1
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                score = sensitivity + specificity - 1
            elif metric == 'pr_optimal':
                # Maximize F1 from precision-recall curve
                score = f1_score(y_true, y_pred)
            else:
                score = f1_score(y_true, y_pred)

            if score > best_score:
                best_score = score
                best_threshold = threshold

        return best_threshold, best_score

    def optimize_threshold(self, y_true, y_pred_proba):
        """
        Optimize and store the best threshold for this model.
        """
        if not self.config.USE_THRESHOLD_TUNING:
            self.optimal_threshold = 0.5
            return 0.5

        threshold, score = self.find_optimal_threshold(
            y_true, y_pred_proba, metric=self.config.THRESHOLD_METRIC
        )

        self.optimal_threshold = threshold

        print(f"\n  → Optimal Threshold Tuning:")
        print(f"     Metric: {self.config.THRESHOLD_METRIC}")
        print(f"     Best threshold: {threshold:.3f}")
        print(f"     Best score: {score:.4f}")

        return threshold

    def predict_with_threshold(self, y_pred_proba, threshold=None):
        """
        Make predictions using the optimal threshold.
        """
        if threshold is None:
            threshold = self.optimal_threshold
        return (y_pred_proba >= threshold).astype(int)

    def get_imbalanced_ensemble(self, base_estimator=None):
        """
        Get specialized ensemble classifier for imbalanced data.

        These ensembles are specifically designed for imbalanced learning:
            - RUSBoost: Random undersampling + AdaBoost
            - EasyEnsemble: Bagging with balanced subsets
            - BalancedBagging: Bagging with random undersampling
        """
        if not self.config.USE_IMBALANCED_ENSEMBLE or not IMBLEARN_AVAILABLE:
            return None

        method = self.config.IMBALANCED_ENSEMBLE_METHOD

        if base_estimator is None:
            from sklearn.tree import DecisionTreeClassifier
            base_estimator = DecisionTreeClassifier(max_depth=5)

        if method == 'rusboost':
            return RUSBoostClassifier(
                estimator=base_estimator,
                n_estimators=50,
                random_state=self.config.SEED
            )
        elif method == 'easy_ensemble':
            return EasyEnsembleClassifier(
                n_estimators=10,
                random_state=self.config.SEED
            )
        elif method == 'balanced_bagging':
            return BalancedBaggingClassifier(
                estimator=base_estimator,
                n_estimators=10,
                random_state=self.config.SEED
            )

        return None


class FocalLoss:
    """
    Focal Loss implementation for gradient boosting models.

    Based on: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
    Adapted for XGBoost/LightGBM custom objectives.

    Formula: FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    where:
        p_t = model probability for the true class
        α_t = weighting factor (alpha)
        γ = focusing parameter (gamma)
    """

    def __init__(self, alpha=0.25, gamma=2.0):
        self.alpha = alpha
        self.gamma = gamma

    def focal_loss_lgb(self, y_true, y_pred):
        """Focal loss for LightGBM"""
        a, g = self.alpha, self.gamma

        def fl(x, t):
            p = 1 / (1 + np.exp(-x))
            return -( a*t + (1-a)*(1-t) ) * (( 1 - ( t*p + (1-t)*(1-p)) )**g) * ( t*np.log(p) + (1-t)*np.log(1-p) )

        partial_fl = lambda x: fl(x, y_true)
        grad = np.gradient(partial_fl(y_pred))
        hess = np.gradient(grad)

        return grad, hess

    def focal_loss_xgb(self, y_pred, y_true):
        """Focal loss for XGBoost"""
        a, g = self.alpha, self.gamma
        p = 1 / (1 + np.exp(-y_pred))

        grad = a * (p - y_true) * ((1 - p) ** g) * (g * p * np.log(p) + p - 1)
        hess = a * ((1 - p) ** g) * (g * p * (2 * p - 1) * np.log(p) + p * (2 - p) - 1)

        return grad, hess


class ImprovedModelTrainer:
    """
    Proper cross-validation model training with:
    - Stratified K-Fold throughout
    - Class weights for imbalance
    - Enhanced hyperparameter spaces
    - No data leakage
    """

    def __init__(self, config: ImprovedConfig):
        self.config = config
        self.models = {}
        self.oof_predictions = {}
        self.test_predictions = {}
        self.feature_importances = {}
        self.imbalance_handler = ImbalancedLearningHandler(config)  # NEW: Imbalanced learning
        self.optimal_thresholds = {}  # Store optimal thresholds per model

    def optimize_class_weights_grid_search(self, X, y, model_name: str, base_params: Dict = None) -> Dict:
        """
        ADVANCED: Grid search to find optimal class weights for maximum accuracy.

        Based on 2024 research showing that tuned class weights significantly outperform
        default balanced weights.

        References:
            - "Balancing the Scales" (2024): Class weight tuning shows moderate improvements
            - XGBoost Docs: scale_pos_weight = sum(negative)/sum(positive) as starting point
            - Adaptive Weight Optimization (Springer): Evolutionary algorithms for weight tuning

        Returns:
            Dictionary with optimal class weights (format depends on model)
        """
        print(f"\n  → Grid Search for Optimal Class Weights ({model_name})")

        # Calculate class distribution
        class_counts = y.value_counts()
        n_neg = class_counts[0]
        n_pos = class_counts[1]
        ratio = n_neg / n_pos

        print(f"     Class distribution: Neg={n_neg}, Pos={n_pos}, Ratio={ratio:.2f}:1")

        # Define search space based on research and imbalance ratio
        if ratio > 10:
            # Severe imbalance: wide search range around ratio
            weight_candidates = [
                ratio * 0.5,   # Conservative
                ratio * 0.75,  # Moderate conservative
                ratio,         # Balanced (default recommendation)
                ratio * 1.25,  # Moderate aggressive
                ratio * 1.5,   # Aggressive
                ratio * 2.0,   # Very aggressive
                ratio * 3.0    # Extremely aggressive
            ]
        else:
            # Mild/moderate imbalance: narrower search
            weight_candidates = [
                ratio * 0.8,
                ratio,
                ratio * 1.2,
                ratio * 1.5,
                ratio * 2.0
            ]

        # Fast 3-fold CV for each weight
        best_weight = ratio
        best_score = 0

        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.config.SEED)

        for weight in weight_candidates:
            scores = []

            for train_idx, val_idx in skf.split(X, y):
                X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

                # Create model with this weight
                try:
                    if model_name in ['xgboost']:
                        model = xgb.XGBClassifier(
                            scale_pos_weight=weight,
                            n_estimators=100,
                            random_state=self.config.SEED,
                            eval_metric='logloss'
                        )
                    elif model_name in ['lightgbm']:
                        model = LGBMClassifier(
                            scale_pos_weight=weight,
                            n_estimators=100,
                            random_state=self.config.SEED,
                            verbose=-1
                        )
                    elif model_name in ['catboost']:
                        model = CatBoostClassifier(
                            scale_pos_weight=weight,
                            iterations=100,
                            random_state=self.config.SEED,
                            verbose=False
                        )
                    else:
                        # sklearn models use class_weight dict
                        class_weights_dict = {0: 1.0, 1: weight}
                        model = ExtraTreesClassifier(
                            n_estimators=100,
                            class_weight=class_weights_dict,
                            random_state=self.config.SEED
                        )

                    # Train and evaluate
                    model.fit(X_tr, y_tr)
                    y_pred_proba = model.predict_proba(X_val)[:, 1]
                    score = roc_auc_score(y_val, y_pred_proba)
                    scores.append(score)
                except Exception as e:
                    print(f"       Error with weight={weight}: {e}")
                    continue

            if len(scores) > 0:
                avg_score = np.mean(scores)
                print(f"     Weight={weight:6.2f} → AUC={avg_score:.6f}")

                if avg_score > best_score:
                    best_score = avg_score
                    best_weight = weight

        print(f"     ✓ OPTIMAL WEIGHT: {best_weight:.2f} (CV AUC={best_score:.6f})")

        # Return format depends on model type
        if model_name in ['xgboost', 'lightgbm', 'catboost']:
            return {'scale_pos_weight': best_weight}
        else:
            return {0: 1.0, 1: best_weight}

    def calculate_class_weights(self, y: pd.Series, method: str = 'effective_num', beta: float = 0.9999) -> Dict:
        """
        Advanced class weight calculation for imbalanced data.

        Implements multiple state-of-the-art methods based on recent research:

        1. Effective Number of Samples (CVPR 2019 - Cui et al.):
           - Paper: "Class-Balanced Loss Based on Effective Number of Samples"
           - Formula: E_n = (1 - β^n) / (1 - β), weights = (1 - β) / E_n
           - Accounts for data overlap and diminishing returns of repeated samples
           - Beta typically: 0.9999 for long-tailed, 0.99-0.999 for moderate imbalance

        2. Inverse Square Root (Modern best practice):
           - Softer weighting than inverse frequency
           - Better generalization in many cases
           - Formula: weight = sqrt(total / class_count)

        3. Balanced (sklearn default):
           - Simple inverse frequency weighting
           - Formula: weight = total / (n_classes * class_count)

        Args:
            y: Target labels
            method: One of 'effective_num', 'sqrt', 'balanced', 'dynamic'
            beta: Hyperparameter for effective_num method (0 < beta < 1)
                  Recommended: 0.9999 (highly imbalanced), 0.999 (moderate), 0.99 (mild)

        Returns:
            Dictionary mapping class labels to weights

        References:
            - CVPR 2019: Class-Balanced Loss Based on Effective Number of Samples
            - Cost-Sensitive Learning for Imbalanced Classification (2024)
            - Focal Loss for Dense Object Detection (Lin et al.)
        """
        class_counts = y.value_counts().sort_index()
        n_samples = len(y)
        n_classes = len(class_counts)

        if method == 'effective_num':
            # Method 1: Effective Number of Samples (CVPR 2019)
            # Accounts for diminishing marginal benefit of additional samples
            effective_num = 1.0 - np.power(beta, class_counts.values)
            weights_array = (1.0 - beta) / effective_num

            # Normalize weights to sum to n_classes (standard practice)
            weights_array = weights_array / weights_array.sum() * n_classes

        elif method == 'sqrt':
            # Method 2: Inverse Square Root
            # Provides softer penalty than inverse frequency
            weights_array = np.sqrt(n_samples / (n_classes * class_counts.values))

        elif method == 'balanced':
            # Method 3: Standard balanced (sklearn default)
            # Simple inverse frequency weighting
            weights_array = n_samples / (n_classes * class_counts.values)

        elif method == 'dynamic':
            # Method 4: Dynamic weighting based on imbalance ratio
            # Automatically selects best method based on imbalance severity
            imbalance_ratio = class_counts.max() / class_counts.min()

            if imbalance_ratio > 100:
                # Severe imbalance: use effective_num with high beta
                effective_num = 1.0 - np.power(0.9999, class_counts.values)
                weights_array = (1.0 - 0.9999) / effective_num
                weights_array = weights_array / weights_array.sum() * n_classes
            elif imbalance_ratio > 10:
                # Moderate imbalance: use effective_num with medium beta
                effective_num = 1.0 - np.power(0.999, class_counts.values)
                weights_array = (1.0 - 0.999) / effective_num
                weights_array = weights_array / weights_array.sum() * n_classes
            else:
                # Mild imbalance: use sqrt method
                weights_array = np.sqrt(n_samples / (n_classes * class_counts.values))
        else:
            raise ValueError(f"Unknown method: {method}. Use 'effective_num', 'sqrt', 'balanced', or 'dynamic'")

        # Convert to dictionary with class labels as keys
        weights = {int(cls): float(weight) for cls, weight in zip(class_counts.index, weights_array)}

        # Log the weighting strategy for transparency
        imbalance_ratio = class_counts.max() / class_counts.min()
        print(f"\n{'='*70}")
        print(f"Class Weight Calculation - Method: {method}")
        print(f"{'='*70}")
        print(f"Dataset Statistics:")
        print(f"  Total samples: {n_samples}")
        print(f"  Number of classes: {n_classes}")
        for cls in class_counts.index:
            pct = (class_counts[cls] / n_samples) * 100
            print(f"  Class {cls}: {class_counts[cls]:,} samples ({pct:.2f}%)")
        print(f"\nImbalance Ratio: {imbalance_ratio:.2f}:1")
        print(f"\nCalculated Weights:")
        for cls in sorted(weights.keys()):
            print(f"  Class {cls}: {weights[cls]:.4f}")
        print(f"{'='*70}\n")

        return weights

    def optimize_class_weights(self, trial, y: pd.Series) -> Dict:
        """
        Optimize class weight method and parameters using Optuna.

        This method can be called within an Optuna objective function to find
        the optimal class weighting strategy for your specific dataset.

        Args:
            trial: Optuna trial object
            y: Target labels

        Returns:
            Dictionary of optimized class weights
        """
        # Let Optuna choose the best method
        method = trial.suggest_categorical('class_weight_method',
                                          ['effective_num', 'sqrt', 'balanced', 'dynamic'])

        # Optimize beta parameter for effective_num method
        if method == 'effective_num':
            beta = trial.suggest_float('class_weight_beta', 0.99, 0.9999, log=True)
        else:
            beta = 0.9999  # Default, won't be used

        return self.calculate_class_weights(y, method=method, beta=beta)

    def get_optuna_params(self, trial, model_name: str) -> Dict:
        """
        Enhanced hyperparameter search spaces based on 2024-2025 research.

        CRITICAL INNOVATION (Nature Scientific Reports 2022):
        Optimizes class_weight TOGETHER with other hyperparameters using Bayesian optimization.
        This achieves ROC-AUC improvement: 0.896 → 0.917 (+2.3%)
        """

        # STEP 1: Optimize class weight as part of Bayesian search (PROVEN approach)
        if self.config.USE_CLASS_WEIGHT_OPTIMIZATION == 'bayesian' and trial is not None:
            # Calculate imbalance ratio for intelligent search range
            # This will be set by the calling function
            pass  # Class weight will be added model-specifically below

        if model_name == 'catboost':
            params = {
                'iterations': trial.suggest_int('iterations', 200, 1000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'depth': trial.suggest_int('depth', 4, 10),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
                'border_count': trial.suggest_int('border_count', 32, 255),
                'random_strength': trial.suggest_float('random_strength', 0.0, 10.0),
                'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
                'random_state': self.config.SEED,
                'verbose': False,
                'task_type': 'CPU',
                'loss_function': 'Logloss',
                'eval_metric': 'AUC',
            }
            # Add class weight optimization (Nature 2022 approach)
            if self.config.USE_CLASS_WEIGHT_OPTIMIZATION == 'bayesian' and trial is not None:
                # Optimize scale_pos_weight in log space for better exploration
                params['scale_pos_weight'] = trial.suggest_float('scale_pos_weight', 0.1, 20.0, log=True)
            return params

        elif model_name == 'lightgbm':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
                'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 1.0),
                'random_state': self.config.SEED,
                'verbose': -1,
                'n_jobs': -1,
            }
            # Add class weight optimization (Nature 2022 approach)
            if self.config.USE_CLASS_WEIGHT_OPTIMIZATION == 'bayesian' and trial is not None:
                params['scale_pos_weight'] = trial.suggest_float('scale_pos_weight', 0.1, 20.0, log=True)
            return params

        elif model_name == 'xgboost':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'gamma': trial.suggest_float('gamma', 0.0, 5.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
                'random_state': self.config.SEED,
                'tree_method': 'hist',
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
            }
            # Add class weight optimization (Nature 2022 approach)
            if self.config.USE_CLASS_WEIGHT_OPTIMIZATION == 'bayesian' and trial is not None:
                params['scale_pos_weight'] = trial.suggest_float('scale_pos_weight', 0.1, 20.0, log=True)
            return params

        elif model_name == 'extratrees':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 200, 1000),
                'max_depth': trial.suggest_int('max_depth', 5, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'random_state': self.config.SEED,
                'n_jobs': -1,
            }

    def create_model(self, model_name: str, params: Dict, class_weights: Dict = None):
        """
        Create model instance with optimized class weights.

        Handles different class weight formats for different model types:
        - XGBoost/LightGBM/CatBoost: use 'scale_pos_weight' directly
        - Sklearn models: use 'class_weight' dictionary
        """

        if model_name == 'catboost':
            if class_weights:
                # Check format: {0: w0, 1: w1} vs {'scale_pos_weight': value}
                if 'scale_pos_weight' in class_weights:
                    params['scale_pos_weight'] = class_weights['scale_pos_weight']
                else:
                    params['class_weights'] = class_weights
            return CatBoostClassifier(**params)

        elif model_name == 'lightgbm':
            if class_weights:
                # Check format
                if 'scale_pos_weight' in class_weights:
                    params['scale_pos_weight'] = class_weights['scale_pos_weight']
                else:
                    params['class_weight'] = class_weights
            return LGBMClassifier(**params)

        elif model_name == 'xgboost':
            if class_weights:
                # Check format
                if 'scale_pos_weight' in class_weights:
                    params['scale_pos_weight'] = class_weights['scale_pos_weight']
                else:
                    # Convert {0: w0, 1: w1} to scale_pos_weight
                    scale_pos_weight = class_weights[1] / class_weights[0]
                    params['scale_pos_weight'] = scale_pos_weight
            return xgb.XGBClassifier(**params)

        elif model_name == 'extratrees':
            if class_weights:
                # sklearn models use class_weight dict
                if 'scale_pos_weight' in class_weights:
                    # Convert scale_pos_weight to class_weight dict
                    params['class_weight'] = {0: 1.0, 1: class_weights['scale_pos_weight']}
                else:
                    params['class_weight'] = class_weights
            return ExtraTreesClassifier(**params)

    def train_with_single_split(self, model_name: str, X_train: pd.DataFrame, y_train: pd.Series,
                                X_val: pd.DataFrame, y_val: pd.Series,
                                X_test: pd.DataFrame, categorical_features: List[str] = None):
        """
        Train model with single 70/30 split (FAST VERSION for quick testing).

        NOW INCLUDES ALL ADVANCED IMBALANCED LEARNING STRATEGIES:
        1. Class weights (effective number of samples)
        2. Hybrid sampling (Borderline-SMOTE + Tomek Links)
        3. Threshold optimization (F1/G-Mean/Youden)
        4. Detailed metrics (Precision, Recall, F1)

        All data is already split - no leakage.
        """
        print(f"\n{'='*70}")
        print(f"  Training {model_name.upper()} with Single 70/30 Split")
        print(f"  ADVANCED IMBALANCED LEARNING ENABLED")
        print(f"{'='*70}")

        # STEP 0: CLASS WEIGHT STRATEGY
        # CRITICAL INSIGHT: Bayesian optimization of class_weight WITH other hyperparameters
        # is the PROVEN best approach (Nature 2022: +2.3% AUC improvement)

        if self.config.USE_CLASS_WEIGHT_OPTIMIZATION == 'bayesian':
            print(f"\n  ╔══════════════════════════════════════════════════════════════════╗")
            print(f"  ║  BAYESIAN OPTIMIZATION: Class Weight + Hyperparameters          ║")
            print(f"  ║  Based on: Nature Scientific Reports 2022 (Drug Discovery)      ║")
            print(f"  ║  Proven Result: ROC-AUC 0.896 → 0.917 (+2.3% improvement)       ║")
            print(f"  ╚══════════════════════════════════════════════════════════════════╝")
            # Class weight will be optimized TOGETHER with other hyperparameters in Optuna
            class_weights = None  # Will be part of Optuna params

        elif self.config.USE_CLASS_WEIGHT_OPTIMIZATION == 'grid':
            print(f"\n  ╔═══════════════════════════════════════════════════════════╗")
            print(f"  ║  GRID SEARCH for Optimal Class Weights                   ║")
            print(f"  ╚═══════════════════════════════════════════════════════════╝")
            # Separate grid search for class weights
            optimal_weights = self.optimize_class_weights_grid_search(
                X_train, y_train, model_name
            )
            class_weights = optimal_weights

        else:  # 'none'
            # Use calculated class weights (formula-based)
            class_weights = self.calculate_class_weights(
                y_train,
                method=self.config.CLASS_WEIGHT_METHOD,
                beta=self.config.CLASS_WEIGHT_BETA
            ) if self.config.USE_CLASS_WEIGHTS else None

        """
        # STEP 1: Apply sampling to training data (NOT validation!)
        X_train_resampled, y_train_resampled = X_train.copy(), y_train.copy()
        if self.config.USE_SAMPLING and IMBLEARN_AVAILABLE:
            X_train_resampled, y_train_resampled = self.imbalance_handler.apply_sampling(
                X_train, y_train
            )
        """

        # Variables to track best model
        best_val_auc = 0
        best_model = None
        best_params = None

        # STEP 2: Optimize hyperparameters with sampled data
        if self.config.USE_OPTUNA:
            print(f"\n  → Optimizing hyperparameters with Optuna ({self.config.N_TRIALS} trials)...")

            def objective(trial):
                nonlocal best_val_auc, best_model, best_params

                params = self.get_optuna_params(trial, model_name)
                model = self.create_model(model_name, params, class_weights)

                # Train
                if model_name == 'catboost' and categorical_features:
                    cat_indices = [X_train.columns.get_loc(c) for c in categorical_features
                                 if c in X_train.columns]
                    model.fit(X_train, y_train, cat_features=cat_indices,
                            verbose=False, eval_set=(X_val, y_val))
                elif model_name in ['lightgbm']:
                    model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
                elif model_name in ['xgboost']:
                    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
                else:
                    model.fit(X_train, y_train)

                # Validate
                val_pred = model.predict_proba(X_val)[:, 1]
                val_auc = roc_auc_score(y_val, val_pred)

                # Track best
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    best_model = model
                    best_params = params

                return val_auc

            study = optuna.create_study(
                direction='maximize',
                sampler=TPESampler(seed=self.config.SEED),
                pruner=HyperbandPruner()
            )
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            study.optimize(objective, n_trials=self.config.N_TRIALS, show_progress_bar=True)

            print(f"    ✓ Best validation AUC: {best_val_auc:.6f}")
            print(f"    ✓ Best params: {study.best_params}")

        else:
            # Use default params - create one for demo
            params = self.get_optuna_params(None, model_name) if self.config.USE_OPTUNA else {}
            best_model = self.create_model(model_name, params, class_weights)

            if model_name == 'catboost' and categorical_features:
                cat_indices = [X_train.columns.get_loc(c) for c in categorical_features
                             if c in X_train.columns]
                best_model.fit(X_train, y_train, cat_features=cat_indices,
                        verbose=False, eval_set=(X_val, y_val))
            elif model_name in ['lightgbm', 'xgboost']:
                best_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            else:
                best_model.fit(X_train, y_train)

            val_pred = best_model.predict_proba(X_val)[:, 1]
            best_val_auc = roc_auc_score(y_val, val_pred)

        # STEP 3: Get predictions
        val_preds = best_model.predict_proba(X_val)[:, 1]
        test_preds = best_model.predict_proba(X_test)[:, 1]

        # STEP 4: THRESHOLD OPTIMIZATION (CRITICAL!)
        if self.config.USE_THRESHOLD_TUNING:
            optimal_threshold = self.imbalance_handler.optimize_threshold(y_val, val_preds)
            self.optimal_thresholds[model_name] = optimal_threshold

            # Predictions with optimal threshold
            val_pred_binary_optimal = self.imbalance_handler.predict_with_threshold(val_preds, optimal_threshold)

            # Calculate comprehensive metrics with optimal threshold
            val_auc = roc_auc_score(y_val, val_preds)
            val_f1_optimal = f1_score(y_val, val_pred_binary_optimal)
            val_precision_optimal = precision_score(y_val, val_pred_binary_optimal)
            val_recall_optimal = recall_score(y_val, val_pred_binary_optimal)
            val_acc_optimal = accuracy_score(y_val, val_pred_binary_optimal)

            # Also show default 0.5 threshold for comparison
            val_pred_binary_default = (val_preds > 0.5).astype(int)
            val_f1_default = f1_score(y_val, val_pred_binary_default)
            val_acc_default = accuracy_score(y_val, val_pred_binary_default)

            print(f"\n  {'─'*70}")
            print(f"  {model_name.upper()} RESULTS:")
            print(f"  {'─'*70}")
            print(f"    Validation AUC: {val_auc:.6f}")
            print(f"\n    OPTIMAL THRESHOLD: {optimal_threshold:.3f} (via {self.config.THRESHOLD_METRIC})")
            print(f"    ├─ F1-Score:       {val_f1_optimal:.6f}")
            print(f"    ├─ Precision:      {val_precision_optimal:.6f}")
            print(f"    ├─ Recall:         {val_recall_optimal:.6f}")
            print(f"    └─ Accuracy:       {val_acc_optimal:.6f}")
            print(f"\n    DEFAULT THRESHOLD: 0.500 (for comparison)")
            print(f"    ├─ F1-Score:       {val_f1_default:.6f}")
            print(f"    └─ Accuracy:       {val_acc_default:.6f}")
            print(f"\n    ⚡ IMPROVEMENT: +{(val_f1_optimal - val_f1_default)*100:.2f}% F1-Score")
            print(f"  {'─'*70}")

            # Confusion matrices
            cm_optimal = confusion_matrix(y_val, val_pred_binary_optimal)
            cm_default = confusion_matrix(y_val, val_pred_binary_default)

            print(f"\n  CONFUSION MATRIX (Optimal Threshold {optimal_threshold:.3f}):")
            print(f"                Predicted Negative    Predicted Positive")
            print(f"    Actual Negative:    {cm_optimal[0,0]:5d}                 {cm_optimal[0,1]:5d}  (FP)")
            print(f"    Actual Positive:    {cm_optimal[1,0]:5d}  (FN)            {cm_optimal[1,1]:5d}")

            print(f"\n  CONFUSION MATRIX (Default Threshold 0.500):")
            print(f"                Predicted Negative    Predicted Positive")
            print(f"    Actual Negative:    {cm_default[0,0]:5d}                 {cm_default[0,1]:5d}  (FP)")
            print(f"    Actual Positive:    {cm_default[1,0]:5d}  (FN)            {cm_default[1,1]:5d}")

        else:
            # Standard results without threshold tuning
            val_acc = accuracy_score(y_val, (val_preds > 0.5).astype(int))

            print(f"\n  {'─'*70}")
            print(f"  {model_name.upper()} Results:")
            print(f"    Validation AUC:      {best_val_auc:.6f}")
            print(f"    Validation Accuracy: {val_acc:.6f}")
            print(f"  {'─'*70}")

            # Confusion matrix
            val_pred_binary = (val_preds > 0.5).astype(int)
            cm = confusion_matrix(y_val, val_pred_binary)
            print(f"\n  Confusion Matrix:")
            print(f"    TN: {cm[0,0]:5d}  FP: {cm[0,1]:5d}")
            print(f"    FN: {cm[1,0]:5d}  TP: {cm[1,1]:5d}")

        # Store results
        self.models[model_name] = best_model
        self.oof_predictions[model_name] = val_preds
        self.test_predictions[model_name] = test_preds

        return {
            'val_auc': best_val_auc,
            'val_acc': val_acc_optimal if self.config.USE_THRESHOLD_TUNING else accuracy_score(y_val, (val_preds > 0.5).astype(int)),
            'val_preds': val_preds,
            'test_preds': test_preds,
            'model': best_model,
            'optimal_threshold': self.optimal_thresholds.get(model_name, 0.5)
        }

    def train_with_cv(self, model_name: str, X: pd.DataFrame, y: pd.Series,
                      X_test: pd.DataFrame, categorical_features: List[str] = None):
        """
        Train model with proper stratified K-fold CV.
        NO DATA LEAKAGE - all preprocessing inside CV loop.
        """
        print(f"\n{'='*70}")
        print(f"  Training {model_name.upper()} with {self.config.N_FOLDS}-Fold CV")
        print(f"{'='*70}")

        # Calculate class weights
        class_weights = self.calculate_class_weights(y) if self.config.USE_CLASS_WEIGHTS else None

        # Initialize OOF predictions
        oof_preds = np.zeros(len(X))
        test_preds = np.zeros(len(X_test))
        fold_scores = []

        # Stratified K-Fold
        skf = StratifiedKFold(n_splits=self.config.N_FOLDS, shuffle=True,
                             random_state=self.config.SEED)

        # Optimize hyperparameters if enabled
        if self.config.USE_OPTUNA:
            print(f"\n  → Optimizing hyperparameters with Optuna ({self.config.N_TRIALS} trials)...")

            def objective(trial):
                params = self.get_optuna_params(trial, model_name)

                # Quick CV to evaluate params
                cv_scores = []
                for train_idx, val_idx in skf.split(X, y):
                    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
                    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

                    model = self.create_model(model_name, params, class_weights)

                    if model_name == 'catboost' and categorical_features:
                        cat_indices = [X.columns.get_loc(c) for c in categorical_features
                                     if c in X.columns]
                        model.fit(X_tr, y_tr, cat_features=cat_indices,
                                verbose=False, eval_set=(X_val, y_val))
                    elif model_name in ['lightgbm']:
                        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])
                    elif model_name in ['xgboost']:
                        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
                    else:
                        model.fit(X_tr, y_tr)

                    val_pred = model.predict_proba(X_val)[:, 1]
                    cv_scores.append(roc_auc_score(y_val, val_pred))

                return np.mean(cv_scores)

            study = optuna.create_study(
                direction='maximize',
                sampler=TPESampler(seed=self.config.SEED),
                pruner=HyperbandPruner()
            )
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            study.optimize(objective, n_trials=self.config.N_TRIALS, show_progress_bar=True)

            best_params = study.best_params
            print(f"    ✓ Best CV Score: {study.best_value:.6f}")
            print(f"    ✓ Best params: {best_params}")
        else:
            # Use default params
            best_params = self.get_optuna_params(None, model_name) if hasattr(self, 'get_default_params') else {}

        # Train with best params using full CV
        print(f"\n  → Training {self.config.N_FOLDS} folds with best parameters...")

        fold_models = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Create and train model
            model = self.create_model(model_name, best_params, class_weights)

            if model_name == 'catboost' and categorical_features:
                cat_indices = [X.columns.get_loc(c) for c in categorical_features
                             if c in X.columns]
                model.fit(X_tr, y_tr, cat_features=cat_indices,
                        verbose=False, eval_set=(X_val, y_val))
            elif model_name in ['lightgbm']:
                model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])
            elif model_name in ['xgboost']:
                model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            else:
                model.fit(X_tr, y_tr)

            # OOF predictions
            oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]

            # Test predictions
            test_preds += model.predict_proba(X_test)[:, 1] / self.config.N_FOLDS

            # Fold score
            fold_score = roc_auc_score(y_val, oof_preds[val_idx])
            fold_scores.append(fold_score)

            #print(f"    Fold {fold}: AUC = {fold_score:.6f}")

            fold_models.append(model)

        # Overall OOF score
        oof_score = roc_auc_score(y, oof_preds)

        print(f"\n  {'─'*70}")
        print(f"  {model_name.upper()} OOF AUC: {oof_score:.6f} (±{np.std(fold_scores):.6f})")
        print(f"  {'─'*70}")

        # Store results
        self.models[model_name] = fold_models
        self.oof_predictions[model_name] = oof_preds
        self.test_predictions[model_name] = test_preds

        return {
            'oof_score': oof_score,
            'fold_scores': fold_scores,
            'oof_preds': oof_preds,
            'test_preds': test_preds
        }


# ═══════════════════════════════════════════════════════════════════════════
# STACKING ENSEMBLE
# ═══════════════════════════════════════════════════════════════════════════

class StackingEnsemble:
    """
    Stacking ensemble with diverse base learners.
    Based on 2024 research showing 95.9% R² with proper diversity.
    """

    def __init__(self, config: ImprovedConfig):
        self.config = config
        self.meta_model = None

    def train_meta_model(self, oof_predictions: Dict, y: pd.Series):
        """Train meta-learner on OOF predictions"""
        print(f"\n{'='*70}")
        print("  TRAINING STACKING META-LEARNER")
        print(f"{'='*70}")

        # Create meta-features from OOF predictions
        meta_features = pd.DataFrame(oof_predictions)

        print(f"\n  Meta-features shape: {meta_features.shape}")
        print(f"  Base models: {list(oof_predictions.keys())}")

        # Train meta-model (Logistic Regression with regularization)
        self.meta_model = LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=self.config.SEED
        )

        self.meta_model.fit(meta_features, y)

        # Meta-model predictions
        meta_preds = self.meta_model.predict_proba(meta_features)[:, 1]
        meta_score = roc_auc_score(y, meta_preds)

        print(f"\n  Meta-model coefficients:")
        for model_name, coef in zip(oof_predictions.keys(), self.meta_model.coef_[0]):
            print(f"    {model_name:15s}: {coef:+.4f}")

        print(f"\n  Stacking Ensemble OOF AUC: {meta_score:.6f}")
        print(f"{'='*70}")

        return meta_score

    def predict(self, test_predictions: Dict) -> np.ndarray:
        """Make predictions on test set"""
        meta_features = pd.DataFrame(test_predictions)
        return self.meta_model.predict_proba(meta_features)[:, 1]


config = ImprovedConfig()
os.makedirs(config.MODELS_DIR, exist_ok=True)
np.random.seed(config.SEED)

print(f"[1/10] Configuration")
print(f"  → Seed: {config.SEED}")
print(f"  → Mode: FAST - Single 70/30 Split")
print(f"  → Models: {', '.join(config.MODELS_TO_USE)}")
print(f"  → Optuna trials: {config.N_TRIALS}")
print(f"\n  ADVANCED IMBALANCED LEARNING (2024-2025 SOTA):")
print(f"  ├─ Class Weights: {config.USE_CLASS_WEIGHTS}")
if config.USE_CLASS_WEIGHTS:
    print(f"  │  ├─ Method: {config.CLASS_WEIGHT_METHOD}")
    if config.CLASS_WEIGHT_METHOD == 'effective_num':
        print(f"  │  └─ Beta: {config.CLASS_WEIGHT_BETA}")
print(f"  ├─ Sampling: {config.USE_SAMPLING}")
if config.USE_SAMPLING:
    print(f"  │  ├─ Strategy: {config.SAMPLING_STRATEGY}")
    if config.SAMPLING_STRATEGY == 'oversample':
        print(f"  │  └─ Method: {config.SAMPLING_METHOD}")
    elif config.SAMPLING_STRATEGY == 'hybrid':
        print(f"  │  └─ Method: {config.HYBRID_METHOD}")
print(f"  └─ Threshold Tuning: {config.USE_THRESHOLD_TUNING}")
if config.USE_THRESHOLD_TUNING:
    print(f"     └─ Metric: {config.THRESHOLD_METRIC}")


train_df = pd.read_csv(config.TRAIN_PATH)
test_df = pd.read_csv(config.TEST_PATH)

print(f"  → Train shape: {train_df.shape}")
print(f"  → Test shape: {test_df.shape}")

target_dist = train_df[config.TARGET].value_counts(normalize=True)
print(f"  → Target: Class 0={target_dist[0]:.2%}, Class 1={target_dist[1]:.2%}")

# Store IDs and separate target
test_ids = test_df[config.ID_COL].copy()
train_df = train_df.drop(columns=[config.ID_COL])
test_df = test_df.drop(columns=[config.ID_COL])

y = train_df[config.TARGET].copy()
X = train_df.drop(columns=[config.TARGET])
X_test = test_df.copy()


X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    train_size=0.70,
    random_state=config.SEED,
    stratify=y
)

print(f"  → Train: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
print(f"  → Val:   {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
print(f"  → Test:  {len(X_test)} samples")


# Label encode categoricals
label_encoders = {}
for col in config.CATEGORICAL_FEATURES:
    if col in X_train.columns:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col].astype(str))
        X_val[col] = X_val[col].astype(str).apply(
            lambda x: le.transform([x])[0] if x in le.classes_ else -1
        )
        X_test[col] = X_test[col].astype(str).apply(
            lambda x: le.transform([x])[0] if x in le.classes_ else -1
        )
        label_encoders[col] = le

# Handle missing values - fit on train only
train_medians = X_train.median()
X_train = X_train.fillna(train_medians)
X_val = X_val.fillna(train_medians)
X_test = X_test.fillna(train_medians)

print(f"  ✓ Preprocessing complete")


engineer = MedicalFeatureEngineer(config)
#X_train = engineer.create_features(X_train)
#X_val = engineer.create_features(X_val)
#X_test = engineer.create_features(X_test)

# Ensure all datasets have same columns
for df_name, df in [('val', X_val), ('test', X_test)]:
    missing_cols = set(X_train.columns) - set(df.columns)
    for col in missing_cols:
        df[col] = 0

X_val = X_val[X_train.columns]
X_test = X_test[X_train.columns]

# Fill any NaNs
X_train = X_train.fillna(0)
X_val = X_val.fillna(0)
X_test = X_test.fillna(0)

print(f"  → Features: {X_train.shape[1]}")


target_encoder = KFoldTargetEncoder(
    categorical_features=config.CATEGORICAL_FEATURES,
    n_splits=config.N_FOLDS,
    seed=config.SEED
)

X_train = target_encoder.fit_transform(X_train, y_train)
X_val = target_encoder.transform(X_val)
X_test = target_encoder.transform(X_test)

print(f"  ✓ Target encoding complete")
print(f"  → Features after encoding: {X_train.shape[1]}")


scaler = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=X_train.columns,
    index=X_train.index
)
X_val_scaled = pd.DataFrame(
    scaler.transform(X_val),
    columns=X_val.columns,
    index=X_val.index
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    columns=X_test.columns,
    index=X_test.index
)

print(f"  ✓ Transformation complete")


selector = ShapFeatureSelector(
    n_estimators=100,
    threshold=0.001,
    max_features=60
)

#X_train_selected = selector.fit_transform(X_train_scaled, y_train)
#X_val_selected = selector.transform(X_val_scaled)
#X_test_selected = selector.transform(X_test_scaled)

#print(f"  ✓ Feature selection complete")
#print(f"  → Selected features: {X_train_selected.shape[1]}")

X_train_selected = X_train_scaled
X_val_selected = X_val_scaled
X_test_selected = X_test_scaled
print(f"  → Features: {X_train_selected.shape[1]}")


imbalance_handler = ImbalancedLearningHandler(config)
#X_train_resampled, y_train_resampled = imbalance_handler.apply_sampling(
#    X_train_selected, y_train)

X_train_resampled = X_train_selected
y_train_resampled = y_train


trainer = ImprovedModelTrainer(config)

results = {}
for model_name in config.MODELS_TO_USE:
    result = trainer.train_with_single_split(
        model_name=model_name,
        X_train=X_train_resampled,
        y_train=y_train_resampled,
        X_val=X_val_selected,
        y_val=y_val,
        X_test=X_test_selected,
        categorical_features=None  # Already encoded
    )
    results[model_name] = result


# Find best model
best_model_name = max(results.items(), key=lambda x: x[1]['val_auc'])[0]
best_result = results[best_model_name]

print(f"\n{'='*70}")
print("MODEL COMPARISON")
print(f"{'='*70}")

for model_name, result in sorted(results.items(), key=lambda x: x[1]['val_auc'], reverse=True):
    marker = " ⭐ BEST" if model_name == best_model_name else ""
    print(f"  {model_name.upper():12s} | AUC: {result['val_auc']:.6f} | "
          f"Acc: {result['val_acc']:.6f}{marker}")

print(f"\n{'='*70}")
print(f"BEST MODEL: {best_model_name.upper()}")
print(f"{'='*70}")
print(f"  Validation AUC:      {best_result['val_auc']:.6f}")
print(f"  Validation Accuracy: {best_result['val_acc']:.6f}")


final_test_preds = best_result['test_preds']

submission = pd.DataFrame({
    config.ID_COL: test_ids,
    config.TARGET: final_test_preds
})

submission.to_csv(config.SUBMISSION_PATH, index=False)

print(f"\n  → Submission saved to: {config.SUBMISSION_PATH}")
print(f"\n  Prediction Statistics:")
print(f"    Mean: {final_test_preds.mean():.6f}")
print(f"    Std:  {final_test_preds.std():.6f}")
print(f"    Min:  {final_test_preds.min():.6f}")
print(f"    Max:  {final_test_preds.max():.6f}")

