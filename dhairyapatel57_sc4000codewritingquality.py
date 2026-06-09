# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from collections import Counter
from typing import List, Dict, Tuple
from lightgbm import LGBMRegressor
import lightgbm as lgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import xgboost as xgb
import optuna
from optuna.samplers import TPESampler
import re
from scipy.stats import skew
from catboost import CatBoostRegressor
import warnings
import statistics
warnings.filterwarnings('ignore')



# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# ============================================================================
# CONSTANTS AND CONFIGURATION
# ============================================================================

class FeatureConfig:
    """Configuration class for feature extraction parameters."""
    
    # Column definitions
    NUMERICAL_FEATURES = ['down_time', 'up_time', 'action_time', 'cursor_position', 'word_count']
    ACTIVITY_CATEGORIES = ['Input', 'Remove/Cut', 'Nonproduction', 'Replace', 'Paste']
    
    # Event types to track
    KEY_EVENTS = [
        'q', 'Space', 'Backspace', 'Shift', 'ArrowRight', 'Leftclick', 'ArrowLeft',
        '.', ',', 'ArrowDown', 'ArrowUp', 'Enter', 'CapsLock', "'", 'Delete', 'Unidentified'
    ]
    
    # Text change patterns
    TEXT_PATTERNS = ['q', ' ', '.', ',', '\n', "'", '"', '-', '?', ';', '=', '/', '\\', ':']
    
    # Statistical aggregations
    @staticmethod
    def get_aggregation_functions():
        """Return list of aggregation functions with proper naming."""
        return [
            'count', 'mean', 'min', 'max', 'first', 'last',
            lambda x: x.quantile(0.25), 'median',
            lambda x: x.quantile(0.75), 'sum'
        ]


# ============================================================================
# PART 1: BEHAVIORAL KEYSTROKE FEATURES
# ============================================================================

class KeystrokeFeatureExtractor:
    """Extracts behavioral features from keystroke logs using Polars."""
    
    def __init__(self, config: FeatureConfig):
        self.config = config
    
    def extract_all(self, logs_df: pl.LazyFrame) -> pd.DataFrame:
        """
        Main method to extract all keystroke features.
        
        Args:
            logs_df: Polars LazyFrame with keystroke logs
            
        Returns:
            DataFrame with extracted features
        """
        print("Extracting keystroke behavioral features...")
        
        # Initialize with unique IDs
        student_ids = logs_df.select(pl.col('id').unique(maintain_order=True))
        
        # Extract each feature group
        features_df = student_ids
        features_df = self._add_event_frequency_features(logs_df, features_df)
        features_df = self._add_input_word_statistics(logs_df, features_df)
        features_df = self._add_temporal_statistics(logs_df, features_df)
        features_df = self._add_diversity_metrics(logs_df, features_df)
        features_df = self._add_pause_analysis(logs_df, features_df)
        features_df = self._add_burst_patterns(logs_df, features_df)
        
        return features_df.collect().to_pandas()
    
    def _add_event_frequency_features(self, logs_df: pl.LazyFrame, 
                                     features_df: pl.LazyFrame) -> pl.LazyFrame:
        """Count frequency of specific events, activities, and text changes."""
        print("  â†’ Event frequencies")
        
        # Activity frequencies
        for idx, activity in enumerate(self.config.ACTIVITY_CATEGORIES):
            activity_counts = logs_df.group_by('id').agg(
                pl.col('activity').eq(activity).sum().alias(f'activity_{idx}_cnt')
            )
            features_df = features_df.join(activity_counts, on='id', how='left')
        
        # Text change frequencies
        for idx, pattern in enumerate(self.config.TEXT_PATTERNS):
            text_counts = logs_df.group_by('id').agg(
                pl.col('text_change').eq(pattern).sum().alias(f'text_change_{idx}_cnt')
            )
            features_df = features_df.join(text_counts, on='id', how='left')
        
        # Key event frequencies (down and up)
        for event_type in ['down_event', 'up_event']:
            for idx, event in enumerate(self.config.KEY_EVENTS):
                event_counts = logs_df.group_by('id').agg(
                    pl.col(event_type).eq(event).sum().alias(f'{event_type}_{idx}_cnt')
                )
                features_df = features_df.join(event_counts, on='id', how='left')
        
        return features_df
    
    def _add_input_word_statistics(self, logs_df: pl.LazyFrame, features_df: pl.LazyFrame) -> pl.LazyFrame:
        """Extract statistics about input word patterns using a fully Polars-only pipeline."""
    
        print("  â†’ Input word statistics")
    
        # 1. Filter for genuine text inputs (ignore "a => b", "NoChange")
        text_input = logs_df.filter(
            (~pl.col("text_change").str.contains("=>")) &
            (pl.col("text_change") != "NoChange")
        )
    
        # 2. Extract characters actually typed: keep only single-word-building characters
        # Accept letters, digits, apostrophes, hyphens
        cleaned = text_input.with_columns([
            pl.col("text_change")
            .str.extract(r"([A-Za-z0-9'-]+)", group_index=1)
            .alias("token")
        ]).filter(pl.col("token").is_not_null())
    
        # 3. Group tokens into word lists per id
        # We treat each keystroke token as part of a word if it is alpha-numeric
        token_lists = cleaned.group_by("id").agg(
            pl.col("token").alias("tokens")
        )
    
        # 4. Compute word lengths using Polars arr.eval
        token_lists = token_lists.with_columns([
            pl.col("tokens").arr.eval(pl.element().str.len()).alias("lengths")
        ])
    
        # 5. Compute statistics
        word_stats = token_lists.with_columns([
            pl.col("tokens").list.len().alias("input_word_count"),
    
            pl.col("lengths").list.mean().alias("input_word_length_mean"),
            pl.col("lengths").list.max().alias("input_word_length_max"),
            pl.col("lengths").list.std().alias("input_word_length_std"),
            pl.col("lengths").list.median().alias("input_word_length_median"),
    
            # Skewness: (mean - median) / std
            (
                (pl.col("lengths").list.mean() - pl.col("lengths").list.median()) /
                (pl.col("lengths").list.std() + 1e-6)
            ).alias("input_word_length_skew")
        ]).drop(["tokens", "lengths"])
    
        return features_df.join(word_stats, on="id", how="left")

    
    def _add_temporal_statistics(self, logs_df: pl.LazyFrame, 
                                 features_df: pl.LazyFrame) -> pl.LazyFrame:
        """Compute statistics on temporal numerical columns."""
        print("  â†’ Temporal statistics")
        
        # Aggregate numerical columns with multiple statistics
        aggregations = []
        
        # Special case: action_time gets sum
        aggregations.append(pl.sum('action_time').alias('action_time_sum'))
        
        # For all numerical columns: mean, std, median, min, max, quantile
        for col in self.config.NUMERICAL_FEATURES:
            aggregations.extend([
                pl.mean(col).alias(f'{col}_mean'),
                pl.std(col).alias(f'{col}_std'),
                pl.median(col).alias(f'{col}_median'),
                pl.min(col).alias(f'{col}_min'),
                pl.max(col).alias(f'{col}_max'),
                pl.col(col).quantile(0.5).alias(f'{col}_quantile')
            ])
        
        temporal_stats = logs_df.group_by('id').agg(aggregations)
        return features_df.join(temporal_stats, on='id', how='left')
    
    def _add_diversity_metrics(self, logs_df: pl.LazyFrame, 
                              features_df: pl.LazyFrame) -> pl.LazyFrame:
        """Calculate diversity (unique count) of categorical variables."""
        print("  â†’ Diversity metrics")
        
        diversity_stats = logs_df.group_by('id').agg([
            pl.n_unique('activity').alias('activity'),
            pl.n_unique('down_event').alias('down_event'),
            pl.n_unique('up_event').alias('up_event'),
            pl.n_unique('text_change').alias('text_change')
        ])
        
        return features_df.join(diversity_stats, on='id', how='left')
    
    def _add_pause_analysis(self, logs_df: pl.LazyFrame, 
                           features_df: pl.LazyFrame) -> pl.LazyFrame:
        """Analyze pause patterns between keystrokes."""
        print("  â†’ Pause patterns")
        
        # Calculate inter-keystroke intervals
        pause_data = logs_df.with_columns(
            pl.col('up_time').shift(1).over('id').alias('prev_up_time')
        ).with_columns(
            ((pl.col('down_time') - pl.col('prev_up_time')).abs() / 1000)
            .fill_null(0)
            .alias('interval_seconds')
        ).filter(
            pl.col('activity').is_in(['Input', 'Remove/Cut'])
        )
        
        # Aggregate pause statistics
        pause_stats = pause_data.group_by('id').agg([
            pl.max('interval_seconds').alias('inter_key_largest_lantency'),
            pl.median('interval_seconds').alias('inter_key_median_lantency'),
            pl.mean('interval_seconds').alias('mean_pause_time'),
            pl.std('interval_seconds').alias('std_pause_time'),
            pl.sum('interval_seconds').alias('total_pause_time'),
            # Categorized pauses
            pl.col('interval_seconds').filter(
                (pl.col('interval_seconds') > 0.5) & (pl.col('interval_seconds') < 1)
            ).count().alias('pauses_half_sec'),
            pl.col('interval_seconds').filter(
                (pl.col('interval_seconds') > 1) & (pl.col('interval_seconds') < 1.5)
            ).count().alias('pauses_1_sec'),
            pl.col('interval_seconds').filter(
                (pl.col('interval_seconds') > 1.5) & (pl.col('interval_seconds') < 2)
            ).count().alias('pauses_1_half_sec'),
            pl.col('interval_seconds').filter(
                (pl.col('interval_seconds') > 2) & (pl.col('interval_seconds') < 3)
            ).count().alias('pauses_2_sec'),
            pl.col('interval_seconds').filter(
                pl.col('interval_seconds') > 3
            ).count().alias('pauses_3_sec')
        ])
        
        return features_df.join(pause_stats, on='id', how='left')
    
    def _add_burst_patterns(self, logs_df: pl.LazyFrame, 
                           features_df: pl.LazyFrame) -> pl.LazyFrame:
        """Identify and characterize keystroke bursts."""
        print("  â†’ Burst patterns (P-bursts and R-bursts)")
        
        # P-bursts: rapid keystroke sequences
        features_df = self._compute_p_bursts(logs_df, features_df)
        
        # R-bursts: removal/deletion sequences
        features_df = self._compute_r_bursts(logs_df, features_df)
        
        return features_df
    
    def _compute_p_bursts(self, logs_df: pl.LazyFrame, 
                      features_df: pl.LazyFrame) -> pl.LazyFrame:
        """Compute production burst statistics (rapid keystrokes < 2s apart)."""

        # Step 1: Add prev_up_time, interval, and is_rapid
        burst_base = (
            logs_df
            .with_columns(
                pl.col('up_time').shift(1).over('id').alias('prev_up_time')
            )
            .with_columns(
                ((pl.col('down_time') - pl.col('prev_up_time')).abs() / 1000)
                .fill_null(0)
                .alias('interval')
            )
            .filter(
                pl.col('activity').is_in(['Input', 'Remove/Cut'])
            )
            .with_columns(
                (pl.col('interval') < 2).alias('is_rapid')
            )
            .with_columns(
                pl.col('is_rapid').rle_id().alias('grp')
            )
        )
    
        # Step 2: Compute burst lengths
        burst_groups = (
            burst_base
            .filter(pl.col('is_rapid'))
            .group_by(['id', 'grp'])
            .agg(pl.len().alias('P-bursts'))
        )
    
        # Step 3: Aggregate per user id
        p_burst_stats = (
            burst_groups
            .group_by('id')
            .agg([
                pl.mean('P-bursts').alias('P-bursts_mean'),
                pl.std('P-bursts').alias('P-bursts_std'),
                pl.count('P-bursts').alias('P-bursts_count'),
                pl.median('P-bursts').alias('P-bursts_median'),
                pl.max('P-bursts').alias('P-bursts_max'),
                pl.first('P-bursts').alias('P-bursts_first'),
                pl.last('P-bursts').alias('P-bursts_last')
            ])
        )
    
        return features_df.join(p_burst_stats, on='id', how='left')

    def _compute_r_bursts(self, logs_df: pl.LazyFrame, 
                      features_df: pl.LazyFrame) -> pl.LazyFrame:
        """Compute removal burst statistics (consecutive deletions)."""
    
        # Step 1: Filter and mark removal events
        removal_base = (
            logs_df
            .filter(pl.col('activity').is_in(['Input', 'Remove/Cut']))
            .with_columns(
                pl.col('activity').eq('Remove/Cut').alias('is_removal')
            )
            .with_columns(
                pl.col('is_removal').rle_id().alias('grp')
            )
        )
    
        # Step 2: Compute removal burst lengths (consecutive deletions)
        r_bursts = (
            removal_base
            .filter(pl.col('is_removal'))
            .group_by(['id', 'grp'])
            .agg(pl.len().alias('R-bursts'))
        )
    
        # Step 3: Aggregate per user
        r_burst_stats = (
            r_bursts
            .group_by('id')
            .agg([
                pl.mean('R-bursts').alias('R-bursts_mean'),
                pl.std('R-bursts').alias('R-bursts_std'),
                pl.median('R-bursts').alias('R-bursts_median'),
                pl.max('R-bursts').alias('R-bursts_max'),
                pl.first('R-bursts').alias('R-bursts_first'),
                pl.last('R-bursts').alias('R-bursts_last')
            ])
        )
    
        return features_df.join(r_burst_stats, on='id', how='left')

    


# ============================================================================
# PART 2: ESSAY RECONSTRUCTION ENGINE
# ============================================================================

class EssayReconstructor:
    """Reconstructs final essay text from keystroke event logs."""
    
    @staticmethod
    def reconstruct_from_logs(event_sequence: pd.DataFrame) -> str:
        """
        Reconstruct essay from sequence of editing events.
        
        Args:
            event_sequence: DataFrame with columns [activity, cursor_position, text_change]
            
        Returns:
            Reconstructed essay text
        """
        essay = ""
        
        for _, event in event_sequence.iterrows():
            action_type = event['activity']
            cursor_pos = int(event['cursor_position'])
            content = str(event['text_change'])
            
            try:
                if action_type == 'Replace':
                    essay = EssayReconstructor._handle_replace(essay, cursor_pos, content)
                elif action_type == 'Paste':
                    essay = EssayReconstructor._handle_paste(essay, cursor_pos, content)
                elif action_type == 'Remove/Cut':
                    essay = EssayReconstructor._handle_removal(essay, cursor_pos, content)
                elif action_type.startswith('Move From'):
                    essay = EssayReconstructor._handle_move(essay, action_type)
                else:
                    essay = EssayReconstructor._handle_input(essay, cursor_pos, content)
            except Exception:
                continue
        
        return essay
    
    @staticmethod
    def _handle_replace(text: str, pos: int, content: str) -> str:
        """Handle text replacement operation."""
        if ' => ' not in content:
            return text
        old_text, new_text = content.split(' => ', 1)
        start_pos = pos - len(new_text)
        end_pos = start_pos + len(old_text)
        return text[:start_pos] + new_text + text[end_pos:]
    
    @staticmethod
    def _handle_paste(text: str, pos: int, content: str) -> str:
        """Handle paste operation."""
        insertion_point = pos - len(content)
        return text[:insertion_point] + content + text[insertion_point:]
    
    @staticmethod
    def _handle_removal(text: str, pos: int, content: str) -> str:
        """Handle deletion operation."""
        return text[:pos] + text[pos + len(content):]
    
    @staticmethod
    def _handle_move(text: str, move_description: str) -> str:
        """Handle text move operation."""
        # Parse move coordinates from description
        match = re.search(r'\((\d+), (\d+)\) To \((\d+), (\d+)\)', move_description)
        if not match:
            return text
        
        from_start, from_end, to_start, to_end = map(int, match.groups())
        
        if from_start == to_start:
            return text
        
        # Extract the segment to move
        moving_text = text[from_start:from_end]
        text_without_segment = text[:from_start] + text[from_end:]
        
        # Insert at new position
        if from_start < to_start:
            adjusted_pos = to_start - (from_end - from_start)
            return text_without_segment[:adjusted_pos] + moving_text + text_without_segment[adjusted_pos:]
        else:
            return text_without_segment[:to_start] + moving_text + text_without_segment[to_start:]
    
    @staticmethod
    def _handle_input(text: str, pos: int, content: str) -> str:
        """Handle regular text input."""
        insertion_point = max(0, pos - len(content))
        return text[:insertion_point] + content + text[insertion_point:]
    
    @staticmethod
    def reconstruct_all_essays(logs_df: pd.DataFrame) -> pd.DataFrame:
        """Reconstruct essays for all students in the logs."""
        # Filter out non-productive actions
        productive_logs = logs_df[logs_df['activity'] != 'Nonproduction']
        
        essays = []
        for student_id in logs_df['id'].unique():
            student_data = productive_logs[productive_logs['id'] == student_id]
            essay_text = EssayReconstructor.reconstruct_from_logs(
                student_data[['activity', 'cursor_position', 'text_change']]
            )
            essays.append({'id': student_id, 'essay': essay_text})
        
        return pd.DataFrame(essays)


# ============================================================================
# PART 3: LINGUISTIC FEATURE EXTRACTION
# ============================================================================

class LinguisticFeatureExtractor:
    """Extracts linguistic features from reconstructed essays."""
    
    def __init__(self):
        self.agg_funcs = FeatureConfig.get_aggregation_functions()
    
    def extract_all(self, essay_df: pd.DataFrame) -> pd.DataFrame:
        """Extract all linguistic features from essays."""
        print("Extracting linguistic features from essays...")
        
        features = essay_df[['id']].copy()
        
        print("  â†’ Word-level features")
        word_features = self._extract_word_features(essay_df.copy())
        features = features.merge(word_features, on='id', how='left')
        
        print("  â†’ Sentence-level features")
        sentence_features = self._extract_sentence_features(essay_df.copy())
        features = features.merge(sentence_features, on='id', how='left')
        
        print("  â†’ Paragraph-level features")
        paragraph_features = self._extract_paragraph_features(essay_df.copy())
        features = features.merge(paragraph_features, on='id', how='left')
        
        print("  â†’ Character-level features")
        character_features = self._extract_character_features(essay_df.copy())
        features = features.merge(character_features, on='id', how='left')
        
        return features
    
    def _extract_word_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract statistical features about word lengths."""
        # Split essays into words
        df['words'] = df['essay'].apply(
            lambda text: [w for w in re.split(r'[ \n.?!]', text) if w]
        )
        df = df.explode('words')
        df['word_length'] = df['words'].str.len()
        df = df[df['word_length'] > 0]
        
        # Aggregate word length statistics
        word_stats = df.groupby('id')['word_length'].agg(self.agg_funcs)
        
        # Properly name columns
        new_cols = []
        for i, col in enumerate(word_stats.columns):
            if callable(col):
                if i == 6:  # First lambda (q1)
                    new_cols.append('word_len_q1')
                elif i == 8:  # Second lambda (q3)
                    new_cols.append('word_len_q3')
                else:
                    new_cols.append(f'word_len_{col.__name__}')
            else:
                new_cols.append(f'word_len_{col}')
        
        word_stats.columns = new_cols
        return word_stats.reset_index()
    
    def _extract_sentence_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract statistical features about sentences."""
        # Split essays into sentences
        df['sentences'] = df['essay'].apply(lambda text: re.split(r'[.?!]', text))
        df = df.explode('sentences')
        df['sentences'] = df['sentences'].str.replace('\n', '').str.strip()
        df = df[df['sentences'].str.len() > 0]
        
        # Calculate sentence metrics
        df['sentence_length'] = df['sentences'].str.len()
        df['sentence_word_count'] = df['sentences'].apply(lambda s: len(s.split()))
        
        # Aggregate sentence statistics
        sent_len_stats = df.groupby('id')['sentence_length'].agg(self.agg_funcs)
        sent_word_stats = df.groupby('id')['sentence_word_count'].agg(self.agg_funcs)
        
        # Combine and rename
        sent_stats = pd.concat([sent_len_stats, sent_word_stats], axis=1)
        sent_stats.columns = self._generate_column_names('sent', sent_len_stats, sent_word_stats)
        
        # Rename count column and drop duplicate
        if 'sent_word_count_count' in sent_stats.columns:
            sent_stats = sent_stats.drop(columns=['sent_word_count_count'])
        sent_stats = sent_stats.rename(columns={'sent_len_count': 'sent_count'})
        
        return sent_stats.reset_index()
    
    def _extract_paragraph_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract statistical features about paragraphs."""
        # Split essays into paragraphs
        df['paragraphs'] = df['essay'].apply(lambda text: text.split('\n'))
        df = df.explode('paragraphs')
        df = df[df['paragraphs'].str.len() > 0]
        
        # Calculate paragraph metrics
        df['para_length'] = df['paragraphs'].str.len()
        df['para_word_count'] = df['paragraphs'].apply(lambda p: len(p.split()))
        
        # Aggregate paragraph statistics
        para_len_stats = df.groupby('id')['para_length'].agg(self.agg_funcs)
        para_word_stats = df.groupby('id')['para_word_count'].agg(self.agg_funcs)
        
        # Combine and rename
        para_stats = pd.concat([para_len_stats, para_word_stats], axis=1)
        para_stats.columns = self._generate_column_names('paragraph', para_len_stats, para_word_stats)
        
        # Clean up column names
        if 'paragraph_word_count_count' in para_stats.columns:
            para_stats = para_stats.drop(columns=['paragraph_word_count_count'])
        para_stats = para_stats.rename(columns={'paragraph_len_count': 'paragraph_count'})
        
        return para_stats.reset_index()
    
    def _extract_character_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract character-level and pattern features."""
        features = df[['id']].copy()
        
        # Basic character counts
        features['no_char'] = df['essay'].str.count(r'q')
        features['no_comma'] = df['essay'].str.count(',')
        features['no_quotes'] = df['essay'].str.count('"') + df['essay'].str.count("'")
        features['no_spaces'] = df['essay'].str.count(' ')
        features['no_mul_spaces'] = df['essay'].str.count(r' {2,}')
        features['no_dot'] = df['essay'].str.count(r'\.')
        features['no_exclamation'] = df['essay'].str.count('!')
        features['no_question'] = df['essay'].str.count(r'\?')
        features['no_semicolon'] = df['essay'].str.count(':')
        features['no_hyphen'] = df['essay'].str.count('-')
        features['no_line_break'] = df['essay'].str.count(r'\n')
        
        # Derived counts
        features['no_spec_char'] = df['essay'].str.len() - features['no_char']
        features['no_dot_space'] = np.log1p(features['no_dot']) * np.log1p(features['no_spaces'])
        features['no_typos'] = df['essay'].fillna('').str.count(r' ,| \.|,[^\s]|\.[^\s\.\)\"]')
        
        # Probability features
        essay_length = df['essay'].str.len().replace(0, 1)
        features['prob_comma'] = features['no_comma'] / essay_length
        features['prob_spec_char'] = features['no_spec_char'] / essay_length
        features['prob_spaces'] = features['no_spaces'] / essay_length
        
        # Word-length pattern features (critical for matching silver-bullet)
        df_clean = df['essay'].str.replace("'", "", regex=False)
        for word_len in range(1, 11):
            pattern = ' ' + 'q' * word_len + ' '
            features[f'no_q{word_len}'] = df_clean.str.count(re.escape(pattern))
        
        return features
    
    def _generate_column_names(self, prefix, stats1, stats2):
        """Generate properly formatted column names for aggregated features."""
        cols = []
        for i, col in enumerate(stats1.columns):
            if callable(col):
                if i == 6:  # First lambda (q1)
                    name = 'q1'
                elif i == 8:  # Second lambda (q3)
                    name = 'q3'
                else:
                    name = col.__name__
            else:
                name = str(col)
            cols.append(f'{prefix}_len_{name}')
        
        for i, col in enumerate(stats2.columns):
            if callable(col):
                if i == 6:  # First lambda (q1)
                    name = 'q1'
                elif i == 8:  # Second lambda (q3)
                    name = 'q3'
                else:
                    name = col.__name__
            else:
                name = str(col)
            cols.append(f'{prefix}_word_count_{name}')
        
        return cols


# ============================================================================
# PART 4: EFFICIENCY AND PERFORMANCE METRICS
# ============================================================================

class EfficiencyMetricsCalculator:
    """Calculates writing efficiency and performance metrics."""
    
    @staticmethod
    def calculate_all_metrics(logs_df: pd.DataFrame, 
                              essay_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all efficiency metrics."""
        print("Calculating efficiency metrics...")
        
        # Typing speed
        speed_metrics = EfficiencyMetricsCalculator._calculate_typing_speed(logs_df)
        
        # Production efficiency
        efficiency_metrics = EfficiencyMetricsCalculator._calculate_production_efficiency(
            logs_df, essay_df
        )
        
        return speed_metrics.merge(efficiency_metrics, on='id', how='outer')
    
    @staticmethod
    def _calculate_typing_speed(logs_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate keys pressed per second."""
        input_events = logs_df[logs_df['activity'].isin(['Input', 'Remove/Cut'])]
        
        speed_data = input_events.groupby('id').agg({
            'event_id': 'count',
            'down_time': 'min',
            'up_time': 'max'
        }).reset_index()
        
        speed_data['time_span_seconds'] = (
            (speed_data['up_time'] - speed_data['down_time']) / 1000
        )
        speed_data['keys_per_second'] = (
            speed_data['event_id'] / speed_data['time_span_seconds'].replace(0, 1)
        )
        
        return speed_data[['id', 'keys_per_second']]
    
    @staticmethod
    def _calculate_production_efficiency(logs_df: pd.DataFrame, 
                                        essay_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate ratio of final essay length to total keystrokes."""
        # Count productive keystrokes
        keystroke_counts = logs_df[
            logs_df['activity'].isin(['Input', 'Remove/Cut'])
        ].groupby('id').size().reset_index(name='total_keystrokes')
        
        # Get essay lengths
        essay_lengths = essay_df.copy()
        essay_lengths['essay_length'] = essay_lengths['essay'].str.len()
        
        # Calculate efficiency ratio
        efficiency = essay_lengths.merge(keystroke_counts, on='id', how='left')
        efficiency['product_to_keys'] = (
            efficiency['essay_length'] / efficiency['total_keystrokes'].replace(0, 1)
        )
        
        return efficiency[['id', 'product_to_keys']]


# ============================================================================
# MAIN PIPELINE ORCHESTRATOR
# ============================================================================

class WritingProcessFeaturePipeline:
    """Main pipeline for extracting all features from writing process data."""
    
    def __init__(self):
        self.config = FeatureConfig()
        self.keystroke_extractor = KeystrokeFeatureExtractor(self.config)
        self.essay_reconstructor = EssayReconstructor()
        self.linguistic_extractor = LinguisticFeatureExtractor()
        self.efficiency_calculator = EfficiencyMetricsCalculator()
    
    def extract_features(self, logs_csv_path: str) -> pd.DataFrame:
        """
        Complete feature extraction pipeline.
        
        Args:
            logs_csv_path: Path to keystroke logs CSV file
            
        Returns:
            DataFrame with all extracted features
        """
        print(f"\n{'='*80}")
        print("WRITING PROCESS FEATURE EXTRACTION PIPELINE")
        print(f"{'='*80}\n")
        
        # Step 1: Load data
        print("ğŸ“‚ Loading keystroke logs...")
        logs_lazy = pl.scan_csv(logs_csv_path)
        logs_pandas = logs_lazy.collect().to_pandas()
        print(f"   Loaded {len(logs_pandas):,} keystroke events")
        
        # Step 2: Extract keystroke behavioral features
        print("\nğŸ�¹ Extracting behavioral keystroke features...")
        behavioral_features = self.keystroke_extractor.extract_all(logs_lazy)
        print(f"   Extracted {behavioral_features.shape[1]-1} behavioral features")
        
        # Step 3: Reconstruct essays
        print("\nğŸ“� Reconstructing essays from keystroke logs...")
        essay_df = self.essay_reconstructor.reconstruct_all_essays(logs_pandas)
        print(f"   Reconstructed {len(essay_df)} essays")
        
        # Step 4: Extract linguistic features
        print("\nğŸ“Š Extracting linguistic features...")
        linguistic_features = self.linguistic_extractor.extract_all(essay_df)
        print(f"   Extracted {linguistic_features.shape[1]-1} linguistic features")
        
        # Step 5: Calculate efficiency metrics
        print("\nâš¡ Calculating efficiency metrics...")
        efficiency_features = self.efficiency_calculator.calculate_all_metrics(
            logs_pandas, essay_df
        )
        print(f"   Calculated {efficiency_features.shape[1]-1} efficiency metrics")
        
        # Step 6: Combine all features
        print("\nğŸ”— Combining all feature sets...")
        all_features = behavioral_features.merge(linguistic_features, on='id', how='left')
        all_features = all_features.merge(efficiency_features, on='id', how='left')
        all_features = all_features.fillna(0)
        
        print(f"\n{'='*80}")
        print("âœ… FEATURE EXTRACTION COMPLETE")
        print(f"   Total features: {all_features.shape[1] - 1}")
        print(f"   Total samples: {all_features.shape[0]}")
        print(f"{'='*80}\n")
        
        return all_features


# ============================================================================
# CONVENIENCE FUNCTION FOR DIRECT USE
# ============================================================================

def extract_writing_features(train_logs_path: str, 
                            test_logs_path: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience function to extract features from train and test logs.
    
    Args:
        train_logs_path: Path to training logs CSV
        test_logs_path: Path to test logs CSV (optional)
        
    Returns:
        Tuple of (train_features, test_features) DataFrames
    """
    pipeline = WritingProcessFeaturePipeline()
    
    print("ğŸš€ Extracting TRAINING features...")
    train_features = pipeline.extract_features(train_logs_path)
    
    test_features = None
    if test_logs_path:
        print("\nğŸš€ Extracting TEST features...")
        test_features = pipeline.extract_features(test_logs_path)
    
    return train_features, test_features

# ============================================================================
# USAGE EXAMPLE
# ============================================================================
DATA_DIR = '/kaggle/input/linking-writing-processes-to-writing-quality/'
    
# Extract features from both train and test sets
train_features, test_features = extract_writing_features(
    train_logs_path=DATA_DIR + 'train_logs.csv',
    test_logs_path=DATA_DIR + 'test_logs.csv'
)

print("\nğŸ“ˆ SUMMARY:")
print(f"Training set: {train_features.shape}")
print(f"Test set: {test_features.shape}")
print(f"\nFeature columns: {list(train_features.columns[:10])}...")

train_features.to_csv('train_features.csv', index=False)
test_features.to_csv('test_features.csv', index=False)


train_logs = pd.read_csv('/kaggle/input/linking-writing-processes-to-writing-quality/train_logs.csv')
train_scores = pd.read_csv('/kaggle/input/linking-writing-processes-to-writing-quality/train_scores.csv')
test_logs = pd.read_csv('/kaggle/input/linking-writing-processes-to-writing-quality/test_logs.csv')


# Prepare data
train_data = train_features.merge(train_scores, on='id', how='left')
X_train = train_data.drop(['id', 'score'], axis=1)
y_train = train_data['score']

test_ids = test_features['id']
X_test = test_features.drop('id', axis=1)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

X_train = X_train.fillna(0)
X_test = X_test.fillna(0)

print(f" Ready: X_train={X_train.shape}, y_train={y_train.shape}, X_test={X_test.shape}")


print("\nFeature selection...")

from sklearn.feature_selection import SelectKBest, f_regression, VarianceThreshold

# Step 1: Variance threshold (higher threshold)
selector = VarianceThreshold(threshold=0.001) 
X_var = X_train.loc[:, selector.fit(X_train).get_support()]
X_test_var = X_test[X_var.columns]

# Step 2: Correlation filter
corr = X_var.corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
to_drop = [col for col in upper.columns if any(upper[col] > 0.999)]
X_corr = X_var.drop(columns=to_drop)
X_test_corr = X_test_var.drop(columns=to_drop)

# Step 3: SelectKBest (keeps most predictive features)
k_features = min(200, X_corr.shape[1])  # Target 160 features
selector_kbest = SelectKBest(f_regression, k=k_features)
selector_kbest.fit(X_corr, y_train)

selected_cols = X_corr.columns[selector_kbest.get_support()].tolist()
X_selected = X_corr[selected_cols]
X_test_selected = X_test_corr[selected_cols]

print(f"Features: {X_train.shape[1]} â†’ {X_selected.shape[1]} (removed {X_train.shape[1] - X_selected.shape[1]})")


y_bins = pd.cut(y_train, bins=5, labels=False)

def objective_lgbm(trial):
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'num_leaves': trial.suggest_int('num_leaves', 20, 60),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.03, log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 0.95),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 0.95),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 30),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 1.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 1.0, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 9),
        'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0, 0.5),
        'n_estimators': 5000,
        'verbose': -1
    }
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, val_idx in skf.split(X_selected, y_bins):
        X_tr, X_val = X_selected.iloc[train_idx], X_selected.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        model = LGBMRegressor(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50)])
        preds = model.predict(X_val)
        scores.append(np.sqrt(mean_squared_error(y_val, preds)))
    
    return np.mean(scores)

print("\n" + "="*80)
print("OPTIMIZING LGBM (20 trials)")
print("="*80)
study_lgbm = optuna.create_study(direction='minimize', sampler=TPESampler(seed=42))
study_lgbm.optimize(objective_lgbm, n_trials=20, show_progress_bar=True)

print(f"\nBest LGBM Score: {study_lgbm.best_value:.6f}")
print(f"Best Params: {study_lgbm.best_params}")
lgbm_best_params = study_lgbm.best_params
lgbm_best_params.update({'objective': 'regression', 'metric': 'rmse', 'n_estimators': 5000, 'verbose': -1})



def objective_xgb(trial):
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': trial.suggest_int('max_depth', 3, 9),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.03, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 0.95),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.95),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 0, 0.5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 1.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 1.0, log=True),
        'n_estimators': 5000,
        'random_state': 42
    }
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, val_idx in skf.split(X_selected, y_bins):
        X_tr, X_val = X_selected.iloc[train_idx], X_selected.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        model = xgb.XGBRegressor(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=False)
        preds = model.predict(X_val)
        scores.append(np.sqrt(mean_squared_error(y_val, preds)))
    
    return np.mean(scores)

print("\n" + "="*80)
print("OPTIMIZING XGBOOST (20 trials)")
print("="*80)
study_xgb = optuna.create_study(direction='minimize', sampler=TPESampler(seed=42))
study_xgb.optimize(objective_xgb, n_trials=20, show_progress_bar=True)

print(f"\nBest XGB Score: {study_xgb.best_value:.6f}")
print(f"Best Params: {study_xgb.best_params}")
xgb_best_params = study_xgb.best_params
xgb_best_params.update({'objective': 'reg:squarederror', 'eval_metric': 'rmse', 'n_estimators': 5000, 'random_state': 42})



def objective_cat(trial):
    params = {
        'loss_function': 'RMSE',
        'iterations': 5000,
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.03, log=True),
        'depth': trial.suggest_int('depth', 4, 9),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.1, 10.0, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 0.95),
        'random_strength': trial.suggest_float('random_strength', 0, 10),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1),
        'random_seed': 42,
        'verbose': False,
        'early_stopping_rounds': 50
    }
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, val_idx in skf.split(X_selected, y_bins):
        X_tr, X_val = X_selected.iloc[train_idx], X_selected.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        model = CatBoostRegressor(**params)
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)
        preds = model.predict(X_val)
        scores.append(np.sqrt(mean_squared_error(y_val, preds)))
    
    return np.mean(scores)

print("\n" + "="*80)
print("OPTIMIZING CATBOOST (20 trials)")
print("="*80)
study_cat = optuna.create_study(direction='minimize', sampler=TPESampler(seed=42))
study_cat.optimize(objective_cat, n_trials=20, show_progress_bar=True)

print(f"\nBest CatBoost Score: {study_cat.best_value:.6f}")
print(f"Best Params: {study_cat.best_params}")
cat_best_params = study_cat.best_params
cat_best_params.update({'loss_function': 'RMSE', 'iterations': 5000, 'random_seed': 42, 'verbose': False})



N_FOLDS = 10
SEEDS = [42, 123, 456]

print("\n" + "="*80)
print(f"TRAINING ENSEMBLE: 3 MODELS Ã— {N_FOLDS} FOLDS Ã— {len(SEEDS)} SEEDS")
print("="*80)

# Storage
oof_lgbm = np.zeros(len(X_selected))
oof_xgb = np.zeros(len(X_selected))
oof_cat = np.zeros(len(X_selected))

test_lgbm = np.zeros(len(X_test_selected))
test_xgb = np.zeros(len(X_test_selected))
test_cat = np.zeros(len(X_test_selected))

for seed in SEEDS:
    print(f"\n{'='*60}")
    print(f"SEED {seed}")
    print('='*60)
    
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_selected, y_bins), 1):
        X_tr, X_val = X_selected.iloc[train_idx], X_selected.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        # LGBM
        lgbm_params_copy = lgbm_best_params.copy()
        lgbm_params_copy['random_state'] = seed
        model_lgbm = LGBMRegressor(**lgbm_params_copy)
        model_lgbm.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50)])
        oof_lgbm[val_idx] += model_lgbm.predict(X_val) / len(SEEDS)
        test_lgbm += model_lgbm.predict(X_test_selected) / (N_FOLDS * len(SEEDS))
        
        # XGBoost
        xgb_params_copy = xgb_best_params.copy()
        xgb_params_copy['random_state'] = seed
        model_xgb = xgb.XGBRegressor(**xgb_params_copy)
        model_xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=False)
        oof_xgb[val_idx] += model_xgb.predict(X_val) / len(SEEDS)
        test_xgb += model_xgb.predict(X_test_selected) / (N_FOLDS * len(SEEDS))
        
        
        # CatBoost
        cat_params_copy = cat_best_params.copy()
        cat_params_copy['random_seed'] = seed
        model_cat = CatBoostRegressor(**cat_params_copy)
        model_cat.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)
        oof_cat[val_idx] += model_cat.predict(X_val) / len(SEEDS)
        test_cat += model_cat.predict(X_test_selected) / (N_FOLDS * len(SEEDS))
        
        
        if fold % 2 == 0:
            print(f"  Fold {fold}/{N_FOLDS} done")

# OOF scores
score_lgbm = np.sqrt(mean_squared_error(y_train, oof_lgbm))
score_xgb = np.sqrt(mean_squared_error(y_train, oof_xgb))
score_cat = np.sqrt(mean_squared_error(y_train, oof_cat))

print(f"\n{'='*80}")
print("OOF SCORES")
print("="*80)
print(f"LGBM:     {score_lgbm:.6f}")
print(f"XGBoost:  {score_xgb:.6f}")
print(f"CatBoost: {score_cat:.6f}")

# Weighted ensemble (based on inverse RMSE)
weights = 1 / np.array([score_lgbm, score_xgb, score_cat])
weights = weights / weights.sum()

oof_ensemble = (oof_lgbm * weights[0] + oof_xgb * weights[1] + oof_cat * weights[2])
test_ensemble = (test_lgbm * weights[0] + test_xgb * weights[1] + test_cat * weights[2])

score_ensemble = np.sqrt(mean_squared_error(y_train, oof_ensemble))
print(f"ENSEMBLE: {score_ensemble:.6f}")
print(f"\nWeights: LGBM={weights[0]:.3f}, XGB={weights[1]:.3f}, CAT={weights[2]:.3f}")



# Clip predictions
test_ensemble_clipped = np.clip(test_ensemble, 1, 6)

submission = pd.DataFrame({
    'id': test_ids,
    'score': test_ensemble_clipped
})
submission.to_csv('submission.csv', index=False)

print(f"\n Submission saved: {len(submission)} predictions")
print(f"   Range: [{test_ensemble_clipped.min():.3f}, {test_ensemble_clipped.max():.3f}]")
print(f"\nPrediction statistics:")
print(f"  Mean: {submission['score'].mean():.2f}")
print(f"  Std:  {submission['score'].std():.2f}")
print(f"  Min:  {submission['score'].min():.2f}")
print(f"  Max:  {submission['score'].max():.2f}")

