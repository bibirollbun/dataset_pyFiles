import os
import ast
import numpy as np
import pandas as pd
import shutil
import copy
import random
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass
from functools import reduce
from bokeh.plotting import figure, gridplot 
from bokeh.io import output_file, show, output_notebook
output_notebook()

def get_visualization_colors(config_dict, theme_name):
    """
    Generate color palette for visualizations based on selected theme.
    Returns a list of colors trimmed to match the number of submissions.
    """
    # Define multiple color palettes for different visualization needs
    palette_map = {
        'Two': ['crimson', 'mediumblue'],
        'Two2': ['crimson', 'darkgreen'],
        'alls': ['silver', 'crimson', 'forestgreen'],
        'alls2': ['red', 'green', 'blue', 'silver', 'gold'],
        'alls3': ['darkmagenta', 'forestgreen', 'mediumblue'],
        'alls4': ['crimson', 'darkgreen', 'forestgreen', 'limegreen'],
        'alls4r': ['darkgreen', 'forestgreen', 'crimson', 'limegreen'],
        'alls4m': ['red', 'forestgreen', 'mediumblue', 'darkmagenta'],
        'alls4i': ['crimson', 'green', 'mediumblue', 'chocolate'],
        'alls4j': ['red', 'green', 'blue', 'sienna'],
        'alls5': ['red', 'forestgreen', 'mediumblue', 'darkmagenta', 'crimson'],
        'gold': ['gainsboro', 'silver', 'darkgray', 'gray', 'gold'],
        'red3': ['olivedrab', 'gold', 'lemonchiffon'],
        'red4': ['firebrick', 'orangered', 'crimson', 'red'],
        'red5': ['crimson', 'red', 'orangered', 'tomato', 'darkmagenta'],
        'red6': ['crimson', 'red', 'orangered', 'tomato', 'green', 'mediumblue'],
        'red13': ['gold', 'crimson', 'red'],
        'red31': ['crimson', 'red', 'gold'],
        'red52': ['silver', 'darkgray', 'gray', 'crimson', 'crimson'],
        'red53': ['silver', 'darkgray', 'gray', 'dimgray', 'crimson'],
        'red54': ['forestgreen', 'limegreen', 'lime', 'crimson'],
        'red55': ['silver', 'darkgray', 'gray', 'crimson'],
        'green': ['limegreen', 'forestgreen', 'mediumseagreen', 'green', 'darkgreen'],
        'green2': ['olivedrab', 'darkgreen', 'forestgreen'],
        'green3': ['darkmagenta', 'olivedrab', 'darkgreen'],
        'green4': ['darkgreen', 'forestgreen', 'limegreen', 'lime'],
        'green5': ['crimson', 'darkgreen', 'forestgreen', 'limegreen', 'lime'],
        'blue': ['midnightblue', 'royalblue', 'mediumblue', 'blue', 'steelblue', 'cyan'],
        'blue4': ['midnightblue', 'royalblue', 'mediumblue', 'deepskyblue'],
        'blue5': ['firebrick', 'navy', 'mediumblue', 'royalblue', 'deepskyblue'],
        'brown': ['maroon', 'firebrick', 'chocolate', 'sienna', 'sandybrown'],
        'brown3': ['maroon', 'sienna', 'sandybrown'],
        'brown4': ['maroon', 'sienna', 'chocolate', 'sandybrown'],
        'brown5': ['maroon', 'sienna', 'chocolate', 'sandybrown', 'gold'],
    }
    # Combine palettes for custom themes
    palette_map['tes3'] = ['limegreen', 'magenta', 'red']
    palette_map['tes3b'] = ['darkmagenta', 'magenta', 'red']
    palette_map['tes5'] = ['mediumblue', 'crimson', 'crimson', 'crimson', 'mediumblue']
    palette_map['tes6'] = ['limegreen'] + palette_map['brown']
    palette_map['tes7'] = palette_map['brown4'] + ['mediumblue'] + ['crimson'] + ['red']
    palette_map['tes8'] = palette_map['red4'] + palette_map['blue4']
    palette_map['tes9'] = palette_map['red4'] + ['darkmagenta'] + palette_map['blue4']
    palette_map['tes10'] = palette_map['brown'] + palette_map['green']
    palette_map['tes11'] = palette_map['brown'] + palette_map['green'] + ['blue']
    
    # Get number of submissions and trim color list accordingly
    num_submissions = len(config_dict['subm'])
    selected_colors = palette_map.get(theme_name, ['red', 'green', 'blue'])    
    return selected_colors[0:num_submissions]


def create_interactive_plots(config, ensemble_result, color_list, 
                             display_plot1, display_plot2, width_param, highlight_color):
    """
    Generate interactive Bokeh visualizations for model ensemble analysis.
    Creates bar charts showing submission rankings and relationships.
    """
    
    def build_submission_profile(index, submission_list, column_data):
        """Build frequency distribution for a submission across sorted columns."""
        def compute_frequency(col_idx, row_idx, submissions, cols):
            occurrences = sum(1 for item in cols[col_idx] if item == submissions[row_idx])
            return {"column_id": col_idx, "frequency": occurrences}
        
        return {
            'submission_name': submission_list[index],
            'frequencies': [compute_frequency(i, index, submission_list, column_data) 
                          for i in range(len(submission_list))]
        }
    # Load and parse the ranking data
    ranking_data = pd.read_csv('tida_desc.csv')
    ranking_matrix = [ast.literal_eval(str(row.alls)) for row in ranking_data.itertuples()]
    submission_names = sorted(ranking_matrix[0])
    # Transpose matrix to get column-wise data
    columns_transposed = [[data[i] for data in ranking_matrix] for i in range(len(submission_names))]
    # Build DataFrame for analysis
    df_submissions = pd.DataFrame({
        f'col_{i}': [x[i] for x in ranking_matrix] 
        for i in range(len(submission_names))
    })
    
    # Generate profiles for each submission
    submission_profiles = [build_submission_profile(j, submission_names, columns_transposed) 
                          for j in range(len(submission_names))]
    
    display_names = [profile['submission_name'] for profile in submission_profiles]
    # Create individual bar charts for each submission
    plot_list = []
    frequency_arrays = []
    # Calculate plot dimensions based on number of colors
    if len(color_list) == 2:
        plot_height = 85
    elif len(color_list) == 3:
        plot_height = 134
    elif len(color_list) == 4:
        plot_height = 154
    else:
        plot_height = 174
    
    for idx, profile in enumerate(submission_profiles):
        chart_title = f'alls. {profile["frequencies"][idx]["column_id"]}'
        freq_values = [item['frequency'] for item in profile['frequencies']]
        # Clean up display names
        x_labels = [name.replace("Group", "").replace("subm_", "") for name in display_names]
        # Determine width based on number of submissions
        if len(color_list) == 5:
            plot_width = 157
        elif len(color_list) == 8:
            plot_width = 121
        elif len(color_list) == 9:
            plot_width = 131
        elif len(color_list) == 10:
            plot_width = 141
        elif len(color_list) == 11:
            plot_width = 171
        else:
            plot_width = 133
        
        # Create bar chart
        chart = figure(x_range=x_labels, width=plot_width, height=plot_height, title=chart_title)
        chart.vbar(x=x_labels, width=0.585, top=freq_values, color=color_list)
        plot_list.append(chart)
        frequency_arrays.append(freq_values)
    # Display first grid of plots
    grid_layout = gridplot([plot_list])
    output_file('tida_alls.html')
    if display_plot1:
        show(grid_layout)
    # Calculate weighted masses for submissions
    adjustment_weights = config['subwts']
    primary_weights = [sub['weight'] for sub in config['subm']]
    weighted_masses = []
    total_masses = []
    for j in range(len(submission_profiles)):
        profile = submission_profiles[j]
        frequencies = [item['frequency'] for item in profile['frequencies']]
        # Apply dual weighting system
        mass_components = [
            frequencies[h] * (primary_weights[j] + adjustment_weights[h]) 
            for h in range(len(adjustment_weights))
        ]
        total_mass = sum(mass_components)
        
        weighted_masses.append(mass_components)
        total_masses.append(round(total_mass))
    # Create horizontal bar chart for total masses
    y_labels_with_mass = [f"{name} - {mass}" 
                         for name, mass in zip(display_names, total_masses)]
    chart1 = figure(y_range=y_labels_with_mass, width=313, height=plot_height, 
                   title='relations of general masses')
    chart1.hbar(y=y_labels_with_mass, height=0.585, right=total_masses, left=0, color=color_list)
    output_file('tida_alls2.html')
    # Create stacked horizontal bar charts
    column_labels = [f'alls.{i}' for i in range(len(submission_profiles))]
    submission_keys = [f'sub{i}' for i in range(len(submission_profiles))]
    # Prepare data for mass distribution chart
    masses_transposed = np.asarray(weighted_masses).T
    mass_data = {'cols': column_labels}
    for i in range(len(submission_profiles)):
        mass_data[f'sub{i}'] = masses_transposed[i, :]
    
    chart2 = figure(y_range=column_labels, height=plot_height, width=274, 
                   title=" ( relations of columns masses )")
    chart2.hbar_stack(submission_keys, y='cols', height=0.585, 
                     color=color_list, source=mass_data)
    # Prepare data for frequency ratio chart
    frequencies_transposed = np.asarray(frequency_arrays).T
    freq_data = {'cols': column_labels}
    for i in range(len(submission_profiles)):
        freq_data[f'sub{i}'] = frequencies_transposed[i, :]
    chart3 = figure(y_range=column_labels, height=plot_height, width=215, 
                   title="ratios in columns")
    chart3.hbar_stack(submission_keys, y='cols', height=0.585, 
                     color=color_list, source=freq_data)
    # Display combined charts
    combined_grid = gridplot([[chart3, chart2, chart1]])
    show(combined_grid)
    # Create time series comparison plot if requested
    if display_plot2:
        def load_submission_file(params, index):
            filename = params["path"] + params["subm"][index]["name"] + ".csv"
            column_mapping = {'target': params["target"], 'pred': params["target"]}
            return pd.read_csv(filename).rename(columns=column_mapping)
        
        # Load all submission files plus ensemble result
        dataframes = [load_submission_file(config, i) 
                     for i in range(len(config["subm"]))] + [ensemble_result]
        # Create line plot
        line_chart = figure(width=800, height=254)
        line_chart.title.text = 'Click on legend entries to mute the corresponding lines'
        # Select sample range for visualization
        start_idx, end_idx = 21000, 21121
        # Extract x and y data for each submission
        x_data = [df[start_idx:end_idx]['id'] for df in dataframes]
        y_data = [df[start_idx:end_idx]['loan_paid_back'] for df in dataframes]
        # Configure visual properties
        line_colors = color_list + [highlight_color]
        alpha_values = [0.8] * (len(dataframes) - 1) + [0.95]
        line_widths = [1.0] * (len(dataframes) - 1) + [1.00]
        legend_labels = display_names + ['cross']
        # Add lines to chart
        for i in range(len(legend_labels)):
            line_chart.line(x_data[i], y_data[i], line_width=line_widths[i], 
                          color=line_colors[i], alpha=alpha_values[i],
                          muted_color='white', legend_label=legend_labels[i])
        line_chart.legend.location = "top_left"
        line_chart.legend.click_policy = "mute"
        show(line_chart)

def weighted_ensemble_blend(config, color_theme, cross_color='silver',
                            show_plot1=False, show_plot2=False, width_fig2=555,
                            show_details=False):
    """
    Main ensemble function that combines multiple predictions using weighted blending.
    Supports both ascending and descending sort strategies with configurable weights.
    """
    
    # Initialize configuration
    working_config = copy.deepcopy(config)
    # Parse configuration parameters
    sort_strategy = config['type_sort'][0]
    working_config['asc'] = config['type_sort'][1]
    working_config['desc'] = config['type_sort'][2]
    working_config['id'] = config['id_target'][0]
    working_config['target'] = config['id_target'][1]
    # Helper function to load individual submission files
    def load_submission(cfg, index):
        submission_name = cfg["subm"][index]["name"]
        file_path = cfg["path"] + submission_name + ".csv"
        # Rename columns to use submission name as identifier
        column_rename = {
            'target': submission_name,
            'pred': submission_name,
            cfg["target"]: submission_name
        }
        return pd.read_csv(file_path).rename(columns=column_rename)
    
    # Merge all submissions on ID column
    def merge_submissions(submission_dfs):
        merged_df = pd.merge(submission_dfs[0], submission_dfs[1], 
                            on=[working_config['id']])
        for i in range(2, len(working_config["subm"])):
            merged_df = pd.merge(merged_df, submission_dfs[i], 
                                on=[working_config['id']])
        return merged_df
    # Core blending algorithm with directional sorting
    def apply_directional_blend(cfg, sort_direction, show_info):
        """
        Apply weighted blending with specified sort direction.
        Sorts predictions and applies position-based weights.
        """
        
        # Load and merge all submissions
        merged_data = merge_submissions([load_submission(cfg, i) 
                                        for i in range(len(cfg["subm"]))])
        # Get prediction columns (exclude ID)
        pred_columns = [col for col in merged_data.columns if col != cfg['id']]
        short_names = [c for c in pred_columns]
        # Define sorting function
        def sort_predictions_desc_asc(row, direction=sort_direction, cols=pred_columns):
            """Sort predictions for each row by value."""
            use_reverse = True if direction == 'desc' else False
            value_dict = {c: row[c] for c in cols}
            sorted_names = [item[0] for item in sorted(value_dict.items(), 
                                                       key=lambda k: k[1], 
                                                       reverse=use_reverse)]
            return sorted_names
        def randomize_predictions(row, direction=sort_direction, cols=pred_columns):
            """Alternative strategy: randomize prediction order."""
            value_dict = {c: row[c] for c in cols}
            name_list = [item[0] for item in value_dict.items()]
            random.shuffle(name_list)
            return name_list
        # Select sorting strategy
        if sort_strategy == 'asc/desc':
            sort_function = sort_predictions_desc_asc
        else:
            sort_function = randomize_predictions
        # Extract weights from configuration
        primary_weights = [entry['weight'] for entry in cfg["subm"]]
        position_weights = [w for w in cfg["subwts"]]
        # Calculate weighted sum based on position in sorted order
        def calculate_weighted_prediction(row, cols=pred_columns, 
                                         w_primary=primary_weights, 
                                         w_position=position_weights):
            """
            Compute final prediction using dual weighting:
            - Primary weight: based on submission quality
            - Position weight: based on rank in sorted order
            """
            position_indices = [row['alls'].index(c) for c in short_names]
            weighted_values = [
                row[pred_columns[j]] * (w_primary[j] + w_position[position_indices[j]]) 
                for j in range(len(pred_columns))
            ]
            return sum(weighted_values)
        # Calculate prediction spread (max - min)
        def calculate_spread(row, cols=pred_columns):
            """Measure disagreement between predictions."""
            values = row[cols].to_list()
            return abs(max(values) - min(values))
        # Apply transformations
        merged_data['alls'] = merged_data.apply(lambda x: sort_function(x), axis=1)
        merged_data[cfg["target"]] = merged_data.apply(
            lambda x: calculate_weighted_prediction(x), axis=1)
        # Rename columns for clarity
        merged_data = merged_data.rename(columns={cfg["target"]: "ensemble"})
        # Add separator columns for display
        merged_data.insert(loc=1, column=' _ ', value=['   '] * len(merged_data))
        merged_data[' _ '] = merged_data[' _ '].astype(str)
        
        # Configure pandas display options
        pd.set_option('display.max_rows', 100)
        pd.set_option('display.float_format', '{:.5f}'.format)
        
        # Select columns for display
        display_cols = ([cfg['id']] + [' _ '] + short_names + 
                       [' _ '] + ['alls'] + [' _ '] + ['ensemble'])
        merged_data = merged_data[display_cols]
        
        # Show sample if requested
        if show_info and sort_direction == 'desc':
            display(merged_data.head(5))
        
        # Save intermediate results
        merged_data = merged_data.rename(columns={"ensemble": cfg["target"]})
        merged_data.to_csv(f'tida_{sort_direction}.csv', index=False)
        
        return merged_data[[cfg['id'], cfg['target']]]
    
    # Combine ascending and descending strategies
    def create_bidirectional_ensemble(cfg, show_info):
        """
        Create ensemble by combining desc and asc sorted predictions.
        Weights the two strategies according to configuration.
        """
        result_desc = apply_directional_blend(cfg, 'desc', show_info)
        result_asc = apply_directional_blend(cfg, 'asc', show_info)
        
        # Blend the two strategies
        result_asc[cfg['target']] = (cfg['desc'] * result_desc[cfg['target']] + 
                                     result_asc[cfg['target']] * cfg['asc'])
        return result_asc
    
    # Execute ensemble creation
    final_ensemble = create_bidirectional_ensemble(working_config, show_details)
    
    # Generate visualizations
    color_palette = get_visualization_colors(working_config, color_theme)
    create_interactive_plots(working_config, final_ensemble, color_palette, 
                           show_plot1, show_plot2, width_fig2, cross_color)
    
    return final_ensemble

def compute_pairwise_distances(file_path, submission_names):
    """
    Calculate pairwise prediction distances between all submissions.
    Returns a matrix showing divergence between each pair of predictions.
    """
    
    def load_all_submissions(path, names):
        """Load all submission files and merge them."""
        dataframes = [pd.read_csv(path + name + '.csv') for name in names]
        # Rename target column to submission name
        for i in range(len(dataframes)):
            dataframes[i] = dataframes[i].rename(
                columns={"loan_paid_back": f'{names[i]}'})
        # Merge all dataframes on ID
        merged = pd.merge(dataframes[0], dataframes[1], on="id")
        for i in range(2, len(dataframes)):
            merged = pd.merge(merged, dataframes[i], on='id')
        return merged
    
    def generate_comparison_pairs(names):
        """Create list of all pairwise comparisons."""
        pairs = []
        for i in range(len(names) - 1):
            for j in range(i + 1, len(names)):
                pairs.append(f"{names[i]}_vs_{names[j]}")
        return pairs
    
    def calculate_all_distances(df, pair_list):
        """Compute absolute distance for each pair of predictions."""
        def get_absolute_diff(row, name1, name2):
            return abs(row[name1] - row[name2])
        
        for pair in pair_list:
            names = pair.split('_vs_')
            df[pair] = df.apply(
                lambda x: get_absolute_diff(x, names[0], names[1]), axis=1)
        return df
    
    def extract_distances_for_submission(target_name, all_names, pairs, df):
        """Get total distance from target to all other submissions."""
        distance_list = []
        for name in all_names:
            pair_key = f"{target_name}_vs_{name}"
            if pair_key not in pairs:
                distance_list.append(0)
            else:
                distance_list.append(round(df[pair_key].sum()))
        return distance_list
    
    # Execute distance calculations
    merged_df = load_all_submissions(file_path, submission_names)
    comparison_pairs = generate_comparison_pairs(submission_names)
    distance_df = calculate_all_distances(merged_df, comparison_pairs)
    
    # Build distance matrix
    names_column = pd.DataFrame({'subm': submission_names})
    distance_columns = pd.DataFrame({
        name: extract_distances_for_submission(name, submission_names, 
                                               comparison_pairs, distance_df)
        for name in submission_names
    })
    distance_matrix = pd.concat([names_column, distance_columns], axis=1)
    return distance_matrix

def show_submission_distances(config):
    """Display pairwise distance matrix for all submissions in config."""
    submission_list = [sub['name'] for sub in config['subm']]
    dist_matrix = compute_pairwise_distances(config['path'], submission_list)
    display(dist_matrix)

def simple_weighted_blend(dataframe_1, dataframe_2, weights=[0.50, 0.50], 
                          output_file='submission.csv'):
    """
    Perform simple weighted average of two prediction dataframes.
    Useful for quick blending of two models.
    """
    target_col = 'loan_paid_back'    
    # Apply weighted average
    dataframe_1[target_col] = (dataframe_1[target_col] * weights[0] + 
                               dataframe_2[target_col] * weights[1])
    # Save result
    dataframe_1.to_csv(output_file, index=False)
    print(f'{output_file} - ready to use')


# Configuration for first ensemble - using 4 submissions with specific weights
# Paths remain exactly as original for compatibility
ensemble_config = {
    'path': f"/kaggle/input/9-november-2025-ps-s5e11/submission ",            
    'id_target': ['id', "loan_paid_back"],          
    'type_sort': ['asc/desc', 0.30, 0.70],  # 30% ascending, 70% descending
    'subwts': [+0.11, -0.01, -0.03, -0.07],  # Position-based weight adjustments
    'subm': [
        {'name': f'0.92661', 'weight': +0.15},    # Model 1 - crimson in viz
        {'name': f'0.92694', 'weight': +0.25},    # Model 2 - darkgreen in viz
        {'name': f'0.92698', 'weight': +0.35},    # Model 3 - forestgreen in viz
        {'name': f'0.92712', 'weight': +0.25},# Model 4 - limegreen in viz (highest weight)
    ]
}
# Execute weighted ensemble with visualizations
final_predictions = weighted_ensemble_blend(
    ensemble_config, 
    color_theme='alls4', 
    show_plot1=True, 
    show_plot2=True, 
    show_details=True
)
# Display pairwise prediction distances
show_submission_distances(ensemble_config)


# Alternative configuration - different weight distribution
# Testing more balanced weights with adjusted asc/desc ratio
ensemble_config_v2 = {
    'path': f"/kaggle/input/9-november-2025-ps-s5e11/submission ",            
    'id_target': ['id', "loan_paid_back"],          
    'type_sort': ['asc/desc', 0.35, 0.65],  # 35% ascending, 65% descending
    'subwts': [+0.11, -0.01, -0.03, -0.07],  # Same position adjustments
    'subm': [
        {'name': f'0.92661', 'weight': +0.25},    # Model 1 - increased weight
        {'name': f'0.92694', 'weight': +0.22},    # Model 2 - increased weight
        {'name': f'0.92698', 'weight': +0.23},    # Model 3 - increased weight
        {'name': f'0.92712', 'weight': +0.30},    # Model 4 - decreased weight (more balanced)
    ]
}

# Execute alternative ensemble configuration
final_predictions_v2 = weighted_ensemble_blend(
    ensemble_config_v2, 
    color_theme='alls4', 
    show_plot1=True, 
    show_plot2=True
)


# Third configuration - different model selection
# Using model 0.92712 instead of 0.92661 with reversed color scheme
ensemble_config_v3 = {
    'path': f"/kaggle/input/9-november-2025-ps-s5e11/submission ",            
    'id_target': ['id', "loan_paid_back"],          
    'type_sort': ['asc/desc', 0.30, 0.70],  # Back to 30/70 split
    'subwts': [+0.11, -0.01, -0.03, -0.07],  # Position weights unchanged
    'subm': [    
        {'name': f'0.92694', 'weight': +0.21},    # Model 2 - darkgreen in viz
        {'name': f'0.92698', 'weight': +0.25},    # Model 3 - forestgreen in viz
        {'name': f'0.92712', 'weight': +0.25},    # Different model - crimson in viz
        {'name': f'0.92732', 'weight': +0.29},    # Model 4 - limegreen (dominant)
    ]
}
# Execute third ensemble with reversed color palette
final_predictions_v3 = weighted_ensemble_blend(
    ensemble_config_v3, 
    color_theme='alls4r',  # Reversed color scheme 
    show_plot1=True, 
    show_plot2=True, 
    show_details=True
)


# Save the final ensemble predictions file
final_predictions_v3.to_csv('submission.csv', index=False)
final_predictions_v3


# id	loan_paid_back
# 0	593994	0.94759
# 1	593995	0.96582

# id	loan_paid_back
# 0	593994	0.94531
# 1	593995	0.96564

# id	loan_paid_back
# 0	593994	0.94596
# 1	593995	0.96668

