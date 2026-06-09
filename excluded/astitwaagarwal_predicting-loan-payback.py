# Essential libraries for data analysis
import numpy as np 
import pandas as pd 

# Listing files in the input directory to verify data availability
import os
for root_dir, _, file_list in os.walk('/kaggle/input'):
    for file_name in file_list:
        print(os.path.join(root_dir, file_name))


import numpy as np
import pandas as pd
import os, ast, shutil, copy
from bokeh.plotting import figure, gridplot 
from bokeh.io import output_file, show, output_notebook

# Initialize Bokeh for notebook output
output_notebook()

def render_bokeh_plot(
        config,
        cross_df,
        display_fig1, 
        display_fig2, 
        wps_f2,
        cross_color):

    # Extract colors from configuration
    palette = [item['color'] for item in config['submissions']]
    
    def create_dossier(idx, sub_list, col_data):
        def calculate_quantity(i, idx, sub_list, col_data):
            return {"c" : i, "q" : sum([1 for s in col_data[i] if s == sub_list[idx]])}
        return {
            'name' : sub_list[idx],
            'q_in' : [calculate_quantity(i, idx, sub_list, col_data) for i in range(len(sub_list))]
        }

    # Load description data
    desc_data = pd.read_csv(f'tida_desc.csv')
    matrix_data = [ast.literal_eval(str(r.alls)) for r in desc_data.itertuples()]
    sorted_subs = sorted(matrix_data[0])
    columns_data = [[m[i] for m in matrix_data] for i in range(len(sorted_subs))]
    
    # Create dossiers for each submission
    dossier_list = [create_dossier(j, sorted_subs, columns_data) for j in range(len(sorted_subs))]
    sub_names = [d['name'] for d in dossier_list]
    
    figures_list, quantity_list, counter = [], [], 0
    
    # Determine plot height based on number of colors
    plot_height = 85 if len(palette)==2 else 134 if len(palette)==3 else (154 if len(palette)==4 else 174)
    
    for d in dossier_list: 
        col_title = 'alls. ' + str(d['q_in'][counter]['c'])
        quantities = [q['q'] for q in d['q_in']]
        x_labels = [n.replace("Group","").replace("subm_","") for n in sub_names]
        
        # Determine plot width
        plot_width = 157 if len(palette) == 5 else (121 if len(palette) == 8 else (131 if len(palette) == 9 else (141 if len(palette) == 10 else (171 if len(palette) == 11 else 133))))
        
        fig = figure(x_range=x_labels, width=plot_width, height=plot_height, title=col_title)
        fig.vbar(x=x_labels, width=0.585, top=quantities, color=palette)
        figures_list.append(fig)
        quantity_list.append(quantities)
        counter += 1
        
    grid_layout = gridplot([figures_list])
    output_file('tida_alls.html')
    if display_fig1: show(grid_layout)
    
    sub_weights = config['sub_weights']
    main_weights = [s['weight'] for s in config['submissions']]
    mass_matrix, accumulated_mass = [], []
    
    for j in range(len(dossier_list)):
        current_dossier = dossier_list[j]
        qs = [item['q'] for item in current_dossier['q_in']]
        mm = [qs[h] * (main_weights[j] + sub_weights[h]) for h in range(len(sub_weights))]
        total_mass = sum(mm)
        mass_matrix.append(mm)
        accumulated_mass.append(round(total_mass))
        
    y_labels = [n + " - " + str(m) for n, m in zip(sub_names, accumulated_mass)]
    f1 = figure(y_range=y_labels, width=313, height=plot_height, title='relations of general masses')
    f1.hbar(y=y_labels, height=0.585, right=accumulated_mass, left=0, color=palette)
    output_file('tida_alls2.html')
    
    alls_labels = [f'alls.{i}' for i in range(len(dossier_list))]
    sub_labels = [f'sub{i}' for i in range(len(dossier_list))] 
    mass_matrix_T = np.asarray(mass_matrix).T
    
    data_source = {'cols' : alls_labels}
    for i in range(len(dossier_list)): data_source[f'sub{i}'] = mass_matrix_T[i,:]
    
    f2 = figure(y_range=alls_labels, height=plot_height, width=274, title=" ( relations of columns masses )")
    f2.hbar_stack(sub_labels, y='cols', height=0.585, color=palette, source=data_source)
    
    quantity_matrix_T = np.asarray(quantity_list).T
    data_source_q = {'cols' : alls_labels}
    for i in range(len(dossier_list)): data_source_q[f'sub{i}'] = quantity_matrix_T[i,:]
    
    f3 = figure(y_range=alls_labels, height=plot_height, width=210, title="ratios in columns")
    f3.hbar_stack(sub_labels, y='cols', height=0.585, color=palette, source=data_source_q)
    
    final_grid = gridplot([[f3, f2, f1]])
    show(final_grid)
    
    if display_fig2:
        def read_submission(cfg, idx):
            file_path = cfg["input_path"] + cfg["submissions"][idx]["name"] + ".csv"
            rename_map = {'target': cfg["target_col"], 'pred': cfg["target_col"]}
            return pd.read_csv(file_path).rename(columns=rename_map)
            
        dfs = [read_submission(config, i) for i in range(len(config["submissions"]))] + [cross_df]
        fig_height = 334 if len(config["submissions"]) == 9 else 254
        
        f = figure(width=785, height=fig_height)
        f.title.text = 'Click on legend entries to mute the corresponding lines'
        
        start_idx, end_idx = 21000, 21121
        x_vals = [dfs[i][start_idx:end_idx]['id'] for i in range(len(dfs))]
        y_vals = [dfs[i][start_idx:end_idx]['loan_paid_back'] for i in range(len(dfs))]
        
        plot_colors = palette + [cross_color]
        alphas = [0.8 for _ in range(len(dfs)-1)] + [0.95]
        line_widths = [1.0 for _ in range(len(dfs)-1)] + [1.00]
        legend_labels = sub_names + ['cross']
        
        for i in range(len(legend_labels)):
            f.line(x_vals[i], y_vals[i], line_width=line_widths[i], color=plot_colors[i], alpha=alphas[i],
                   muted_color='white', legend_label=legend_labels[i])
                   
        f.legend.location = "top_left"
        f.legend.click_policy = "mute"
        show(f)


def calculate_matrix_differences(path, file_names):
    def load_files(p, names):
        loaded_dfs = [pd.read_csv(p + name + '.csv') for name in names]
        for i in range(len(loaded_dfs)):
            loaded_dfs[i] = loaded_dfs[i].rename(columns={"loan_paid_back": f'{names[i]}'})
            
        merged_df = pd.merge(loaded_dfs[0], loaded_dfs[1], on="id")
        for i in range(2, len(loaded_dfs)):
            merged_df = pd.merge(merged_df, loaded_dfs[i], on='id')
        return merged_df   
        
    def generate_pairs(names):
        pairs = []
        for i in range(0, len(names)-1):
            for j in range(i+1, len(names)):
                pairs.append(names[i] + "_vs_" + names[j])
        return pairs
        
    def compute_diffs(df, pairs):
        def abs_diff(row, col1, col2):
            return abs(row[col1] - row[col2])
        for pair in pairs:
            c1, c2 = pair.split('_vs_')
            df[pair] = df.apply(lambda row: abs_diff(row, c1, c2), axis=1)
        return df   
        
    def sum_distances(target_name, all_names, pairs, df):
        dists = []
        for name in all_names:
            pair_key = target_name + "_vs_" + name
            if pair_key not in pairs:
                dists.append(0)
            else:
                dists.append(round(df[pair_key].sum()))
        return dists
        
    dfs = load_files(path, file_names)
    pairs_list = generate_pairs(file_names)
    diff_df = compute_diffs(dfs, pairs_list)
    
    df_names = pd.DataFrame({'subm': file_names})
    df_dists = pd.DataFrame({name: sum_distances(name, file_names, pairs_list, diff_df) for name in file_names})
    
    result_matrix = pd.concat([df_names, df_dists], axis=1)
    return result_matrix


def show_distance_metrics(config):
    f_names = [s['name'] for s in config['submissions']]
    dist_matrix = calculate_matrix_differences(config['input_path'], f_names)
    display(dist_matrix)


def get_color_palette(scheme_code):
    base_grays = ['silver', 'gainsboro']
    if scheme_code == 'silver' or scheme_code == 'S': return ['gray', 'silver', 'gold'] + base_grays
    if scheme_code == 'red'    or scheme_code == 'R': return ['darkorchid', 'orangered', 'crimson'] + base_grays
    if scheme_code == 'green'  or scheme_code == 'G': return ['darkorchid', 'limegreen', 'darkgreen'] + base_grays
    if scheme_code == 'blue'   or scheme_code == 'B': return ['darkorchid', 'mediumblue', 'midnightblue'] + base_grays
    return ['black', 'dimgray', 'gray'] + base_grays


def format_schema(raw_schema):
    palette = get_color_palette(raw_schema[2])
    formatted_list = [
        {'name': raw_schema[0][i], 'weight': raw_schema[1][i], 'color': palette[i]} 
        for i in range(len(raw_schema[0]))
    ]
    return {'submissions': formatted_list}


def hybrid_ensemble(
        config, _update_params={},
        cross_color='silver',
        show_details=False,
        show_fig1=False, show_fig2=False, wf2=555, 
        enable_dtls=False, enable_dist=False, output_filename=''):

    # Convert list format to dictionary if necessary
    if isinstance(config, list): config = format_schema(config)

    if 'input_path' in _update_params: config.update(_update_params)
    
    c_color, cfg_copy = cross_color, copy.deepcopy(config)

    if show_details == True:
        enable_dist = True
        is_details, is_fig1, is_fig2 = True, True, True
    else:
        is_details, is_fig1, is_fig2 = enable_dtls, show_fig1, show_fig2
        
    short_names = [s['name'] for s in config['submissions']]
    sort_mode = config['sort_strategy'][0]
    cfg_copy['asc_w'] = config['sort_strategy'][1]
    cfg_copy['desc_w'] = config['sort_strategy'][2]
    cfg_copy['id_col'] = config['id_target'][0]
    cfg_copy['target_col'] = config['id_target'][1]

    # ------------------------------------------------------------------------
    def read_file(c, i):
        name = c["submissions"][i]["name"]
        full_path = c["input_path"] + name + ".csv"
        return pd.read_csv(full_path).rename(columns={
            'target': name, 'pred': name, c["target_col"]: name})
        
    def merge_dfs(df_list):
        merged = pd.merge(df_list[0], df_list[1], on=[cfg_copy['id_col']])
        for i in range(2, len(cfg_copy["submissions"])): 
            merged = pd.merge(merged, df_list[i], on=[cfg_copy['id_col']])
        return merged
        
    def process_data(c, sort_dir, details_flag):
        
        merged_df = merge_dfs([read_file(c, i) for i in range(len(c["submissions"]))])
        columns = [col for col in merged_df.columns if col != c['id_col']]
        short_cols = [col for col in columns]
        
        def sort_cols_by_val(row, direction=sort_dir, cols=columns):
            is_reverse = True if direction=='desc' else False
            items = {col: row[col] for col in cols}.items()
            sorted_keys = [k for k, v in sorted(items, key=lambda item: item[1], reverse=is_reverse)]
            return sorted_keys

        import random

        def shuffle_cols(row, direction=sort_dir, cols=columns):
            items = {col: row[col] for col in cols}.items()
            keys = [k for k, v in items]
            random.shuffle(keys)
            return keys

        sorter = sort_cols_by_val if sort_mode == 'asc/desc' else shuffle_cols
            
        def weighted_sum(row, cols, weights, indices): 
            return sum([row[cols[j]] * (weights[0][j] + weights[1][indices[j]]) for j in range(len(cols))])
            
        weights_list = [[[s['weight'] for s in c["submissions"]], [w for w in c["sub_weights"]]]]
          
        def correct_vals(row, cols=columns, wts=weights_list):
            idx_list = [row['alls'].index(col) for col in short_cols]
            return weighted_sum(row, cols, wts[0], idx_list)

        if len(weights_list) == 1:
            sub_w = [w for w in c["sub_weights"]]
            main_w = [s['weight'] for s in c["submissions"]]
            def correct_vals(row, cols=columns, mw=main_w, sw=sub_w):
                idx_list = [row['alls'].index(col) for col in short_cols]
                val_sum = [row[cols[j]] * (mw[j] + sw[idx_list[j]]) for j in range(len(cols))]
                return sum(val_sum)
                   
        def calc_range(row, cols=columns):
            vals = row[cols].to_list()
            return abs(max(vals) - min(vals))

        if len(weights_list) > 1:
            merged_df['mx-m'] = merged_df.apply(lambda r: calc_range(r), axis=1)
            
        merged_df['alls'] = merged_df.apply(lambda r: sorter(r), axis=1)
        merged_df[c["target_col"]] = merged_df.apply(lambda r: correct_vals(r), axis=1)
        
        rename_map = {old: new for old, new in zip(columns, short_cols)}
        merged_df = merged_df.rename(columns=rename_map)
        merged_df = merged_df.rename(columns={c["target_col"]: "ensemble"})
        
        merged_df.insert(loc=1, column=' _ ', value=['   '] * len(merged_df))
        merged_df[' _ '] = merged_df[' _ '].astype(str)
        
        pd.set_option('display.max_rows', 100)
        pd.set_option('display.float_format', '{:.5f}'.format)
        
        view_cols = [c['id_col']] + [' _ '] + short_cols + [' _ '] + ['alls'] + [' _ '] + ['ensemble']
        if len(weights_list) > 1: view_cols.append([' _ '] + ['mx-m'])
        
        merged_df = merged_df[view_cols]
        if details_flag and sort_dir == 'desc': display(merged_df.head(5))
        
        pd.set_option('display.float_format', '{:.5f}'.format)
        merged_df = merged_df.rename(columns={"ensemble": c["target_col"]})
        
        if sort_dir == 'desc': 
            merged_df.to_csv(f'tida_{sort_dir}.csv', index=False)
            
        return merged_df[[c['id_col'], c['target_col']]]
   
    def run_ensemble(c, details_flag): 
        df_desc = process_data(c, 'desc', details_flag)
        df_asc = process_data(c, 'asc', details_flag)
        df_asc[c['target_col']] = c['desc_w'] * df_desc[c['target_col']] + df_asc[c['target_col']] * c['asc_w']
        return df_asc

    final_df = run_ensemble(cfg_copy, is_details)
    render_bokeh_plot(cfg_copy, final_df, is_fig1, is_fig2, wf2, c_color)
    
    if enable_dist == True: show_distance_metrics(config)
    if output_filename != '': final_df.to_csv(output_filename, index=False)
    return final_df


config_params = {
  'input_path'     : f"/kaggle/input/22-november-2025-ps-s5e11/submission_",   
  'id_target': ['id', "loan_paid_back"],
  'sort_strategy': ['asc/desc', 0.30, 0.70],
  'sub_weights'   : [ -0.02, +0.02, +0.02, -0.02 ],
  'submissions'     : [
      {'name': f'015', 'LB': 0.92756, 'weight': +0.31, 'color': 'navy'},
      {'name': f'017', 'LB': 0.92743, 'weight': +0.09, 'color': 'royalblue'},
      {'name': f'018', 'LB': 0.92764, 'weight': +0.29, 'color': 'deepskyblue'},
      {'name': f'019', 'LB': 0.92768, 'weight': +0.31, 'color': 'dodgerblue'},]
}

blended_df = hybrid_ensemble(config_params, show_details=True)


blended_df.to_csv('submission.csv', index=False)
blended_df




