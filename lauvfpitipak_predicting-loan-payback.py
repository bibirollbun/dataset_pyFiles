import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import os,ast,shutil,copy

from bokeh.plotting import figure, gridplot 
from bokeh.io import output_file, show, output_notebook
output_notebook()
from scipy import stats
from scipy.stats import rankdata
from itertools import combinations
import os

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import init_notebook_mode
init_notebook_mode(connected=True)

from scipy.optimize import minimize

colors = ['#f6f5f5', '#fe346e', '#512b58', '#2c003e']
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette([colors[1], colors[2], colors[3]])


train = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')
orig = pd.read_csv('/kaggle/input/loan-prediction-dataset-2025/loan_dataset_20000.csv')
TARGET = 'loan_paid_back'


train.info()


train['loan_paid_back'].value_counts()


COLS = list(test)
NUM_COLS = test.select_dtypes(include=['int64','float64']).columns.tolist()
CAT_COLS = test.select_dtypes(include=[ 'object']).columns.tolist()


test['loan_paid_back']=0
train.shape, test.shape


train.shape[0]


numerical_cols = ['annual_income', 'debt_to_income_ratio', 'credit_score', 
                  'loan_amount', 'interest_rate']

for col in numerical_cols:
    Q1 = train[col].quantile(0.25)
    Q3 = train[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    train[col] = train[col].clip(lower=lower_bound, upper=upper_bound)
    test[col] = test[col].clip(lower=lower_bound, upper=upper_bound)


#find distribution of annual income and loan amount

test['annual_income'].plot(kind='kde')
plt.title('annual_income distribution (KDE)')
plt.xlabel('annual_income')
plt.show()

test['loan_amount'].plot(kind='kde')
plt.title('loan_amount distribution (KDE)')
plt.xlabel('loan_amount')
plt.show()


fig = go.Figure()
target_counts = train[TARGET].value_counts()
target_pct = (target_counts / len(train) * 100).round(1)

fig.add_trace(go.Bar(y=['Status'], x=[target_pct[0]], name='Default', orientation='h',
    marker=dict(color=colors[1]), text=f'{target_pct[0]:.1f}%', textposition='inside',
    textfont=dict(size=20, color='white', family='Arial Black')))

fig.add_trace(go.Bar(y=['Status'], x=[target_pct[1]], name='Paid Back', orientation='h',
    marker=dict(color=colors[2]), text=f'{target_pct[1]:.1f}%', textposition='inside',
    textfont=dict(size=20, color='white', family='Arial Black')))

fig.update_layout(title='Loan Repayment Status Distribution', barmode='stack', height=400,
    showlegend=True, plot_bgcolor='#f6f5f5', paper_bgcolor='#f6f5f5',
    annotations=[dict(text='Created By Ozan M.', xref='paper', yref='paper', x=1, y=-0.15, 
    showarrow=False, font=dict(size=10, color='gray'))])
fig.show(renderer='iframe_connected')


fig = make_subplots(rows=1, cols=2, subplot_titles=('Income Distribution', 'Income vs Default'))

for status in [0, 1]:
    subset = train[train[TARGET] == status]
    fig.add_trace(go.Violin(y=subset['annual_income'], 
        name='Default' if status == 0 else 'Paid Back',
        marker_color=colors[1] if status == 0 else colors[2]), row=1, col=1)

income_bins = pd.cut(train['annual_income'], bins=10)
default_rate = train.groupby(income_bins)[TARGET].agg(['mean', 'count'])
bin_centers = [interval.mid for interval in default_rate.index]

fig.add_trace(go.Scatter(x=bin_centers, y=(1 - default_rate['mean']) * 100,
    mode='markers+lines', marker=dict(size=default_rate['count']/50, color=colors[1]),
    line=dict(color=colors[2], width=3)), row=1, col=2)

fig.update_layout(height=500, plot_bgcolor='#f6f5f5', paper_bgcolor='#f6f5f5',
    annotations=[dict(text='Created By Ozan M.', xref='paper', yref='paper', 
    x=1, y=-0.1, showarrow=False, font=dict(size=10, color='gray'))])
fig.show(renderer='iframe_connected')


train['credit_bin'] = pd.cut(train['credit_score'], 
    bins=[0, 580, 670, 740, 800, 1000],
    labels=['Poor', 'Fair', 'Good', 'Very Good', 'Excellent'])

credit_analysis = train.groupby('credit_bin').agg({TARGET: ['mean', 'count']}).reset_index()
credit_analysis.columns = ['credit_bin', 'default_rate', 'count']

fig = make_subplots(rows=1, cols=2, subplot_titles=('Credit Score by Status', 'Default Rate'))

for status in [0, 1]:
    subset = train[train[TARGET] == status]
    fig.add_trace(go.Box(y=subset['credit_score'], 
        name='Default' if status == 0 else 'Paid Back',
        marker_color=colors[1] if status == 0 else colors[2]), row=1, col=1)

fig.add_trace(go.Bar(x=credit_analysis['credit_bin'], 
    y=(1 - credit_analysis['default_rate']) * 100,
    marker=dict(color=(1 - credit_analysis['default_rate']) * 100,
    colorscale=[[0, colors[1]], [1, colors[2]]], showscale=False)), row=1, col=2)

fig.update_layout(height=500, plot_bgcolor='#f6f5f5', paper_bgcolor='#f6f5f5',
    annotations=[dict(text='Created By Ozan M.', xref='paper', yref='paper',
    x=1, y=-0.1, showarrow=False, font=dict(size=10, color='gray'))])
fig.show(renderer='iframe_connected')

train.drop('credit_bin', axis=1, inplace=True)


purpose_counts = train['loan_purpose'].value_counts()
purpose_default = train.groupby('loan_purpose')[TARGET].mean()

fig = go.Figure(go.Bar(y=purpose_counts.index, x=purpose_counts.values, orientation='h',
    marker=dict(color=purpose_counts.values, 
    colorscale=[[0, colors[2]], [0.5, colors[1]], [1, colors[3]]], showscale=False),
    text=[f"{val} loans" for val in purpose_counts.values], textposition='outside'))

fig.update_layout(title='Loan Purpose Distribution', height=500, 
    plot_bgcolor='#f6f5f5', paper_bgcolor='#f6f5f5',
    annotations=[dict(text='Created By Ozan M.', xref='paper', yref='paper',
    x=1, y=-0.1, showarrow=False, font=dict(size=10, color='gray'))])
fig.show(renderer='iframe_connected')


train['grade'] = train['grade_subgrade'].str[0]
test['grade'] = test['grade_subgrade'].str[0]

grade_analysis = train.groupby('grade').agg({
    'interest_rate': ['mean', 'std', 'count'], TARGET: 'mean'}).reset_index()
grade_analysis.columns = ['grade', 'mean_rate', 'std_rate', 'count', 'default_rate']

fig = go.Figure(go.Scatter(x=grade_analysis['grade'], y=grade_analysis['mean_rate'],
    mode='markers+lines', marker=dict(size=grade_analysis['count']/100, 
    color=grade_analysis['mean_rate'], colorscale=[[0, colors[2]], [1, colors[1]]], 
    showscale=True), line=dict(color=colors[2], width=3),
    error_y=dict(type='data', array=grade_analysis['std_rate'])))

fig.update_layout(title='Interest Rate by Loan Grade', height=500,
    plot_bgcolor='#f6f5f5', paper_bgcolor='#f6f5f5',
    annotations=[dict(text='Created By Ozan M.', xref='paper', yref='paper',
    x=1, y=-0.12, showarrow=False, font=dict(size=10, color='gray'))])
fig.show(renderer='iframe_connected')


dti_bins = pd.cut(train['debt_to_income_ratio'], bins=10)
dti_analysis = train.groupby(dti_bins).agg({
    TARGET: ['mean', 'count']}).reset_index()
dti_analysis.columns = ['dti_bin', 'default_rate', 'count']

# FIX: Convert intervals to float for multiplication
dti_analysis['bin_center'] = [interval.mid for interval in dti_analysis['dti_bin']]

fig = make_subplots(rows=2, cols=1, subplot_titles=('Default Rate', 'DTI Distribution'),
    row_heights=[0.6, 0.4])

fig.add_trace(go.Scatter(x=np.array(dti_analysis['bin_center']) * 100, 
    y=(1 - dti_analysis['default_rate']) * 100, mode='markers+lines',
    marker=dict(size=dti_analysis['count']/30, color=(1 - dti_analysis['default_rate']) * 100,
    colorscale=[[0, colors[1]], [1, colors[2]]], showscale=False),
    line=dict(color=colors[2], width=3), fill='tozeroy'), row=1, col=1)

for status in [0, 1]:
    subset = train[train[TARGET] == status]
    fig.add_trace(go.Histogram(x=subset['debt_to_income_ratio'] * 100,
        name='Default' if status == 0 else 'Paid Back',
        marker_color=colors[1] if status == 0 else colors[2], opacity=0.7), row=2, col=1)

fig.update_layout(height=700, plot_bgcolor='#f6f5f5', paper_bgcolor='#f6f5f5',
    barmode='overlay', annotations=[dict(text='Created By Ozan M.', xref='paper',
    yref='paper', x=1, y=-0.05, showarrow=False, font=dict(size=10, color='gray'))])
    
fig.update_xaxes(title_text='Debt-to-Income Ratio (%)', row=1, col=1)
fig.update_xaxes(title_text='Debt-to-Income Ratio (%)', row=2, col=1)
fig.update_yaxes(title_text='Default Rate (%)', row=1, col=1)
fig.update_yaxes(title_text='Frequency', row=2, col=1)

fig.show(renderer='iframe_connected')


employment_marital = pd.crosstab(train['employment_status'], train['marital_status'],
    values=(1 - train[TARGET]) * 100, aggfunc='mean')

fig = go.Figure(go.Heatmap(z=employment_marital.values, x=employment_marital.columns,
    y=employment_marital.index, colorscale=[[0, colors[1]], [0.5, colors[3]], [1, colors[2]]],
    text=employment_marital.values.round(1), texttemplate='%{text}%',
    textfont=dict(size=14, color='white', family='Arial Black')))

fig.update_layout(title='Employment Ã— Marital Status Risk Matrix', height=500,
    plot_bgcolor='#f6f5f5', paper_bgcolor='#f6f5f5',
    annotations=[dict(text='Created By Ozan M.', xref='paper', yref='paper',
    x=1, y=-0.12, showarrow=False, font=dict(size=10, color='gray'))])
fig.show(renderer='iframe_connected')


fig = make_subplots(rows=1, cols=2, subplot_titles=('Loan Amount Distribution', 'Amount vs Default'))

for status in [0, 1]:
    subset = train[train[TARGET] == status]
    fig.add_trace(go.Histogram(x=subset['loan_amount'],
        name='Default' if status == 0 else 'Paid Back',
        marker_color=colors[1] if status == 0 else colors[2], opacity=0.7), row=1, col=1)

loan_bins = pd.cut(train['loan_amount'], bins=15)
loan_analysis = train.groupby(loan_bins).agg({TARGET: ['mean', 'count']}).reset_index()
loan_analysis.columns = ['loan_bin', 'default_rate', 'count']
loan_analysis['bin_center'] = loan_analysis['loan_bin'].apply(lambda x: x.mid)

fig.add_trace(go.Scatter(x=loan_analysis['bin_center'], 
    y=(1 - loan_analysis['default_rate']) * 100, mode='markers+lines',
    marker=dict(size=loan_analysis['count']/40, color=(1 - loan_analysis['default_rate']) * 100,
    colorscale=[[0, colors[1]], [1, colors[2]]], showscale=False),
    line=dict(color=colors[2], width=3)), row=1, col=2)

fig.update_layout(height=500, plot_bgcolor='#f6f5f5', paper_bgcolor='#f6f5f5',
    barmode='overlay', annotations=[dict(text='Created By Ozan M.', xref='paper',
    yref='paper', x=1, y=-0.1, showarrow=False, font=dict(size=10, color='gray'))])
fig.show(renderer='iframe_connected')


edu_analysis = train.groupby('education_level').agg({
    TARGET: ['mean', 'count'], 'annual_income': 'mean', 'credit_score': 'mean'}).reset_index()
edu_analysis.columns = ['education', 'default_rate', 'count', 'avg_income', 'avg_credit']
edu_analysis = edu_analysis.sort_values('default_rate', ascending=False)

fig = make_subplots(rows=1, cols=2, subplot_titles=('Default Rate', 'Portfolio Distribution'),
    specs=[[{'type': 'bar'}, {'type': 'pie'}]])

fig.add_trace(go.Bar(x=edu_analysis['education'], 
    y=(1 - edu_analysis['default_rate']) * 100,
    marker=dict(color=(1 - edu_analysis['default_rate']) * 100,
    colorscale=[[0, colors[1]], [1, colors[2]]], showscale=False),
    text=[f"{val:.1f}%" for val in (1 - edu_analysis['default_rate']) * 100],
    textposition='outside'), row=1, col=1)

fig.add_trace(go.Pie(labels=edu_analysis['education'], values=edu_analysis['count'],
    marker=dict(colors=[colors[2], colors[1], colors[3], colors[2]]),
    textinfo='label+percent', hole=0.4), row=1, col=2)

fig.update_layout(height=500, plot_bgcolor='#f6f5f5', paper_bgcolor='#f6f5f5',
    annotations=[dict(text='Created By Ozan M.', xref='paper', yref='paper',
    x=1, y=-0.1, showarrow=False, font=dict(size=10, color='gray'))])
fig.show(renderer='iframe_connected')


corr_matrix = train[numerical_cols + [TARGET]].corr()

fig = go.Figure(go.Heatmap(z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.columns,
    colorscale=[[0, colors[1]], [0.5, colors[0]], [1, colors[2]]], zmid=0,
    text=corr_matrix.values.round(2), texttemplate='%{text}', textfont=dict(size=11)))

fig.update_layout(title='Feature Correlation Matrix', height=600, width=700,
    plot_bgcolor='#f6f5f5', paper_bgcolor='#f6f5f5',
    annotations=[dict(text='Created By Ozan M.', xref='paper', yref='paper',
    x=1.2, y=-0.20, showarrow=False, font=dict(size=10, color='gray'))])
fig.show(renderer='iframe_connected')


def bokeh_show(
        params,
        df_cross,
        show_figures1, 
        show_figures2, wps_fig2,
        color_cross):

    colors = [subm['color'] for subm in params['subm']]
    
    def dossier(js,subms,cols):
        def quant(i,js,subms,cols):
            return {"c" : i, "q" : sum([1 for subm in cols[i] if subm == subms[js]])}
        return {
            'name' : subms[js],
            'q_in' : [quant(i,js,subms,cols) for i in range(len(subms))]
        }
    alls = pd.read_csv(f'tida_desc.csv')
    matrix = [ast.literal_eval(str(row.alls)) for row in alls.itertuples()]
    subms = sorted(matrix[0])
    cols = [[data[i] for data in matrix] for i in range(len(subms))]
    df_subms = pd.DataFrame({f'col_{i}': [x[i] for x in matrix] for i in range(len(subms))})
    dossiers = [dossier(js,subms,cols) for js in range(len(subms))]
    subm_names = [one_dossier['name'] for one_dossier in dossiers]
    figures1,qss,i = [],[],0
    height = 85 if len(colors)==2\
        else 134 if len(colors)==3 else (154 if len(colors)==4 else 174)
    for one_dossier in dossiers: 
        i_col = 'alls. ' + str(one_dossier['q_in'][i]['c'])
        qs = [one['q'] for one in one_dossier['q_in']]
        x_names = [name.replace("Group","").replace("subm_","") for name in subm_names]
        width = 157  if len(colors) == 5\
            else (121 if len(colors) == 8\
            else (131 if len(colors) == 9\
            else (141 if len(colors) == 10\
            else (171 if len(colors) == 11 else 140))))
        f = figure(x_range=x_names,width=width, height=height, title=i_col)
        f.vbar(x=x_names, width=0.585, top=qs, color=colors)
        figures1.append(f)
        qss.append(qs)
        i+=1
    grid = gridplot([figures1])
    output_file('tida_alls.html')
    if show_figures1 == True: show(grid)
    sub_wts = params['subwts']
    main_wts = [subm['weight'] for subm in params['subm']]
    mms,acc_mass = [],[]
    for j in range(len(dossiers)):
        one_dossier = dossiers[j]
        qs = [one['q'] for one in one_dossier['q_in']]
        mm = [qs[h] * (main_wts[j] + sub_wts[h]) for h in range(len(sub_wts))]
        mass = sum(mm)
        mms.append(mm)
        acc_mass.append(round(mass))                        #subm_names[::-1]
    y_names = [name + " - " + str(mass) for name,mass in zip(subm_names,acc_mass)]
    f1 = figure(y_range=y_names, width=270, height=height, title='relations of general masses')
    f1.hbar(y=y_names, height=0.555, right=acc_mass, left=0, color=colors)
    output_file('tida_alls2.html')
    alls = [f'alls.{i}' for i in range(len(dossiers))]
    subm = [f'sub{i}'   for i in range(len(dossiers))] 
    mmsT  = np.asarray(mms).T
    data = {'cols' : alls}
    for i in range(len(dossiers)): data[f'sub{i}'] = mmsT[i,:]
    f2 = figure(y_range=alls, height=height, width=270, title="relations of columns masses")
    f2.hbar_stack(subm, y='cols', height=0.555, color=colors, source=data)
    qssT  = np.asarray(qss).T
    data = {'cols' : alls}
    for i in range(len(dossiers)): data[f'sub{i}'] = qssT[i,:]
    f3 = figure(y_range=alls, height=height, width=245, title="ratios in columns")
    f3.hbar_stack(subm, y='cols', height=0.555, color=colors, source=data)
    grid = gridplot([[f3,f2,f1]])
    show(grid)
    if show_figures2 == True:
        def read(params,i):
            FiN = params["path"] + params["subm"][i]["name"] + ".csv"
            target_name_back = {'target':params["target"],'pred':params["target"]}
            return pd.read_csv(FiN).rename(columns=target_name_back)
        dfs = [read(params,i) for i in range(len(params["subm"]))] + [df_cross]
        _height = 358 if len(params["subm"]) == 11 else 254
        f   = figure(width=785, height=_height)
        f.title.text = 'Click on legend entries to mute the corresponding lines'
        b,e        = 21000,21121
        line_x     = [dfs[i][b:e]['id']             for i in range(len(dfs))]
        line_y     = [dfs[i][b:e]['loan_paid_back'] for i in range(len(dfs))]
        color      = colors + [color_cross]
        alpha      = [0.8 for i in range(len(dfs)-1)] + [0.95]
        lws        = [1.0 for i in range(len(dfs)-1)] + [1.00]
        legend = subm_names + ['cross']
        for i in range(len(legend)):
            f.line(line_x[i], line_y[i], line_width=lws[i], color=color[i], alpha=alpha[i],
                   muted_color='white',legend_label=legend[i])
        f.legend.location = "top_left"
        f.legend.click_policy="mute"
        show(f)


def matrix_vs(path,fs_names):
    def load(path,fs_names):
        dfs = [pd.read_csv(path + name_subm +'.csv') for name_subm in fs_names]
        for i in range(len(dfs)):
            dfs[i] = dfs[i].rename(columns={"loan_paid_back": f'{fs_names[i]}'})
        dfsm = pd.merge(dfs[0], dfs[1], on="id")
        for i in range(2,len(dfs)):
            dfsm = pd.merge(dfsm,dfs[i],on='id')
        return dfsm   
    def make_list_vs(fs_names):
        list = []
        for i in range(0,len(fs_names)-1):
            for j in range(i+1,len(fs_names)):
                list.append(fs_names[i] + "_vs_" + fs_names[j])
        return list
    def get_mvs(dfs, list_vs):
        def get_abs_distance(x,t1,t2):
            return abs(x[t1]-x[t2])
        for vs in list_vs:
            t = vs.split('_vs_')
            dfs[vs] = dfs.apply(lambda x: get_abs_distance(x,t[0],t[1]), axis=1)
        return dfs   
    def distance_vs(name, st_names, list_vs, dfs):
        distances = []
        for st in st_names:
            vs_between = name + "_vs_" + st
            if vs_between not in list_vs:
                distances.append(0)
            else: distances.append(round(dfs[vs_between].sum()))
        return distances
    dfs = load(path,fs_names)
    list_vs = make_list_vs(fs_names)
    mvs = get_mvs(dfs, list_vs)
    m1 = pd.DataFrame({'subm':fs_names})
    m2 = pd.DataFrame({ name :distance_vs(name, fs_names, list_vs, mvs) for name in fs_names})
    matrix = pd.concat([m1,m2],axis=1)
    return matrix


def display_distances(params):
    files = [subm['name'] for subm in params['subm']]
    distances = matrix_vs ( params['path'], files )            
    display(distances)


def arr_colors(color):
    sg = ['silver','gainsboro']
    if color=='red'   or color=='r': return ['red','crimson','firebrick']            + sg
    if color=='Red'   or color=='R': return ['red','tomato','crimson']               + sg
    if color=='Green' or color=='G': return ['forestgreen','limegreen', 'darkgreen'] + sg
    if color=='Blue'  or color=='B': return ['blue','royalblue','mediumblue']        + sg
    if color=='RGB'   or color=='S': return ['mediumblue','darkgreen','crimson']     + sg
    return ['black','dimgray','gray'] + sg


def convert(schema):
    colors = arr_colors(schema[2])
    dicts  = [
        {'name': schema[0][i],'weight':schema[1][i],'color':colors[i]} 
        for i in range(len(schema[0]))
    ]
    return {'subm':dicts}


def h_blend(
        params, _update={},
        cross='silver',
        details=False,
        fig1=False, fig2=False, wf2=555, 
        dtls=False, dist=False, subm=''):

    if isinstance(params, list): params = convert(params)

    if 'path' in _update: params.update(_update)
    
    color_cross, dk  = cross, copy.deepcopy(params)

    if details == True:
        dist = True
        show_details,show_figures1,show_figures2 = True,True,True
    else:
        show_details,show_figures1,show_figures2 = dtls,fig1,fig2
        
    file_short_names = [subm['name'] for subm in params['subm']]
    type_sort    = params['type_sort'][0]
    dk['asc']    = params['type_sort'][1]
    dk['desc']   = params['type_sort'][2]
    dk['id']     = params['id_target'][0]
    dk['target'] = params['id_target'][1]
# ------------------------------------------------------------------------
    def read(dk,i):
        tnm = dk["subm"][i]["name"]
        FiN = dk["path"] + tnm + ".csv"
        return pd.read_csv(FiN).rename(columns={
            'target':tnm, 'pred':tnm, dk["target"]:tnm})
        
    def merge(dfs_subm):
        df_subms = pd.merge(dfs_subm[0],  dfs_subm[1], on=[dk['id']])
        for i in range(2, len(dk["subm"])): 
            df_subms = pd.merge(df_subms, dfs_subm[i], on=[dk['id']])
        return df_subms
        
    def da(dk,sorting_direction,show_details):
        
        df_subms = merge([read(dk,i) for i in range(len(dk["subm"]))])
        cols = [col for col in df_subms.columns if col != dk['id']]
        short_name_cols = [c for c in cols]
        
        def alls1(x, sd=sorting_direction,cs=cols):
            reverse = True if sd=='desc' else False
            tes = {c: x[c] for c in cs}.items()
            subms_sorted = [t[0] for t in sorted(tes,key=lambda k:k[1],reverse=reverse)]
            return subms_sorted

        import random

        def alls2(x, sd=sorting_direction,cs=cols):
            reverse = True if sd=='desc' else False
            tes = {c: x[c] for c in cs}.items()
            subms_random = [t[0] for t in tes]
            random.shuffle(subms_random)
            return subms_random

        alls = alls1 if type_sort == 'asc/desc' else alls2
            
        def summa(x,cs,wts,ic_alls): 
            return sum([x[cs[j]] * (wts[0][j] + wts[1][ic_alls[j]]) for j in range(len(cs))])
            
        wts = [[[e['weight'] for e in dk["subm"]], [w for w in dk["subwts"]]]]
          
        def correct(x, cs=cols, wts=wts):
            i = [x['alls'].index(c) for c in short_name_cols]
            return summa(x,cs,wts[0],i)

        if len(wts) == 1:
            correct_sub_weights = [wt for wt in dk["subwts"]]
            weights = [subm['weight'] for subm in dk["subm"]]
            def correct(x, cs=cols, w=weights, cw=correct_sub_weights):
                ic = [x['alls'].index(c) for c in short_name_cols]
                cS = [x[cols[j]] * (w[j] + cw[ic[j]]) for j in range(len(cols))]
                return sum(cS)
                
        if len(wts) > 1 or "subwts2" in dk:

            wts = [
                [[e['weight'] for e in dk["subm"]], [w for w in dk["subwts" ]]],
                [[e['weight'] for e in dk["subm2"]],[w for w in dk["subwts2"]]],
                [[e['weight'] for e in dk["subm3"]],[w for w in dk["subwts3"]]],
            ]

            def correct(x, cs=cols, wts=wts):
                i = [x['alls'].index(c) for c in short_name_cols]
                if   0.0540 < x['mx-m'] <= 0.0740: return summa(x,cs,wts[2],i)
                if   0.0000 < x['mx-m'] <= 0.0050: return summa(x,cs,wts[1],i)
                else:                              return summa(x,cs,wts[0],i)
                   
        def amxm(x, cs=cols):
            list_values = x[cs].to_list()
            mxm = abs(max(list_values)-min(list_values))
            return mxm

        if len(wts) > 1 or "subwts2" in dk:
            df_subms['mx-m']   = df_subms.apply(lambda x: amxm   (x), axis=1)
        df_subms['alls']       = df_subms.apply(lambda x: alls   (x), axis=1)
        df_subms[dk["target"]] = df_subms.apply(lambda x: correct(x), axis=1)
        schema_rename = { old_nc:new_shnc for old_nc, new_shnc in zip(cols, short_name_cols) }
        df_subms = df_subms.rename(columns=schema_rename)
        df_subms = df_subms.rename(columns={dk["target"]:"ensemble"})
        df_subms.insert(loc=1, column=' _ ', value=['   '] * len(df_subms))
        df_subms[' _ '] = df_subms[' _ '].astype(str)
        pd.set_option('display.max_rows',100)
        pd.set_option('display.float_format', '{:.5f}'.format)
        vcols = [dk['id']]+[' _ '] + short_name_cols + [' _ ']+['alls']+[' _ ']+['ensemble']
        if len(wts) > 1: 
            vcols = [dk['id']] + [' _ '] + short_name_cols + [' _ '] + ['mx-m'] + [' _ '] + ['alls'] + [' _ '] + ['ensemble']
        df_subms = df_subms[vcols]
        if show_details and sorting_direction=='desc': display(df_subms.head(5))
        pd.set_option('display.float_format', '{:.5f}'.format)
        df_subms = df_subms.rename(columns={"ensemble":dk["target"]})
        if sorting_direction=='desc': 
            df_subms.to_csv(f'tida_{sorting_direction}.csv', index=False)
        return df_subms[[dk['id'],dk['target']]]
   
    def ensemble_da(dk,        show_details): 
        dfD    = da(dk,'desc', show_details)
        dfA    = da(dk,'asc',  show_details)
        dfA[dk['target']] = dk['desc']*dfD[dk['target']] + dfA[dk['target']]*dk['asc']
        return dfA

    da = ensemble_da(dk,show_details)
    bokeh_show(dk, da, show_figures1, show_figures2, wf2, color_cross)
    if dist == True: display_distances(params)
    if subm != '': da.to_csv(subm, index=False)
    return  da


def bl(file_names, weights, file_name_subm):
    path = '/kaggle/input/25-november-2025-ps-s5e11/'
    dfs = [pd.read_csv(f'{path}{file_name}.csv') for file_name in file_names]
    w = [weight/100 for weight in weights]
    t = 'loan_paid_back'
    df = pd.read_csv('/kaggle/input/playground-series-s5e11/sample_submission.csv')
    if len(dfs) == 2: df[t] = dfs[0][t]*w[0] + dfs[1][t]*w[1]
    if len(dfs) == 3: df[t] = dfs[0][t]*w[0] + dfs[1][t]*w[1]+ dfs[2][t]*w[2]
    if len(dfs) == 4: df[t] = dfs[0][t]*w[0] + dfs[1][t]*w[1]+ dfs[2][t]*w[2]+ dfs[3][t]*w[3]
    if file_name_subm != "":
        if '.csv' not in file_name_subm: file_name_subm += '.csv'
        df.to_csv(file_name_subm, index=False)
        # print(f'{file_name_submission} - ready to use')
        display(df.head(3))


up1 = {'path'     : '/kaggle/input/27-november-2025-ps-s5e11/',            
      'id_target': ['id','loan_paid_back'],          
      'type_sort': ['asc/desc',0.32,0.68 ],
      'subwts'   : [w/150 for w in [10,-3,-7]],
}
up_ = {'path'     : '/kaggle/working/',          
      'id_target': ['id','loan_paid_back'],          
      'type_sort': ['asc/desc',0.32,0.68 ],
      'subwts'   : [w/150 for w in [10,-3,-7]],
}

# version 1 LB=0.92_778

local_1_weights = [0.09, 0.27, 0.51]
group_1_weights = [0.60, 0.30, 0.10]

local_4_weights = [0.30, 0.40, 0.30]
group_4_weights = [0.30, 0.40, 0.30]

# version 2 LB>0.92778

local_1_weights = [0.08, 0.25, 0.52]  
group_1_weights = [0.61, 0.29, 0.10]  

local_4_weights = [0.25, 0.40, 0.35]  
group_4_weights = [0.35, 0.40, 0.25]  

# version 3 LB> V2

local_1_weights = [0.07, 0.23, 0.55]  
group_1_weights = [0.62, 0.28, 0.10]  

local_4_weights = [0.20, 0.35, 0.45]  
group_4_weights = [0.45, 0.35, 0.20]   

# version 4 LB>V3

local_1_weights = [0.08, 0.26, 0.50]  
group_1_weights = [0.60, 0.30, 0.10]   

local_4_weights = [0.25, 0.45, 0.30]                              
group_4_weights = [0.25, 0.45, 0.30]  

# version 5 LB>V4

local_1_weights = [0.09, 0.26, 0.51]  
group_1_weights = [0.65, 0.25, 0.10]  

local_4_weights = [0.33, 0.33, 0.34]  
group_4_weights = [0.34, 0.33, 0.33]  

# version 6 LB>V5

local_1_weights = [0.05, 0.25, 0.70]  
group_1_weights = [0.60, 0.30, 0.10]  

local_4_weights = [0.30, 0.40, 0.30]  
group_4_weights = [0.30, 0.40, 0.30]  

# version 7 LB>V6

local_1_weights = [0.09, 0.27, 0.51] 
group_1_weights = [0.60, 0.30, 0.10] 

local_4_weights = [0.45, 0.30, 0.25]  
group_4_weights = [0.25, 0.30, 0.45]  

# version 8 LB>V7

local_1_weights = [0.15, 0.35, 0.50] 
group_1_weights = [0.50, 0.30, 0.20] 

local_4_weights = [0.30, 0.30, 0.40] 
group_4_weights = [0.40, 0.30, 0.30]

# version 9 LB>V8   

local_1_weights = [0.07, 0.20, 0.73] 
group_1_weights = [0.60, 0.30, 0.10] 

local_4_weights = [0.30, 0.40, 0.30] 
group_4_weights = [0.30, 0.40, 0.30]

# version 10 LB>= 0.92800

local_1_weights = [0.33, 0.33, 0.34]
group_1_weights = [0.34, 0.33, 0.33]

local_4_weights = [0.33, 0.33, 0.34]
group_4_weights = [0.34, 0.33, 0.33]


%%time
df3 = h_blend([['12','13','14'],local_1_weights,'r'],up1,subm='g3.csv')
df2 = h_blend([['16','17','18'],local_1_weights,'G'],up1,subm='g2.csv')
df1 = h_blend([['44','45','46'],local_1_weights,'B'],up1,subm='g1.csv')
df_ = h_blend([['g1','g2','g3'],group_1_weights,'S'],up_,subm='_1.csv')
print(f'\n-------------------------------------------------------------2')
df3 = h_blend([['22','23','24'],local_1_weights,'r'],up1,subm='g3.csv')
df2 = h_blend([['26','27','28'],local_1_weights,'G'],up1,subm='g2.csv')
df1 = h_blend([['47','48','49'],local_1_weights,'B'],up1,subm='g1.csv')
df_ = h_blend([['g1','g2','g3'],group_1_weights,'S'],up_,subm='_2.csv')
print(f'\n-------------------------------------------------------------3')
df3 = h_blend([['32','33','34'],local_1_weights,'r'],up1,subm='g3.csv')
df2 = h_blend([['36','37','38'],local_1_weights,'G'],up1,subm='g2.csv')
df1 = h_blend([['41','42','43'],local_1_weights,'B'],up1,subm='g1.csv')
df_ = h_blend([['g1','g2','g3'],group_1_weights,'S'],up_,subm='_3.csv')
print(f'\n-------------------------------------------------------------4')
df3 = h_blend([['51','52','56'],local_4_weights,'R'],up1,subm='g3.csv')
df2 = h_blend([['57','58','59'],local_4_weights,'G'],up1,subm='g2.csv')
df1 = h_blend([['53','54','55'],local_4_weights,'B'],up1,subm='g1.csv')
df_ = h_blend([['g1','g2','g3'],group_4_weights,'S'],up_,subm='_4.csv')


params = {
      'path'     : '/kaggle/working/',            
      'id_target': ['id',"loan_paid_back"],          
      'type_sort': ['asc/desc',0.32,0.68 ],
      'subwts'   : [e/200 for e in [11, -1,-3,-7]],
      'subm'     : [
         { 'name': f'_1', 'weight':+0.60, 'color':'brown'       },
         { 'name': f'_2', 'weight':+0.32, 'color':'chocolate'   },
         { 'name': f'_3', 'weight':+0.08, 'color':'sandybrown'  },
         { 'name': f'_4', 'weight':+0.00, 'color':'sienna'      },]
}
df = h_blend(params, details=True)

for file in '_1,_2,_3,_4,g1,g2,g3'.split(','): os.remove(f'/kaggle/working/{file}.csv')


df.to_csv('submission.csv',index=False)
df

