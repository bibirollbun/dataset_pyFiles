import kagglehub

MK_PATH = kagglehub.dataset_download("kaggle/meta-kaggle")

print("Path to Meta-Kaggle dataset files:", MK_PATH)


import pandas as pd
import os
from datetime import datetime
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import duckdb
from plotly.offline import init_notebook_mode 
init_notebook_mode(connected=True) # to fix plotly graphs not showing in viewer mode
# from IPython.display import IFrame 
import plotly.io as pio
pio.renderers.default = 'iframe' # use default iframe renderer if not specified
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, module='pandas.io.formats.format')
warnings.simplefilter('ignore', category=FutureWarning)


segmentOrder = {'CompSegment': ['Playground', 'Featured', 'Research']}
segmentColors = {
    'Playground': '#1f77b4',  # blue
    'Featured': '#ff7f0e',    # orange
    'Research': '#2ca02c',    # green
    #'Community': '#d62728',   # red
    #'Recruitment': '#9467bd'  # purple
}

medalOrder = {'MedalType': ['None', 'Bronze', 'Silver','Gold']}
medalColors = {
    'Gold':   '#FFD700',  # gold
    'Silver': '#C0C0C0',  # silver
    'Bronze': '#CD7F32',  # bronze
    'None':   '#CCCCFF'   # Periwinkle
}

tierOrder = {'UserPerformanceTierName':['Unranked','Expert','Master','Grandmaster']}  #,'Staff'
tierColors = {
    'Unranked': '#20BEFF',      # Kaggle blue
    'Expert':      '#8751FD',  # Purple
    'Master':      '#F97B48',  # Orange-Red
    'Grandmaster': '#E5D050',  # White Gold
    #'Staff':       '#8B0000'   # dark red
}

accelOrder = {'AccelGroup':['Unknown','None','Entry GPU (K80/T4x2/L4x1)', 'High-End GPU (P100/A100/L4x4)','TPU (v2-32/v3-8/VM v3-8)']}
accelColors = {
    'None': '#CCCCFF',  # periwinkle
    'Unknown':'#cccccc',
    'Entry GPU (K80/T4x2/L4x1)': '#91bfdb',  # light blue
    'High-End GPU (P100/A100/L4x4)': '#fc8d59',  # warm orange
    'TPU (v2-32/v3-8/VM v3-8)': '#d73027'  # strong red
}
teamSizeOrder = {'TeamSizeCat':['Solo','Small (2-3)','Big (4+)']}
teamSizeColors = {
    'Solo': '#52b69a',
    'Small (2-3)': '#1a759f',
    'Big (4+)': '#184e77'
}


def get_user_info_by_Id(user_id: int) -> pd.Series:
    if user_id in users.index:
        return users.loc[user_id]
    else:
        return pd.Series(index=users.columns, dtype=object)


def fig_heatmap_cat_cat(users_data, cat1, cat2, title, xaxis_title=None, yaxis_title=None,  normalize='row',level='UserId'):
    """
    Creates a heatmap visualizing the relationship between two categorical variables 
    (`cat1` vs `cat2`) in the given data, with counts aggregated by either unique users 
    (`UserId`) or total submissions (`ScriptId`).

    Normalization can be applied by row, column, or total to show proportions instead of raw counts.
    Axes are optionally reordered based on predefined category orderings.

    Returns a Plotly heatmap figure.
    """
    df = users_data[[level, cat1, cat2]].dropna()

    if level == 'UserId':
        # Count unique users per cat1-cat2 combo
        count_matrix = df.groupby([cat1, cat2])[level].nunique().unstack(fill_value=0)
    elif level == 'ScriptId':
        # Count submissions (rows) per cat1-cat2 combo
        count_matrix = df.groupby([cat1, cat2]).size().unstack(fill_value=0)
    
    # Define orders
    orders = {
        **segmentOrder,
        **medalOrder,
        **tierOrder,
        **accelOrder,
        **teamSizeOrder
    }

    # Reorder rows and columns if orders are known
    if cat1 in orders:
        count_matrix = count_matrix.reindex(orders[cat1])
    if cat2 in orders:
        count_matrix = count_matrix[orders[cat2]]

    # Normalize
    if normalize == 'row':
        norm_matrix = count_matrix.div(count_matrix.sum(axis=1), axis=0)
    elif normalize == 'column':
        norm_matrix = count_matrix.div(count_matrix.sum(axis=0), axis=1)
    elif normalize == 'total':
        norm_matrix = count_matrix / count_matrix.values.sum()
    else:
        norm_matrix = count_matrix  # No normalization

    # Plot
    fig = px.imshow(
        norm_matrix,
        text_auto='.2f',
        labels=dict(x=cat2, y=cat1, color='Proportion'),
        color_continuous_scale='Blues',
        title=title
    )
    # change to default axis title if None
    if xaxis_title == None: xaxis_title = cat2
    if yaxis_title == None: yaxis_title = cat1
    
    fig.update_layout(
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        template='plotly_white',
        height=600
    )

    return fig


def fig_heatmap_cat_cat_grid(users_data, cat1, cat2, years, normalize='row',level='UserId',title=None):
    """
    Input:
      - users_data (DataFrame)
      - cat1, cat2 (str): categorical columns
      - years (list of int): years to facet by
      - normalize (str): normalization method for counts
      - level (str): 'UserId' or 'ScriptId' aggregation
      - title (str, optional): main title
    Output: Plotly figure with grid of heatmaps by year
    """

    cols = len(years)
    fig = make_subplots(rows=1, 
                        cols=cols, 
                        subplot_titles=[f'Submissions in {y}' for y in years],
                       horizontal_spacing=0.02)
    if title == None: title = f'Heatmaps of {cat1} vs {cat2} by {year_col}'
    # update y axis title
    #fig.update_yaxes(title_text=cat1, row=1, col=1)
    
    for i, year in enumerate(years):
        subset = users_data[users_data['SubmissionDate'].dt.year == year]
        heatmap_fig = fig_heatmap_cat_cat(
            subset, cat1, cat2,
            title='',  # Title is handled by subplot_titles
            normalize=normalize,
            level=level
        )

        # Extract heatmap trace(s) from the generated fig
        for trace in heatmap_fig.data:
            fig.add_trace(trace, row=1, col=i+1)

        # Update x axes title per subplot
        fig.update_xaxes(title=dict(text=cat2,font_size=16), showticklabels=True,tickfont_size=15, row=1, col=i+1)
                # Update y axis title only for first plot
        if i == 0:
            fig.update_yaxes(title=dict(text=cat1,font_size=16), showticklabels=True,tickfont_size=15, row=1, col=i+1)
        else:
            fig.update_yaxes(showticklabels=False, row=1, col=i+1)


    fig.update_layout(
        title_text=title,
        title_font_size=18,
        template='plotly_white',
        #height=600,
        width=500 * cols,  # width per number of cols
        coloraxis=dict(colorscale='Blues')
    )
   

    return fig



def fig_comp_completed_counts(users_data, title, group='CompSegment'):
    """
    Input:
      - users_data (DataFrame)
      - title (str): figure title
      - group (str): categorical grouping column
    Output: Plotly stacked bar chart of competition completion counts with drop-off line
    """   
    # change palettes and order based on group
    if group == 'CompSegment':
        colormap = segmentColors
        categoryOrder = segmentOrder
        legend = 'Competition Segment'
    elif group =='UserPerformanceTierName':
        colormap = tierColors
        categoryOrder = tierOrder
        legend = 'Performance Tier'
    elif group == 'MedalType':
        colormap = medalColors
        categoryOrder = medalOrder
        legend = 'Medal Rank'
    elif group =='AccelGroup':
        colormap = accelColors
        categoryOrder = accelOrder
        legend = 'Accelerator Type'
    elif group =='TeamSizeCat':
        colormap = teamSizeColors
        categoryOrder = teamSizeOrder
        legend = 'Team Size'
    else:
        colormap = None
        categoryOrder = None
        legend = group
    # prepare data
    compCountsBySegment = (
        users_data
        .groupby(['UserCompNumber', group])
        .size()
        .reset_index(name='Count')
    )
    compCounts = (
        users_data
        .groupby(['UserCompNumber'])
        .size()
        .reset_index(name='Count')
    )
    compCounts['DropRate'] = compCounts['Count'].pct_change().fillna(0) * -100
    compCounts['Count'] = compCounts['Count'].astype(int)

    # stacked bar base
    fig_counts = px.bar(
        compCountsBySegment,
        x='UserCompNumber',
        y='Count',
        color=group,
        color_discrete_map=colormap,
        category_orders=categoryOrder,
        labels={
            'UserCompNumber':  'n<sup>th</sup> Competition Completed',
            'Count': 'Number of Participants'
        },
        title=title,
        text='Count'
    )
    fig_counts.update_traces(textposition='inside')
    fig_counts.update_layout(
        barmode='stack',
        legend_title_text=legend,
        xaxis=dict(dtick=1, range=[0, 25.5])
    )

    # layer traces
    fig_counts = go.Figure(fig_counts)

    # add drop-off line
    fig_counts.add_trace(go.Scatter(
        x=compCounts['UserCompNumber'],
        y=compCounts['DropRate'],
        mode='lines+markers',
        name='Drop-Off Rate (%)',
        yaxis='y2',
        line=dict(color='firebrick', width=2, dash='dash')
    ))

    # add totals on top of each bar as a text
    fig_counts.add_trace(go.Scatter(
        x=compCounts['UserCompNumber'],
        y=compCounts['Count'],
        text=compCounts['Count'],
        mode='text',
        textposition='top center',
        textfont=dict(size=10, color='black', style='italic',weight='bold'),
        showlegend=False
    ))

    # final axes
    fig_counts.update_layout(
        width=1000,
        margin=dict(l=80, r=100, t=80, b=80),
        legend=dict(
            title_text=legend,
            x=1.08, xanchor='left', yanchor='top'
        ),
        xaxis=dict(dtick=1, range=[0, 15.5]),
        yaxis=dict(title='Number of Participants',
                  rangemode='tozero'),
        yaxis2=dict(
            title='Drop-Off Rate (%)',
            overlaying='y',
            side='right',
            #range=[0, 100],
            tickmode='sync',
            showgrid=True,
            rangemode='tozero'
        )
    )

    return fig_counts


def fig_comp_completed_props(users_data, title, group='CompSegment'):
    """
    Input:
      - users_data (DataFrame)
      - title (str): figure title
      - group (str): categorical grouping column
    Output: Plotly stacked bar chart of competition completion proportions
    """    
    # change palettes and order based on group
    if group == 'CompSegment':
        colormap = segmentColors
        categoryOrder = segmentOrder
        legend = 'Competition Segment'
    elif group =='UserPerformanceTierName':
        colormap = tierColors
        categoryOrder = tierOrder
        legend = 'Performance Tier'
    elif group == 'MedalType':
        colormap = medalColors
        categoryOrder = medalOrder
        legend = 'Medal Rank'
    elif group =='AccelGroup':
        colormap = accelColors
        categoryOrder = accelOrder
        legend = 'Accelerator Type'
    elif group =='TeamSizeCat':
        colormap = teamSizeColors
        categoryOrder = teamSizeOrder
        legend = 'Team Size'
    else:
        colormap = None
        categoryOrder = None
        legend = group    
        
    # Compute proportions within each CompNumber
    compCountsBySegment = users_data.groupby(['UserCompNumber', group]).size().reset_index(name='Count')
    compCountsBySegment['Proportion'] = (
        compCountsBySegment
        .groupby('UserCompNumber')['Count']
        .transform(lambda x: x / x.sum())
    )
    
    fig_props = px.bar(
        compCountsBySegment,
        x='UserCompNumber',
        y='Proportion',
        color=group,
        color_discrete_map=colormap,
        category_orders=categoryOrder,
        text=compCountsBySegment['Proportion'].apply(lambda x: f'{x:.0%}'),
        labels={
            'UserCompNumber': ' n<sup>th</sup> Competition Completed',
            'Proportion': f'{legend} Share'
        },
        title=title
    )
    
    fig_props.update_traces(textposition='inside')
    fig_props.update_layout(
        barmode='stack',
        legend_title_text=legend,
        xaxis=dict(dtick=1, range=[0, 10.5]),
        yaxis=dict(tickformat='.0%', range=[0, 1])
    )
    
    return fig_props

def fig_comp_completed_combined(fig_counts,fig_props,title,xlim=[0,20.5],legend='Competition Segment'):
    """
    Input:
      - fig_counts: Plotly figure of counts
      - fig_props: Plotly figure of proportions
      - title (str): main figure title
      - xlim (list): x-axis limits
      - legend (str): legend title
    Output: Combined Plotly figure with counts + drop-off line above proportions bar chart
    """
    combined = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[0.3, 0.2],
        specs=[[{"secondary_y": True}], [{}]]
    )
    
    # trace from fig_counts into row1
    for trace in fig_counts.data:
        # determine if this trace should go on secondary y:
        use_secondary = getattr(trace, "yaxis", "") == "y2"
        combined.add_trace(trace, row=1, col=1, secondary_y=use_secondary)
    
    # trace from fig_props into row2
    for trace in fig_props.data:
        trace.showlegend = False  # Disable duplicate legend
        combined.add_trace(trace, row=2, col=1)
    
    # figure level layout settings
    combined.update_layout(
        #width=1000,
        height=800,
        barmode="stack",
        #legend=fig_counts.layout.legend,
        legend_title_text=legend,
        title_text=title,
        title_x=0.05,               # 0 = left, 0.5 = center, 1 = right
        title_y=0.96,              
        title_font=dict(size=20)  
    )
    
    # main titles from both figs
    combined.add_annotation(
        text=fig_counts.layout.title.text,
        xref="paper", yref="paper",
        x=0, y=1.05,  # top-left, above row 1
        showarrow=False,
        font=dict(size=12, family="Arial", color="black"),
        align="left"
    )
    
    combined.add_annotation(
        text=fig_props.layout.title.text,
        xref="paper", yref="paper",
        x=0, y=0.39, 
        showarrow=False,
        font=dict(size=12, family="Arial", color="black"),
        align="left"
    )
    
    # axes titles and ticks
    combined.update_xaxes(title_text="n<sup>th</sup> Competition Completed", dtick=1, row=2, col=1)
    combined.update_xaxes(showticklabels=True, row=1, col=1)
    combined.update_xaxes(range=xlim)
    combined.update_xaxes(
        dtick=1,                  # tick every 1 unit
        tickmode='linear',        
        row=1, col=1
    )
    
    combined.update_xaxes(
        dtick=1,
        tickmode='linear',
        row=2, col=1
    )
    
    combined.update_yaxes(title_text="Number of Participants", rangemode='tozero', row=1, col=1, secondary_y=False)
    combined.update_yaxes(title_text="Drop-Off Rate (%)", rangemode='tozero', row=1, col=1,tickmode='sync',secondary_y=True)
    combined.update_yaxes(title_text=f"{legend} Share", tickformat=".0%", row=2, col=1)
    
    return combined

def fig_time_since_last_comp(users_data,title,xlim=[1.5, 10.5],ylim=None,legend_text=None,stat='Median'):    
    """
    Input:
      - users_data (DataFrame) with 'UserCompNumber', 'UserDaysSinceLastComp', 'UserPerformanceTierName'
      - title (str): figure title
      - xlim, ylim (list, optional): axis limits
      - stat (str): 'Median' or 'Mean' for aggregation
    Output: Plotly line chart of median/mean days since last competition by competition number and tier
    """
    # filter data
    time_since_last_comp = users_data[users_data['UserIsFirstComp'] == False][[
        'UserCompNumber', 'UserIsMedalFirstComp', 'UserDaysSinceLastComp', 'UserPerformanceTierName'
    ]].dropna(subset=['UserCompNumber', 'UserDaysSinceLastComp'])

    # choose to extract mean or median data based on stat parameter
    if stat=='Median':
        # Group by CompNumber and Performance Tier, compute median
        time_since_last_comp_agg = time_since_last_comp.groupby(
            ['UserCompNumber', 'UserPerformanceTierName']
        )['UserDaysSinceLastComp'].median().reset_index()
        
    elif stat=='Mean':
        # Group by CompNumber and Performance Tier, compute mean
        time_since_last_comp_agg = time_since_last_comp.groupby(
            ['UserCompNumber', 'UserPerformanceTierName']
        )['UserDaysSinceLastComp'].mean().reset_index()
    
    # Line plot
    fig = px.line(
        time_since_last_comp_agg,
        x='UserCompNumber',
        y='UserDaysSinceLastComp',
        color='UserPerformanceTierName',
        color_discrete_map=tierColors,
        category_orders=tierOrder,
        markers=True,  
        labels={'UserDaysSinceLastComp': f"{stat} Days Since Prior Competition",
               'UserCompNumber':'n<sup>th</sup> Competition Submission'},
        title=title
    )
    
    # Set axis limits
    fig.update_layout(
        xaxis=dict(range=xlim,
                  dtick=1),
        yaxis=dict(range=ylim),        
        legend=dict(
            title_text='Performance Tier',
                x=1.08,
                xanchor='left',
                yanchor='top'
            )
    )
    return fig
    

def fig_time_since_last_comp_by_cohort(users_data, years, title=None, xlim=[1.5, 4.5], ylim=None,stat='Median'):
    """
    Input:
      - users_data (DataFrame)
      - years (list): cohort years
      - title (str, optional): figure title
      - xlim, ylim (list, optional): axis limits
      - stat (str): 'Median' or 'Mean'
    Output: Plotly subplot figure with time since last competition by cohort year
    """   
    if title is None:
        title = "Median Days Since Last Competition by Cohorts"

    subplot_titles = [f"<b>First Competition in {year}</b>" for year in years]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=subplot_titles,
        #shared_xaxes=False,
        shared_yaxes=True,
        vertical_spacing=0.16,
        row_heights=[0.2, 0.2],
    )

    positions = {
        years[0]: (1, 1),
        years[1]: (1, 2),
        years[2]: (2, 1),
        years[3]: (2, 2)
    }

    for i, year in enumerate(years):
        row, col = positions[year]
        data_subset = users_data[users_data['UserFirstCompYear'] == year]

        subfig = fig_time_since_last_comp(
            data_subset,
            title='',
            xlim=xlim,
            ylim=ylim,
            legend_text='Performance Tier',
            stat=stat
        )

        for trace in subfig['data']:
            if i > 0:
                trace.showlegend = False  # Only show legend once
            fig.add_trace(trace, row=row, col=col)

        fig.update_xaxes(range=xlim, title_text="n-th Competition", row=row, col=col, dtick=1)

        # Only set y-axis title on left column
        if col == 1:
            fig.update_yaxes(
                title_text=f"{stat} Days",
                range=ylim,
                row=row,
                col=col
            )
        else:
            fig.update_yaxes(range=ylim, row=row, col=col)

    fig.update_layout(
        height=600,
        title_text=title,
        showlegend=True,
        legend=dict(
            title='Performance Tier',
            x=1.05,
            y=1,
            xanchor='left',
            yanchor='top'
        )
    )

    fig.update_annotations(font=dict(size=12))

    return fig


def fig_comp_completed_by_cohort(users_data,years,title=None,group='CompSegment',legend='Competition Segment',ylim=None):  
    """
    Input:
      - users_data (DataFrame)
      - years (list): cohort years
      - title (str, optional): main title
      - group (str): grouping column for category
      - legend (str): legend title
      - ylim (list, optional): y-axis limits
    Output: Plotly subplot figure of competition completion counts and proportions by cohort
    """    
    subplot_titles = []
    if title == None:
        title = f"<b>Participation and {legend} Share Based On First Year of Competition<b>"
    for y in years:
        subplot_titles.append(f"<b>First Competition in {y}</b>")
        subplot_titles.append(f"<b>First Competition in {y}</b>")
        
    # Create 12 rows × 2 cols
    fig = make_subplots(
        rows=len(years),
        cols=2,
        specs=[[{"secondary_y": True}, {}] for _ in years], # col 1 has 2 axis, col 2 is normal
        shared_xaxes=False,
        shared_yaxes=False,
        vertical_spacing=0.09,
        horizontal_spacing=0.2,
        subplot_titles=subplot_titles
    )
    
    # loop over years
    for i, year in enumerate(years, start=1): #enumerate() produces (index, value) pairs, index start at 1
        users_data_year = users_data[users_data["UserFirstCompYear"] == year]
        
        # Generate the two figures
        fig_counts = fig_comp_completed_counts(
            users_data_year,
            group=group,
            title=""
        )
        fig_props = fig_comp_completed_props(
            users_data_year,
            group=group,
            title=""
        )
        
        # left column: Counts (1st axis) + Drop-off rate (2nd axis)
        for trace in fig_counts.data:
            is_secondary = getattr(trace, "yaxis", "") == "y2"
            fig.add_trace(trace, row=i, col=1, secondary_y=is_secondary)
        
        # right column: Proportions
        for trace in fig_props.data:
            fig.add_trace(trace, row=i, col=2, secondary_y=False)
    
        # Update axes
        fig.update_xaxes(title_text="n-th Competition", title_font=dict(size=11),tickfont=dict(size=9), dtick=1,range=[0, 10.5], row=i, col=1)
        fig.update_xaxes(title_text="n-th Competition", title_font=dict(size=11),tickfont=dict(size=9), dtick=1, range=[0, 10.5], row=i, col=2)
        fig.update_yaxes(title_text="Participants", title_font=dict(size=11),tickfont=dict(size=9), row=i, col=1, range=ylim, secondary_y=False)
        fig.update_yaxes(title_text="Drop-Off Rate (%)", title_font=dict(size=11),tickfont=dict(size=9), range=[0, 100], row=i, col=1, secondary_y=True, tickmode='sync')
        fig.update_yaxes(title_text=f"{legend} Share", title_font=dict(size=11), tickfont=dict(size=9),tickformat=".0%", row=i, col=2)
    
    # Final layout
    fig.update_layout(
        barmode="stack",
        height=900,
        title={
            "text": title,
            "x": 0.05,            
            "xanchor": "left",  # anchor the left edge of the text at x=0.1
            "font": {
                "size": 20
            }
        },
        #legend_title_text=legend,
        legend=dict(
            font=dict(size=11),
            title=dict(text=f"<b>{legend}</b>")
    )
    )
    
    fig.update_annotations(font=dict(size=12))

    # handle duplicate legends
    seen = set()
    for t in fig.data:
        if t.name in seen:
            t.showlegend = False
        else:
            seen.add(t.name)
    
    return fig

    
def fig_ecdf_overall(users_data,title,group='UserPerformanceTierName'):
    """
    Input:
      - users_data (DataFrame) with columns ['TeamId','CompDaysUsedPct', group]
      - title (str): figure title
      - group (str): categorical grouping column
    Output: Plotly ECDF plot of competition duration used proportion by group
    """
    # change palettes and order based on group
    if group == 'CompSegment':
        colormap = segmentColors
        categoryOrder = segmentOrder
        legend = 'Competition Segment'
    elif group =='UserPerformanceTierName':
        colormap = tierColors
        categoryOrder = tierOrder
        legend = 'Performance Tier'
    elif group == 'MedalType':
        colormap = medalColors
        categoryOrder = medalOrder
        legend = 'Medal Rank'
    elif group =='AccelGroup':
        colormap = accelColors
        categoryOrder = accelOrder
        legend = 'Accelerator Type'
    elif group =='TeamSizeCat':
        colormap = teamSizeColors
        categoryOrder = teamSizeOrder
        legend = 'Team Size'
    elif group == 'UserFirstCompYear':
        colormap = None  
        categoryOrder = {'UserFirstCompYear': [2019, 2020, 2023, 2024]}  # this is a fix for ordering the legend of one of the plot 
        legend = 'Cohort Year'
    else:
        colormap = None
        categoryOrder = None
        legend = group
    # select team-level submissions data only  
    users_data = users_data[['TeamId','CompDaysUsedPct',group]].drop_duplicates()
    fig = px.ecdf(
        data_frame=users_data,
        x="CompDaysUsedPct",
        color=group,
        color_discrete_map=colormap,
        category_orders=categoryOrder,
        markers=True,   
        title=title,
    )
    
    fig.update_layout(
        xaxis_title="Percentage of Competition Duration Used",
        yaxis_title="Cumulative Proportion of Submissions",
        legend=dict(
            font=dict(size=11),
            title=dict(text=f"<b>{legend}</b>")
        )
    )
    return fig



def fig_ecdf_by_submission_year(users_data, years, title, group='UserPerformanceTierName'):
    """
    Input:
      - users_data (DataFrame) with submission date column
      - years (list): list of years for facetting
      - title (str): figure title
      - group (str): grouping variable
    Output: Plotly 2x2 grid of ECDF plots by submission year
    """    
    # change palettes and order based on group
    if group=='UserPerformanceTierName':
        colormap = tierColors
        categoryOrder = tierOrder
        legend = 'Performance Tier'
    elif group == 'CompSegment':
        colormap = segmentColors
        categoryOrder = segmentOrder
        legend = 'Competition Segment'
    elif group == 'MedalType':
        colormap = medalColors
        categoryOrder = medalOrder
        legend = 'Medal Rank'
    elif group =='AccelGroup':
        colormap = accelColors
        categoryOrder = accelOrder
        legend = 'Accelerator Type'
    elif group =='TeamSizeCat':
        colormap = teamSizeColors
        categoryOrder = teamSizeOrder
        legend = 'Team Size'
    else:
        colormap = None
        categoryOrder = None
        legend = group
    # Filter only submissions from selected years
    # select team-level submissions data only  
    data_all_year = (
        users_data.loc[users_data['SubmissionDate'].dt.year.isin(years), 
                       ['TeamId', 'CompDaysUsedPct', 'SubmissionDate',group]].drop_duplicates()
    )
    # Create subplot figure: 2x2 grid
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[f"<b>Submissions in {y}</b>" for y in years],
        shared_xaxes=True,
        shared_yaxes=True,
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    # Row/column mapping
    positions = {
        years[0]: (1, 1),
        years[1]: (1, 2),
        years[2]: (2, 1),
        years[3]: (2, 2)
    }
    
    # Loop through years and add ECDF plots
    for year in years:
        row, col = positions[year]
        data_year = data_all_year[data_all_year['SubmissionDate'].dt.year == year]
    
        ecdf_fig = px.ecdf(
            data_frame=data_year,
            x="CompDaysUsedPct",
            color=group,
            color_discrete_map=colormap,
            category_orders=categoryOrder,
            markers=True,
        )
    
        for trace in ecdf_fig.data:
            fig.add_trace(trace, row=row, col=col)
    
    # Final layout
    fig.update_layout(
        height=800,
        width=900,
        title_text=title,
        title_x=0.05,
        showlegend=True,
        legend=dict(
            font=dict(size=11),
            title=dict(text=f"<b>{legend}</b>")
        )
    )
    
    # Update axis labels
    fig.update_xaxes(title_text="Competition Duration Used (%)")
    fig.update_yaxes(title_text="Cumulative Proportion")
    
    # Optional: Reduce duplicate legends
    seen = set()
    for trace in fig.data:
        if trace.name in seen:
            trace.showlegend = False
        else:
            seen.add(trace.name)
    return fig


def fig_time_series_submissions_ma(users_data, group, title, min_year=2019, window=30, min_periods=1):
    """
    Input:
      - users_data (DataFrame) with 'SubmissionDate', group column, and 'ScriptId'
      - group (str): categorical grouping column
      - title (str): figure title
      - min_year (int): filter submissions from this year onwards
      - window (int): rolling window size (days)
      - min_periods (int): minimum periods for rolling average
    Output: Plotly line chart of rolling mean submission counts over time by group
    """    
    # Pick color map and category order
    if group == 'UserPerformanceTierName':
        colormap = tierColors
        category_order = tierOrder[group]
        legend_title = 'Performance Tier'
    elif group == 'CompSegment':
        colormap = segmentColors
        category_order = segmentOrder[group]
        legend_title = 'Competition Segment'
    elif group == 'MedalType':
        colormap = medalColors
        category_order = medalOrder[group]
        legend_title = 'Medal Rank'
    elif group == 'AccelGroup':
        colormap = accelColors
        category_order = accelOrder[group]
        legend_title = 'Accelerator Type'
    elif group == 'TeamSizeCat':
        colormap = teamSizeColors
        category_order = teamSizeOrder[group]
        legend_title = 'Team Size'
    else:
        colormap = None
        category_order = sorted(users_data[group].dropna().unique())
        legend_title = group

    # Prep time series in long format
    df = users_data[users_data['SubmissionDate'].dt.year >= min_year]
    df = df.groupby(['SubmissionDate', group])['ScriptId'].count().reset_index(name='SubmissionCount')
    
    # compute moving average in wide format
    df_wide = df.pivot(index='SubmissionDate', columns=group, values='SubmissionCount').sort_index()
    df_wide = df_wide.rolling(window=window, min_periods=min_periods).mean()
    
    # pivot back to long format for line plot
    df_long = df_wide.reset_index().melt(id_vars='SubmissionDate', var_name=group, value_name='SubmissionCount_MA')
    fig = px.line(
        df_long, 
        x="SubmissionDate", 
        y="SubmissionCount_MA", 
        color=group,
        color_discrete_map=colormap,
        category_orders={group: category_order},
        title=title,
        labels={
            'SubmissionDate': 'Submission Date',
            'SubmissionCount_MA': f'Submission Count - {window}-Day MA',
            group: legend_title  # label for legend
        }
    )
    
    fig.update_layout(
        legend_title_text=legend_title
    )
    
    return fig


def fig_time_series_submissions_ma_grid(users_data,groups,titles,
                                        main_title,
                                        window=60,
                                        min_periods=1,
                                        min_year=2019,
                                        height=1200):
    """
    Input:
      - users_data (DataFrame)
      - groups (list of str): list of grouping columns for separate plots
      - titles (list of str): subplot titles
      - main_title (str): overall figure title
      - window, min_periods, min_year: parameters for rolling mean calculation
      - height (int): figure height
    Output: Plotly vertical grid subplot figure of rolling mean submission counts by multiple groups
    """
    # Create 3-row subplot
    fig_subplots = make_subplots(
        rows=len(groups), cols=1,
        shared_xaxes=False,
        subplot_titles=titles,
        vertical_spacing=0.1
    )
    
    for i, group in enumerate(groups):
        fig = fig_time_series_submissions_ma(
            users_data,
            group=group,
            title='',
            min_year=min_year,
            window=window,
            min_periods=min_periods
        )
    
        seen = set()  # Track which legend labels already added
        for trace in fig.data:
            show_legend = trace.name not in seen
            seen.add(trace.name)
    
            fig_subplots.add_trace(
                go.Scatter(
                    x=trace.x,
                    y=trace.y,
                    name=trace.name,
                    mode='lines',
                    line=dict(color=trace.line.color),
                    legendgroup=f"{group}",      # keep legend group per subplot
                    showlegend=show_legend       # only show once per label
                ),
                row=i+1, col=1
            )
    
        # Manually update axis titles per row
        fig_subplots.update_yaxes(title_text=f'Submissions {window}-Days MA', row=i+1, col=1)
        fig_subplots.update_xaxes(title_text='Submission Date', row=i+1, col=1)
    
    # calculate gap between legend groups
    fig_subplots.update_layout(
        margin=dict(t=100, b=40)
    )
    
    usable_height = height - fig_subplots.layout.margin.t - fig_subplots.layout.margin.b
    row_height = usable_height / len(groups)
    legend_tracegroupgap = row_height * 0.75
    
    # Final layout 
    fig_subplots.update_layout(
        height=height,
        title_text=main_title,
        template='plotly_white',
        legend_tracegroupgap=legend_tracegroupgap,  
        margin=dict(t=100, b=40)
    )
    
    return fig_subplots


def fig_time_series_mean_ma(users_data, group, title, y='KernelEngagement',yname='Engagement Score', min_year=2019, window=30, min_periods=1):
    """
    Input:
      - users_data (DataFrame) with date, group column, and metric column y
      - group (str): grouping column
      - title (str): figure title
      - y (str): metric column name
      - yname (str): metric display name for labels
      - min_year (int): filter start year
      - window (int): rolling window size
      - min_periods (int): minimum periods for rolling mean
    Output: Plotly line chart of rolling mean metric over time by group
    """
    # Pick color map and category order
    if group == 'UserPerformanceTierName':
        colormap = tierColors
        category_order = tierOrder[group]
        legend_title = 'Performance Tier'
    elif group == 'CompSegment':
        colormap = segmentColors
        category_order = segmentOrder[group]
        legend_title = 'Competition Segment'
    elif group == 'MedalType':
        colormap = medalColors
        category_order = medalOrder[group]
        legend_title = 'Medal Rank'
    elif group == 'AccelGroup':
        colormap = accelColors
        category_order = accelOrder[group]
        legend_title = 'Accelerator Type'
    elif group == 'TeamSizeCat':
        colormap = teamSizeColors
        category_order = teamSizeOrder[group]
        legend_title = 'Team Size'
    else:
        colormap = None
        category_order = sorted(users_data[group].dropna().unique())
        legend_title = group

    # Prep time series in long format
    df = users_data[users_data['SubmissionDate'].dt.year >= min_year]
    df = df.groupby(['SubmissionDate', group])[y].mean().reset_index(name='AvgMetric')
    
    # compute moving average in wide format
    df_wide = df.pivot(index='SubmissionDate', columns=group, values='AvgMetric').sort_index()
    df_wide = df_wide.rolling(window=window, min_periods=min_periods).mean()
    
    # pivot back to long format for line plot
    df_long = df_wide.reset_index().melt(id_vars='SubmissionDate', var_name=group, value_name='AvgMetric_MA')
    fig = px.line(
        df_long, 
        x="SubmissionDate", 
        y="AvgMetric_MA", 
        color=group,
        color_discrete_map=colormap,
        category_orders={group: category_order},
        title=title,
        labels={
            'SubmissionDate': 'Submission Date',
            'AvgMetric_MA': f'Mean {yname} - {window}-Day MA',
            group: legend_title  # label for legend
        }
    )
    
    fig.update_layout(
        legend_title_text=legend_title
    )
    
    return fig



# getting fields needed:            
submissions = pd.read_csv(f"{MK_PATH}/Submissions.csv",
                          usecols=["Id","TeamId","SourceKernelVersionId","SubmissionDate","IsAfterDeadline"])  #"PrivateScoreLeaderboardDisplay","IsSelected",
submissions['SubmissionDate'] = pd.to_datetime(submissions['SubmissionDate'])


teamMemberships = pd.read_csv(f"{MK_PATH}/TeamMemberships.csv",
                             usecols=["Id","TeamId","UserId"])


competitions = pd.read_csv(f"{MK_PATH}/Competitions.csv",
                          usecols=['Id','HostSegmentTitle','EnabledDate','DeadlineDate','Title'])
competitions['EnabledDate'] = pd.to_datetime(competitions['EnabledDate']).dt.floor('D') # keep only the date component to avoid negative comp days
competitions['DeadlineDate'] = pd.to_datetime(competitions['DeadlineDate']).dt.floor('D') 


kernels = pd.read_csv(f"{MK_PATH}/Kernels.csv",
                     usecols=['Id','Medal','TotalVotes','TotalComments','TotalViews']) #'CurrentUrlSlug',


versions = pd.read_csv(f"{MK_PATH}/KernelVersions.csv",
                      usecols=['Id','ScriptId','AcceleratorTypeId']) #,'RunningTimeInMilliseconds','TotalLines'

accelerators = pd.read_csv(f"{MK_PATH}/KernelAcceleratorTypes.csv")

teams = pd.read_csv(f"{MK_PATH}/Teams.csv",
                   usecols=['Id','CompetitionId','TeamLeaderId','Medal','IsBenchmark'])


users = pd.read_csv(f"{MK_PATH}/Users.csv",
                   usecols=["Id","UserName","PerformanceTier"])
users.set_index("Id", inplace = True)

tier_map = {
    #0: 'Unranked',  # Originally 'Novice', Retired
    1: 'Unranked',  # Originally 'Contributor', Retired
    2: 'Expert',
    3: 'Master',
    4: 'Grandmaster',
    5: 'Staff'  
}
users['PerformanceTierName'] = users['PerformanceTier'].map(tier_map)


team_submissions = duckdb.query(
    '''
    SELECT *
    FROM (
        SELECT 
            submissions.TeamId,
            submissions.SourceKernelVersionId,
            submissions.SubmissionDate,
            ROW_NUMBER() OVER (PARTITION BY TeamId ORDER BY SubmissionDate DESC) AS row,
            teams.CompetitionId,
            teams.Medal
        FROM submissions LEFT JOIN teams ON submissions.TeamId = teams.Id
        WHERE 
            submissions.IsAfterDeadline == False AND
            submissions.SourceKernelVersionId IS NOT NULL AND
            teams.IsBenchmark == False
    ) teams
    WHERE row = 1;
    ''').to_df()

team_submissions.head()


user_team_submissions = duckdb.query(
    '''
    SELECT 
        teamMemberships.TeamId AS TeamId,
        teamMemberships.UserId AS UserId,
        team_submissions.CompetitionId AS CompetitionId,
        team_submissions.Medal AS TeamMedal,
        team_submissions.SubmissionDate AS SubmissionDate,
        versions.ScriptId AS ScriptId,
        accelerators.Label AS Accelerator,
        competitions.HostSegmentTitle AS CompSegment,
        competitions.Title AS CompTitle,
        competitions.EnabledDate AS EnabledDate,
        competitions.DeadlineDate AS DeadlineDate,
        kernels.Medal AS KernelMedal,
        kernels.TotalViews AS KernelViews,
        kernels.TotalVotes AS KernelVotes,
        kernels.TotalComments AS KernelComments
        --versions.RunningTimeInMilliseconds AS KernelRuntime,
        --versions.TotalLines AS KernelTotalLines
    FROM teamMemberships 
    INNER JOIN team_submissions  ON teamMemberships.TeamId = team_submissions.TeamId
    INNER JOIN versions ON team_submissions.SourceKernelVersionId = versions.Id
    INNER JOIN competitions ON team_submissions.CompetitionId = competitions.Id
    LEFT JOIN kernels ON versions.ScriptId = kernels.Id
    LEFT JOIN accelerators ON versions.AcceleratorTypeId = accelerators.Id
    ''').to_df()

# get more columns
user_info = user_team_submissions['UserId'].apply(get_user_info_by_Id)
user_team_submissions = user_team_submissions.join(user_info)

# filter out staff users
user_team_submissions = user_team_submissions[~(user_team_submissions['PerformanceTierName']=='Staff')]

# filter out 'Community','Getting Started', 'Analytics' and 'Recruitment' competitions
user_team_submissions = user_team_submissions[~user_team_submissions['CompSegment'].isin(['Community', 'Recruitment','Analytics','Getting Started'])]

user_team_submissions.head()


users_comp_stats = duckdb.query(
    '''
    SELECT 
        UserId,
        UserName,
        TeamId,
        ScriptId,
        Accelerator,
        CompetitionId,
        PerformanceTierName AS UserPerformanceTierName, 
        SubmissionDate,
        CompSegment,
        CompTitle,
        EnabledDate,
        DeadlineDate,
        ROW_NUMBER() OVER (PARTITION BY UserId ORDER BY SubmissionDate ASC) AS UserCompNumber,
        TeamMedal AS Medal,
        KernelViews,
        KernelVotes,
        KernelComments
        --KernelRuntime,
        --KernelTotalLines
    FROM user_team_submissions
    ''').to_df()
users_comp_stats['Medal'] = users_comp_stats['Medal'].fillna(0)
users_comp_stats['Accelerator'] = users_comp_stats['Accelerator'].fillna('None')
#users_comp_stats['KernelMedal'] = users_comp_stats['KernelMedal'].fillna(0)

# map medal type
medal_mapping = {0: 'None', 1: 'Gold', 2: 'Silver', 3: 'Bronze'}
users_comp_stats['MedalType'] = users_comp_stats['Medal'].map(medal_mapping)

# map accelerators into 4 main types
accel_group_map = {
    'None': 'None',
    # Entry GPUs
    'GPU K80': 'Entry GPU (K80/T4x2/L4x1)',
    'GPU T4 x2': 'Entry GPU (K80/T4x2/L4x1)',
    'GPU L4 x1': 'Entry GPU (K80/T4x2/L4x1)',
    # High-End GPUs
    'GPU P100': 'High-End GPU (P100/A100/L4x4)',
    'GPU A100': 'High-End GPU (P100/A100/L4x4)',
    'GPU L4 x4': 'High-End GPU (P100/A100/L4x4)',
    # TPUs
    'TPU v2-32': 'TPU (v2-32/v3-8/VM v3-8)',
    'TPU v3-8': 'TPU (v2-32/v3-8/VM v3-8)',
    'TPU VM v3-8': 'TPU (v2-32/v3-8/VM v3-8)'
}
users_comp_stats['AccelGroup'] = users_comp_stats['Accelerator'].map(accel_group_map).fillna('Unknown')

# calculate days since last competition
#users_comp_stats['SubmissionDate'] = pd.to_datetime(users_comp_stats['SubmissionDate'])
users_comp_stats = users_comp_stats.sort_values(by=['UserId', 'SubmissionDate'])
users_comp_stats['UserDaysSinceLastComp'] = users_comp_stats.groupby('UserId')['SubmissionDate'].diff().dt.days
users_comp_stats['UserDaysSinceLastComp'] = users_comp_stats['UserDaysSinceLastComp'].fillna(0)

# compute days used for competition
users_comp_stats['CompDaysUsed'] = (users_comp_stats['SubmissionDate'] - users_comp_stats['EnabledDate']).dt.days

# compute days used as percentage of whole comp duration
users_comp_stats['CompDaysUsedPct'] = users_comp_stats['CompDaysUsed'] / (users_comp_stats['DeadlineDate'] - users_comp_stats['EnabledDate']).dt.days 
users_comp_stats['CompDaysUsedPct'] = users_comp_stats['CompDaysUsedPct'].clip(upper=1) # cap at 1 and 0
users_comp_stats['CompDaysUsedPct'] = users_comp_stats['CompDaysUsedPct'].clip(lower=0)

# compute IsSolo indicator
team_sizes = users_comp_stats.groupby('TeamId')['UserId'].nunique().reset_index(name='TeamSize')
users_comp_stats = users_comp_stats.merge(team_sizes, on='TeamId')
users_comp_stats['IsSolo'] = users_comp_stats['TeamSize'] == 1

# team size categories
users_comp_stats['TeamSizeCat'] = pd.cut(
    users_comp_stats['TeamSize'],
    bins=[0,1,3,float('inf')],
    labels=['Solo','Small (2-3)','Big (4+)'],
    right=True,
    include_lowest=True
)

# compute IsFirstComp indicator
users_comp_stats['UserIsFirstComp'] = users_comp_stats['UserCompNumber'] == 1

# compute IsMedal indicator
users_comp_stats['IsMedal'] = users_comp_stats['Medal'] > 0

# compute the year of first competition completion
first_comp_dates = users_comp_stats[users_comp_stats['UserCompNumber']==1][['UserId','SubmissionDate']]
first_comp_dates['UserFirstCompYear'] = first_comp_dates['SubmissionDate'].dt.year
users_comp_stats = users_comp_stats.merge(first_comp_dates[['UserId', 'UserFirstCompYear']], 
                          on=['UserId'],
                          how='left')

# exclude submission records < 2015 
users_comp_stats = users_comp_stats[users_comp_stats['SubmissionDate'].dt.year >= 2015]

# compute proxy engagement score
users_comp_stats['KernelEngagement'] = (users_comp_stats['KernelVotes'] + users_comp_stats['KernelComments']) / (users_comp_stats['KernelViews'] + 1)

# compute adjusted version of engagement (log-denominator)
users_comp_stats['KernelAdjustedEngagement'] = (users_comp_stats['KernelVotes'] + users_comp_stats['KernelComments']) / np.log(users_comp_stats['KernelViews'] + 1)

# drop any duplicates
users_comp_stats = users_comp_stats.drop_duplicates()

users_comp_stats.head()


unique_users = len(users_comp_stats['UserId'].unique())
na_users = len(users_comp_stats[(users_comp_stats['UserPerformanceTierName'].isna())]['UserId'].unique())
negative_days_used = len(users_comp_stats[users_comp_stats['CompDaysUsed']<0]['UserId'].unique())
print(f"Among the {unique_users} unique users who have joined at least one competition:\n")
print(f"{na_users} of them have missing values in 'PerformanceTier'")
print(f"{negative_days_used} of them have negative days used for competition\n")

print("I will now remove rows with or negative day values.")
# clean nulls and negative days
users_comp_stats = users_comp_stats.dropna()
users_comp_stats = users_comp_stats[~(users_comp_stats['CompDaysUsed']<0)]

print(f"There are {len(users_comp_stats['TeamId'].unique())} unique teams and {len(users_comp_stats['UserId'].unique())} unique users after cleaning")
print(f"\nMinimum number of competitions joined: {users_comp_stats['UserCompNumber'].min()}")
print(f"\nMaximum number of competitions joined: {users_comp_stats['UserCompNumber'].max()}")
print(f"\nTotal Medals combined for all users:\n{users_comp_stats['Medal'].value_counts()}")
print(f"\nTotal Competition participants in different segments:\n{users_comp_stats['CompSegment'].value_counts()}")
print(f"\nCount of Accelerator usage by competition participants:\n{users_comp_stats['AccelGroup'].value_counts()}")



# create indicator columns for each medal type
users_comp_stats['GoldMedal'] = (users_comp_stats['Medal'] == 1).astype(int)
users_comp_stats['SilverMedal'] = (users_comp_stats['Medal'] == 2).astype(int)
users_comp_stats['BronzeMedal'] = (users_comp_stats['Medal'] == 3).astype(int)
users_comp_stats['IsMedalInt'] = (users_comp_stats['IsMedal']).astype(int)

# compute total medals and comps currently:
user_medal_totals = (
    users_comp_stats.groupby('UserId')
    .agg(
        UserTotalMedalsFinal = ('IsMedalInt', 'sum'),
        UserGoldsFinal = ('GoldMedal', 'sum'),
        UserSilversFinal = ('SilverMedal', 'sum'),
        UserBronzesFinal = ('BronzeMedal', 'sum'),
        UserTotalComps = ('UserCompNumber', 'max')
    )
    .reset_index()
)
users_comp_stats = users_comp_stats.merge(
    user_medal_totals,
    on='UserId',
    how='left'
)
'''
# compute cumulative gold, silver, bronze medals for the user as of submission date:
users_comp_stats = users_comp_stats.sort_values(['UserId', 'SubmissionDate'])
users_comp_stats['UserCumulativeGolds'] = users_comp_stats.groupby('UserId')['GoldMedal'].cumsum()
users_comp_stats['UserCumulativeSilvers'] = users_comp_stats.groupby('UserId')['SilverMedal'].cumsum()
users_comp_stats['UserCumulativeBronzes'] = users_comp_stats.groupby('UserId')['BronzeMedal'].cumsum()
users_comp_stats['UserCumulativeTotalMedals'] = users_comp_stats.groupby('UserId')['IsMedalInt'].cumsum()
'''
# compute MedalNumber (n-th medal of user)
user_medals = duckdb.query(
    '''
        SELECT 
            UserId, 
            UserCompNumber,
            IsMedal,
            ROW_NUMBER() OVER (PARTITION BY UserId ORDER BY UserCompNumber) AS UserMedalNumber
        FROM users_comp_stats
        WHERE IsMedal = TRUE
    ''').to_df()
users_comp_stats = users_comp_stats.merge(user_medals[['UserId', 'UserCompNumber','UserMedalNumber']], 
                          on=['UserId', 'UserCompNumber'],
                          how='left')
users_comp_stats['UserMedalNumber'] = users_comp_stats['UserMedalNumber'].fillna(0)

# compute IsFirstMedal indicator
users_comp_stats['UserIsFirstMedal'] = users_comp_stats['UserMedalNumber'] == 1

# compute IsMedalFirstComp indicator (getting a medal on the first competition)
medal_first_comp_ids = (users_comp_stats[users_comp_stats['UserIsFirstComp'] & users_comp_stats['UserIsFirstMedal']]['UserId'].unique())
users_comp_stats['UserIsMedalFirstComp'] = users_comp_stats['UserId'].isin(medal_first_comp_ids)

# compute submission year
users_comp_stats['SubmissionYear'] = users_comp_stats['SubmissionDate'].dt.year



# Show users with Medal but still marked Novice
suspicious = users_comp_stats[
    (users_comp_stats['UserPerformanceTierName']=='Unranked') &
    (users_comp_stats['IsMedal'] == True)
]
sus_users = (suspicious[['UserId','UserName','CompetitionId','MedalType','UserPerformanceTierName','CompTitle','UserTotalMedalsFinal']].drop_duplicates())
sus_users.head()


# Show users with Medal from Playground Competitions
suspicious2 = users_comp_stats[
    (users_comp_stats['CompSegment']=='Playground') &
    (users_comp_stats['IsMedal'] == True)
]

sus2_users = (suspicious2[['UserId','UserName','CompetitionId','MedalType','UserPerformanceTierName','CompTitle','UserTotalMedalsFinal']].drop_duplicates())
sus2_users.head(5)


# check final dataframe
users_comp_stats.sort_values(by=['UserId','UserCompNumber']).head(5)


users_comp_stats.describe(include='all')


print(users_comp_stats.info())


users_comp_stats.info()


users_comp_stats.sort_values(by=['UserId','UserCompNumber']).head(5)


groups = ['CompSegment', 'TeamSizeCat', 'AccelGroup','MedalType']
titles = [
    'By Competition Segment',
    'By Team Size',
    'By Accelerator usage',
    'By Medal Awarded'
]

fig = fig_time_series_submissions_ma_grid(
    users_comp_stats,
    groups=groups,
    titles=titles,
    min_year=2019,
    main_title='<b>Figure 1: Daily Final Submission Counts Over Time by Attributes</b><br><sup>60-Day Moving Average Line Chart</sup>',
    window=60,
    min_periods=1,
    height=900)
fig.write_html('figure1.html')

fig.show(renderer='iframe')
#IFrame('figure1.html', width='100%',height=600)


fig = fig_time_series_submissions_ma(users_comp_stats[(users_comp_stats['UserFirstCompYear']>=2019)&(users_comp_stats['IsSolo']==True)],
                   group='UserFirstCompYear',
                   title='<b>Figure 2: Daily Final Submission Counts by Cohorts</b><br><sup>60-Day Moving Average Line Chart. Data includes Kernel Submissions of the <b>Final Version and from Solo Teams only.</sup>',
                   min_year=2019,
                   min_periods=1,
                   window=60)
fig.update_layout(height=500)
fig.write_html('figure2.html')

fig.show(renderer='iframe')
#IFrame('figure2.html', width='100%',height=600)


fig = fig_time_series_mean_ma(users_comp_stats[users_comp_stats['IsSolo']==True],
                   group='UserPerformanceTierName', y='KernelEngagement', yname='Engagement Score',
                   title='<b>Figure 3: Daily Mean Engagement Score by Current User Tiers</b><br><sup>60-Day Moving Average Line Chart. Data includes Kernel Submissions of the <b>Final Version and from Solo Teams only.</b><br><b>Engagement Score </b>= (Upvotes + Comments)/(Views + 1)</sup>',
                   min_year=2018,
                   min_periods=1,
                   window=60)
fig.update_layout(height=500)
fig.write_html('figure3.html')

fig.show(renderer='iframe')
#IFrame('figure3.html', width='100%',height=600)


fig = fig_time_series_mean_ma(users_comp_stats[users_comp_stats['IsSolo']==True],
                   group='UserPerformanceTierName',y='KernelAdjustedEngagement',yname='Adjusted Engagement',
                   title='<b>Figure 4: Daily Mean Adjusted Engagement Score by Current User Tiers</b><br><sup>60-Day Moving Average Line Chart. Data includes Kernel Submissions of the <b>Final Version and from Solo Teams only.</b><br><b>Adjusted Engagement Score</b> = (Upvotes + Comments)/log(Views + 1)</sup>',
                   min_year=2019,
                   min_periods=1,
                   window=60)
fig.update_layout(height=500)
fig.write_html('figure4.html')
fig.show(renderer='iframe')
#IFrame('figure4.html', width='100%',height=600)


fig = fig_time_series_mean_ma(users_comp_stats,
                   group='CompSegment',y='KernelEngagement',yname='Engagement Score',
                   title='<b>Figure 5: Daily Mean Engagement Score by Competition Segments</b><br><sup>60-Day Moving Average Line Chart. Data includes Kernel Submissions of the Final Version Only.<br><b>Engagement Score </b>= (Upvotes + Comments)/(Views + 1)</sup>',
                   min_year=2019,
                   min_periods=1,
                   window=60)
fig.update_layout(height=500)
fig.write_html('figure5.html')

fig.show(renderer='iframe')
#IFrame('figure5.html', width='100%',height=600)


fig = fig_time_series_mean_ma(users_comp_stats,
                   group='CompSegment',y='KernelAdjustedEngagement',yname='Adjusted Engagement',
                   title='<b>Figure 6: Daily Mean Adjusted Engagement Score by Competition Segments</b><br><sup>60-Day Moving Average Line Chart. Data includes Kernel Submissions of the Final Version Only.<br><b>Adjusted Engagement Score </b>= (Upvotes + Comments)/log(Views + 1)</sup>',
                   min_year=2019,
                   min_periods=1,
                   window=60)
fig.update_layout(height=500)
fig.write_html('figure6.html')

fig.show(renderer='iframe')
#IFrame('figure6.html', width='100%',height=600)


fig_counts = fig_comp_completed_counts(users_comp_stats,
                                       title="How Many Kagglers Have Completed Their n-th Competition?<br><sup>Most People Stop at the First One.</sup>",
                                      group='CompSegment')
fig_props = fig_comp_completed_props(users_comp_stats,
                                     title="Which Competition Segments do Kagglers join in their n<sup>th</sup> Competition?<br><sup>Segment Share is the percentage of competitions from each segment across all users’ n<sup>th</sup> competition entries.</sup>",
                                    group='CompSegment')

fig = fig_comp_completed_combined(fig_counts,fig_props,title="<b>Figure 7: Competition Participation drop-off and Segment shares</b><br><sup>Users who started competing in 2015 or later, by Competition Segment Joined.</sup>",xlim=[0, 20.5],legend='Competition Segment')
fig.update_layout(height=700)
fig.write_html('figure7.html')
fig.show(renderer='iframe')
#IFrame('figure7.html', width='100%',height=600)


fig_counts = fig_comp_completed_counts(users_comp_stats,
                                       title="Count of Current Performance Tier of Kagglers on their n-th Competition<br><sup>Most of the Current Novices Stop at the First One.</sup>",
                                      group='UserPerformanceTierName')
fig_props = fig_comp_completed_props(users_comp_stats,
                                     title="Which Performance Tiers Compete More Often?<br><sup>Share of Competitions by Tier at Each n<sup>th</sup> Entry</sup>",
                                    group='UserPerformanceTierName')

fig = fig_comp_completed_combined(fig_counts,fig_props,
                            title="<b>Figure 8: Participation Patterns by Current Performance Tier</b><br><sup>Users who started competing in 2015 or later, grouped by Present Tier (as of 2025)</sup>",
                            xlim=[0, 20.5],
                           legend='Performance Tier')
fig.update_layout(height=700)
fig.write_html('figure8.html')
fig.show(renderer='iframe')
#IFrame('figure8.html', width='100%',height=600)


users_total_comp = users_comp_stats[['UserId','UserPerformanceTierName','UserTotalComps']].drop_duplicates()
x0 = users_total_comp[users_total_comp['UserPerformanceTierName']=='Grandmaster']['UserTotalComps']
x1 = users_total_comp[users_total_comp['UserPerformanceTierName']=='Master']['UserTotalComps']
x2 = users_total_comp[users_total_comp['UserPerformanceTierName']=='Expert']['UserTotalComps']
x3 = users_total_comp[users_total_comp['UserPerformanceTierName']=='Unranked']['UserTotalComps']


fig = go.Figure()
fig.add_trace(go.Histogram(
    x=x0,
    histnorm='percent',
    name='Grandmaster', # name used in legend and hover labels
    xbins=dict( # bins used for histogram
        start=0.5,
        end=7.5,
        size=1
    ),
    marker_color=tierColors['Grandmaster'],
    #opacity=0.75
))
fig.add_trace(go.Histogram(
    x=x1,
    histnorm='percent',
    name='Master',
    xbins=dict(
        start=0.5,
        end=7.5,
        size=1
    ),
    marker_color=tierColors['Master'],
    #opacity=0.75
))
fig.add_trace(go.Histogram(
    x=x2,
    histnorm='percent',
    name='Expert',
    xbins=dict(
        start=0.5,
        end=7.5,
        size=1
    ),
    marker_color=tierColors['Expert'],
    #opacity=0.75
))

fig.add_trace(go.Histogram(
    x=x3,
    histnorm='percent',
    name='Unranked',
    xbins=dict(
        start=0.5,
        end=7.5,
        size=1
    ),
    marker_color=tierColors['Unranked'],
    #opacity=0.75
))

fig.update_layout(
    title_text='<b>Figure 9: Normalized Histogram of Total Competitions Completed by Tiers<br></b><sup>For Users With First Competition Starting 2015 or Later</sup>', # title of plot
    xaxis_title_text='Total Competitions', # xaxis label
    xaxis_range=[0,7.5],
    yaxis_title_text='Percent', # yaxis label
    bargap=0.2, # gap between bars of adjacent location coordinates
    bargroupgap=0.1 # gap between bars of the same location coordinates
)
fig.write_html('figure9.html')

fig.show(renderer='iframe')
#IFrame('figure9.html', width='100%',height=600)


fig = fig_comp_completed_by_cohort(users_comp_stats,
                       title="<b>Figure 10: Participation and Performance Tier Share Based On Cohorts",
                       years=[2019,2020,2023,2024],group='UserPerformanceTierName',legend='Performance Tier', ylim=[0,4000])
fig.update_layout(height=800)
fig.write_html('figure10.html')

fig.show(renderer='iframe') 
#IFrame('figure10.html', width='100%',height=600)


fig = fig_heatmap_cat_cat(
    users_comp_stats, 
    cat1='UserFirstCompYear', 
    cat2='UserPerformanceTierName',
    xaxis_title='Performance Tier',
    yaxis_title='First Competition Year (Cohort)',
    normalize='column',
    level='UserId',
    title="<b>Figure 11: Distribution of First Competition Year By Performance Tier</b><br><sup>Count Heatmap of Users in Each Cohort vs Performance Tier, <b>normalized by columns.</sup>"
)
fig.update_yaxes(dtick=1)  
fig.update_layout(height=500)
fig.write_html('figure11.html')

fig.show(renderer='iframe')
#IFrame('figure11.html', width='100%',height=600)


fig = fig_comp_completed_by_cohort(users_comp_stats,
                       title="<b>Figure 12: Participation Patterns Based On Cohorts",
                       years=[2019,2020,2023,2024],group='CompSegment',ylim=[0,4000])
fig.update_layout(height=800)
fig.write_html('figure12.html')

fig.show(renderer='iframe')  
#IFrame('figure12.html', width='100%',height=600)


fig = fig_comp_completed_by_cohort(users_comp_stats,
                       title="<b>Figure 13: Participation and Team Size Share Based On Cohorts",
                       years=[2019,2020,2023,2024],group='TeamSizeCat',legend='Team Size', ylim=[0,4000])
fig.update_layout(height=800)
fig.write_html('figure13.html')

fig.show(renderer='iframe')  
#IFrame('figure13.html', width='100%',height=600)


fig = fig_heatmap_cat_cat_grid(
    users_comp_stats,
    cat1='IsSolo',
    cat2='IsMedal',
    years=[2019,2020,2023,2024],
    normalize='column',
    level='ScriptId', # kernel level count
    title='<b>Figure 14: Count Heatmap of Solo Indicator vs Medal Indicator</b><br><sup>Normalized by Columns.</sup>'
)
fig.update_layout(height=350,
                 width=1000)
fig.write_html('figure14.html')

fig.show(renderer='iframe')
#IFrame('figure14.html', width='100%',height=600)


fig = fig_comp_completed_by_cohort(users_comp_stats,
                       title="<b>Figure 15: Participation and Accelerator Usage Based On Cohorts",
                       years=[2019,2020,2023,2024],group='AccelGroup',legend='Accelerator Used', ylim=[0,4000])
fig.update_layout(height=800)
fig.write_html('figure15.html')

fig.show(renderer='iframe') 
#IFrame('figure15.html', width='100%',height=600)


fig = fig_time_since_last_comp(users_comp_stats,
                         title="<b>Figure 16.1: Median Days Since Last Competition's Submission (Overall)</b><br><sup>For All Users Starting in 2015 or Later by Performance Tier Currently.</sup>",
                        xlim=[1.5,7.5],ylim=[0,300])
fig.update_layout(height=450)
fig.write_html('figure16_1.html')

fig.show(renderer='iframe') 
#IFrame('figure16_1.html', width='100%',height=600)


fig = fig_time_since_last_comp(users_comp_stats,
                         title="<b>Figure 16.2: Mean Days Since Last Competition's Submission (Overall)</b><br><sup>For All Users Starting in 2015 or Later by Performance Tier Currently.</sup>",
                         stat='Mean',
                        xlim=[1.5,7.5],ylim=[0,300])
fig.update_layout(height=450)
fig.write_html('figure16_2.html')

fig.show(renderer='iframe') 
#IFrame('figure16_2.html', width='100%',height=600)



fig = fig_time_since_last_comp_by_cohort(users_comp_stats,
                                   years=[2019,2020,2023,2024],
                                   ylim=[0,500],
                                   title='<b>Figure 17.1: Median Days Since Last Competition by Cohorts and Tiers')
fig.update_layout(height=600)
fig.write_html('figure17_1.html')

fig.show(renderer='iframe')
#IFrame('figure17_1.html', width='100%',height=600)


fig = fig_time_since_last_comp_by_cohort(users_comp_stats,years=[2019,2020,2023,2024],
                                   ylim=[0,500],
                                   stat='Mean',
                                   title='<b>Figure 17.2: Mean Days Since Last Competition by Cohorts and Tiers')
fig.update_layout(height=600)
fig.write_html('figure17_2.html')

fig.show(renderer='iframe')
#IFrame('figure17_2.html', width='100%',height=600)


fig = fig_ecdf_overall(users_comp_stats,
                 group='MedalType',
                 title="<b>Figure 18: Time to Final Submission by Medal Awarded (Overall)</b><br><sup>ECDF Plot of Final Submission Timing From Submissions Data (2015 or Later).</sup>")
fig.update_layout(height=500)
fig.write_html('figure18.html')

fig.show(renderer='iframe')
#IFrame('figure18.html', width='100%',height=600)


fig = fig_ecdf_overall(users_comp_stats[(users_comp_stats['UserFirstCompYear'].isin([2019,2020,2023,2024])) &
                       (users_comp_stats['UserCompNumber']<=3)],
                 group='UserFirstCompYear',
                 title="<b>Figure 19: Time to Final Submission By Cohorts (First Three Competitions)</b><br><sup>ECDF Plot of Final Submission Timing From Submissions Data (2015 or Later).</sup>")
fig.update_layout(height=500)
fig.write_html('figure19.html')

fig.show(renderer='iframe')
#IFrame('figure19.html', width='100%',height=600)


fig = fig_ecdf_by_submission_year(users_comp_stats, 
                       years=[2021,2022,2023,2024], 
                       title="<b>Figure 20: Time to Final Submission by Competition Segment</b><br><sup>ECDF Plots for Team Submissions Made in 2021–2024, Faceted by Submission Year.</sup>",
                       group='CompSegment')
fig.update_layout(height=650)
fig.write_html('figure20.html')

fig.show(renderer='iframe')
#IFrame('figure20.html', width='100%',height=600)


fig = fig_ecdf_by_submission_year(users_comp_stats[users_comp_stats['IsSolo']==True], 
                       years=[2021,2022,2023,2024], 
                       title="<b>Figure 21: Time to Final Submission by Performance Tier</b><br><sup>ECDF Plots for Team Submissions Made in 2021–2024, Faceted by Submission Year. <b>(This Plot Include Solo Teams Only)</b></sup>",
                       group='UserPerformanceTierName')
fig.update_layout(height=650)
fig.write_html('figure21.html')

fig.show(renderer='iframe')
#IFrame('figure21.html', width='100%',height=600)


fig = fig_ecdf_by_submission_year(users_comp_stats, 
                       years=[2021,2022,2023,2024], 
                       title="<b>Figure 22: Time to Final Submission by Team Size</b><br><sup>ECDF Plots for Team Submissions Made in 2021–2024, Faceted by Submission Year.</sup>",
                       group='TeamSizeCat')
fig.update_layout(height=650)
fig.write_html('figure22.html')

fig.show(renderer='iframe')
#IFrame('figure22.html', width='100%',height=600)


fig = fig_ecdf_by_submission_year(users_comp_stats, 
                       years=[2021,2022,2023,2024], 
                       title="<b>Figure 23: Time to Final Submission by Accelerators Usage</b><br><sup>ECDF Plots for Team Submissions Made in 2021–2024, Faceted by Submission Year.</sup>",
                       group='AccelGroup')
fig.update_layout(height=650)
fig.write_html('figure23.html')

fig.show(renderer='iframe')
#IFrame('figure23.html', width='100%',height=600)


fig = fig_ecdf_by_submission_year(users_comp_stats, 
                       years=[2021,2022,2023,2024], 
                       title="<b>Figure 24: Time to Final Submission by Medal Awarded</b><br><sup>ECDF Plots for Team Submissions Made in 2021–2024, Faceted by Submission Year.</sup>",
                       group='MedalType')
fig.update_layout(height=650)
fig.write_html('figure24.html')

fig.show(renderer='iframe')
#IFrame('figure24.html', width='100%',height=600)

