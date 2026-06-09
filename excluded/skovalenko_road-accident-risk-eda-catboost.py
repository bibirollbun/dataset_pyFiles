import os
import math
import shap
import optuna
import scipy

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import matthews_corrcoef, mean_squared_error
from sklearn.model_selection import train_test_split, KFold
from scipy.stats import chi2_contingency

from catboost import CatBoostRegressor, Pool

import matplotlib.pyplot as plt
%matplotlib inline

import seaborn as sns
sns.set(color_codes=True)

import warnings
warnings.filterwarnings("ignore")


optuna.logging.set_verbosity(optuna.logging.WARNING)


def num_var_distribution_float(df,
                               title: str,
                               x1: str,
                               y1: str,
                               x1_label: str,
                               y1_label: str,
                               x2_label: str,
                               y2_label: str):
    
    figure, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (16, 6))
    figure.suptitle(title,
                    x = 0.5, y = 0.95, fontsize = 16, fontweight ='bold')

    # Figure 1: box-plot
    dir_order = ['train', 'test']
    my_pal = {'train': 'orange', 'test': 'royalblue'}
    box_plot = sns.boxplot(data = df, 
                           x = x1, y = y1,
                           order = dir_order,
                           palette = my_pal,
                           ax = axes[0])
    axes[0].set_xlabel(x1_label, fontsize = 14, fontweight ='bold')
    axes[0].set_ylabel(y1_label, fontsize = 14, fontweight ='bold')
    axes[0].set_xticklabels(labels = dir_order, rotation = 0, ha = 'center', size = 12)

    medians = df.groupby([x1]).agg(
      Med = (y1, np.median)
    ).reset_index()
    medians['Med'] = medians['Med'].round(2)
    medians['Tick'] = range(len(medians))
    
    medians['Cat'] = 0
    for i in range(len(medians)):
        if medians.loc[i, x1] == 'train':
            medians.loc[i, 'Cat'] = 0
        if medians.loc[i, x1] == 'test':
            medians.loc[i, 'Cat'] = 1
    
    medians = medians.sort_values(['Cat'])
    ticks = list(medians['Tick'])
    medians = list(medians['Med'])
    vertical_offset = [median * 0.025 for median in medians]
    
    for xtick in ticks:
        box_plot.text(xtick, medians[xtick] + vertical_offset[xtick], medians[xtick], 
                      horizontalalignment = 'center', 
                      size = 10, 
                      color = 'black', 
                      weight = 'semibold')
    
    
    # Figure 2: distplots
    kde_1 = sns.distplot(a = df.loc[df[x1] == 'train', y1],
                         kde_kws = {'color': 'orange', 'lw': 2.0, 'linestyle': '--'},
                         hist = False,
                         label = 'train',
                         ax = axes[1])
    kde_2 = sns.distplot(a = df.loc[df[x1] == 'test', y1],
                         kde_kws = {'color': 'royalblue', 'lw': 2.0, 'linestyle': '--'},
                         hist = False,
                         label = 'test',
                         ax = axes[1])
    
    axes[1].set_xlabel(x2_label, fontsize = 14, fontweight ='bold')
    axes[1].set_ylabel(y2_label, fontsize = 14, fontweight ='bold')
    
    
    plt.plot()


def num_var_distribution_int(df,
                             title: str,
                             x1: str,
                             y1: str,
                             x1_label: str,
                             y1_label: str,
                             x2_label: str,
                             y2_label: str):
    
    figure, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (16, 6))
    figure.suptitle(title,
                    x = 0.5, y = 0.95, fontsize = 16, fontweight ='bold')

    # Figure 1: box-plot
    dir_order = ['train', 'test']
    my_pal = {'train': 'orange', 'test': 'royalblue'}
    box_plot = sns.boxplot(data = df, 
                           x = x1, y = y1,
                           order = dir_order,
                           palette = my_pal,
                           ax = axes[0])
    axes[0].set_xlabel(x1_label, fontsize = 14, fontweight ='bold')
    axes[0].set_ylabel(y1_label, fontsize = 14, fontweight ='bold')
    axes[0].set_xticklabels(labels = dir_order, rotation = 0, ha = 'center', size = 12)

    medians = df.groupby([x1]).agg(
      Med = (y1, np.median)
    ).reset_index()
    medians['Med'] = medians['Med'].round(2)
    medians['Tick'] = range(len(medians))
    
    medians['Cat'] = 0
    for i in range(len(medians)):
        if medians.loc[i, x1] == 'train':
            medians.loc[i, 'Cat'] = 0
        if medians.loc[i, x1] == 'test':
            medians.loc[i, 'Cat'] = 1
    
    medians = medians.sort_values(['Cat'])
    ticks = list(medians['Tick'])
    medians = list(medians['Med'])
    vertical_offset = [median * 0.025 for median in medians]
    
    for xtick in ticks:
        box_plot.text(xtick, medians[xtick] + vertical_offset[xtick], medians[xtick], 
                      horizontalalignment = 'center', 
                      size = 10, 
                      color = 'black', 
                      weight = 'semibold')
    
    
    # Figure 2: histplot
    df_copy = df.copy()
    df_copy[y1] = df_copy[y1].astype('str')

    elements_lst = list(df[y1].unique())
    elements_lst.sort()
    elements_order = [str(elm) for elm in elements_lst]
    
    hist_plot = sns.histplot(data = df_copy,
                             x = y1,
                             hue = x1,
                             tick_label = elements_order,
                             multiple = 'dodge',
                             shrink = 0.8,
                             discrete = True,
                             stat = 'percent',
                             common_norm = False,
                             palette = my_pal,
                             hue_order = dir_order,
                             legend = False,
                             ax = axes[1])
    
    axes[1].set_xlabel(x2_label, fontsize = 14, fontweight ='bold')
    axes[1].set_ylabel(y2_label, fontsize = 14, fontweight ='bold')
    
    
    plt.plot()


def y_var_distribution_boolean(df,
                               title: str,
                               x1: str,
                               y1: str,
                               x1_label: str,
                               y1_label: str,
                               x2_label: str,
                               y2_label: str):
    
    figure, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (16, 6))
    figure.suptitle(title,
                    x = 0.5, y = 0.95, fontsize = 16, fontweight ='bold')

    # Figure 1: box-plot
    dir_order = [True, False]
    my_pal = {True: 'orange', False: 'royalblue'}
    box_plot = sns.boxplot(data = df, 
                           x = x1, y = y1,
                           order = dir_order,
                           palette = my_pal,
                           ax = axes[0])
    axes[0].set_xlabel(x1_label, fontsize = 14, fontweight ='bold')
    axes[0].set_ylabel(y1_label, fontsize = 14, fontweight ='bold')
    axes[0].set_xticklabels(labels = dir_order, rotation = 0, ha = 'center', size = 12)

    medians = df.groupby([x1]).agg(
      Med = (y1, np.median)
    ).reset_index()
    medians['Med'] = medians['Med'].round(2)
    medians['Tick'] = range(len(medians))
    
    medians['Cat'] = 0
    for i in range(len(medians)):
        if medians.loc[i, x1] == True:
            medians.loc[i, 'Cat'] = 0
        if medians.loc[i, x1] == False:
            medians.loc[i, 'Cat'] = 1
    
    medians = medians.sort_values(['Cat'])
    ticks = list(medians['Tick'])
    medians = list(medians['Med'])
    vertical_offset = [median * 0.025 for median in medians]
    
    for xtick in ticks:
        box_plot.text(xtick, medians[xtick] + vertical_offset[xtick], medians[xtick], 
                      horizontalalignment = 'center', 
                      size = 10, 
                      color = 'black', 
                      weight = 'semibold')
    
    
    # Figure 2: distplots
    kde_1 = sns.distplot(a = df.loc[df[x1] == True, y1],
                         kde_kws = {'color': 'orange', 'lw': 2.0, 'linestyle': '--'},
                         hist = False,
                         label = 'True',
                         ax = axes[1])
    kde_2 = sns.distplot(a = df.loc[df[x1] == False, y1],
                         kde_kws = {'color': 'royalblue', 'lw': 2.0, 'linestyle': '--'},
                         hist = False,
                         label = 'False',
                         ax = axes[1])
    
    axes[1].set_xlabel(x2_label, fontsize = 14, fontweight ='bold')
    axes[1].set_ylabel(y2_label, fontsize = 14, fontweight ='bold')
    
    
    plt.plot()


def y_var_distribution_categorical(df,
                                   title: str,
                                   x1: str,
                                   y1: str,
                                   x1_label: str,
                                   y1_label: str,
                                   x2_label: str,
                                   y2_label: str):
    
    figure, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (16, 6))
    figure.suptitle(title,
                    x = 0.5, y = 0.95, fontsize = 16, fontweight ='bold')

    # Figure 1: box-plot
    if x1 == 'road_type':
        dir_order = ['rural', 'urban', 'highway']
    elif x1 == 'lighting':
        dir_order = ['night', 'dim', 'daylight']
    elif x1 == 'weather':
        dir_order = ['rainy', 'foggy', 'clear']
    elif x1 == 'time_of_day':
        dir_order = ['morning', 'afternoon', 'evening']
    box_plot = sns.boxplot(data = df, 
                           x = x1, y = y1,
                           order = dir_order,
                           ax = axes[0])
    axes[0].set_xlabel(x1_label, fontsize = 14, fontweight ='bold')
    axes[0].set_ylabel(y1_label, fontsize = 14, fontweight ='bold')
    axes[0].set_xticklabels(labels = dir_order, rotation = 0, ha = 'center', size = 12)

    medians = df.groupby([x1]).agg(
      Med = (y1, np.median)
    ).reset_index()
    medians['Med'] = medians['Med'].round(2)
    medians['Tick'] = range(len(medians))
    
    medians['Cat'] = 0
    for i in range(len(medians)):
        if medians.loc[i, x1] == dir_order[0]:
            medians.loc[i, 'Cat'] = 0
        elif medians.loc[i, x1] == dir_order[1]:
            medians.loc[i, 'Cat'] = 1
        elif medians.loc[i, x1] == dir_order[2]:
            medians.loc[i, 'Cat'] = 2
    
    medians = medians.sort_values(['Cat'])
    ticks = list(medians['Tick'])
    medians = list(medians['Med'])
    vertical_offset = [median * 0.025 for median in medians]
    
    for xtick in ticks:
        box_plot.text(xtick, medians[xtick] + vertical_offset[xtick], medians[xtick], 
                      horizontalalignment = 'center', 
                      size = 10, 
                      color = 'black', 
                      weight = 'semibold')
    
    
    # Figure 2: distplots
    kde_1 = sns.distplot(a = df.loc[df[x1] == dir_order[0], y1],
                         kde_kws = {'lw': 2.0, 'linestyle': '--'},
                         hist = False,
                         label = dir_order[0],
                         ax = axes[1])
    kde_2 = sns.distplot(a = df.loc[df[x1] == dir_order[1], y1],
                         kde_kws = {'lw': 2.0, 'linestyle': '--'},
                         hist = False,
                         label = dir_order[1],
                         ax = axes[1])
    kde_3 = sns.distplot(a = df.loc[df[x1] == dir_order[2], y1],
                         kde_kws = {'lw': 2.0, 'linestyle': '--'},
                         hist = False,
                         label = dir_order[2],
                         ax = axes[1])
    
    
    axes[1].set_xlabel(x2_label, fontsize = 14, fontweight ='bold')
    axes[1].set_ylabel(y2_label, fontsize = 14, fontweight ='bold')
    
    
    plt.plot()


def y_var_distribution_int(df,
                           title: str,
                           x1: str,
                           y1: str,
                           x1_label: str,
                           y1_label: str):

    figure, axes = plt.subplots(nrows = 1, ncols = 1, figsize = (16, 6))

    medians = df.groupby([x1]).agg(
      Med = (y1, np.median)
    ).reset_index()
    medians["Med"] = medians["Med"].round(2)
    medians = medians.sort_values(by = x1)
    medians["Tick"] = range(len(medians))
    
    dir_order = list(medians[x1])
    ticks = list(medians["Tick"])
    medians = list(medians["Med"])
    vertical_offset = [median * 0.025 for median in medians]
    
    
    # Figure 1: Box Plot
    box_plot = sns.boxplot(data = df, 
                           x = x1, y = y1,
                           order = dir_order,
                           palette = "Blues_d",
                           ax = axes)
    axes.set_title(title, fontsize = 16, fontweight = 'bold')
    axes.set_xlabel(x1_label, fontsize = 14, fontweight ='bold')
    axes.set_ylabel(y1_label, fontsize = 14, fontweight ='bold')
    axes.set_xticklabels(labels = dir_order, rotation = 0, ha = 'center', size = 12)
    
    for xtick in ticks:
        box_plot.text(xtick, medians[xtick] + vertical_offset[xtick], medians[xtick], 
                      horizontalalignment = 'center', 
                      size = 10, 
                      color = 'white', 
                      weight = 'semibold')
    
    
    plt.plot()


def corr_plot(df_1, df_2, title):

    figure, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (16, 6))
    figure.suptitle(title,
                    x = 0.5, y = 0.95, fontsize = 18, fontweight ='bold')
     
    sns.heatmap(df_1, 
                annot = True, 
                vmin = -1, 
                vmax = 1, 
                center = 0, 
                cmap = 'coolwarm',
                linewidths = 3, 
                linecolor = 'black',
                ax = ax[0])
    
    sns.heatmap(df_2, 
                annot = True, 
                vmin = -1, 
                vmax = 1, 
                center = 0, 
                cmap = 'coolwarm',
                xticklabels = True,
                yticklabels = False,
                linewidths = 3, 
                linecolor = 'black',
                ax = ax[1])
    
    ax[0].set_title("train", fontsize = 16)
    ax[1].set_title("test", fontsize = 16)
    
    plt.show()


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


df_train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
df_orig = pd.read_csv('/kaggle/input/simulated-roads-accident-data/synthetic_road_accidents_100k.csv')


df_1 = df_train.drop(columns = ['accident_risk'])
df_1['data_type'] = 'train'

df_2 = df_test.copy()
df_2['data_type'] = 'test'

df = pd.concat([df_1, df_2], ignore_index = True)


df_train.info()


df_train.head(3)


df_train_no_id = df_train.drop(columns = ['id'])
df_train_no_id.drop_duplicates(keep = 'first', inplace = True, ignore_index = True)

print('Number of duplicates in the train_df: ', len(df_train) - len(df_train_no_id))





df_test.info()


df_test.head(3)


figure, axes = plt.subplots(nrows = 1, ncols = 1, figsize = (12, 6))

hist_plot = sns.histplot(data = df_train, x = 'accident_risk',
                         stat = 'percent', bins = 25,
                         ax = axes)
axes.set_title('Accident Risk Distribution', fontsize = 16, fontweight = 'bold')
axes.set_xlabel('Accident Risk', fontsize = 14, fontweight ='bold')
axes.set_ylabel('Percent', fontsize = 14, fontweight ='bold')

plt.show()


df_train['accident_risk'].describe()


num_variables = ['num_lanes', 'curvature', 'speed_limit', 'num_reported_accidents']


df_train[num_variables].describe()


df_test[num_variables].describe()


num_var_distribution_int(df = df,
                         title = 'Number of Lanes by type of data',
                         x1 = 'data_type',
                         y1 = 'num_lanes',
                         x1_label = 'Data Type',
                         y1_label = 'Number of Lanes',
                         x2_label = 'Number of Lanes',
                         y2_label = 'Percents')


num_var_distribution_int(df = df,
                         title = 'Number of Reported Accidents by type of data',
                         x1 = 'data_type',
                         y1 = 'num_reported_accidents',
                         x1_label = 'Data Type',
                         y1_label = 'Number of Accidents',
                         x2_label = 'Number of Accidents',
                         y2_label = 'Percents')


df_train['num_reported_accidents'].value_counts()


num_var_distribution_int(df = df,
                         title = 'Speed Limit by type of data',
                         x1 = 'data_type',
                         y1 = 'speed_limit',
                         x1_label = 'Data Type',
                         y1_label = 'Speed Limit',
                         x2_label = 'Speed Limit',
                         y2_label = 'Percents')


num_var_distribution_float(df = df,
                           title = 'Curvature Distributions by type of data',
                           x1 = 'data_type',
                           y1 = 'curvature',
                           x1_label = 'Data Type',
                           y1_label = 'Curvature',
                           x2_label = 'Curvature',
                           y2_label = 'Density')


corr_plot(df_1 = df_train[['num_lanes', 
                           'curvature', 
                           'speed_limit', 
                           'num_reported_accidents']].corr(method = 'spearman'), 
          df_2 = df_test[['num_lanes', 
                          'curvature', 
                          'speed_limit', 
                          'num_reported_accidents']].corr(method = 'spearman'), 
          title = "Spearman's rank correlation")


categorical_variables = ['road_type', 'lighting', 'weather', 'time_of_day']
boolean_variables = ['road_signs_present', 'public_road', 'holiday', 'school_season']


df_train[categorical_variables + boolean_variables].describe()


df_test[categorical_variables + boolean_variables].describe()


df_phi_train = pd.DataFrame(columns = boolean_variables, 
                            index = boolean_variables, 
                            dtype = np.float32)
df_phi_test = pd.DataFrame(columns = boolean_variables, 
                           index = boolean_variables,
                           dtype = np.float32)
for i in range(len(boolean_variables)):
    var_i = boolean_variables[i]
    for j in range(len(boolean_variables)):
        var_j = boolean_variables[j]
        df_phi_train.loc[var_i, var_j] = matthews_corrcoef(df_train[var_i], df_train[var_j])
        df_phi_test.loc[var_i, var_j] = matthews_corrcoef(df_test[var_i], df_test[var_j])


corr_plot(df_1 = df_phi_train,
          df_2 = df_phi_test,
          title = "Matthews' correlation coefficients")


df_cramers_train = pd.DataFrame(columns = categorical_variables, 
                                index = categorical_variables, 
                                dtype = np.float32)
df_cramers_test = pd.DataFrame(columns = categorical_variables, 
                               index = categorical_variables,
                               dtype = np.float32)
for i in range(len(categorical_variables)):
    var_i = categorical_variables[i]
    for j in range(len(categorical_variables)):
        var_j = categorical_variables[j]
        
        df_temp_train = pd.crosstab(df_train[var_i], df_train[var_j])
        chi2_train, _, _, _ = chi2_contingency(df_temp_train)
        df_cramers_train.loc[var_i, var_j] = math.sqrt(chi2_train / (df_temp_train.values.sum() * min(df_temp_train.shape[0]-1, df_temp_train.shape[1]-1)))
        
        df_temp_test = pd.crosstab(df_test[var_i], df_test[var_j])
        chi2_test, _, _, _ = chi2_contingency(df_temp_test)
        df_cramers_test.loc[var_i, var_j] = math.sqrt(chi2_test / (df_temp_test.values.sum() * min(df_temp_test.shape[0]-1, df_temp_test.shape[1]-1)))


corr_plot(df_1 = df_cramers_train,
          df_2 = df_cramers_test,
          title = "Cramers' V correlation coefficients")


y_var_distribution_boolean(df = df_train,
                           title = 'Accident Risk Distribution by road_signs_present',
                           x1 = 'road_signs_present',
                           y1 = 'accident_risk',
                           x1_label = 'Road Signs',
                           y1_label = 'Accident Risk',
                           x2_label = 'Accident Risk',
                           y2_label = 'Density')


y_var_distribution_boolean(df = df_train,
                           title = 'Accident Risk Distribution by public_road',
                           x1 = 'public_road',
                           y1 = 'accident_risk',
                           x1_label = 'Public Road',
                           y1_label = 'Accident Risk',
                           x2_label = 'Accident Risk',
                           y2_label = 'Density')


y_var_distribution_boolean(df = df_train,
                           title = 'Accident Risk Distribution by holiday',
                           x1 = 'holiday',
                           y1 = 'accident_risk',
                           x1_label = 'Holiday',
                           y1_label = 'Accident Risk',
                           x2_label = 'Accident Risk',
                           y2_label = 'Density')


y_var_distribution_boolean(df = df_train,
                           title = 'Accident Risk Distribution by school_season',
                           x1 = 'school_season',
                           y1 = 'accident_risk',
                           x1_label = 'School Season',
                           y1_label = 'Accident Risk',
                           x2_label = 'Accident Risk',
                           y2_label = 'Density')


y_var_distribution_categorical(df = df_train,
                               title = 'Accident Risk Distribution by Road Type',
                               x1 = 'road_type',
                               y1 = 'accident_risk',
                               x1_label = 'Road Type',
                               y1_label = 'Accident Risk',
                               x2_label = 'Accident Risk',
                               y2_label = 'Density')


y_var_distribution_categorical(df = df_train,
                               title = 'Accident Risk Distribution by Lighting',
                               x1 = 'lighting',
                               y1 = 'accident_risk',
                               x1_label = 'Lighting',
                               y1_label = 'Accident Risk',
                               x2_label = 'Accident Risk',
                               y2_label = 'Density')


y_var_distribution_categorical(df = df_train,
                               title = 'Accident Risk Distribution by Weather',
                               x1 = 'weather',
                               y1 = 'accident_risk',
                               x1_label = 'Weather',
                               y1_label = 'Accident Risk',
                               x2_label = 'Accident Risk',
                               y2_label = 'Density')


y_var_distribution_categorical(df = df_train,
                               title = 'Accident Risk Distribution by Time of Day',
                               x1 = 'time_of_day',
                               y1 = 'accident_risk',
                               x1_label = 'Time of Day',
                               y1_label = 'Accident Risk',
                               x2_label = 'Accident Risk',
                               y2_label = 'Density')


y_var_distribution_int(df = df_train,
                       title = 'Accident Risk Distribution by Number of Lanes',
                       x1 = 'num_lanes',
                       y1 = 'accident_risk',
                       x1_label = 'Number of Lanes',
                       y1_label = 'Accident Risk')


y_var_distribution_int(df = df_train,
                       title = 'Accident Risk Distribution by Speed Limit',
                       x1 = 'speed_limit',
                       y1 = 'accident_risk',
                       x1_label = 'Speed Limit',
                       y1_label = 'Accident Risk')


y_var_distribution_int(df = df_train,
                       title = 'Accident Risk Distribution by Number of Reported Accidents',
                       x1 = 'num_reported_accidents',
                       y1 = 'accident_risk',
                       x1_label = 'Number of Reported Accidents',
                       y1_label = 'Accident Risk')


figure, axes = plt.subplots(nrows = 1, ncols = 1, figsize = (16, 8))

box_plot = sns.lineplot(data = df_train, 
                        x = 'curvature', y = 'accident_risk',
                        ax = axes)
reg_plot = sns.regplot(data = df_train,
                       x = 'curvature', y = 'accident_risk',
                       scatter = False, 
                       line_kws = {'color': 'red', 'lw': 1.5, 'linestyle': "--"},
                       ax = axes)
axes.set_title('Accident Risk Distribution by Curvature', fontsize = 16, fontweight = 'bold')
axes.set_xlabel('Curvature', fontsize = 14, fontweight ='bold')
axes.set_ylabel('Accident Risk', fontsize = 14, fontweight ='bold')

plt.show()


categorical_features = ['road_type', 'lighting', 'weather', 'time_of_day']
numerical_features = ['num_lanes', 'curvature', 'speed_limit', 'num_reported_accidents']
boolean_features = ['road_signs_present', 'public_road', 'holiday', 'school_season']

predictors = categorical_features + numerical_features + boolean_features

target = 'accident_risk'


X_train, X_val, y_train, y_val = train_test_split(df_train[predictors],
                                                  df_train[target],
                                                  train_size = 0.8,
                                                  random_state = 42)


categorical_var_indices = np.where(df_train[predictors].dtypes == object)[0]
categorical_var_indices


catboost_model = CatBoostRegressor(loss_function = 'RMSE',
                                   eval_metric = 'RMSE',
                                   one_hot_max_size = 3)


catboost_model.fit(X = X_train, y = y_train,
                   eval_set = (X_val, y_val),
                   early_stopping_rounds = 10,
                   use_best_model = True,
                   cat_features = categorical_var_indices,
                   verbose = 50,
                   plot = False)


shap.initjs()


explainer = shap.TreeExplainer(catboost_model)
shap_values = explainer.shap_values(Pool(X_train, y_train,
                                         cat_features = categorical_var_indices))


shap.summary_plot(shap_values, X_train, plot_type = "bar")


df_train = df_train_no_id.copy()


original_columns = []
global_orig_mean = df_orig[target].mean()
for col in predictors:
    new_col_name = f"orig_{col}"
    original_columns.append(new_col_name)
    
    df_tmp = df_orig.groupby(col)[target].mean()
    df_tmp.name = new_col_name
    df_train = df_train.merge(df_tmp, on = col, how = 'left')
    df_test = df_test.merge(df_tmp, on = col, how = 'left')

    df_train[new_col_name] = df_train[new_col_name].fillna(global_orig_mean)
    df_test[new_col_name] = df_test[new_col_name].fillna(global_orig_mean)


def f(X):
    return \
    0.3 * X["curvature"] + \
    0.2 * (X["lighting"] == "night").astype(int) + \
    0.1 * (X["weather"] != "clear").astype(int) + \
    0.2 * (X["speed_limit"] >= 60).astype(int) + \
    0.1 * (X["num_reported_accidents"] > 2).astype(int)


df_train['orig_target'] = f(df_train)
df_test['orig_target'] = f(df_test)


for col in categorical_features:
    df_train[col], _ = df_train[col].factorize(sort = True)
    df_train[col] = df_train[col].astype('int32')

    df_test[col], _ = df_test[col].factorize(sort = True)
    df_test[col] = df_test[col].astype('int32')


predictors += original_columns + ['orig_target']


df_train = df_train.loc[: len(df_train) - len(df_train) % 5 - 1,:].copy()


def objective(trial):
    params = {
        "grow_policy": trial.suggest_categorical("grow_policy", ['Depthwise', 'Lossguide']),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-3, 10.0, log = True),
        "depth": trial.suggest_int("depth", 4, 16),
        "subsample": trial.suggest_float("subsample", 0.1, 1.0, step = 0.01),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.1, 1.0, step = 0.01),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 100)}
    
    kf = KFold(n_splits = 5, shuffle = True, random_state = 42)
    a = kf.split(X = df_train[['curvature']], y = df_train[target])
    
    oof_pred = np.zeros(len(df_train))
    for i, (train_index, test_index) in enumerate(a): 
        X_train = df_train.loc[train_index, :].copy()
        X_test = df_train.loc[test_index, :].copy()

        X_train, X_val, y_train, y_val = train_test_split(X_train[predictors],
                                                          X_train[target],
                                                          train_size = 0.95,
                                                          random_state = 42)

        lr_model = LinearRegression().fit(X = X_train[['curvature']], y = y_train)
        X_train['curvature_reg'] = lr_model.predict(X_train[['curvature']])
        X_val['curvature_reg'] = lr_model.predict(X_val[['curvature']])
        X_test['curvature_reg'] = lr_model.predict(X_test[['curvature']])
        
        eval_set = (X_val[predictors + ['curvature_reg']], y_val)
        
        alg = CatBoostRegressor(**params,
                                learning_rate = 0.025,
                                n_estimators = 10000,
                                bootstrap_type = 'Bernoulli',
                                loss_function = 'RMSE',
                                eval_metric = 'RMSE')

        alg.fit(X_train[predictors + ['curvature_reg']], y_train,
                early_stopping_rounds = 50, 
                eval_set = eval_set,
                use_best_model = True,
                verbose = False,
                plot = False)
        oof_pred[test_index] = alg.predict(X_test[predictors + ['curvature_reg']])
        
    
    return mean_squared_error(df_train[target], oof_pred, squared = False)


study = optuna.create_study(direction = 'minimize')
study.optimize(func = objective, 
               n_trials = 25,
               n_jobs = 1,
               gc_after_trial = False,
               show_progress_bar = False)


print('Best hyper-parameters:', study.best_params)
print('---------')
print('Best RMSE:', study.best_value)


kf = KFold(n_splits = 5, shuffle = True, random_state = 42)
a = kf.split(X = df_train[['curvature']], y = df_train[target])


params = study.best_params


oof_pred = np.zeros(len(df_train))
test_pred = np.zeros(len(df_test))
for i, (train_index, test_index) in enumerate(a): 
    X_train = df_train.loc[train_index, :].copy()
    X_test = df_train.loc[test_index, :].copy()
    df_test_copy = df_test.copy()

    X_train, X_val, y_train, y_val = train_test_split(X_train[predictors],
                                                      X_train[target],
                                                      train_size = 0.95,
                                                      random_state = 42)

    lr_model = LinearRegression().fit(X = X_train[['curvature']], y = y_train)
    X_train['curvature_reg'] = lr_model.predict(X_train[['curvature']])
    X_val['curvature_reg'] = lr_model.predict(X_val[['curvature']])
    X_test['curvature_reg'] = lr_model.predict(X_test[['curvature']])
    df_test_copy['curvature_reg'] = lr_model.predict(df_test_copy[['curvature']])
    
    eval_set = (X_val[predictors + ['curvature_reg']], y_val)
    
    alg = CatBoostRegressor(**params,
                            learning_rate = 0.025,
                            n_estimators = 10000,
                            bootstrap_type = 'Bernoulli',
                            loss_function = 'RMSE',
                            eval_metric = 'RMSE')

    alg.fit(X_train[predictors + ['curvature_reg']], y_train,
            early_stopping_rounds = 50, 
            eval_set = eval_set,
            use_best_model = True,
            verbose = False,
            plot = False)
    
    oof_pred[test_index] = alg.predict(X_test[predictors + ['curvature_reg']])
    test_pred += alg.predict(df_test_copy[predictors + ['curvature_reg']])


rmse = mean_squared_error(df_train[target], oof_pred, squared = False)
print("5-fold CV RMSE: ", rmse)


test_pred = test_pred / 5


submission = pd.DataFrame({'id': df_test['id'], 'accident_risk': test_pred})
submission.to_csv('/kaggle/working/submission.csv', index=False)

