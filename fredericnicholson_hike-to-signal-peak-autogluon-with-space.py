import numpy as np # linear algebra
import polars as pl # data processing, CSV file I/O (e.g. pd.read_csv)
import polars.selectors as cs
import seaborn as sns
import matplotlib.pyplot  as plt
import warnings

warnings.simplefilter("ignore")

print ('Libraries loaded')



raw_train = pl.scan_csv("/kaggle/input/playground-series-s5e2/train.csv").collect()
more_raw_train = pl.scan_csv("/kaggle/input/playground-series-s5e2/training_extra.csv").collect()
raw_test = pl.scan_csv("/kaggle/input/playground-series-s5e2/test.csv").collect()
sample_submission = pl.scan_csv("/kaggle/input/playground-series-s5e2/sample_submission.csv").collect()

# raw_train = pl.concat ([raw_train, more_raw_train])


run_EDA = False


with pl.Config(tbl_rows=20):
    display (raw_train.select (cs.numeric()).describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.95])) 


if run_EDA :
    for col in raw_train.select(cs.string()).columns :
        print (f"values for {col = }")
        display_me = raw_train.group_by(col).len()
        sns.barplot (data = display_me.to_pandas(), x = col, y = "len")
        plt.show()


if run_EDA : 
    for col in raw_train.select(cs.string()).columns :
        print (f"values for {col = }")
        display_me = raw_train.group_by(col).agg (pl.col("Price").mean())
        print (display_me)
        print (f" number of null value : {raw_train.get_column(col).is_null().sum()}")
        sns.barplot (data = display_me.to_pandas(), x = col, y = "Price")
        plt.show()


if run_EDA : 
    plt.figure(figsize=(8, 30))
    sns.scatterplot(data = raw_train.to_pandas(), x = "Compartments", y = "Price")


if run_EDA :
    plt.figure(figsize=(22, 8))
    sns.scatterplot(data = raw_train.to_pandas(), x = "Weight Capacity (kg)", y = "Price")


if run_EDA : 
    with pl.Config(tbl_rows=20): 
        display ( raw_train.get_column ("Price").describe( [0.1, 0.25,0.5, 0.75, 0.9]))

    plt.figure(figsize=(22, 8))
    sns.histplot(data = raw_train.to_pandas(), x =  "Price", binwidth = 5)


if run_EDA :
    print (raw_train.columns )

    double = raw_train.group_by (["Brand", "Material", 'Compartments', 'Laptop Compartment', "Waterproof", "Size", "Style", "Color",'Weight Capacity (kg)']).len()

    with pl.Config (tbl_rows = 20):
        display (double.sort("len").tail()) 


if run_EDA :
    raw_train.filter ((pl.col("Brand") == "Adidas") & (pl.col("Material") == "Nylon") & (pl.col("Weight Capacity (kg)") == 5 ) & 
                 (pl.col("Style") == "Messenger") & (pl.col("Color") == "Blue") & (pl.col("Compartments") == 1) & 
                 (pl.col("Size") == "Small") &  (pl.col("Waterproof") == "No") & (pl.col("Laptop Compartment") == "No"))


if run_EDA :
    print (double.filter (pl.col("len")  == 2.0).group_by("Weight Capacity (kg)").len())
    print (raw_train.filter (pl.col("Weight Capacity (kg)") == 5.0))



# not very intelligent, but we add all categorical feature combinations with two cat features 

import itertools 

cat_features = ["Brand", "Material", 'Laptop Compartment', "Waterproof", "Size", "Style", "Color"]

# these unused feature list comes from Autogluon runs, where the logs indicate that these features are not used.   
unused_features = ['Material_Brand', 'p_Material_Brand', 'Laptop Compartment_Brand', 'p_Laptop Compartment_Brand', 
                   'Laptop Compartment_Material', 'p_Laptop Compartment_Material', 'Waterproof_Brand', 
                   'p_Waterproof_Brand', 'Waterproof_Material','p_Waterproof_Material', 'Waterproof_Laptop Compartment', 
                   'p_Waterproof_Laptop Compartment',  'Size_Brand', 'p_Size_Brand', 'Size_Material', 
                   'p_Size_Material', 'Size_Laptop Compartment', 'p_Size_Laptop Compartment', 'Size_Waterproof',
                   'p_Size_Waterproof', 'Style_Brand', 'p_Style_Brand', 'Style_Material', 'p_Style_Material',
                   'Style_Laptop Compartment', 'p_Style_Laptop Compartment', 'Style_Waterproof', 'p_Style_Waterproof',
                   'Style_Size', 'p_Style_Size', 'Color_Brand', 'p_Color_Brand', 'Color_Material', 'p_Color_Material',
                   'Color_Laptop Compartment', 'p_Color_Laptop Compartment', 'Color_Waterproof', 
                   'p_Color_Waterproof', 'Color_Size', 'p_Color_Size', 'Color_Style', 'p_Color_Style', 'weight_cat_Brand', 
                   'p_weight_cat_Brand', 'weight_cat_Material', 'p_weight_cat_Material', 'weight_cat_Laptop Compartment', 
                   'p_weight_cat_Laptop Compartment', 'weight_cat_Waterproof', 'p_weight_cat_Waterproof', 
                   'weight_cat_Size', 'p_weight_cat_Size', 'weight_cat_Style', 'p_weight_cat_Style', 'weight_cat_Color', 
                   'p_weight_cat_Color', 'compartment_cat_Brand', 'p_compartment_cat_Brand', 'compartment_cat_Material',
                   'p_compartment_cat_Material', 'compartment_cat_Laptop Compartment', 
                   'p_compartment_cat_Laptop Compartment', 'compartment_cat_Waterproof', 'p_compartment_cat_Waterproof', 
                   'compartment_cat_Size', 'p_compartment_cat_Size', 'compartment_cat_Style', 'p_compartment_cat_Style',
                   'compartment_cat_Color', 'p_compartment_cat_Color', 'compartment_cat_weight_cat', 
                   'p_compartment_cat_weight_cat']

def add_features (df : pl.DataFrame ) -> pl.DataFrame :
    
    result = df.with_columns ((pl.col('Weight Capacity (kg)').round (2).cast(pl.String) + "_kg").alias ("weight_cat"),
                               (pl.col('Compartments').cast(pl.String) + "_").alias ("compartment_cat"))
    cat_features.append ("weight_cat")
    cat_features.append ("compartment_cat")
    # result = result.with_columns ((pl.col('Weight Capacity (kg)') / 
    #                            pl.col('Compartments')).alias ("weight_per_compartment"), 
    #                            (pl.col('Weight Capacity (kg)') + 
    #                            pl.col('Compartments')).alias ("weight_plus_compartment"), )
    
    for feature in cat_features :
        result = result.with_columns (pl.col(feature).is_null().alias (f"{feature}_null")) 
    
    added_columns = result.select(cs.ends_with ("_null"))
    result = result.with_columns ((added_columns.sum_horizontal()).alias ("total_null")) 
    
    iter= itertools.permutations (cat_features, 2)
    for (a,b) in iter : 
        result = result.with_columns ((pl.col(a).fill_null("empty") + "_" + 
                                       pl.col(b).fill_null("empty")  ).cast (pl.Categorical).alias (f"{a}_{b}"))
        column_occurance = result.group_by (f"{a}_{b}").len()
        column_occurance = column_occurance.with_columns (pl.col("len") / result.shape [0])
        result = result.join (column_occurance, how = "left", on = f"{a}_{b}")
        result = result.rename ({"len" : f"p_{a}_{b}"})
        result = result.drop (f"{a}_{b}")
    # result = result.drop(unused_features)
    for col in cat_features :
       result = result.with_columns (pl.col(col).cast(pl.Categorical).alias (col)) 
    
    return result

new_features = add_features (raw_train)

# most models predict around the median, so we are interested in backback with extreme prices   
new_features_train  = new_features.with_columns (
                 ((pl.col("Price") - pl.col("Price").mean()).abs() **2).alias ("my_weight")) 


new_features_train


print (new_features_train.columns)

print (raw_train.columns )


!pip install ray==2.10.0
!pip install autogluon.tabular --no-cache-dir -q
!pip install -U ipywidgets


from autogluon.tabular import TabularPredictor

from autogluon.common import space


import gc

def search_space (hyper_search : dict) :
    gc.collect()
    predictor = TabularPredictor(path = '/kaggle/working/Autogluon1',
                                       label='Price', 
                               problem_type = 'regression', 
                               eval_metric =  'root_mean_squared_error',  
                               sample_weight = 'my_weight',
                               verbosity  = 2,
                               learner_kwargs = {'ignored_columns' : [
                                   'id',
                                   'Price'
                               #   'my_weight'
                                    ]})
    num_trials = 7  # try at most 7 different hyperparameter configurations for each type of model

    search_strategy = 'auto'  # to tune hyperparameters using random search routine with a local scheduler

    hyperparameter_tune_kwargs = {  # HPO is not performed unless hyperparameter_tune_kwargs is specified
        'num_trials': num_trials,
        'scheduler' : 'local',
        'searcher': search_strategy,
    } 
    try :
        predictor.fit(train_data= new_features_train.to_pandas(), 
                        presets= 'medium_quality',
    # best_quality, high_quality, medium_quality, 'experimental_quality',                         
                        time_limit = 1200,
                        num_gpus=0,
                        raise_on_no_models_fitted = True,
                        dynamic_stacking=False, 
                        num_stack_levels=0,
                        hyperparameters=hyper_search,
#                         hyperparameters = my_search_hyperparameters  ,
                        hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
                        )
    except :
        print ("exception")
        return {}
    leaderboard_tuning = predictor.leaderboard()
    display (leaderboard_tuning)
    best_model_name = leaderboard_tuning[leaderboard_tuning['stack_level'] == 1]['model'].iloc[0]

    predictor_info = predictor.info()
    best_model_info = predictor_info['model_info'][best_model_name]

    hyperparameters = best_model_info['hyperparameters']
    return hyperparameters     


cat_search  =    {'CAT': [{'depth': space.Int(lower=3, upper=10, default=6),  
                  'grow_policy': 'SymmetricTree', 
                  'l2_leaf_reg': 1.3, 
                  'learning_rate': space.Real(1e-4, 1e-2, default=5e-4, log=True),
                  'max_ctr_complexity': 4, 
                  'one_hot_max_size': 40, 
                  'ag_args': {'name_suffix': '_learnrate', 'priority': -1}}, 
                  {'depth': 8, 
                   'grow_policy': 'Depthwise', 
                   'l2_leaf_reg': space.Real(1, 4, default = 2.7997999596449104), 
                   'learning_rate': 0.031375015734637225, 
                   'max_ctr_complexity': 2, 
                   'one_hot_max_size': 30, 'ag_args': {'name_suffix': '_l2_leaf_reg', 'priority': -5}}]}

cat_result = search_space (cat_search)

print ("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
print (type (cat_result))
print (cat_result)



# adding the tuned hyperparameters to be preselected config for the  default models:
custom_hyperparameters = {
	'NN_TORCH': [{}, {'activation': 'elu', 'dropout_prob': 0.10077639529843717, 'hidden_size': 108, 'learning_rate': 0.002735937344002146, 'num_layers': 4, 'use_batchnorm': True, 'weight_decay': 1.356433327634438e-12, 'ag_args': {'name_suffix': '_r79', 'priority': -2}}, {'activation': 'elu', 'dropout_prob': 0.11897478034205347, 'hidden_size': 213, 'learning_rate': 0.0010474382260641949, 'num_layers': 4, 'use_batchnorm': False, 'weight_decay': 5.594471067786272e-10, 'ag_args': {'name_suffix': '_r22', 'priority': -57}}],
	'GBM': [{'extra_trees': True, 'ag_args': {'name_suffix': 'XT'}}, {}, {'learning_rate': 0.03, 'num_leaves': 128, 'feature_fraction': 0.9, 'min_data_in_leaf': 3, 'ag_args': {'name_suffix': 'Large', 'priority': 0, 'hyperparameter_tune_kwargs': None}}],
	'CAT': [{}, {'depth': 6, 'grow_policy': 'SymmetricTree', 'l2_leaf_reg': 2.1542798306067823, 'learning_rate': 0.06864209415792857, 'max_ctr_complexity': 4, 'one_hot_max_size': 10, 'ag_args': {'name_suffix': '_r177', 'priority': -1}}, {'depth': 8, 'grow_policy': 'Depthwise', 'l2_leaf_reg': 2.7997999596449104, 'learning_rate': 0.031375015734637225, 'max_ctr_complexity': 2, 'one_hot_max_size': 3, 'ag_args': {'name_suffix': '_r9', 'priority': -75}}],
	'XGB': [{}, {'colsample_bytree': 0.6917311125174739, 'enable_categorical': False, 'learning_rate': 0.018063876087523967, 'max_depth': 10, 'min_child_weight': 0.6028633586934382, 'ag_args': {'name_suffix': '_r33', 'priority': -8}}, {'colsample_bytree': 0.6628423832084077, 'enable_categorical': False, 'learning_rate': 0.08775715546881824, 'max_depth': 5, 'min_child_weight': 0.6294123374222513, 'ag_args': {'name_suffix': '_r89', 'priority': -86}}],
	'FASTAI': [{}, {'bs': 256, 'emb_drop': 0.5411770367537934, 'epochs': 43, 'layers': [800, 400], 'lr': 0.01519848858318159, 'ps': 0.23782946566604385, 'ag_args': {'name_suffix': '_r191', 'priority': -4}}, {'bs': 2048, 'emb_drop': 0.05070411322605811, 'epochs': 29, 'layers': [200, 100], 'lr': 0.08974235041576624, 'ps': 0.10393466140748028, 'ag_args': {'name_suffix': '_r102', 'priority': -91}}],
	'RF': [{'criterion': 'gini', 'ag_args': {'name_suffix': 'Gini', 'problem_types': ['binary', 'multiclass']}}, {'criterion': 'entropy', 'ag_args': {'name_suffix': 'Entr', 'problem_types': ['binary', 'multiclass']}}, {'criterion': 'squared_error', 'ag_args': {'name_suffix': 'MSE', 'problem_types': ['regression', 'quantile']}}],
	'XT': [{'criterion': 'gini', 'ag_args': {'name_suffix': 'Gini', 'problem_types': ['binary', 'multiclass']}}, {'criterion': 'entropy', 'ag_args': {'name_suffix': 'Entr', 'problem_types': ['binary', 'multiclass']}}, {'criterion': 'squared_error', 'ag_args': {'name_suffix': 'MSE', 'problem_types': ['regression', 'quantile']}}],
	'KNN': [{'weights': 'uniform', 'ag_args': {'name_suffix': 'Unif'}}, {'weights': 'distance', 'ag_args': {'name_suffix': 'Dist'}}],
}

cat_result ['ag_args'] = {'name_suffix': '_HPO', 'priority': 102}
custom_hyperparameters['CAT'].append (cat_result)

display (custom_hyperparameters)

print (custom_hyperparameters['CAT']) 



xgb_search = {'XGB': [{'colsample_bytree': 0.6917311125174739, 
                       'enable_categorical': False, 
                       'learning_rate':  space.Real(1e-5, 1e-2, default=0.018063876087523967, log=True), 
                       'max_depth': 10, 
                       'min_child_weight': 0.6028633586934382, 
                       'ag_args': {'name_suffix': '_HPO_LR', 'priority': 1}}, 
                      {'colsample_bytree': 0.6628423832084077, 
                       'enable_categorical': False, 
                       'learning_rate': 0.08775715546881824, 
                       'max_depth': space.Int (3, 20, default = 8), 
                       'min_child_weight': 0.6294123374222513, 
                       'ag_args': {'name_suffix': '_HPO_depth', 'priority': 2}}]}

xgb_result = search_space (xgb_search)

xgb_result ['ag_args'] = {'name_suffix': '_HPO', 'priority': 101}

custom_hyperparameters['XGB'].append (xgb_result)

print (custom_hyperparameters['XGB']) 


gbm_search = {'GBM': [

  {'learning_rate' : space.Real(1e-5, 1e-2, default=0.03, log=True), 
   'num_leaves':  128,
   'feature_fraction': 0.9,
   'min_data_in_leaf': 3,
   'ag_args': {'name_suffix': '_HPO',
    'priority': 1,
    }}, 
    {'learning_rate' : 0.03,
   'num_leaves': space.Int (20, 512, default =128),
   'feature_fraction': 0.9,
   'min_data_in_leaf': 3,
   'ag_args': {'name_suffix': '_HPO_leaves',
    'priority': 2,
    }}]}

gbm_result = search_space (gbm_search)

gbm_result ['ag_args'] = {'name_suffix': '_HPO', 'priority': 103}

custom_hyperparameters['GBM'].append (gbm_result)

print (custom_hyperparameters['GBM']) 



nn_torch_search = {
	 'NN_TORCH': [
                  {'activation': space.Categorical('relu', 'softrelu'), 
                       'dropout_prob': 0.1,
                       'hidden_size': 108, 
                       'learning_rate': space.Real(1e-6, 1e-2, default=5e-4, log=True), 
                       'num_layers': space.Int(lower=3, upper=10, default=6), 
                       'use_batchnorm': True, 
                       'weight_decay': 1.356433327634438e-12, 
                       'ag_args': {'name_suffix': '_torch_drop_layer_lr', 'priority': -2}}, 
                   {'activation': space.Categorical('softrelu', 'tanh'), 
                         'hidden_size': 213, 
                        'learning_rate': 0.0010474382260641949, 
                        'num_layers': space.Int(lower=3, upper=10, default=3) , 
                        'use_batchnorm': False, 
                        'weight_decay': 5.594471067786272e-10, 
                        'ag_args': {'name_suffix': '_torch_num_layer', 'priority': -7}}],
}

nn_torch_result  = search_space (nn_torch_search)

nn_torch_result ['ag_args'] = {'name_suffix': '_HPO', 'priority': 104}

custom_hyperparameters['NN_TORCH'].append (nn_torch_result)

print (custom_hyperparameters['NN_TORCH']) 


fastai_search = { 'FASTAI': [ 
               {'bs': space.Int(lower=5, upper=5024, default= 1024),   
                'emb_drop': 0.5411770367537934, 
                'epochs': 43, 
                'layers': [800, 400], 
                'lr': 0.01519848858318159, 
                'ps': 0.23782946566604385, 
                'ag_args': {'name_suffix': '_bs_search', 'priority': 1}}, 
               {'bs': 2048, 
                'emb_drop': 0.05070411322605811,
                'epochs': 80, 
                'layers': [200, 100], 
                'lr': space.Real(0.000001, 0.5, default=0.0897423 , log=True), 
                'ps': 0.10393466140748028,
                'ag_args': {'name_suffix': '_lr_search', 'priority': 2}},
                 {'bs': 1024,   
                'emb_drop': space.Real(0.001, 0.99, default=0.8423 ), 
                'epochs': 53, 
                'layers': [800, 400], 
                'lr': 0.01519848858318159, 
                'ps': space.Real(0.001, 0.99, default=0.23 ), 
                'ag_args': {'name_suffix': '_emb_ps_search', 'priority': 3}}
                
           ]}

fastai_result = search_space (fastai_search)

fastai_result ['ag_args'] = {'name_suffix': '_HPO', 'priority': 105}

custom_hyperparameters['FASTAI'].append (fastai_result)

print (custom_hyperparameters['FASTAI']) 


print (custom_hyperparameters)

final_predictor = TabularPredictor(path = '/kaggle/working/Autogluon2',
                                       label='Price', 
                               problem_type = 'regression', 
                               eval_metric =  'root_mean_squared_error',  
                               sample_weight = 'my_weight',
                               verbosity  = 2,
                               learner_kwargs = {'ignored_columns' : [
                                   'id',
                                   'Price'
                               #   'my_weight'
                                    ]})

final_predictor.fit(train_data= new_features_train.to_pandas(), 
                        presets= 'medium_quality',
    # best_quality, high_quality, medium_quality, 'experimental_quality',                         
                        time_limit = 10000,
                        num_gpus=0,
                        raise_on_no_models_fitted = True,
                        dynamic_stacking=False, 
                        num_stack_levels=0,
                        hyperparameters=custom_hyperparameters,
#                         hyperparameters = my_search_hyperparameters  ,
#                         hyperparameter_tune_kwargs=hyperparameter_tune_kwargs,
                        )

final_predictor.leaderboard()


final_predictor.leaderboard()


#new_features_test = add_features (raw_train)

#new_features_test = new_features_test.drop(["Price"])


#y_pred =  final_predictor.predict (new_features_test.to_pandas())


#sns.histplot(y_pred)


new_features_test = add_features (raw_test)




y_pred = pl.Series ("Price", final_predictor.predict (new_features_test.to_pandas()))

y_pred



sample_submission = sample_submission.with_columns (y_pred.alias ("Price"))

sample_submission.head()


sample_submission.write_csv('submission.csv')
print ('submission finished')


print (y_pred.describe( [0.1, 0.25,0.5, 0.75, 0.9]))

sns.histplot(y_pred.to_numpy())
plt.show()




