import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


test=pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
train=pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
df_all=pd.concat([train, test], ignore_index  = True)


train.describe()


test.describe()


for column in df_all.columns:
        print(f"{column}: {df_all[column].nunique()} unique valuies")


print("\nCount of missing values:")
print(df_all.isnull().sum())


import matplotlib as plt
train.drop(columns='id').hist(bins=50, figsize=(20,13))


train1=pd.get_dummies(train, columns=['Sex'])
test1=pd.get_dummies(test, columns=['Sex'])


def spread_line(tbl, col, num=10):
    return np.linspace(min(tbl[f'{col}'])-1, max(tbl[f'{col}'])+1, num=num)

train1['Age_expand']=pd.cut(train1['Age'], spread_line(train1, 'Age', 20))
train1['Height_expand']=pd.cut(train1['Height'], spread_line(train1, 'Height', 20))
train1['Weight_expand']=pd.cut(train1['Weight'], spread_line(train1, 'Weight', 20))
train1['Duration_expand']=pd.cut(train1['Duration'], spread_line(train1, 'Duration', 20))
train1['Heart_Rate_expand']=pd.cut(train1['Heart_Rate'], spread_line(train1, 'Heart_Rate', 20))
train1['Body_Temp_expand']=pd.cut(train1['Body_Temp'], spread_line(train1, 'Body_Temp', 20))


test1['Age_expand']=pd.cut(test1['Age'], spread_line(test1, 'Age', 20))
test1['Height_expand']=pd.cut(test1['Height'], spread_line(test1, 'Height', 20))
test1['Weight_expand']=pd.cut(test1['Weight'], spread_line(test1, 'Weight', 20))
test1['Duration_expand']=pd.cut(test1['Duration'], spread_line(test1, 'Duration', 20))
test1['Heart_Rate_expand']=pd.cut(test1['Heart_Rate'], spread_line(test1, 'Heart_Rate', 20))
test1['Body_Temp_expand']=pd.cut(test1['Body_Temp'], spread_line(test1, 'Body_Temp', 20))



train1.drop(columns=['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp'], inplace=True)
test1.drop(columns=['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp'], inplace=True)

features=['Sex_female', 
          'Sex_male', 
          'Age_expand',
          'Height_expand', 
          'Weight_expand', 
          'Duration_expand',
          'Heart_Rate_expand',
          'Body_Temp_expand']


train1.columns


train1


#combinations 
import itertools
#three_features=itertools.combinations(enumerate(features), 3)
four_features=itertools.combinations(enumerate(features), 4)



def analyze_feature_combinations(features_list, train_data, test_data):
    feature_results = []
    
    for feature_combo in features_list:
        # Extract feature names
        feature_names = [f[1] for f in feature_combo]
        
        # Calculate statistics
        stats = (train_data.groupby(feature_names)['Calories']
                 .agg(['median', 'count'])
                 .reset_index()
                 .sort_values('median', ascending=False))
        
        # Filter by threshold
        #high_survival = stats.loc[stats['mean']]
        
        if not stats.empty:
            # Dynamically rename columns
            rename_dict = {
                'median': 'score',
                'count': 'cnt'
            }
            for i, col in enumerate(feature_names, 1):
                rename_dict[col] = f'val{i}'
                stats[f'columns{i}'] = col
            
            stats = stats.rename(columns=rename_dict)
            feature_results.append(stats)
    
    # Combine all results
    result_df = pd.concat(feature_results, ignore_index=True) if feature_results else pd.DataFrame()
    return result_df

def apply_to_test_data(result_df, test_data):
    output_data = []
    
    if not result_df.empty:
        # Dynamically generate columns based on the number of features
        num_features = max(int(col[7:]) for col in result_df.columns if col.startswith('columns'))
        
        for _, row in result_df.iterrows():
            # Create filter mask dynamically
            mask = pd.Series(True, index=test_data.index)
            for i in range(1, num_features + 1):
                mask &= (test_data[row[f'columns{i}']] == row[f'val{i}'])
            
            # Get matching passenger IDs
            matching_passengers = test_data.loc[mask, 'id']
            
            # Prepare output
            for pid in matching_passengers:
                output_row = [pid]
                for i in range(1, num_features + 1):
                    output_row.extend([row[f'val{i}'], row[f'columns{i}']])
                output_row.extend([row['score'], row['cnt']])
                output_data.append(output_row)
    
    # Dynamically generate column names
    columns = ['id']
    if result_df.empty:
        return pd.DataFrame(columns=columns)
    
    num_features = max(int(col[7:]) for col in result_df.columns if col.startswith('columns'))
    for i in range(1, num_features + 1):
        columns.extend([f'val{i}', f'columns{i}'])
    columns.extend(['score', 'cnt'])
    
    return pd.DataFrame(output_data, columns=columns)

def create_final_result(df_output, num_features):
    # Create the combined 'columns' string
    df_result = df_output.copy()
    
    # Combine all columns into one string
    cols_to_combine = [f'columns{i}' for i in range(1, num_features + 1)]
    df_result['columns'] = df_result[cols_to_combine].apply(lambda x: ', '.join(x), axis=1)
    
    return df_result



def prepare_final_dataframe(df_output, max_features=2):
   
    if not df_output.empty:
        # Detect features 
        num_features = max(int(col[7:]) for col in df_output.columns 
                          if col.startswith('columns'))
        df_result = create_final_result(df_output, num_features)
    else:
        # Create Dataframe with empty columns
        columns = ['id'] + \
                 [f'val{i}' for i in range(1, max_features + 1)] + \
                 ['score', 'cnt'] + \
                 [f'columns{i}' for i in range(1, max_features + 1)] + \
                 ['columns']
        df_result = pd.DataFrame(columns=columns)
    
    return df_result


def add_prediction_to_test_table(x, min_limit=0):
    x=pd.pivot_table(x, values='score', index='id', columns='columns', aggfunc='max')
    x['Calories']=x.mean(axis=1)
    x.fillna(0, inplace=True)
    return x


import warnings
warnings.filterwarnings("ignore")


#df_three_features = analyze_feature_combinations(three_features, train1, test1)
#df_result_3 = apply_to_test_data(df_three_features, test1)
#df_result_3 = prepare_final_dataframe(df_result_3)
#df_result_3 = add_prediction_to_test_table(df_result_3)
#df_result_3


#df_result_3=df_result_3.iloc[:, -1].to_frame().reset_index()
#df_result_3


df_four_features = analyze_feature_combinations(four_features, train1, test1)
df_result_4 = apply_to_test_data(df_four_features, test1)
df_result_4 = prepare_final_dataframe(df_result_4)
df_result_4 = add_prediction_to_test_table(df_result_4)
df_result_4=df_result_4.iloc[:, -1].to_frame().reset_index()
df_result_4


#result=pd.concat([df_result_4, df_result_3], axis=1).drop(columns=['Calories']) 
#result['Calories']=result.mean(axis=1)
#submission=result



submission=df_result_4
submission.to_csv('submission.csv', index=False)

