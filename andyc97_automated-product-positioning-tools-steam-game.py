# Install google-cloud-bigquery-storage for running BigQuery SQL without error
# Add -q to suppress verbose for the sake of readability 
!pip uninstall -q -y bigframes
!pip install -q google-cloud-bigquery-storage


# Import all libraries required for this project
import pandas as pd

from google.cloud import bigquery
from datetime import datetime, timedelta


# Initialize BigQuery client with Google Cloud's project id
project_id = 'analog-delight-470708-d0'
client = bigquery.Client(project=project_id)

# We also define dataset and table ids
dataset_id = 'steam'
game_list_data = 'steam_game_list'
review_data = 'steam_reviews'

# We also define the name of text embedding models
embedding_model_name = 'llm_steam'


# Check whether a column exist in the table schema
def check_column_exists(dataset_id, table_id, name):
    # Given dataset_id and table_id, we retrieve its schema
    table_ref = client.dataset(dataset_id).table(table_id)
    table_schema = client.get_table(table_ref).schema

    # Loop through each field in the schema to determine whether a column exists.
    for field in table_schema:
        if field.name == name:
            return True
    return False


# Check whether the primary key for game_list_data exists
game_list_data_pk = 'app_id'
exist_app_id = check_column_exists(dataset_id, game_list_data, game_list_data_pk)
print('Does the primary key exist? ' + str(exist_app_id))

# If it does not exist, generate it
if not exist_app_id:
    query = f"""
    alter table {project_id}.{dataset_id}.{game_list_data}
    add column if not exists {game_list_data_pk} integer;
    
    update {project_id}.{dataset_id}.{game_list_data}
    set {game_list_data_pk} = cast(`App ID` as integer)
    where true;
    
    alter table {project_id}.{dataset_id}.{game_list_data}
    add primary key ({game_list_data_pk}) not enforced;
    """
    result_pk = client.query(query)
    print(result_pk.result())


# This function generate embeddings using AI model if not exists
def create_embeddings(embeddings_name, embeddings_model_name, table_name, column_name):
    query = f"""
    alter table `{project_id}.{dataset_id}.{table_name}`
    add column if not exists {embeddings_name} array<float64>;

    update `{project_id}.{dataset_id}.{table_name}` as t
    set t.{embeddings_name} = e.ml_generate_embedding_result
    from (
        select distinct
            ml_generate_embedding_result,
            content
        from ml.generate_embedding(
            model `{project_id}.{dataset_id}.{embeddings_model_name}`,
            (select ifnull({column_name}, ' ') as content
              from `{project_id}.{dataset_id}.{table_name}`
            )
        )
    ) e
    where ifnull(t.{column_name}, ' ') = e.content
    """
    return client.query(query)


# Create text embeddings for 'short description' of each game available on steam
exist_desc = check_column_exists(dataset_id, game_list_data, "desc_embeddings")
print('Does the embeddings for short description of game exist? ' + str(exist_desc))
if not exist_desc:
    result_desc = create_embeddings("desc_embeddings", embedding_model_name, game_list_data, "`Short Description`")
    print(result_desc.result())

# Create text embeddings for 'tags' fof each game available on steam 
exist_tags = check_column_exists(dataset_id, game_list_data, "tags_embeddings")
print('Does the embeddings for tags of game exist? ' + str(exist_tags))
if not exist_tags:
    result_tags = create_embeddings("tags_embeddings", embedding_model_name, game_list_data, "tags")
    print(result_tags.result())


def create_table_reviews_embeddings(out_table_name, in_table_name):
    query = f"""
    create or replace table `{project_id}.{dataset_id}.{out_table_name}` as
    select *
    from ml.generate_embeddings(
    model `{project_id}.{dataset_id}.{embedding_model_name}`,
    (select Review as content, app_id from `{project_id}.{dataset_id}.{in_table_name}`),
    struct(
        true AS flatten_json_output, 
        'RETRIEVAL_DOCUMENT' as task_type
        )
    );

    create or replace table {dataset_id}.{out_table_name}_notnull as 
    select * from {dataset_id}.{out_table_name} 
    where ARRAY_LENGTH(ml_generate_embedding_result) > 0;

    create vector index if not exists embedding_index
    on {dataset_id}.{out_table_name}_notnull(ml_generate_embedding_result)
    storing(app_id)
    options(distance_type='COSINE', index_type='IVF');
    """
    return client.query(query)


table_review_embeddings = 'review_embeddings'
try:
    client.dataset(dataset_id).table(table_review_embeddings)
    print(f'Table {table_review_embeddings} already exists.')
except: 
    print(f'Table {table_review_embeddings} does not exist. A new table will be created.')
    handler = create_table_reviews_embeddings('review_embeddings', review_data)


def get_list_of_games(user_input, number_of_games, min_reviews):
    embeddings = ["desc_embeddings", "tags_embeddings"]
    query = f"""
    select a.base.*
    from vector_search(
        (select {embeddings[0]}, name, app_id, `short description`, tags, `positive reviews`, `negative reviews` 
        from `{project_id}.{dataset_id}.{game_list_data}`
        where (`positive reviews` > {min_reviews}) or (`negative reviews` > {min_reviews})),
        '{embeddings[0]}',
        (select ml_generate_embedding_result, content as query 
        from ml.generate_embedding(
        model `{project_id}.{dataset_id}.{embedding_model_name}`,
            (select '{user_input}' as content))
        ),
        top_k => {number_of_games},
        distance_type => 'COSINE') as a
    inner join 
    vector_search(
        (select {embeddings[1]}, name, app_id, `short description`, tags, `positive reviews`, `negative reviews` 
        from `{project_id}.{dataset_id}.{game_list_data}`
        where (`positive reviews` > {min_reviews}) or (`negative reviews` > {min_reviews})),
        '{embeddings[1]}',
        (select ml_generate_embedding_result, content as query 
        from ml.generate_embedding(
        model `{project_id}.{dataset_id}.{embedding_model_name}`,
            (select '{user_input}' as content))
        ),
        top_k => {number_of_games},
        distance_type => 'COSINE') as b
        on a.base.app_id = b.base.app_id
    """
    df = client.query(query).to_dataframe()
    return df


# Parameters to set
number_of_games = 100
min_reviews = 1000

# Example 1
user_input = "I would like to find a multi-person strategic game on farming in an open-world setting."
df_retrieve1 = get_list_of_games(user_input, number_of_games, min_reviews)

# Example 2
user_input = "I would like to find a multi-person action RPG game with dragon"
df_retrieve2 = get_list_of_games(user_input, number_of_games, min_reviews)


# Convert pd.Series to python list object for postprocessing 
app_ids_1 = df_retrieve['app_id'].values.tolist()
print('The list of relevant games (app_ids) retrieved by the user query: ', app_ids_1)

# Convert pd.Series to python list object for postprocessing 
app_ids_2 = df_retrieve2['app_id'].values.tolist()
print('The list of relevant games (app_ids) retrieved by the user query: ', app_ids_2)


# Calculate the metric column to measure good rating and sort the dataset in descending order.
df_retrieve1['odd'] = df_retrieve1['positive reviews']/df_retrieve1['negative reviews']
df_sort = df_retrieve.iloc[:, 1:].sort_values('odd', ascending=False)
df_sort


df_retrieve2['odd'] = df_retrieve2['positive reviews']/df_retrieve2['negative reviews']
df_sort2 = df_retrieve2.iloc[:, 1:].sort_values('odd', ascending=False)
df_sort2


number_of_reviews = 20
user_input_age = 'Is this game easy to play for elderly?'
user_input_music = 'Tell me the music and visual aspects of this game.'
user_input_price = 'Is the price expensive or cheap or just affordable given the quality?'

def search_reviews(user_input, number_of_reviews):
    query = f"""
    select a.base.*
    from vector_search(
        (select ml_generate_embedding_result, app_id, content
        from `{project_id}.{dataset_id}.{table_review_embeddings}_notnull`
        where app_id in ({','.join([str(app_id) for app_id in app_ids_1])})),
        'ml_generate_embedding_result',
        (select ml_generate_embedding_result, content as query 
        from ml.generate_embedding(
        model `{project_id}.{dataset_id}.{embedding_model_name}`,
            (select '{user_input}' as content))
        ),
        top_k => {number_of_reviews},
        distance_type => 'COSINE') a
    """
    df = client.query(query).to_dataframe()
    return df


df_age = search_reviews(user_input_age, number_of_reviews)
df_music = search_reviews(user_input_music, number_of_reviews)
df_price = search_reviews(user_input_price, number_of_reviews)


def merge_dataset(df):
    return df.iloc[:, 1:].\
        merge(df_retrieve[['app_id', 'name', 'odd']], on='app_id').\
        sort_values('name')


pd.set_option('display.max_colwidth', None)
print(f'Analysis: {user_input_age}')
merge_dataset(df_age)


print(f'Analysis: {user_input_music}')
merge_dataset(df_music)


print(f'Analysis: {user_input_price}')
merge_dataset(df_price)

