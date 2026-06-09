import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from google.cloud import bigquery
from google.api_core.exceptions import NotFound, Conflict
import logging

# Configure logging to provide more detailed output
logging.basicConfig(level=logging.INFO)
logging.getLogger("google.cloud").setLevel(logging.DEBUG)

def create_bigquery_dataset(project_id: str, dataset_id: str, location: str = "US"):
    """
    Creates a new BigQuery dataset if it does not already exist.

    Args:
        project_id (str): The Google Cloud project ID.
        dataset_id (str): The ID of the dataset to create.
        location (str): The geographic location for the dataset (e.g., 'US', 'EU').
                        This cannot be changed after creation.
    """
    try:
        logging.info(f"Connecting to BigQuery client for project '{project_id}'...")
        client = bigquery.Client(project=project_id)

        # Construct a full Dataset object
        dataset = bigquery.Dataset(f"{project_id}.{dataset_id}")
        dataset.location = location

        logging.info(f"Attempting to create dataset '{dataset_id}' in location '{location}'...")
        dataset = client.create_dataset(dataset, timeout=30)
        logging.info(f"Successfully created dataset '{dataset.dataset_id}'.")
    except Conflict:
        logging.info(f"Dataset '{dataset_id}' already exists. Skipping creation.")
    except Exception as e:
        logging.error(f"An error occurred while creating the dataset: {e}")
        raise


def write_table_from_query(project_id: str, dataset_id: str, table_id: str, query: str, overwrite: bool = True):
    """
    Executes a SQL query and writes the results to a new or existing BigQuery table.

    Args:
        project_id (str): The Google Cloud project ID.
        dataset_id (str): The ID of the target dataset.
        table_id (str): The ID of the target table.
        query (str): The SQL query to execute.
        overwrite (bool): If True, the table will be overwritten (truncated). If False,
                          the results will be appended to the table.
    """
    try:
        logging.info(f"Connecting to BigQuery client for project '{project_id}'...")
        client = bigquery.Client(project=project_id)

        # Define the fully qualified destination table ID
        table_ref = client.dataset(dataset_id).table(table_id)
        
        # Determine the write disposition based on the 'overwrite' parameter
        if overwrite:
            write_disposition = bigquery.WriteDisposition.WRITE_TRUNCATE
            logging.info(f"Setting job disposition to WRITE_TRUNCATE (overwrite).")
        else:
            write_disposition = bigquery.WriteDisposition.WRITE_APPEND
            logging.info(f"Setting job disposition to WRITE_APPEND (append).")

        # Configure the query job to save results to the specified table
        job_config = bigquery.QueryJobConfig(
            destination=table_ref,
            write_disposition=write_disposition
        )

        logging.info(f"Starting query job to populate table '{table_id}'...")
        # Start the query and wait for it to complete
        job = client.query(query, job_config=job_config)
        
        logging.info(f"Waiting for job {job.job_id} to complete...")
        job.result()  # Waits for the job to finish

        logging.info(f"Job {job.job_id} completed. Table populated successfully.")
        table = client.get_table(table_ref)
        logging.info(
            f"Table '{table.table_id}' now contains {table.num_rows} rows."
        )

    except NotFound:
        logging.error(f"Dataset '{dataset_id}' not found. Please create the dataset first.")
        raise
    except Exception as e:
        logging.error(f"An error occurred during the table creation from query: {e}")
        raise


def run_query(project_id: str, query: str) -> pd.DataFrame:
    """
    Executes a SQL query and returns the results as a Pandas DataFrame.

    Args:
        project_id (str): The Google Cloud project ID.
        query (str): The SQL query to execute.

    Returns:
        pd.DataFrame: A DataFrame containing the query results.
    """
    try:
        logging.info(f"Connecting to BigQuery client for project '{project_id}'...")
        client = bigquery.Client(project=project_id)

        logging.info("Starting query job...")
        # Start the query and wait for it to complete
        job = client.query(query)

        logging.info(f"Waiting for job {job.job_id} to complete...")
        results = job.result()  # Waits for the job to finish

        logging.info(f"Job {job.job_id} completed. Fetching results as a DataFrame...")
        df = results.to_dataframe()
        logging.info(f"Query returned {len(df)} rows.")
        return df

    except Exception as e:
        logging.error(f"An error occurred while running the query: {e}")
        raise


def import_parquet_file(project_id: str, dataset_id: str, table_id: str, parquet_file_path: str, overwrite: bool = True):
    """
    Loads data from a local Parquet file into a BigQuery table.
    The table will be created if it does not exist.

    Args:
        project_id (str): The Google Cloud project ID.
        dataset_id (str): The ID of the target dataset.
        table_id (str): The ID of the target table.
        parquet_file_path (str): The local path to the Parquet file.
        overwrite (bool): If True, the table will be overwritten (truncated). If False,
                          the results will be appended to the table.
    """
    if not os.path.exists(parquet_file_path):
        logging.error(f"Parquet file not found at path: {parquet_file_path}")
        raise FileNotFoundError(f"File not found: {parquet_file_path}")

    try:
        logging.info(f"Connecting to BigQuery client for project '{project_id}'...")
        client = bigquery.Client(project=project_id)
        
        # Define the fully qualified destination table ID
        table_ref = client.dataset(dataset_id).table(table_id)

        # Determine the write disposition based on the 'overwrite' parameter
        if overwrite:
            write_disposition = bigquery.WriteDisposition.WRITE_TRUNCATE
            logging.info(f"Setting job disposition to WRITE_TRUNCATE (overwrite).")
        else:
            write_disposition = bigquery.WriteDisposition.WRITE_APPEND
            logging.info(f"Setting job disposition to WRITE_APPEND (append).")

        # Configure the load job for a Parquet file
        job_config = bigquery.LoadJobConfig(
            source_format=bigquery.SourceFormat.PARQUET,
            write_disposition=write_disposition
        )

        logging.info(f"Opening local Parquet file from '{parquet_file_path}'...")
        with open(parquet_file_path, "rb") as source_file:
            logging.info(f"Starting load job for table '{table_id}'...")
            job = client.load_table_from_file(
                source_file,
                table_ref,
                location="US",  # Must match the dataset location
                job_config=job_config
            )

        # Wait for the job to complete
        logging.info(f"Waiting for job {job.job_id} to complete...")
        job.result()

        logging.info(f"Job {job.job_id} completed. Data loaded successfully.")
        table = client.get_table(table_ref)
        logging.info(
            f"Loaded {table.num_rows} rows and {len(table.schema)} columns to table '{table_id}'."
        )

    except NotFound:
        logging.error(f"Dataset '{dataset_id}' not found. Please create the dataset first.")
        raise
    except Exception as e:
        logging.error(f"An error occurred during the table load job: {e}")
        raise

def import_csv_file(project_id: str, dataset_id: str, table_id: str, csv_file_path: str, overwrite: bool = True):
    """
    Loads data from a local CSV file into a BigQuery table.
    The table will be created with an autodetected schema if it does not exist.

    Args:
        project_id (str): The Google Cloud project ID.
        dataset_id (str): The ID of the target dataset.
        table_id (str): The ID of the target table.
        csv_file_path (str): The local path to the CSV file.
        overwrite (bool): If True, the table will be overwritten (truncated). If False,
                          the results will be appended to the table.
    """
    if not os.path.exists(csv_file_path):
        logging.error(f"CSV file not found at path: {csv_file_path}")
        raise FileNotFoundError(f"File not found: {csv_file_path}")

    try:
        logging.info(f"Connecting to BigQuery client for project '{project_id}'...")
        client = bigquery.Client(project=project_id)
        
        # Define the fully qualified destination table ID
        table_ref = client.dataset(dataset_id).table(table_id)

        # Determine the write disposition based on the 'overwrite' parameter
        if overwrite:
            write_disposition = bigquery.WriteDisposition.WRITE_TRUNCATE
            logging.info(f"Setting job disposition to WRITE_TRUNCATE (overwrite).")
        else:
            write_disposition = bigquery.WriteDisposition.WRITE_APPEND
            logging.info(f"Setting job disposition to WRITE_APPEND (append).")

        # Configure the load job for a CSV file with auto-detection
        job_config = bigquery.LoadJobConfig(
            source_format=bigquery.SourceFormat.CSV,
            skip_leading_rows=1, # Skips the header row
            autodetect=True,
            write_disposition=write_disposition
        )

        logging.info(f"Opening local CSV file from '{csv_file_path}'...")
        with open(csv_file_path, "rb") as source_file:
            logging.info(f"Starting load job for table '{table_id}'...")
            job = client.load_table_from_file(
                source_file,
                table_ref,
                location="US",  # Must match the dataset location
                job_config=job_config
            )

        # Wait for the job to complete
        logging.info(f"Waiting for job {job.job_id} to complete...")
        job.result()

        logging.info(f"Job {job.job_id} completed. Data loaded successfully.")
        table = client.get_table(table_ref)
        logging.info(
            f"Loaded {table.num_rows} rows and {len(table.schema)} columns to table '{table_id}'."
        )

    except NotFound:
        logging.error(f"Dataset '{dataset_id}' not found. Please create the dataset first.")
        raise
    except Exception as e:
        logging.error(f"An error occurred during the table load job: {e}")
        raise

def import_json_file(project_id: str, dataset_id: str, table_id: str, json_file_path: str, overwrite: bool = True):
    """
    Loads data from a local JSON file (newline-delimited) into a BigQuery table.
    The table will be created with an autodetected schema if it does not exist.

    Args:
        project_id (str): The Google Cloud project ID.
        dataset_id (str): The ID of the target dataset.
        table_id (str): The ID of the target table.
        json_file_path (str): The local path to the JSON file.
        overwrite (bool): If True, the table will be overwritten (truncated). If False,
                          the results will be appended to the table.
    """
    if not os.path.exists(json_file_path):
        logging.error(f"JSON file not found at path: {json_file_path}")
        raise FileNotFoundError(f"File not found: {json_file_path}")

    try:
        logging.info(f"Connecting to BigQuery client for project '{project_id}'...")
        client = bigquery.Client(project=project_id)
        
        # Define the fully qualified destination table ID
        table_ref = client.dataset(dataset_id).table(table_id)

        # Determine the write disposition based on the 'overwrite' parameter
        if overwrite:
            write_disposition = bigquery.WriteDisposition.WRITE_TRUNCATE
            logging.info(f"Setting job disposition to WRITE_TRUNCATE (overwrite).")
        else:
            write_disposition = bigquery.WriteDisposition.WRITE_APPEND
            logging.info(f"Setting job disposition to WRITE_APPEND (append).")

        # Configure the load job for a JSON file
        job_config = bigquery.LoadJobConfig(
            source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
            autodetect=True,
            write_disposition=write_disposition
        )

        logging.info(f"Opening local JSON file from '{json_file_path}'...")
        with open(json_file_path, "rb") as source_file:
            logging.info(f"Starting load job for table '{table_id}'...")
            job = client.load_table_from_file(
                source_file,
                table_ref,
                location="US",  # Must match the dataset location
                job_config=job_config
            )

        # Wait for the job to complete
        logging.info(f"Waiting for job {job.job_id} to complete...")
        job.result()

        logging.info(f"Job {job.job_id} completed. Data loaded successfully.")
        table = client.get_table(table_ref)
        logging.info(
            f"Loaded {table.num_rows} rows and {len(table.schema)} columns to table '{table_id}'."
        )

    except NotFound:
        logging.error(f"Dataset '{dataset_id}' not found. Please create the dataset first.")
        raise
    except Exception as e:
        logging.error(f"An error occurred during the table load job: {e}")
        raise



project_id = 
project_number = 
dataset_id =
location = 


create_bigquery_dataset(project_id=project_id, dataset_id=dataset_id, location=location)


!bq mk --connection --location={location} --project_id={project_id} \
    --connection_type=CLOUD_RESOURCE {dataset_id}


!bq show --connection --project_id=kaggle-hackathon-project --location=US --format=json geo_intent


!gcloud projects add-iam-policy-binding {project_id} \
    --member=serviceAccount: <PUT BQ SERVICE ACCOUNT EMAIL> \
    --role=roles/aiplatform.user


test_ai_command = f"""
SELECT
  AI.GENERATE(
    ('Give a short, one sentence description of Austin'),
    connection_id => 'us.{dataset_id}',
    endpoint => 'gemini-2.0-flash').result
"""
run_query(project_id=project_id, query=test_ai_command)


demographics_table_creation_query = f"""
SELECT
  a.zip_code,
  a.city,
  a.county,
  a.state_name AS state,
  a.area_land_meters + a.area_water_meters AS area,
  a.internal_point_geom AS point_geom,
  b.total_pop,
  b.households,
  b.median_age,
  b.pop_25_64,
  b.median_income,
  b.income_per_capita,
  b.housing_units,
  b.occupied_housing_units,
  b.owner_occupied_housing_units,
  b.million_dollar_housing_units,
  b.housing_units_renter_occupied,
  b.median_year_structure_built,
  b.family_households,
  b.median_rent,
  b.percent_income_spent_on_rent,
  (b.commute_less_10_mins + b.commute_10_14_mins + b.commute_15_19_mins + b.commute_20_24_mins + b.commute_25_29_mins ) AS commute_within_30_min,
  b.commute_60_more_mins,
  b.commuters_16_over,
  b.walked_to_work,
  b.worked_at_home,
  b.commuters_by_public_transportation,
  b.commuters_by_car_truck_van,
  b.associates_degree,
  b.bachelors_degree,
  b.high_school_diploma,
  b.masters_degree,
  b.graduate_professional_degree,
  b.employed_pop,
  b.unemployed_pop,
  b.workers_16_and_over,
  b.in_school,
  b.in_undergrad_college
FROM
  `bigquery-public-data.geo_us_boundaries.zip_codes` AS a
JOIN
  `bigquery-public-data.census_bureau_acs.zip_codes_2018_5yr` AS b
ON
  a.zip_code = b.geo_id;"""

write_table_from_query(project_id=project_id, dataset_id=dataset_id, table_id='demographic_data', query=demographics_table_creation_query, overwrite=True)


column_descriptions = {
    "zip_code": "The 5-digit ZIP code tabulation area (ZCTA) identifier.",
    "city": "The default city name associated with the ZIP code.",
    "county": "The name of the county containing the ZIP code.",
    "state": "The full name of the state.",
    "area": "Total area of the ZIP code in square meters, including both land and water.",
    "point_geom": "A GEOGRAPHY point representing the internal center of the ZIP code's boundaries.",
    "total_pop": "Total population within the ZIP code. Source: ACS 2018 5-year estimates.",
    "households": "Total number of households within the ZIP code. Source: ACS 2018 5-year estimates.",
    "median_age": "The median age of the population. Source: ACS 2018 5-year estimates.",
    "pop_25_64": "Population aged 25 to 64 years. Source: ACS 2018 5-year estimates.",
    "median_income": "Median household income in the past 12 months (in 2018 inflation-adjusted dollars). Source: ACS 2018 5-year estimates.",
    "income_per_capita": "Per capita income in the past 12 months (in 2018 inflation-adjusted dollars). Source: ACS 2018 5-year estimates.",
    "housing_units": "Total number of housing units. Source: ACS 2018 5-year estimates.",
    "occupied_housing_units": "Number of housing units that are occupied. Source: ACS 2018 5-year estimates.",
    "owner_occupied_housing_units": "Number of occupied housing units that are owner-occupied. Source: ACS 2018 5-year estimates.",
    "million_dollar_housing_units": "Owner-occupied housing units valued at $1,000,000 or more. Source: ACS 2018 5-year estimates.",
    "housing_units_renter_occupied": "Number of occupied housing units that are renter-occupied. Source: ACS 2018 5-year estimates.",
    "median_year_structure_built": "The median year in which housing structures were built. Source: ACS 2018 5-year estimates.",
    "family_households": "Total number of family households. Source: ACS 2018 5-year estimates.",
    "median_rent": "Median gross rent for renter-occupied units. Source: ACS 2018 5-year estimates.",
    "percent_income_spent_on_rent": "Median percentage of household income spent on rent. Source: ACS 2018 5-year estimates.",
    "commute_within_30_min": "Total commuters with a travel time to work of less than 30 minutes. Source: ACS 2018 5-year estimates.",
    "commute_60_more_mins": "Commuters with a travel time to work of 60 minutes or more. Source: ACS 2018 5-year estimates.",
    "commuters_16_over": "Total number of commuters aged 16 and over. Source: ACS 2018 5-year estimates.",
    "walked_to_work": "Number of commuters who walked to work. Source: ACS 2018 5-year estimates.",
    "worked_at_home": "Number of people who worked from home. Source: ACS 2018 5-year estimates.",
    "commuters_by_public_transportation": "Number of commuters using public transportation. Source: ACS 2018 5-year estimates.",
    "commuters_by_car_truck_van": "Number of commuters using a car, truck, or van. Source: ACS 2018 5-year estimates.",
    "associates_degree": "Population with an Associate's degree as their highest level of education. Source: ACS 2018 5-year estimates.",
    "bachelors_degree": "Population with a Bachelor's degree as their highest level of education. Source: ACS 2018 5-year estimates.",
    "high_school_diploma": "Population with a high school diploma or equivalent as their highest level of education. Source: ACS 2018 5-year estimates.",
    "masters_degree": "Population with a Master's degree as their highest level of education. Source: ACS 2018 5-year estimates.",
    "graduate_professional_degree": "Population with a graduate or professional degree. Source: ACS 2018 5-year estimates.",
    "employed_pop": "Number of the population aged 16 and over that is employed. Source: ACS 2018 5-year estimates.",
    "unemployed_pop": "Number of the population aged 16 and over that is unemployed. Source: ACS 2018 5-year estimates.",
    "workers_16_and_over": "Total workers aged 16 and over. Source: ACS 2018 5-year estimates.",
    "in_school": "Population 3 years and over enrolled in school. Source: ACS 2018 5-year estimates.",
    "in_undergrad_college": "Population 15 years and over enrolled in undergraduate college. Source: ACS 2018 5-year estimates."
}

# Build the list of ALTER COLUMN clauses
alter_clauses = []
for column, description in column_descriptions.items():
    # Escape double quotes in description
    escaped_description = description.replace('"', '\\"')
    alter_clauses.append(f'ALTER COLUMN {column} SET OPTIONS(description="{escaped_description}")')

# Join the clauses with a comma
joined_clauses = ",\n".join(alter_clauses)

# Construct the final single ALTER TABLE query
alter_query = f"""
ALTER TABLE `{project_id}.{dataset_id}.demographic_data`
{joined_clauses}
"""

# Run the single query
print("Setting all column descriptions in a single statement...")
run_query(project_id=project_id, query=alter_query)
print("All column descriptions have been updated.")



us_places_category="""SELECT
  a.*,
  AI.GENERATE(
    ('Generate a description of business that fall in the category: ', category, '. Describe how it affects the business of a cafe if any.'),
    connection_id => 'us.geo_intent',
    endpoint => 'gemini-2.5-flash').result as category_description,
  AI.GENERATE_BOOL(
    ('Decide if a place categorized as ', category, ' is a direct competitor to a cafe.'),
    connection_id => 'us.geo_intent',
    endpoint => 'gemini-2.5-flash').result as competition,
  AI.GENERATE_INT(
    ('Decide the magnitude of competition between a place categorized as', category, ' and a cafe. Output a single float between 0 and 100, where 0 means no competition and 100 means they are direct competitors selling the same products.'),
    connection_id => 'us.geo_intent',
    endpoint => 'gemini-2.5-flash').result as competition_magnitude,
  AI.GENERATE_BOOL(
    ('Decide if a place categorized as ', category, 'represents an opportunity for more sales to a cafe. Output only true or false.'),
    connection_id => 'us.geo_intent',
    endpoint => 'gemini-2.5-flash').result as opportunity,
  AI.GENERATE_INT(
    ('Decide the magnitude of competition between a place categorized as', category, ' and a cafe. Output a single float between 0 and 100, where 0 means no opportunity and 100 means maximum opportunity for additional sales.'),
    connection_id => 'us.geo_intent',
    endpoint => 'gemini-2.5-flash').result as opportunity_magnitude,
    
FROM (
  SELECT
    categories.`primary` AS category,
    COUNT(DISTINCT id) AS number_of_places
  FROM
    bigquery-public-data.overture_maps.place
  WHERE
    addresses.`list`[SAFE_OFFSET(0)].element.country = 'US'
  GROUP BY
    1
  ORDER BY
    2 DESC) as a"""
write_table_from_query(project_id=project_id, dataset_id=dataset_id, table_id='us_places_category', query=us_places_category, overwrite=True)


us_places_table_creation_query = f"""
SELECT
  *
FROM (
  SELECT
    id,
    geometry,
    names.`primary` AS name,
    categories.`primary` AS category,
    brand.names.`primary` AS brand,
    TO_JSON_STRING(ARRAY(
      SELECT
        DISTINCT x
      FROM
        UNNEST(websites.`list`) AS x)) AS website,
    TO_JSON_STRING(ARRAY(
      SELECT
        DISTINCT x
      FROM
        UNNEST(socials.`list`) AS x)) AS socials,
  IF
    (ARRAY_LENGTH(addresses.`list`)>0, TO_JSON_STRING(addresses.`list`[SAFE_OFFSET(0)].element), NULL) AS address
  FROM
    bigquery-public-data.overture_maps.place
  WHERE
    addresses.`list`[SAFE_OFFSET(0)].element.country = 'US') as a
LEFT JOIN
  kaggle-hackathon-project.geo_intent.us_places_category as b
  USING(category)"""

write_table_from_query(project_id=project_id, dataset_id=dataset_id, table_id='us_places', query=us_places_table_creation_query, overwrite=True)


column_descriptions = {
    "id": "A unique identifier for the place, sourced from Overture Maps.",
    "geometry": "The geographical coordinates (point or polygon) of the place's location.",
    "name": "The primary, common name of the place.",
    "category": "The primary business category of the place (e.g., 'restaurant', 'book_store').",
    "brand": "The brand associated with the place, if applicable (e.g., 'Starbucks').",
    "website": "A JSON string containing a list of official websites for the place.",
    "socials": "A JSON string containing a list of social media profile links for the place.",
    "address": "A JSON string containing the detailed primary address of the place.",
    "number_of_places": "The total count of places within the same primary category across the US.",
    "category_description": "An LLM-generated description of the business category from the perspective of a nearby cafe.",
    "competition": "An LLM-generated boolean indicating if the category is considered direct competition for a cafe.",
    "competition_magnitude": "An LLM-generated score (0-100) of how competitive the category is to a cafe.",
    "opportunity": "An LLM-generated boolean indicating if the category represents a sales opportunity (e.g., foot traffic).",
    "opportunity_magnitude": "An LLM-generated score (0-100) of the level of sales opportunity the category presents."
}

# Build the list of ALTER COLUMN clauses
alter_clauses = []
for column, description in column_descriptions.items():
    # Escape double quotes in description
    escaped_description = description.replace('"', '\\"')
    alter_clauses.append(f'ALTER COLUMN {column} SET OPTIONS(description="{escaped_description}")')

# Join the clauses with a comma
joined_clauses = ",\n".join(alter_clauses)

# Construct the final single ALTER TABLE query
alter_query = f"""
ALTER TABLE `{project_id}.{dataset_id}.us_places`
{joined_clauses}
"""

# Run the single query
print("Setting all column descriptions in a single statement...")
run_query(project_id=project_id, query=alter_query)
print("All column descriptions have been updated.")


