import os
import json
import time
import datetime
from google.cloud import bigquery
import pandas as pd


## ------------------------ VARIABLES ------------------------ 
PROJECT_ID = "gcpbees" # Change this project ID with yours
DATASET_ID = "insurance_risk"

# VERTEX AI could failed to generate the inference text with message “finishReason” : “MAX_TOKENS” on large set of data
# while charging you, so for tests purpose you can samplify the data before the inferences steps
SAMPLIFY = True
SAMPLE_SUFFIX = "_sample"
SAMPLE_SIZE = 100
SET_SAMPLE = ""

# Vertex AI Connections Configurations
VERTEX_AI_CONNECTION_ID = "vertex_ai_connection"
VERTEX_AI_CONNECTION_MODEL = "gemini-2.5-flash"
VERTEX_AI_CONNECTION_MODEL_ENDPOINT = "gemini_model"
VERTEX_AI_CONNECTION_EMBEDDING = "gemini-embedding-001"
VERTEX_AI_CONNECTION_EMBEDDING_ENDPOINT = "embedding_model"
VERTEX_AI_CONNECTION_LOCATION = "us"

TEMPERATURE = 1
MAX_OUTPUT_TOKEN = 3072
TOP_P = 0.9
TOP_K = 1

EMBEDDING_SIZE = 768


# Helper Library 
def execute_bigquery_with_polling(project_id, query, max_results=100, timeout=300):
    """
    Execute BigQuery query with job polling and sample results retrieval.
    
    Parameters:
    - query: SQL query string to execute
    - max_results: Maximum number of sample results to return
    - timeout: Maximum time to wait for job completion in seconds
    
    Returns:
    - Dictionary with sample results and metadata
    """
    
    try:
        # Step 1: Get credentials and create client
        print("Step 1")
        
        client = bigquery.Client(project=project_id)
        
        # Step 2: Submit query job
        print("Step 2")
        job_config = bigquery.QueryJobConfig(
            use_query_cache=True,
            use_legacy_sql=False
        )
        
        query_job = client.query(query, job_config=job_config)
        job_id = query_job.job_id
        location = query_job.location or 'US'
        
        print(f"Job submitted: {job_id}")
        
        # Step 3: Poll until job completes
        print("Step 3")
        start_time = time.time()
        
        while (time.time() - start_time) < timeout:
            # Get current job status
            job = client.get_job(job_id, location=location)
            
            if job.state == 'DONE':
                print(f"Job completed in {time.time() - start_time:.2f}s")
                
                # Check for errors
                if job.errors:
                    raise RuntimeError(f"Query failed: {job.errors[0]['message']}")
                break
            if job.state == 'RUNNING':
                print(">> The job is in progress....")
            
            # Wait before next poll
            time.sleep(10)
        else:
            raise TimeoutError(f"Job timeout after {timeout}s")
        
        # Step 4: Get sample results using getQueryResults
        print("Step 4")
        query_results = client._connection.api_request(
            method='GET',
            path=f'/projects/{project_id}/queries/{job_id}',
            query_params={
                'maxResults': max_results,
                'location': location
            }
        )
        
        # Step 5: Parse results to DataFrame
        print("Step 5")
        df = pd.DataFrame()
        
        if 'rows' in query_results:
            # Extract column names
            columns = [field['name'] for field in query_results['schema']['fields']]
            
            # Extract row data
            data = []
            for row in query_results['rows']:
                row_data = [cell.get('v') for cell in row['f']]
                data.append(row_data)
            
            df = pd.DataFrame(data, columns=columns)
        
        total_rows = int(query_results.get('totalRows', 0))
        bytes_processed = job.total_bytes_processed or 0
        
        return {
            'success': True,
            'dataframe': df,
            'rows_returned': len(df),
            'total_rows': total_rows,
            'job_id': job_id,
            'bytes_processed': bytes_processed,
            'cost_usd': (bytes_processed / 1e12) * 5
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'dataframe': pd.DataFrame()
        }


# Usage example
def execute_query_to_dataframe(project_id, query, max_results=100, timeout=300):
    """
    Run query and get sample of results.
    
    Parameters:
    - query: SQL query string
    - sample_size: Number of sample rows to retrieve
    
    Returns:
    - pandas DataFrame with sample results
    """
    print("execute_query_to_dataframe")
    result = execute_bigquery_with_polling(project_id, query, max_results=max_results, timeout=300)

    print(">> Processing finished at: ", datetime.datetime.now())
    
    if result['success']:
        print(f"Retrieved {result['rows_returned']} of {result['total_rows']} total rows")
        print(f"Query cost: ${result['cost_usd']:.6f}")
        return result['dataframe']
    else:
        print(f"Error: {result['error']}")
        return pd.DataFrame()


HISTORICAL_RISK_QUERY =f"""
-- Create comprehensive historical risk aggregation at county-month level
-- This table serves as the foundation for understanding long-term risk patterns
CREATE OR REPLACE TABLE `{DATASET_ID}.historical_risk_master` AS
WITH storm_aggregation AS (
  -- Aggregate all historical storm events from 1950-2025
  SELECT 
    state_fips_code,
    cz_fips_code AS county_fips,
    cz_name AS county_name,
    state,
    -- Temporal dimensions for seasonality analysis
    EXTRACT(YEAR FROM event_begin_time) AS event_year,
    EXTRACT(MONTH FROM event_begin_time) AS event_month,
    FORMAT_DATE('%Y-%m', DATE(event_begin_time)) AS year_month,
    
    -- Event type and severity metrics
    event_type,
    COUNT(*) AS event_count,
    
    -- Financial impact metrics (handling nulls for $0 damages)
    SUM(IFNULL(damage_property, 0)) AS total_property_damage,
    SUM(IFNULL(damage_crops, 0)) AS total_crop_damage,
    SUM(IFNULL(damage_property, 0) + IFNULL(damage_crops, 0)) AS total_damage,
    AVG(IFNULL(damage_property, 0) + IFNULL(damage_crops, 0)) AS avg_damage_per_event,
    MAX(IFNULL(damage_property, 0) + IFNULL(damage_crops, 0)) AS max_single_event_damage,
    
    -- Human impact metrics for severity assessment
    SUM(IFNULL(deaths_direct, 0) + IFNULL(deaths_indirect, 0)) AS total_deaths,
    SUM(IFNULL(injuries_direct, 0) + IFNULL(injuries_indirect, 0)) AS total_injuries,
    
    -- Event characteristics for pattern analysis
    AVG(CASE WHEN magnitude IS NOT NULL THEN magnitude END) AS avg_magnitude,
    MAX(CASE WHEN magnitude IS NOT NULL THEN magnitude END) AS max_magnitude,
    STRING_AGG(DISTINCT tor_f_scale, ', ' ORDER BY tor_f_scale) AS tornado_scales_observed,
    
    -- Duration metrics for exposure analysis
    AVG(DATETIME_DIFF(event_end_time, event_begin_time, HOUR)) AS avg_event_duration_hours,
    MAX(DATETIME_DIFF(event_end_time, event_begin_time, HOUR)) AS max_event_duration_hours
    
  FROM `bigquery-public-data.noaa_historic_severe_storms.storms_*`
  WHERE _TABLE_SUFFIX BETWEEN '1950' AND '2025'
    AND cz_fips_code IS NOT NULL  -- Ensure geographic identification
  GROUP BY 1, 2, 3, 4, 5, 6, 7, 8
),
-- Add rolling statistics for trend analysis
rolling_metrics AS (
  SELECT 
    *,
    -- 3-year rolling averages for smoothing
    AVG(total_damage) OVER (
      PARTITION BY county_fips, county_name, state
      ORDER BY year_month 
      ROWS BETWEEN 35 PRECEDING AND CURRENT ROW
    ) AS rolling_3yr_avg_damage,
    
    -- Year-over-year change indicators
    LAG(total_damage, 12) OVER (
      PARTITION BY county_fips, county_name, state
      ORDER BY year_month
    ) AS previous_year_damage,
    
    -- Cumulative risk exposure
    SUM(total_damage) OVER (
      PARTITION BY  county_fips, county_name, state
      ORDER BY year_month
    ) AS cumulative_historical_damage
    
  FROM storm_aggregation
)
SELECT 
  *,
  -- Risk trend indicators
  CASE 
    WHEN previous_year_damage > 0 
    THEN ((total_damage - previous_year_damage) / previous_year_damage) * 100
    ELSE NULL 
  END AS year_over_year_change_pct,
  
  -- Seasonal risk flags
  CASE 
    WHEN event_month BETWEEN 4 AND 10 THEN 'HIGH_SEASON'
    ELSE 'LOW_SEASON'
  END AS seasonal_risk_period,
  
  -- Data quality indicator
  CURRENT_TIMESTAMP() AS last_updated_timestamp

FROM rolling_metrics;
"""


# Run Historical Risk Master Table
df = execute_query_to_dataframe(
    project_id=PROJECT_ID,
    query=HISTORICAL_RISK_QUERY, 
    max_results=10
)
print(df.head())
    


PRELIMINARY_EVENTS_REPORTS = f"""
-- Consolidate all preliminary reports into unified format
-- This table captures real-time signals and social media intelligence
CREATE OR REPLACE TABLE `{DATASET_ID}.event_reports_master` AS
WITH standardized_reports AS (
  -- Standardize hail reports
  SELECT 
    'HAIL' AS report_type,
    timestamp,
    EXTRACT(DATE FROM timestamp) AS report_date,
    time AS report_hour_utc,
    size/100.0 AS severity_measure,  -- Convert to inches
    'inches' AS severity_unit,
    location,
    county,
    state,
    latitude,
    longitude,
    comments,
    report_point,
    -- Extract social media indicators
    CASE 
      WHEN UPPER(comments) LIKE '%SOCIAL MEDIA%' THEN TRUE 
      ELSE FALSE 
    END AS from_social_media,
    CASE 
      WHEN UPPER(comments) LIKE '%PHOTO%' THEN TRUE 
      ELSE FALSE 
    END AS has_photo_evidence
  FROM `bigquery-public-data.noaa_preliminary_severe_storms.hail_reports`
  
  UNION ALL
  
  -- Standardize tornado reports
  SELECT 
    'TORNADO' AS report_type,
    timestamp,
    EXTRACT(DATE FROM timestamp) AS report_date,
    time AS report_hour_utc,


    -- Categorize tornado severity by mapping note string values to numeric scale (0-5)
    -- Includes all EF Scale categories from EF0 (weakest) to EF5 (strongest)
    CASE 
      -- Map EF0 string to severity level 0 (Light damage: 65-85 mph)
      WHEN UPPER(f_scale) IN ('EF0', 'EF-0', 'F0') THEN 1
      
      -- Map EF1 string to severity level 1 (Moderate damage: 86-110 mph)
      WHEN UPPER(f_scale) IN ('EF1', 'EF-1', 'F1') THEN 2
      
      -- Map EF2 string to severity level 2 (Considerable damage: 111-135 mph)
      WHEN UPPER(f_scale) IN ('EF2', 'EF-2', 'F2') THEN 3
      
      -- Map EF3 string to severity level 3 (Severe damage: 136-165 mph)
      WHEN UPPER(f_scale) IN ('EF3', 'EF-3', 'F3') THEN 4
      
      -- Map EF4 string to severity level 4 (Devastating damage: 166-200 mph)
      WHEN UPPER(f_scale) IN ('EF4', 'EF-4', 'F4') THEN 5
      
      -- Map EF5 string to severity level 5 (Incredible damage: Over 200 mph)
      WHEN UPPER(f_scale) IN ('EF5', 'EF-5', 'F5') THEN 6
      
      -- Handle unknown/unsurveyed tornadoes (return NULL for unmeasurable events)
      WHEN UPPER(f_scale) IN ('EFU', 'EF-U', 'UNK', 'UNKNOWN') THEN 0
      
      -- Handle NULL values
      WHEN f_scale IS NULL THEN 0
      
      -- Default for unrecognized string values
      ELSE 0
    END AS severity_measure,

    'f_scale' AS severity_unit,
    location,
    county,
    state,
    latitude,
    longitude,
    comments,
    report_point,
    CASE 
      WHEN UPPER(comments) LIKE '%SOCIAL MEDIA%' THEN TRUE 
      ELSE FALSE 
    END AS from_social_media,
    CASE 
      WHEN UPPER(comments) LIKE '%PHOTO%' OR UPPER(comments) LIKE '%VIDEO%' THEN TRUE 
      ELSE FALSE 
    END AS has_photo_evidence
  FROM `bigquery-public-data.noaa_preliminary_severe_storms.tornado_reports`
  WHERE f_scale IS NOT NULL
  
  UNION ALL
  
  -- Standardize wind reports
  SELECT 
    'WIND' AS report_type,
    timestamp,
    EXTRACT(DATE FROM timestamp) AS report_date,
    time AS report_hour_utc,
    IFNULL(0, speed) AS severity_measure,  -- Already in MPH
    'mph' AS severity_unit,
    location,
    county,
    state,
    latitude,
    longitude,
    comments,
    report_point,
    CASE 
      WHEN UPPER(comments) LIKE '%SOCIAL MEDIA%' THEN TRUE 
      ELSE FALSE 
    END AS from_social_media,
    CASE 
      WHEN UPPER(comments) LIKE '%MEASURED%' OR UPPER(comments) LIKE '%ASOS%' THEN TRUE 
      ELSE FALSE 
    END AS has_photo_evidence  -- For wind, this indicates official measurement
  FROM `bigquery-public-data.noaa_preliminary_severe_storms.wind_reports`
),
-- Add temporal clustering to identify storm systems
event_clustering AS (
  SELECT 
    *,
    -- Identify potential storm clusters (events within 6 hours and 50 miles)
    LAG(timestamp) OVER (
      PARTITION BY state, report_date 
      ORDER BY timestamp
    ) AS previous_event_time,
    
    -- Count events in surrounding area
    COUNT(*) OVER (
      PARTITION BY state, report_date, CAST(latitude AS INT64), CAST(longitude AS INT64)
    ) AS local_event_density,
    
    -- Normalized severity for cross-type comparison
    CASE 
      WHEN report_type = 'HAIL' THEN 
        CASE 
          WHEN severity_measure >= 2.0 THEN 'SEVERE'
          WHEN severity_measure >= 1.0 THEN 'MODERATE'
          ELSE 'MINOR'
        END
      WHEN report_type = 'TORNADO' THEN
        CASE 
          WHEN severity_measure >= 3 THEN 'SEVERE'
          WHEN severity_measure >= 1 THEN 'MODERATE'
          ELSE 'MINOR'
        END
      WHEN report_type = 'WIND' THEN
        CASE 
          WHEN severity_measure >= 75 THEN 'SEVERE'
          WHEN severity_measure >= 58 THEN 'MODERATE'
          ELSE 'MINOR'
        END
    END AS severity_category
    
  FROM standardized_reports
)
SELECT 
  *,
  -- Storm system indicator
  CASE 
    WHEN previous_event_time IS NOT NULL 
      AND DATETIME_DIFF(timestamp, previous_event_time, HOUR) <= 6 
    THEN TRUE 
    ELSE FALSE 
  END AS part_of_storm_system,
  
  -- Data freshness indicator
  DATETIME_DIFF(CURRENT_TIMESTAMP(), timestamp, HOUR) AS hours_since_report,
  
  -- Quality score based on evidence
  (CASE WHEN from_social_media THEN 1 ELSE 2 END + 
   CASE WHEN has_photo_evidence THEN 2 ELSE 0 END) AS report_quality_score,
   
  CURRENT_TIMESTAMP() AS last_updated_timestamp
  
FROM event_clustering;
"""


df = execute_query_to_dataframe(
    project_id=PROJECT_ID,
    query=PRELIMINARY_EVENTS_REPORTS, 
    max_results=10
)
print(df.head())


ENRICHED_LOCATION_MASTER = f"""
-- Create comprehensive location profiles combining historical and current signals
-- IMPORTANT: This table aggregates 75 years of NOAA data (1950-2025) at county level
-- All damage figures represent cumulative totals over this period, not annual amounts
-- Classifications are preliminary patterns for investigation, not insurance ratings
CREATE OR REPLACE TABLE `{DATASET_ID}.enriched_location_master` AS
WITH historical_summary AS (
  -- Aggregate historical data at county level (75-year cumulative totals)
  SELECT 
    county_fips,
    county_name,
    state,
    
    -- Overall risk metrics (cumulative since 1950)
    COUNT(DISTINCT year_month) AS months_with_events,
    COUNT(DISTINCT event_type) AS event_type_diversity,
    SUM(event_count) AS total_historical_events,
    SUM(total_damage) AS total_historical_damage,
    -- Correct monthly average: total damage divided by months in dataset
    SAFE_DIVIDE(
      SUM(total_damage), 
      (EXTRACT(YEAR FROM CURRENT_DATE()) - 1950) * 12
    ) AS avg_monthly_damage,
    STDDEV(total_damage) AS damage_volatility,
    MAX(max_single_event_damage) AS worst_case_event,
    
    -- Human impact summary (75-year totals)
    SUM(total_deaths) AS total_historical_deaths,
    SUM(total_injuries) AS total_historical_injuries,
    
    -- Recent trend indicators (last 5 years vs. historical baseline)
    SUM(CASE 
      WHEN event_year >= EXTRACT(YEAR FROM CURRENT_DATE()) - 5 
      THEN total_damage ELSE 0 
    END) AS recent_5yr_damage,
    
    SUM(CASE 
      WHEN event_year >= EXTRACT(YEAR FROM CURRENT_DATE()) - 5 
      THEN event_count ELSE 0 
    END) AS recent_5yr_events,
    
    -- Seasonal patterns
    STRING_AGG(DISTINCT seasonal_risk_period, ', ') AS seasonal_patterns,
    
    -- Most common and severe event types
    ARRAY_AGG(
      STRUCT(event_type, total_damage) 
      ORDER BY total_damage DESC 
      LIMIT 3
    ) AS top_damage_event_types
    
  FROM `{DATASET_ID}.historical_risk_master`
  GROUP BY 1, 2, 3
),
recent_reports_summary AS (
  -- Aggregate recent preliminary reports (real-time monitoring)
  SELECT 
    county,
    state,
    
    -- Recent activity indicators
    COUNT(*) AS preliminary_report_count_30d,
    COUNT(DISTINCT report_type) AS active_threat_types,
    
    -- Severity distributions
    SUM(CASE WHEN severity_category = 'SEVERE' THEN 1 ELSE 0 END) AS severe_reports_30d,
    SUM(CASE WHEN severity_category = 'MODERATE' THEN 1 ELSE 0 END) AS moderate_reports_30d,
    
    -- Evidence quality
    AVG(report_quality_score) AS avg_report_quality,
    SUM(CASE WHEN from_social_media THEN 1 ELSE 0 END) AS social_media_reports,
    
    -- Storm system indicators
    SUM(CASE WHEN part_of_storm_system THEN 1 ELSE 0 END) AS system_events_count,
    MAX(local_event_density) AS max_event_clustering,
    
    -- Most recent activity
    MAX(timestamp) AS last_report_timestamp,
    MIN(hours_since_report) AS hours_since_last_report
    
  FROM `{DATASET_ID}.event_reports_master`
  WHERE hours_since_report <= 720  -- Last 30 days
  GROUP BY 1, 2
),
-- Combine all location intelligence
location_profiles AS (
  SELECT 
    h.county_fips,
    h.county_name,
    h.state,
    
    -- Historical risk profile (75-year cumulative)
    h.total_historical_events,
    h.total_historical_damage,
    h.avg_monthly_damage,  -- Now correctly calculated as historical average
    h.damage_volatility,
    h.worst_case_event,
    h.event_type_diversity,
    h.recent_5yr_damage,
    h.recent_5yr_events,
    
    -- Current activity signals
    IFNULL(r.preliminary_report_count_30d, 0) AS current_activity_level,
    IFNULL(r.severe_reports_30d, 0) AS current_severe_threats,
    IFNULL(r.hours_since_last_report, 999999) AS hours_since_activity,
    
    -- Risk indicators (normalized for meaningful comparison)
    CASE 
      -- Volatility relative to average damage
      WHEN SAFE_DIVIDE(h.damage_volatility, h.avg_monthly_damage) > 100 THEN 'HIGH_VOLATILITY'
      WHEN SAFE_DIVIDE(h.damage_volatility, h.avg_monthly_damage) > 10 THEN 'MODERATE_VOLATILITY'
      ELSE 'LOW_VOLATILITY'
    END AS volatility_category,
    
    -- Trend analysis (comparing recent 5 years to historical average)
    CASE 
      WHEN h.recent_5yr_damage > 
        -- Calculate expected 5-year damage based on historical average
        SAFE_DIVIDE(h.total_historical_damage, EXTRACT(YEAR FROM CURRENT_DATE()) - 1950) * 5 * 2
      THEN 'INCREASING_RISK'  -- Recent damage is 2x+ the historical average
      ELSE 'STABLE_OR_DECREASING'
    END AS risk_trend,
    
    -- Combined risk score components (for AI processing)
    SAFE_DIVIDE(h.total_historical_damage, h.total_historical_events) AS damage_per_event,
    SAFE_DIVIDE(h.recent_5yr_events, 5.0) AS annual_event_frequency,
    
    -- Metadata
    h.top_damage_event_types,
    h.seasonal_patterns,
    CURRENT_TIMESTAMP() AS last_updated_timestamp
    
  FROM historical_summary h
  LEFT JOIN recent_reports_summary r
    ON h.county_name = r.county 
    AND h.state = r.state
)
SELECT 
  *,
  -- PRELIMINARY risk patterns for investigation (not insurance classifications)
  -- Based on 75-year cumulative totals and recent activity patterns
  CASE 
    -- Pattern suggesting need for immediate investigation
    WHEN (total_historical_damage > 500000000  -- $500M cumulative over 75 years
          AND recent_5yr_damage > total_historical_damage * 0.5)  -- 50%+ in last 5 years
      OR current_severe_threats > 10  -- Unusual current activity
      OR (worst_case_event > 100000000 AND recent_5yr_events > 5)  -- High severity + frequency
    THEN 'EXTREME_RISK_ZONE'  -- Flag for detailed actuarial analysis
    
    -- Pattern suggesting elevated attention needed
    WHEN total_historical_damage > 100000000  -- $100M cumulative over 75 years
      OR (current_severe_threats > 3 AND worst_case_event > 10000000)
      OR (recent_5yr_damage > total_historical_damage * 0.3)  -- 30%+ in last 5 years
    THEN 'HIGH_RISK_ZONE'  -- Warrants monitoring and verification
    
    -- Pattern suggesting moderate historical activity
    WHEN total_historical_damage > 10000000  -- $10M cumulative over 75 years
      OR total_historical_events > 50  -- Frequent but lower-severity events
    THEN 'MODERATE_RISK_ZONE'  -- Standard monitoring appropriate
    
    -- Limited historical activity pattern
    ELSE 'LOW_RISK_ZONE'  -- Baseline monitoring
  END AS preliminary_risk_classification
  -- NOTE: These classifications are AI-generated hypotheses for investigation,
  -- not industry-standard insurance risk ratings. Actual risk assessment requires
  -- population normalization, inflation adjustment, and professional actuarial analysis.

FROM location_profiles;
"""


df = execute_query_to_dataframe(
    project_id=PROJECT_ID,
    query=ENRICHED_LOCATION_MASTER, 
    max_results=10
)
print(df.head())


MODEL_REMOTE_CONNEXION = f"""
-- First, create the remote model that all AI functions will use

-- Using latest Gemini Flash for optimal performance and cost
CREATE OR REPLACE MODEL `{DATASET_ID}.{VERTEX_AI_CONNECTION_MODEL_ENDPOINT}`
  REMOTE WITH CONNECTION `projects/{PROJECT_ID}/locations/{VERTEX_AI_CONNECTION_LOCATION}/connections/{VERTEX_AI_CONNECTION_ID}`
  OPTIONS (
    ENDPOINT = '{VERTEX_AI_CONNECTION_MODEL}'
  );

-- Create embedding model for vector operations
CREATE OR REPLACE MODEL `{DATASET_ID}.{VERTEX_AI_CONNECTION_EMBEDDING_ENDPOINT}`
  REMOTE WITH CONNECTION `projects/{PROJECT_ID}/locations/{VERTEX_AI_CONNECTION_LOCATION}/connections/{VERTEX_AI_CONNECTION_ID}`
  OPTIONS (
    ENDPOINT = '{VERTEX_AI_CONNECTION_EMBEDDING}'  -- Latest embedding model
  );
"""


df = execute_query_to_dataframe(
    project_id=PROJECT_ID,
    query=MODEL_REMOTE_CONNEXION, 
    max_results=10
)
print(df.head())


# Prompt use for the 2.1.2 - Prepare Risk Narrative Tables
COUNTY_RISK_EVALUATION_PROMPT = """
# County Territorial Risk Assessment

## Read the Semantic Profile
Extract ALL these values:
- Total events: [number] 
- Total damages: $[amount]
- Worst single event: $[amount]
- Recent 5-year damage: $[amount]
- Risk classification: [ZONE]
- Trend: [DIRECTION]

## Pattern Recognition Rules

### For TOTAL DAMAGE:
- 10 digits ($X,XXX,XXX,XXX) = Billions = EXTREME territory
- 9 digits ($XXX,XXX,XXX) = Hundreds of millions = HIGH territory
- 8 digits or less = MODERATE/LOW territory

### For WORST EVENT vs TOTAL:
- If worst event is also 10 digits (billions) = Requires catastrophe modeling
- If worst event is half of total damage = High reinsurance attachment needed
- If worst event is small fraction of total = Standard reinsurance adequate

### For RECENT TREND:
- If recent damage is millions but total is billions = Territory improving
- If recent damage is large portion of total = Territory deteriorating
- Check Trend field: STABLE_OR_DECREASING vs INCREASING_RISK

### For EVENT FREQUENCY:
- 1000+ events = High frequency territory
- 500-999 = Moderate frequency territory
- Under 500 = Low frequency territory

## Generate County Insurance Output:

### Territory Classification: [STATE the RISK_ZONE from input]

### Key Territory Factors:
1. Loss concentration: [State worst event amount and proportion]
2. Frequency profile: [State total events and classification]
3. Trend analysis: [State if improving/deteriorating with recent vs total]

### Territory Rating Decision:
- EXTREME territory (billions, 1000+ events): Apply 1.35x factor
- If IMPROVING (recent tiny vs total): Reduce to 1.30x factor
- HIGH territory: Apply 1.20-1.25x factor
- MODERATE territory: Apply 1.00-1.10x factor
- LOW territory: Apply 0.85-0.95x factor

### Portfolio Management Strategy:
**Total Insured Value (TIV) Limit**: 
- EXTREME: Cap county at $500M aggregate exposure
- HIGH: Monitor at $1B threshold
- MODERATE/LOW: Standard concentration limits

**Reinsurance Attachment**:
- If worst event >$1B: Low attachment point required
- If worst event $100M-$1B: Standard attachment
- If worst event <$100M: Higher retention acceptable

**Market Appetite**:
- EXTREME + deteriorating: RESTRICT new business
- EXTREME + improving: SELECTIVE underwriting only
- HIGH: MONITOR with enhanced guidelines
- MODERATE/LOW: STANDARD to GROWTH appetite

***
- RISK PATTERN: Notable risk patterns that warrant further investigation
- RISK FACTORS: Potential risk factors not captured in traditional models
- AREAS FOR ANALYSIS: Suggested areas for detailed actuarial analysis
- OBSERVATION: Data quality observations or anomalies to verify
- RISK SCORE: Risk score (0-100)
- COVERAFE LIMITS: Recommended coverage limits
- PREMIUM: Suggested premium adjustment factor
"""


RISK_NARRATIVE_PREP = f"""
-- Step 2.1.2.A: Create risk narrative preparation table
-- This table prepares structured text for LLM processing
CREATE OR REPLACE TABLE `{DATASET_ID}.risk_narrative_prep` AS
SELECT 
  county_fips,
  county_name,
  state,
  current_severe_threats,
  
  -- Construct comprehensive risk narrative
  CONCAT(
    'County: ', county_name, ', ', state, '. ',
    'Historical Profile: ',
    CAST(total_historical_events AS STRING), ' total events recorded, ',
    'causing $', CAST(ROUND(total_historical_damage, 0) AS STRING), ' in damages. ',
    'Average monthly damage: $', CAST(ROUND(avg_monthly_damage, 0) AS STRING), '. ',
    'Worst single event: $', CAST(ROUND(worst_case_event, 0) AS STRING), '. ',
    'Event diversity: ', CAST(event_type_diversity AS STRING), ' different hazard types. ',
    'Recent 5-year trend: ', CAST(recent_5yr_events AS STRING), ' events ',
    'with $', CAST(ROUND(recent_5yr_damage, 0) AS STRING), ' in damages. ',
    'Current status: ', CAST(current_activity_level AS STRING), ' reports in last 30 days, ',
    CAST(current_severe_threats AS STRING), ' severe threats. ',
    'Risk classification: ', preliminary_risk_classification, '. ',
    'Volatility: ', volatility_category, '. ',
    'Trend: ', risk_trend
  ) AS risk_context,
  
  -- Construct risk summary for embedding
  CONCAT(
    preliminary_risk_classification, ' risk zone with ',
    volatility_category, ' and ', risk_trend, '. ',
    'Primary threats from ', 
    ARRAY_TO_STRING(
      ARRAY(SELECT event_type FROM UNNEST(top_damage_event_types) GROUP BY 1), 
      ', '
    )
  ) AS risk_summary,
  
  -- Include key metrics for reference
  total_historical_damage,
  recent_5yr_damage,
  current_activity_level,
  preliminary_risk_classification
  
FROM `{DATASET_ID}.enriched_location_master`;

-- Step 2.1.2.B: Create assessment prompt preparation table
-- Prepare specific prompts for different AI tasks
CREATE OR REPLACE TABLE `{DATASET_ID}.assessment_prompts` AS
SELECT 
    county_fips,
    county_name,
    state,
    risk_context,
    risk_summary,
    current_activity_level,
    current_severe_threats,
    preliminary_risk_classification,
    total_historical_damage,
    
    -- Underwriting assessment prompt
    CONCAT(
      '''{COUNTY_RISK_EVALUATION_PROMPT}''',
      'Context: Analyzing ', CAST( (EXTRACT(YEAR FROM CURRENT_DATE()) - 1950) AS STRING) ,' years of cumulative NOAA severe weather data (1950-', CAST(EXTRACT(YEAR FROM CURRENT_DATE()) AS STRING) ,') at county aggregate level. ',
      'This represents total county exposure, not individual property risk. ',
      'Data includes all reported events and damages for the entire county over this historical period. ',
      'Review this county-level historical risk profile: ',
      risk_context,
      ' Based on this ', CAST( (EXTRACT(YEAR FROM CURRENT_DATE()) - 1950) AS STRING) ,' year aggregate data',
      'Note: Average monthly damage represents historical average over ', CAST( (EXTRACT(YEAR FROM CURRENT_DATE()) - 1950) AS STRING) ,' years, not current monthly exposure.'
    ) AS underwriting_prompt,
    
    -- Monitoring alert prompt  
    CONCAT(
      'Context: Comparing current 30-day activity against ', CAST( (EXTRACT(YEAR FROM CURRENT_DATE()) - 1950) AS STRING) ,' year historical baseline for county-wide events. ',
      'Evaluate recent activity pattern: ',
      'Location has ', CAST(current_activity_level AS STRING), ' reports in last 30 days with ',
      CAST(current_severe_threats AS STRING), ' severe events. ',
      'Historical baseline ( ', CAST( (EXTRACT(YEAR FROM CURRENT_DATE()) - 1950) AS STRING) ,' year county aggregate): ', risk_summary,
      ' Question: Does current activity suggest unusual pattern compared to historical baseline? ',
      'Identify specific factors that merit human analyst attention. ',
      'Distinguish between normal seasonal variation and potentially significant deviation.'
    ) AS monitoring_prompt,
    
    -- Comparative analysis prompt
    CONCAT(
      'Context: Comparing ', CAST( (EXTRACT(YEAR FROM CURRENT_DATE()) - 1950) AS STRING) ,' year cumulative county-level data across ', state, '. ',
      'All figures represent total historical impact, not annual rates. ',
      'Compare this county ', CAST( (EXTRACT(YEAR FROM CURRENT_DATE()) - 1950) AS STRING) ,' year profile to other ', state, ' counties: ',
      risk_summary,
      ' Considering this is cumulative historical data for the entire county: ',
      '1) What distinguishes this county risk pattern from others in the state? ',
      '2) Are there unique hazard combinations or frequencies to investigate? ',
      '3) What additional data would help validate these observations? '
    ) AS comparative_prompt
  
FROM `{DATASET_ID}.risk_narrative_prep`;
"""


df = execute_query_to_dataframe(
    project_id=PROJECT_ID,
    query=RISK_NARRATIVE_PREP, 
    max_results=10
)
print(df.head())


if SAMPLIFY:
    # Set sample prefix and create sample table
    SET_SAMPLE = SAMPLE_SUFFIX = "_sample"


SAMPLIFY_PROMPTS_LIST=f"""
-- Create sample tables --

-- Create a sample of assessment prompts
CREATE OR REPLACE TABLE `{DATASET_ID}.assessment_prompts{SAMPLE_SUFFIX}` AS
SELECT
*
FROM `{DATASET_ID}.assessment_prompts`
ORDER BY county_fips, county_name, state DESC
LIMIT {SAMPLE_SIZE};


-- Create a sample of risk narrative prompts
CREATE OR REPLACE TABLE `{DATASET_ID}.risk_narrative_prep{SAMPLE_SUFFIX}` AS
SELECT
*
FROM `{DATASET_ID}.risk_narrative_prep`
ORDER BY county_fips, county_name, state DESC
LIMIT {SAMPLE_SIZE};
"""


if SAMPLIFY:
    df = execute_query_to_dataframe(
        project_id=PROJECT_ID,
        query=SAMPLIFY_PROMPTS_LIST, 
        max_results=10
    )
    print(df.head())


RISK_NARRATIVE_GEN = f"""
-- Query 2.1.3.A: Generate comprehensive risk narratives
CREATE OR REPLACE TABLE `{DATASET_ID}.risk_narratives_generated` AS
SELECT
  -- Extract the generated text from the nested response structure
  ml_generate_text_result['candidates'][0]['content']['parts'][0]['text'] AS risk_assessment_text,
  
  -- Include all original fields except the raw ML result
  county_fips,
  county_name,
  state,
  risk_context,
  risk_summary,
  underwriting_prompt,
  CURRENT_TIMESTAMP() AS generation_timestamp
  
FROM
  ML.GENERATE_TEXT(
    MODEL `{DATASET_ID}.{VERTEX_AI_CONNECTION_MODEL_ENDPOINT}`,
    (
      -- Subquery must have a column named 'prompt' for ML.GENERATE_TEXT
      SELECT
        county_fips,
        county_name,
        state,
        risk_context,
        risk_summary,
        underwriting_prompt,
        underwriting_prompt AS prompt  -- Required column name for ML.GENERATE_TEXT
      FROM
        `{DATASET_ID}.assessment_prompts{SET_SAMPLE}`

        WHERE county_fips IS NOT NULL 
        AND county_name IS NOT NULL 
        AND state IS NOT NULL
    ),
    STRUCT(
      {TEMPERATURE} AS temperature,
      {MAX_OUTPUT_TOKEN} AS max_output_tokens,
      {TOP_P} AS top_p,
      {TOP_K} AS top_k
    )
  );
"""


df = execute_query_to_dataframe(
    project_id=PROJECT_ID,
    query=RISK_NARRATIVE_GEN, 
    max_results=10,
    timeout=3000
)
print(df.head())


MONITORING_ALERT_GEN = f"""
-- Query 2.1.3.B: Generate monitoring alerts
CREATE OR REPLACE TABLE `{DATASET_ID}.monitoring_alerts_generated` AS
SELECT
  -- Extract the monitoring assessment text
  ml_generate_text_result['candidates'][0]['content']['parts'][0]['text'] AS monitoring_alert_text,
  
  -- Include relevant fields
  county_fips,
  county_name,
  state,
  current_activity_level,
  current_severe_threats,
  monitoring_prompt,
  CURRENT_TIMESTAMP() AS alert_timestamp
  
FROM
  ML.GENERATE_TEXT(
    MODEL `{DATASET_ID}.{VERTEX_AI_CONNECTION_MODEL_ENDPOINT}`,
    (
      SELECT
        county_fips,
        county_name,
        state,
        current_activity_level,
        current_severe_threats,
        monitoring_prompt,
        monitoring_prompt AS prompt  -- Required column name
      FROM
        `{DATASET_ID}.assessment_prompts{SET_SAMPLE}`
      WHERE 

        county_fips IS NOT NULL 
        AND county_name IS NOT NULL 
        AND state IS NOT NULL
        AND current_activity_level > 0  -- Only process active areas
    ),
    STRUCT(
      {TEMPERATURE} AS temperature,  -- Lower temperature for consistent alerts
      {MAX_OUTPUT_TOKEN} AS max_output_tokens,
      {TOP_P} AS top_p
    )
  );
"""


df = execute_query_to_dataframe(
    project_id=PROJECT_ID,
    query=MONITORING_ALERT_GEN, 
    max_results=10,
    timeout=3000
)
print(df.head())


RISK_EMBEDDINGS_GEN = f"""
-- Query 2.3.A: Generate embeddings for risk profiles
CREATE OR REPLACE TABLE `{DATASET_ID}.risk_embeddings` AS
SELECT
  -- The ML result is the embedding array directly
  ml_generate_embedding_result AS risk_embedding_vector,
  
  -- Include all original fields
  county_fips,
  county_name,
  state,
  risk_summary,
  preliminary_risk_classification,
  total_historical_damage,
  CURRENT_TIMESTAMP() AS embedding_timestamp
  
FROM
  ML.GENERATE_EMBEDDING(
    MODEL `{DATASET_ID}.{VERTEX_AI_CONNECTION_EMBEDDING_ENDPOINT}`,
    (
      SELECT
        county_fips,
        county_name,
        state,
        risk_summary,
        preliminary_risk_classification,
        total_historical_damage,
        risk_summary AS content  -- Required column name for ML.GENERATE_EMBEDDING
      FROM
        `{DATASET_ID}.risk_narrative_prep{SET_SAMPLE}`
    ),
    STRUCT(
      'SEMANTIC_SIMILARITY' AS task_type,
      {EMBEDDING_SIZE} AS output_dimensionality
    )
  );

-- Query 2.3.B: Generate context embeddings with validation
CREATE OR REPLACE TABLE `{DATASET_ID}.context_embeddings` AS
SELECT
  -- Embedding result is ARRAY<FLOAT64>
  ml_generate_embedding_result AS context_embedding_vector,
  
  -- Validate embedding
  CASE 
    WHEN ml_generate_embedding_result IS NULL THEN 'FAILED'
    WHEN ARRAY_LENGTH(ml_generate_embedding_result) != {EMBEDDING_SIZE} THEN 'INVALID_DIMENSION'
    ELSE 'SUCCESS'
  END AS embedding_status,
  
  -- Include metadata
  county_fips,
  county_name,
  state,
  risk_context,
  CURRENT_TIMESTAMP() AS embedding_timestamp
  
FROM
  ML.GENERATE_EMBEDDING(
    MODEL `{DATASET_ID}.{VERTEX_AI_CONNECTION_EMBEDDING_ENDPOINT}`,
    (
      SELECT
        county_fips,
        county_name,
        state,
        risk_context,
        risk_context AS content  -- Required column name
      FROM
        `{DATASET_ID}.risk_narrative_prep{SET_SAMPLE}`
      WHERE LENGTH(risk_context) > 0  -- Ensure non-empty content
    ),
    STRUCT(
      'SEMANTIC_SIMILARITY' AS task_type,
      {EMBEDDING_SIZE} AS output_dimensionality
    )
  );

-- Query 2.3.C: Combined embeddings table
CREATE OR REPLACE TABLE `{DATASET_ID}.risk_embeddings_complete` AS
SELECT
  r.county_fips,
  r.county_name,
  r.state,
  r.risk_summary,
  r.risk_embedding_vector,
  c.context_embedding_vector,
  r.preliminary_risk_classification,
  r.total_historical_damage,
  
  -- Validate both embeddings exist
  CASE
    WHEN r.risk_embedding_vector IS NULL OR c.context_embedding_vector IS NULL THEN FALSE
    WHEN ARRAY_LENGTH(r.risk_embedding_vector) != {EMBEDDING_SIZE} OR ARRAY_LENGTH(c.context_embedding_vector) != {EMBEDDING_SIZE} THEN FALSE
    ELSE TRUE
  END AS has_valid_embeddings,
  
  -- Calculate embedding statistics for quality check
  (SELECT AVG(val) FROM UNNEST(r.risk_embedding_vector) AS val) AS risk_embedding_mean,
  (SELECT STDDEV(val) FROM UNNEST(r.risk_embedding_vector) AS val) AS risk_embedding_stddev,
  
  CURRENT_TIMESTAMP() AS creation_timestamp
  
FROM `{DATASET_ID}.risk_embeddings` r
LEFT JOIN `{DATASET_ID}.context_embeddings` c
  ON r.county_fips = c.county_fips AND r.county_name = c.county_name AND r.state = c.state;
"""


df = execute_query_to_dataframe(
    project_id=PROJECT_ID,
    query=RISK_EMBEDDINGS_GEN, 
    max_results=10,
    timeout=3000
)
print(df.head())


# Example of county
COUNTY_FIPS = 0
COUNTY_NAME = "CAZALL"
COUNTY_STATE = "Ca"


VECTOR_SIMILARITY_SEARCH_EXAMPLE = f"""
-- VECTOR_SEARCH usage
CREATE OR REPLACE TABLE `{DATASET_ID}.vector_search_results` AS
WITH query_vectors AS (
  -- Define the counties we want to find similarities for
  SELECT 
    county_fips AS query_county_fips,
    county_name AS query_county_name,
    state AS query_state,
    risk_embedding_vector AS query_vector
  FROM `{DATASET_ID}.risk_embeddings_complete`
  WHERE county_fips = '{COUNTY_FIPS}' 
    AND LOWER(county_name) = 'LOWER({COUNTY_NAME})'
    AND LOWER(state) = 'LOWER({COUNTY_STATE})'
    AND has_valid_embeddings = TRUE
)
SELECT
  qv.query_county_fips,
  qv.query_county_name,
  qv.query_state,
  -- Access fields through the base STRUCT
  search_result.base.county_fips AS similar_county_fips,
  search_result.base.county_name AS similar_county_name,
  search_result.base.state AS similar_county_state,
  search_result.distance AS similarity_distance,
  1 - search_result.distance AS similarity_score,
  search_result.base.preliminary_risk_classification AS similar_risk_class,
  search_result.base.total_historical_damage AS similar_damage_total,
  search_result.base.risk_summary AS similar_risk_summary
FROM
  query_vectors qv,
  VECTOR_SEARCH(
    -- Base table with columns to return
    (SELECT 
      county_fips,
      county_name,
      state,
      risk_embedding_vector,
      preliminary_risk_classification,
      total_historical_damage,
      risk_summary
    FROM `{DATASET_ID}.risk_embeddings_complete`
    WHERE has_valid_embeddings = TRUE),
    'risk_embedding_vector',  -- Column to search
    TABLE query_vectors,  -- Query table
    'query_vector',  -- Query vector column
    top_k => 10,
    distance_type => 'COSINE'
  ) AS search_result  -- Alias the VECTOR_SEARCH result
-- Exclude self-matches
WHERE search_result.base.county_fips != qv.query_county_fips
AND search_result.base.county_name != qv.query_county_name
AND search_result.base.state != qv.query_state
ORDER BY qv.query_county_fips, search_result.distance ASC;
"""


df = execute_query_to_dataframe(
    project_id=PROJECT_ID,
    query=VECTOR_SIMILARITY_SEARCH_EXAMPLE, 
    max_results=10,
    timeout=3000
)
print(df.head())





# Prompt use for the 2.1.5 - Generate Dynamic Risk Classifications
COUNTY_RISK_CLASSIFICATION_PROMPT = """
# County Risk Classification

## Input Structure
You will receive: "Analyze county with these characteristics: Total damage: $[AMOUNT], Event frequency: [NUMBER], Recent activity: [NUMBER] reports, Volatility: [CATEGORY]."

## Extract These Values
- Total damage: Look for dollar amount after "Total damage: $"
- Event frequency: Number after "Event frequency:"
- Recent activity: Number before "reports"
- Volatility: Category after "Volatility:" (HIGH_VOLATILITY/MODERATE_VOLATILITY/LOW_VOLATILITY)

## Pattern Recognition

### DAMAGE PATTERNS
- 10+ digits (billions): EXTREME damage level
- 9 digits (hundreds of millions): HIGH damage level  
- 8 digits (tens of millions): MEDIUM damage level
- 7 digits or less: LOW damage level

### FREQUENCY PATTERNS
- 1000+ events: EXTREME frequency
- 500-999: HIGH frequency
- 100-499: MEDIUM frequency
- <100: LOW frequency

### ACTIVITY PATTERNS
- 20+ reports: Very active
- 10-19 reports: Active
- 1-9 reports: Quiet
- 0 reports: Dormant

## Generate Required Output

### risk_score (0-100):
Base score from damage:
- Billions = 85
- Hundreds of millions = 65
- Tens of millions = 45
- Millions or less = 25

Adjustments:
- Add 10 if frequency >1000
- Add 10 if HIGH_VOLATILITY
- Add 5 if recent activity >10
- Subtract 5 if 0 recent activity

### risk_tier:
- EXTREME: risk_score 80-100
- HIGH: risk_score 60-79
- MEDIUM: risk_score 40-59
- LOW: risk_score 0-39

### confidence_level (0.0-1.0):
- 0.9: Damage and frequency both EXTREME or both LOW
- 0.7: Damage and frequency one level apart
- 0.5: Conflicting signals (EXTREME damage + LOW frequency)

### primary_hazard:
Match patterns:
- 1000+ events + any damage: "HAIL"
- <100 events + billion damage: "HURRICANE"
- 500+ events + HIGH_VOLATILITY: "SEVERE_STORM"
- Moderate frequency + consistent damage: "FLOOD"
- Mixed indicators: "MIXED_PERILS"

### premium_factor (0.5-5.0):
Based on risk_tier:
- EXTREME: 3.0-5.0
- HIGH: 2.0-3.0
- MEDIUM: 1.2-2.0
- LOW: 0.5-1.2

Volatility adjustment:
- HIGH_VOLATILITY: Use upper half of range
- LOW_VOLATILITY: Use lower half of range

## Output Format
risk_score: [number]
risk_tier: [LOW/MEDIUM/HIGH/EXTREME]
confidence_level: [0.0-1.0]
primary_hazard: [hazard type]
premium_factor: [0.5-5.0]
"""


DYNAMIC_RISK_CLASSIFICATIONS_PROMPT =f"""
-- Step 2.4.A: Create classification prompt table with required 'prompt' column
CREATE OR REPLACE TABLE `{DATASET_ID}.classification_prompts` AS
SELECT 
  county_fips,
  county_name,
  state,
  total_historical_damage,
  total_historical_events,
  current_activity_level,
  volatility_category,
  -- AI.GENERATE_TABLE requires a column named 'prompt'
  CONCAT(
    '''{COUNTY_RISK_CLASSIFICATION_PROMPT}''',
    'Analyze this county insurance risk profile:',
    '- Total historical damage: $', CAST(ROUND(total_historical_damage, 0) AS STRING),
    '- Total events recorded: ', CAST(total_historical_events AS STRING),
    '- Recent activity level: ', CAST(current_activity_level AS STRING), ' reports in last 30 days',
    '- Risk volatility: ', volatility_category
  ) AS prompt
FROM `{DATASET_ID}.enriched_location_master`;"""


df = execute_query_to_dataframe(
    project_id=PROJECT_ID,
    query=DYNAMIC_RISK_CLASSIFICATIONS_PROMPT, 
    max_results=10,
    timeout=3000
)
print(df.head())


if SAMPLIFY:
    # Set sample prefix and create sample table
    SET_SAMPLE = SAMPLE_SUFFIX = "_sample"


SAMPLIFY_RISK_PROMPTS=f"""
-- Create sample tables --

-- Create a sample of assessment prompts
CREATE OR REPLACE TABLE `{DATASET_ID}.classification_prompts{SAMPLE_SUFFIX}` AS
SELECT
*
FROM `{DATASET_ID}.classification_prompts`
ORDER BY county_fips, county_name, state DESC
LIMIT {SAMPLE_SIZE};
"""


if SAMPLIFY:
    
    df = execute_query_to_dataframe(
        project_id=PROJECT_ID,
        query=SAMPLIFY_RISK_PROMPTS, 
        max_results=10
    )
    print(df.head())


DYNAMIC_RISK_CLASSIFICATIONS_GEN =f"""
-- Step 2.4: Generate structured risk classifications using AI.GENERATE_TABLE
CREATE OR REPLACE TABLE `{DATASET_ID}.ai_risk_classifications` AS
SELECT 
  -- Original columns from the input table are passed through
  county_fips,
  county_name,
  state,
  total_historical_damage,
  
  -- Generated columns based on output_schema
  risk_score,
  risk_tier,
  confidence_level,
  primary_hazard,
  premium_factor,
  
  -- Add metadata
  CURRENT_TIMESTAMP() AS classification_timestamp
  
FROM
  AI.GENERATE_TABLE(
    MODEL `{DATASET_ID}.{VERTEX_AI_CONNECTION_MODEL_ENDPOINT}`,
    (
      -- Select the required prompt column and other fields to pass through
      SELECT 
        county_fips,
        county_name,
        state,
        total_historical_damage,
        prompt  -- Required column name for AI.GENERATE_TABLE
      FROM 
        `{DATASET_ID}.classification_prompts{SAMPLE_SUFFIX}`
        WHERE county_fips IS NOT NULL 
        AND county_name IS NOT NULL 
        AND state IS NOT NULL
    ),
    STRUCT(
      -- Define the schema as a string literal
      "risk_score INT64, risk_tier STRING, confidence_level FLOAT64, primary_hazard STRING, premium_factor FLOAT64" AS output_schema,
      {MAX_OUTPUT_TOKEN} AS max_output_tokens,
      {TEMPERATURE} AS temperature,
      {TOP_P} AS top_p
    )
  );
"""


df = execute_query_to_dataframe(
    project_id=PROJECT_ID,
    query=DYNAMIC_RISK_CLASSIFICATIONS_GEN, 
    max_results=10,
    timeout=3000
)
print(df.head())


# Prompt use for the 2.1.6 - Generate Alert Triggers
COUNTY_ALERT_CLASSIFICATION_PROMPT="""
# County Alert Evaluation - Pattern Matching

## Input Recognition

### Pattern A: Activity Anomaly
"County [NAME] has [CURRENT] recent reports with [THREATS] severe threats. Historical average: [HISTORICAL] events/year"

### Pattern B: Customer Notification
"With [THREATS] severe threats and historical worst case of $[WORST_CASE]"

### Pattern C: Risk Trend
"County has $[RECENT_5YR] damage in last 5 years compared to $[PRIOR_YEARS] in prior years"

## Digit Pattern Rules

### DOLLAR AMOUNTS:
- 10+ digits = BILLIONS (e.g., $1,234,567,890)
- 9 digits = HUNDREDS OF MILLIONS (e.g., $123,456,789)
- 8 digits = TENS OF MILLIONS (e.g., $12,345,678)
- 7 digits = MILLIONS (e.g., $1,234,567)
- 6 digits or less = UNDER MILLION

### EVENT COUNTS:
- 3+ digits = HUNDREDS OR MORE (100+)
- 2 digits = TENS (10-99)
- 1 digit = SINGLE DIGITS (0-9)

## Alert Decision Patterns

### ABNORMAL ACTIVITY = TRUE when:
- Current reports has MORE digits than historical average
- Current is 2+ digits (10+) AND historical is 1 digit (under 10)
- Severe threats is 5+ regardless of history
- Current is 50+ with any severe threats

### CUSTOMER ALERT = TRUE when:
- Worst case is 10+ digits (BILLIONS) with ANY threats
- Worst case is 9 digits (HUNDREDS OF MILLIONS) with 3+ threats
- Worst case is 8 digits (TENS OF MILLIONS) with 5+ threats
- Severe threats is 10+ regardless of worst case

### INCREASING RISK = TRUE when:
- Recent 5yr has SAME number of digits as prior years
- Recent 5yr has MORE digits than prior years
- Recent 5yr is 9+ digits (HUNDREDS OF MILLIONS+)
- Prior years is 10+ digits BUT recent is also 9+ digits

## FALSE Conditions

### ABNORMAL ACTIVITY = FALSE when:
- Current has FEWER digits than historical
- Both are single digits
- Current is 0 reports

### CUSTOMER ALERT = FALSE when:
- Severe threats is 0
- Worst case is 7 digits or less (MILLIONS) with under 3 threats

### INCREASING RISK = FALSE when:
- Recent 5yr has 2+ FEWER digits than prior (e.g., millions vs billions)
- Recent 5yr is 7 digits or less (MILLIONS or less)
- Prior years has 10+ digits and recent has 8 or fewer digits

## Examples

**Activity Check:**
- "50 reports, historical 15/year" → 50 is 2 digits, 15 is 2 digits → Check if 50 >3x pattern → TRUE
- "3 reports, historical 100/year" → 3 is 1 digit, 100 is 3 digits → FALSE

**Customer Alert:**
- "5 threats, worst $2,000,000,000" → 10 digits + threats → TRUE
- "2 threats, worst $50,000,000" → 8 digits + under 3 threats → FALSE

**Risk Trend:**
- "$800,000,000 recent vs $2,000,000,000 prior" → 9 digits vs 10 digits → TRUE (both high)
- "$5,000,000 recent vs $3,000,000,000 prior" → 7 digits vs 10 digits → FALSE (3 digits difference)

## Output
Answer only: TRUE or FALSE
"""


ALERT_CONDITIONS_PROMPTS =f"""
-- Step 2.1.6.A: Create intermediate alert conditions table with prompts
CREATE OR REPLACE TABLE `{DATASET_ID}.alert_conditions` AS
SELECT 
  county_fips,
  county_name,
  state,
  current_activity_level,
  current_severe_threats,
  annual_event_frequency,
  worst_case_event,
  recent_5yr_damage,
  total_historical_damage,
  
  -- Construct alert evaluation prompts for debugging visibility
  CONCAT(
    '''{COUNTY_ALERT_CLASSIFICATION_PROMPT}''', 
    'County ', county_name, ' has ',
    CAST(current_activity_level AS STRING), ' recent reports with ',
    CAST(current_severe_threats AS STRING), ' severe threats. ',
    'Historical average: ', CAST(ROUND(annual_event_frequency, 1) AS STRING), ' events/year. ',
    'Is this an abnormal situation requiring immediate attention? Answer only TRUE or FALSE.'
  ) AS abnormal_activity_prompt,
  
  CONCAT(
    '''{COUNTY_ALERT_CLASSIFICATION_PROMPT}''',
    'With ', CAST(current_severe_threats AS STRING), ' severe threats reported ',
    'and historical worst case of $', CAST(ROUND(worst_case_event, 0) AS STRING), ', ',
    'should customers be notified of elevated risk? Answer only TRUE or FALSE.'
  ) AS customer_alert_prompt,
  
  CONCAT(
    '''{COUNTY_ALERT_CLASSIFICATION_PROMPT}''',  
    'County has $', CAST(ROUND(recent_5yr_damage, 0) AS STRING), ' damage in last 5 years ',
    'compared to $', CAST(ROUND(total_historical_damage - recent_5yr_damage, 0) AS STRING), ' in prior years. ',
    'Is the risk trend significantly increasing? Answer only TRUE or FALSE.'
  ) AS increasing_risk_prompt,
  
  -- Pre-calculate alert priority for reference
  CASE 
    WHEN current_severe_threats >= 5 THEN 'CRITICAL'
    WHEN current_severe_threats >= 2 THEN 'HIGH'
    WHEN current_activity_level >= 10 THEN 'MEDIUM'
    ELSE 'LOW'
  END AS suggested_priority,
  
  CURRENT_TIMESTAMP() AS prompt_creation_timestamp
  
FROM `{DATASET_ID}.enriched_location_master`
WHERE current_activity_level > 0;  -- Only evaluate counties with recent activity
"""


df = execute_query_to_dataframe(
    project_id=PROJECT_ID,
    query=ALERT_CONDITIONS_PROMPTS, 
    max_results=10,
    timeout=3000
)
print(df.head())


ALERT_TRIGGETS_GEN = f"""
-- Step 2.1.6.B: Generate alert triggers using AI.GENERATE_BOOL on prepared prompts
CREATE OR REPLACE TABLE `{DATASET_ID}.alert_triggers` AS
SELECT
  county_fips,
  county_name,
  state,
  current_activity_level,
  current_severe_threats,
  
  -- Generate boolean alert for abnormal activity
  AI.GENERATE_BOOL(
    abnormal_activity_prompt,
    connection_id => '{PROJECT_ID}.{VERTEX_AI_CONNECTION_LOCATION}.{VERTEX_AI_CONNECTION_ID}',
    endpoint => '{VERTEX_AI_CONNECTION_MODEL}'
  ).result AS is_abnormal_activity,
  
  -- Generate boolean alert for customer notification
  AI.GENERATE_BOOL(
    customer_alert_prompt,
    connection_id => '{PROJECT_ID}.{VERTEX_AI_CONNECTION_LOCATION}.{VERTEX_AI_CONNECTION_ID}',
    endpoint => '{VERTEX_AI_CONNECTION_MODEL}'
  ).result AS should_alert_customers,
  
  -- Generate boolean alert for increasing risk trend
  AI.GENERATE_BOOL(
    increasing_risk_prompt,
    connection_id => '{PROJECT_ID}.{VERTEX_AI_CONNECTION_LOCATION}.{VERTEX_AI_CONNECTION_ID}',
    endpoint => '{VERTEX_AI_CONNECTION_MODEL}'
  ).result AS has_increasing_risk,
  
  -- Include the suggested priority from conditions table
  suggested_priority AS alert_priority,
  
  -- Include context for alert actions
  CASE
    WHEN current_severe_threats > 0 THEN 
      CONCAT('Active severe threats detected: ', CAST(current_severe_threats AS STRING))
    WHEN current_activity_level > annual_event_frequency * 2 THEN
      'Activity level significantly above historical average'
    ELSE 'Monitoring for changes'
  END AS alert_context,
  
  -- Include original prompts for audit trail
  abnormal_activity_prompt AS abnormal_prompt_used,
  customer_alert_prompt AS customer_prompt_used,
  increasing_risk_prompt AS risk_trend_prompt_used,
  
  CURRENT_TIMESTAMP() AS evaluation_timestamp
  
FROM `{DATASET_ID}.alert_conditions`;
"""


df = execute_query_to_dataframe(
    project_id=PROJECT_ID,
    query=ALERT_TRIGGETS_GEN, 
    max_results=10,
    timeout=3000
)
print(df.head())


PREDICTION_SYNTHESIS = f"""
-- Synthesize all AI outputs into unified predictions
CREATE OR REPLACE TABLE `{DATASET_ID}.prediction_synthesis` AS
SELECT
  l.county_fips,
  l.county_name,
  l.state,
  
  -- Combine all AI-generated insights
    n.risk_assessment_text AS risk_narrative,
    c.risk_score,
    c.risk_tier,
    c.confidence_level,
    c.primary_hazard,
    c.premium_factor,
    IFNULL(a.is_abnormal_activity, FALSE) AS has_abnormal_activity,
    IFNULL(a.should_alert_customers, FALSE) AS customer_alert_needed,
    --IFNULL(m.monitoring_alert_text, 'No current threats') AS monitoring_status
  
  -- Include source metrics for validation
  l.total_historical_damage,
  l.recent_5yr_damage,
  l.current_activity_level,
  
  CURRENT_TIMESTAMP() AS synthesis_timestamp
  
FROM `{DATASET_ID}.enriched_location_master` l
LEFT JOIN `{DATASET_ID}.risk_narratives_generated` n
  ON l.county_fips = n.county_fips AND l.county_name = n.county_name AND l.state = n.state
LEFT JOIN `{DATASET_ID}.ai_risk_classifications` c
  ON l.county_fips = c.county_fips  AND l.county_name = c.county_name AND l.state = c.state
LEFT JOIN `{DATASET_ID}.alert_triggers` a
  ON l.county_fips = a.county_fips AND l.county_name = a.county_name AND l.state = a.state
LEFT JOIN `{DATASET_ID}.monitoring_alerts_generated` m
  ON l.county_fips = m.county_fips AND l.county_name = m.county_name AND l.state = m.state;
"""

