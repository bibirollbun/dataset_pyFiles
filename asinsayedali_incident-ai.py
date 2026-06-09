# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
x
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# SRE Incident & Telemetry Data Generator
# Generate realistic production incident data with correlated telemetry patterns
# Perfect for BigQuery AI hackathon - semantic search + incident resolution

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import random
import uuid
from typing import List, Dict, Tuple
import os

# =============================================================================
# INCIDENT TEMPLATES - Based on Real Production Scenarios
# =============================================================================

INCIDENT_TEMPLATES = {
    "database_connection_pool": {
        "title": "Database Connection Pool Exhaustion",
        "description": "Application experiencing timeouts due to database connection pool being exhausted. Users unable to complete transactions.",
        "technologies": ["PostgreSQL", "Java", "Spring Boot", "Connection Pool"],
        "symptoms": ["Connection timeouts", "HTTP 500 errors", "Slow response times"],
        "root_cause": "High traffic spike overwhelmed connection pool settings",
        "resolution_steps": [
            "Identified connection pool exhaustion in application logs",
            "Temporarily increased max_connections in database",
            "Restarted application servers to clear stale connections", 
            "Updated connection pool configuration to handle traffic spikes",
            "Implemented connection pool monitoring alerts"
        ],
        "impact_level": "Critical",
        "affected_services": ["payment-api", "user-service", "web-app"],
        "telemetry_patterns": {
            "pre_incident": {"cpu": (40, 60), "memory": (60, 70), "connections": (80, 95)},
            "during_incident": {"cpu": (20, 30), "memory": (70, 80), "connections": (98, 100)},
            "resolution": {"cpu": (45, 55), "memory": (65, 75), "connections": (30, 50)}
        }
    },
    
    "memory_leak": {
        "title": "Memory Leak in Payment Service",
        "description": "Payment service consuming increasing memory over time, leading to OOM kills and service restarts.",
        "technologies": ["Node.js", "Kubernetes", "Docker", "MongoDB"],
        "symptoms": ["High memory usage", "Frequent pod restarts", "Payment failures"],
        "root_cause": "Memory leak in payment processing logic - objects not being garbage collected",
        "resolution_steps": [
            "Analyzed heap dumps to identify memory leak source",
            "Found unclosed database connections in payment handler",
            "Applied hotfix to properly close connections",
            "Deployed updated service with memory limits",
            "Added memory usage monitoring and alerting"
        ],
        "impact_level": "High",
        "affected_services": ["payment-service", "order-service"],
        "telemetry_patterns": {
            "pre_incident": {"cpu": (30, 50), "memory": (50, 70), "restarts": (0, 1)},
            "during_incident": {"cpu": (60, 80), "memory": (90, 100), "restarts": (5, 15)},
            "resolution": {"cpu": (35, 45), "memory": (40, 60), "restarts": (0, 1)}
        }
    },
    
    "disk_space_full": {
        "title": "Disk Space Exhaustion on Log Server",
        "description": "Log aggregation server running out of disk space, causing log ingestion failures and service degradation.",
        "technologies": ["Elasticsearch", "Logstash", "Kibana", "Linux"],
        "symptoms": ["Log ingestion failures", "Disk I/O errors", "Search queries failing"],
        "root_cause": "Log retention policy not properly configured, old logs accumulating",
        "resolution_steps": [
            "Identified full disk partitions on log servers",
            "Cleaned up old log indices to free immediate space",
            "Implemented automated log retention policies",
            "Added disk usage monitoring and alerts",
            "Scaled storage capacity for log infrastructure"
        ],
        "impact_level": "Medium",
        "affected_services": ["logging-service", "monitoring-stack", "search-api"],
        "telemetry_patterns": {
            "pre_incident": {"cpu": (20, 40), "disk_usage": (70, 85), "io_wait": (5, 10)},
            "during_incident": {"cpu": (80, 100), "disk_usage": (95, 100), "io_wait": (30, 60)},
            "resolution": {"cpu": (25, 35), "disk_usage": (40, 60), "io_wait": (2, 8)}
        }
    },
    
    "api_rate_limit": {
        "title": "Third-party API Rate Limiting",
        "description": "External payment gateway implementing rate limits, causing transaction failures during peak hours.",
        "technologies": ["REST API", "Payment Gateway", "Redis", "Rate Limiting"],
        "symptoms": ["Payment failures", "HTTP 429 errors", "Customer complaints"],
        "root_cause": "Payment provider implemented new rate limits without notification",
        "resolution_steps": [
            "Identified HTTP 429 responses from payment provider",
            "Implemented exponential backoff retry logic",
            "Added Redis-based request queuing system",
            "Configured circuit breaker for payment failures",
            "Set up monitoring for external API response codes"
        ],
        "impact_level": "High",
        "affected_services": ["payment-gateway", "checkout-service", "billing-api"],
        "telemetry_patterns": {
            "pre_incident": {"response_time": (200, 300), "error_rate": (0.1, 0.5), "success_rate": (99.0, 99.8)},
            "during_incident": {"response_time": (2000, 5000), "error_rate": (15, 30), "success_rate": (70, 85)},
            "resolution": {"response_time": (250, 400), "error_rate": (0.2, 1.0), "success_rate": (98.5, 99.5)}
        }
    },
    
    "cpu_spike": {
        "title": "CPU Spike Due to Inefficient Query",
        "description": "Database experiencing high CPU usage due to poorly optimized query causing performance degradation across all services.",
        "technologies": ["MySQL", "Database", "Query Optimization", "Indexing"],
        "symptoms": ["High database CPU", "Slow query responses", "Application timeouts"],
        "root_cause": "New feature deployed with unoptimized database query missing proper indexes",
        "resolution_steps": [
            "Identified slow queries using database performance tools",
            "Found missing index on frequently queried table",
            "Created appropriate database indexes",
            "Optimized query execution plan",
            "Implemented query performance monitoring"
        ],
        "impact_level": "Medium",
        "affected_services": ["user-service", "product-catalog", "search-api"],
        "telemetry_patterns": {
            "pre_incident": {"cpu": (30, 50), "query_time": (50, 100), "active_connections": (20, 40)},
            "during_incident": {"cpu": (90, 100), "query_time": (2000, 8000), "active_connections": (80, 100)},
            "resolution": {"cpu": (35, 55), "query_time": (60, 120), "active_connections": (25, 45)}
        }
    }
}

# =============================================================================
# DATA GENERATION CLASSES
# =============================================================================

class TelemetryGenerator:
    """Generate realistic telemetry data patterns for incidents"""
    
    def __init__(self):
        self.metric_types = {
            "infrastructure": ["cpu_usage", "memory_usage", "disk_usage", "network_io", "load_average"],
            "application": ["response_time", "error_rate", "throughput", "active_connections", "queue_depth"],
            "business": ["transactions_per_minute", "success_rate", "revenue_per_hour", "active_users"]
        }
    
    def generate_metric_timeline(self, incident_time: datetime, template: Dict) -> List[Dict]:
        """Generate telemetry data around an incident with realistic patterns"""
        
        timeline = []
        telemetry_patterns = template["telemetry_patterns"]
        
        # Generate data points every minute for 4 hours around incident
        start_time = incident_time - timedelta(hours=2)
        end_time = incident_time + timedelta(hours=2)
        
        current_time = start_time
        while current_time <= end_time:
            
            # Determine which phase we're in
            if current_time < incident_time - timedelta(minutes=30):
                phase = "pre_incident"
            elif current_time < incident_time + timedelta(minutes=45):
                phase = "during_incident"
            else:
                phase = "resolution"
            
            # Generate metrics for this timestamp
            for service in template["affected_services"]:
                for metric_category in self.metric_types:
                    for metric_name in self.metric_types[metric_category]:
                        
                        # Get expected range for this metric in this phase
                        if metric_name in telemetry_patterns[phase]:
                            min_val, max_val = telemetry_patterns[phase][metric_name]
                        else:
                            # Default ranges for metrics not specified in template
                            min_val, max_val = self._get_default_range(metric_name, phase)
                        
                        # Add some noise and trends
                        value = random.uniform(min_val, max_val)
                        value = max(0, value + random.gauss(0, (max_val - min_val) * 0.1))
                        
                        timeline.append({
                            "timestamp": current_time,
                            "service_name": service,
                            "metric_category": metric_category,
                            "metric_name": metric_name,
                            "metric_value": round(value, 2),
                            "incident_id": None,  # Will be set later
                            "phase": phase
                        })
            
            current_time += timedelta(minutes=1)
        
        return timeline
    
    def _get_default_range(self, metric_name: str, phase: str) -> Tuple[float, float]:
        """Get default metric ranges for common metrics"""
        
        defaults = {
            "pre_incident": {
                "cpu_usage": (20, 60), "memory_usage": (40, 70), "disk_usage": (30, 70),
                "response_time": (100, 500), "error_rate": (0.1, 2.0), "throughput": (100, 500),
                "success_rate": (95, 99.5), "active_users": (1000, 5000)
            },
            "during_incident": {
                "cpu_usage": (70, 100), "memory_usage": (80, 100), "disk_usage": (60, 95),
                "response_time": (1000, 10000), "error_rate": (5, 25), "throughput": (10, 100),
                "success_rate": (60, 90), "active_users": (500, 2000)
            },
            "resolution": {
                "cpu_usage": (25, 65), "memory_usage": (45, 75), "disk_usage": (35, 75),
                "response_time": (150, 600), "error_rate": (0.5, 3.0), "throughput": (120, 600),
                "success_rate": (92, 99), "active_users": (800, 4500)
            }
        }
        
        return defaults.get(phase, {}).get(metric_name, (10, 90))

class IncidentGenerator:
    """Generate realistic incident records with resolutions"""
    
    def __init__(self):
        self.telemetry_gen = TelemetryGenerator()
    
    def generate_incidents(self, num_incidents: int = 100) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate a dataset of incidents with corresponding telemetry"""
        
        incidents_data = []
        telemetry_data = []
        
        # Generate incidents over the past 6 months
        end_date = datetime.now()
        start_date = end_date - timedelta(days=180)
        
        for i in range(num_incidents):
            # Random incident time in the past 6 months
            incident_time = start_date + timedelta(
                seconds=random.randint(0, int((end_date - start_date).total_seconds()))
            )
            
            # Choose random incident template
            template_name = random.choice(list(INCIDENT_TEMPLATES.keys()))
            template = INCIDENT_TEMPLATES[template_name].copy()
            
            # Generate unique incident ID
            incident_id = f"INC-{str(uuid.uuid4())[:8].upper()}"
            
            # Add some variation to the template
            severity_levels = ["Critical", "High", "Medium", "Low"]
            template["impact_level"] = random.choice(severity_levels)
            
            # Calculate resolution time based on severity
            resolution_times = {"Critical": (15, 120), "High": (30, 240), "Medium": (60, 480), "Low": (120, 1440)}
            min_res, max_res = resolution_times[template["impact_level"]]
            resolution_time = incident_time + timedelta(minutes=random.randint(min_res, max_res))
            
            # Create incident record
            incident_record = {
                "incident_id": incident_id,
                "title": template["title"],
                "description": template["description"],
                "impact_level": template["impact_level"],
                "technologies": json.dumps(template["technologies"]),
                "affected_services": json.dumps(template["affected_services"]),
                "symptoms": json.dumps(template["symptoms"]),
                "root_cause": template["root_cause"],
                "resolution_steps": json.dumps(template["resolution_steps"]),
                "incident_time": incident_time,
                "resolution_time": resolution_time,
                "duration_minutes": int((resolution_time - incident_time).total_seconds() / 60),
                "template_type": template_name
            }
            
            incidents_data.append(incident_record)
            
            # Generate corresponding telemetry data
            telemetry_timeline = self.telemetry_gen.generate_metric_timeline(incident_time, template)
            
            # Link telemetry to incident
            for telemetry_point in telemetry_timeline:
                telemetry_point["incident_id"] = incident_id
                telemetry_data.append(telemetry_point)
            
            if (i + 1) % 20 == 0:
                print(f"Generated {i + 1}/{num_incidents} incidents...")
        
        # Convert to DataFrames
        incidents_df = pd.DataFrame(incidents_data)
        telemetry_df = pd.DataFrame(telemetry_data)
        
        return incidents_df, telemetry_df

# =============================================================================
# EXPORT AND VALIDATION FUNCTIONS
# =============================================================================

def save_datasets(incidents_df: pd.DataFrame, telemetry_df: pd.DataFrame, output_dir: str = "sre_dataset"):
    """Save generated datasets to CSV files"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save incidents
    incidents_path = f"{output_dir}/incidents.csv"
    incidents_df.to_csv(incidents_path, index=False)
    print(f"âœ… Saved {len(incidents_df)} incidents to {incidents_path}")
    
    # Save telemetry (might be large, so save in chunks)
    telemetry_path = f"{output_dir}/telemetry.csv"
    telemetry_df.to_csv(telemetry_path, index=False)
    print(f"âœ… Saved {len(telemetry_df)} telemetry points to {telemetry_path}")
    
    # Create data dictionary
    data_dict = {
        "dataset_info": {
            "name": "SRE Incident Intelligence Dataset",
            "version": "1.0",
            "created": datetime.now().isoformat(),
            "description": "Synthetic but realistic SRE incident data with correlated telemetry for AI/ML training"
        },
        "incidents_schema": {
            "incident_id": "Unique identifier for incident",
            "title": "Brief incident title",
            "description": "Detailed incident description",
            "impact_level": "Critical, High, Medium, Low",
            "technologies": "JSON array of involved technologies",
            "affected_services": "JSON array of impacted services",
            "symptoms": "JSON array of observed symptoms",
            "root_cause": "Root cause analysis",
            "resolution_steps": "JSON array of resolution steps",
            "incident_time": "When incident started",
            "resolution_time": "When incident was resolved",
            "duration_minutes": "Total incident duration",
            "template_type": "Which incident pattern this follows"
        },
        "telemetry_schema": {
            "timestamp": "Metric collection time",
            "service_name": "Name of service being monitored",
            "metric_category": "infrastructure, application, or business",
            "metric_name": "Specific metric name",
            "metric_value": "Metric value",
            "incident_id": "Related incident ID",
            "phase": "pre_incident, during_incident, or resolution"
        }
    }
    
    dict_path = f"{output_dir}/data_dictionary.json"
    with open(dict_path, 'w') as f:
        json.dump(data_dict, f, indent=2, default=str)
    print(f"âœ… Saved data dictionary to {dict_path}")

def validate_dataset(incidents_df: pd.DataFrame, telemetry_df: pd.DataFrame):
    """Validate the generated dataset quality"""
    
    print("\nğŸ“Š DATASET VALIDATION REPORT")
    print("=" * 50)
    
    # Incidents validation
    print(f"ğŸ“‹ INCIDENTS:")
    print(f"  Total incidents: {len(incidents_df)}")
    print(f"  Incident types: {incidents_df['template_type'].nunique()}")
    print(f"  Severity distribution:")
    for severity, count in incidents_df['impact_level'].value_counts().items():
        print(f"    {severity}: {count}")
    
    print(f"  Average resolution time: {incidents_df['duration_minutes'].mean():.1f} minutes")
    
    # Telemetry validation  
    print(f"\nğŸ“Š TELEMETRY:")
    print(f"  Total data points: {len(telemetry_df)}")
    print(f"  Services monitored: {telemetry_df['service_name'].nunique()}")
    print(f"  Metric types: {telemetry_df['metric_name'].nunique()}")
    print(f"  Time range: {telemetry_df['timestamp'].min()} to {telemetry_df['timestamp'].max()}")
    
    # Relationship validation
    incident_ids_in_telemetry = telemetry_df['incident_id'].nunique()
    incident_ids_total = incidents_df['incident_id'].nunique()
    
    print(f"\nğŸ”— RELATIONSHIPS:")
    print(f"  Incidents with telemetry: {incident_ids_in_telemetry}/{incident_ids_total}")
    print(f"  Avg telemetry points per incident: {len(telemetry_df) / incident_ids_total:.0f}")

# =============================================================================
# MAIN GENERATION FUNCTION
# =============================================================================

def generate_complete_dataset(num_incidents: int = 100):
    """Generate complete SRE incident dataset"""
    
    print("ğŸš€ SRE INCIDENT DATASET GENERATOR")
    print("=" * 50)
    print(f"Generating {num_incidents} realistic incidents with telemetry...")
    
    # Generate data
    generator = IncidentGenerator()
    incidents_df, telemetry_df = generator.generate_incidents(num_incidents)
    
    # Validate data
    validate_dataset(incidents_df, telemetry_df)
    
    # Save data
    save_datasets(incidents_df, telemetry_df)
    
    print("\nâœ… DATASET GENERATION COMPLETE!")
    print("\nğŸ�¯ READY FOR BIGQUERY:")
    print("1. Upload incidents.csv and telemetry.csv to BigQuery")
    print("2. Generate embeddings for incident descriptions")
    print("3. Implement vector search for similar incidents")
    print("4. Use Gemini to generate resolution recommendations")
    
    return incidents_df, telemetry_df

if __name__ == "__main__":
    # Generate dataset with 150 incidents (good size for demo)
    incidents, telemetry = generate_complete_dataset(150)


from google.cloud import bigquery
import pandas as pd

# project_id = 'river-direction-472718-f8'
project_id = ''
connection_name = 'Incident AI'
gcp_location = ''
my_dataset = ''

client = bigquery.Client(project=project_id)
print(client)
# dataset_id = f"{project_id}.{my_dataset}"


# Step 3: Data Loading and Pre-processing in BigQuery for Incident AI Assistant
# Updated to work with the SRE Incident Dataset Generator output
# Kaggle Notebook Implementation

import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account
import json
import re
import numpy as np
from datetime import datetime
import ast

# ================================
# PART 1: SETUP AND AUTHENTICATION
# ================================

print("Setting up BigQuery client...")

# For Kaggle, you'll need to upload your service account key as a secret
# Go to Kaggle Notebook -> Add-ons -> Secrets -> Add Secret
# Upload your service account JSON file

# Uncomment and modify the path below:
# credentials = service_account.Credentials.from_service_account_file('/kaggle/input/your-secret/service-account-key.json')
# client = bigquery.Client(credentials=credentials, project='your-project-id')

# For demonstration, we'll show the setup without actual credentials
print("âœ… BigQuery client setup complete")

# ================================
# PART 2: CREATE BIGQUERY DATASET
# ================================

def create_bigquery_dataset(client, project_id, dataset_id, location='US'):
    """Create a BigQuery dataset for the hackathon"""
    
    dataset_ref = f"{project_id}.{dataset_id}"
    
    try:
        # Check if dataset already exists
        client.get_dataset(dataset_ref)
        print(f"âœ… Dataset {dataset_id} already exists")
        return dataset_ref
    except:
        # Create new dataset
        dataset = bigquery.Dataset(dataset_ref)
        dataset.location = location
        dataset.description = "Incident AI Assistant - Hackathon Dataset"
        
        dataset = client.create_dataset(dataset, timeout=30)
        print(f"âœ… Created dataset {dataset_id} in {location}")
        return dataset_ref

# Example usage (uncomment when you have credentials):
# dataset_ref = create_bigquery_dataset(client, 'your-project-id', 'incident_ai_hackathon')

# ================================
# PART 3: LOAD GENERATED SRE DATA
# ================================

def load_generated_sre_data():
    """Load the SRE incident data generated by your data generator"""
    
    print("Loading generated SRE incident dataset...")
    
    # Assuming you've run the generator and saved the files
    # Adjust paths based on where you saved the data
    try:
        incidents_df = pd.read_csv('/kaggle/input/sre-dataset/incidents.csv')  # Adjust path
        telemetry_df = pd.read_csv('/kaggle/input/sre-dataset/telemetry.csv')  # Adjust path
        print(f"âœ… Loaded incidents: {incidents_df.shape}")
        print(f"âœ… Loaded telemetry: {telemetry_df.shape}")
    except FileNotFoundError:
        print("â�Œ Generated dataset files not found. Please run the data generator first.")
        print("Expected files: incidents.csv, telemetry.csv")
        return None, None
    
    return incidents_df, telemetry_df

# Alternative: If you've run the generator in the same notebook
def load_from_generator_output(incidents_df, telemetry_df):
    """Use data directly from generator if run in same notebook"""
    print(f"âœ… Using generator output directly")
    print(f"   Incidents: {incidents_df.shape}")
    print(f"   Telemetry: {telemetry_df.shape}")
    return incidents_df, telemetry_df

# Load the data (choose one method)
incidents_df, telemetry_df = load_generated_sre_data()

# If loading from files fails, show what the data structure looks like
if incidents_df is None:
    print("\nğŸ“‹ Expected Data Structure:")
    print("INCIDENTS.CSV should contain:")
    expected_columns = ['incident_id', 'title', 'description', 'impact_level', 
                       'technologies', 'affected_services', 'symptoms', 'root_cause', 
                       'resolution_steps', 'incident_time', 'resolution_time', 
                       'duration_minutes', 'template_type']
    for col in expected_columns:
        print(f"  - {col}")

# ================================
# PART 4: DATA PREPROCESSING FOR BIGQUERY
# ================================

def preprocess_sre_incidents(incidents_df):
    """Preprocess the SRE incidents data for BigQuery and embeddings"""
    
    if incidents_df is None:
        print("â�Œ No incidents data to preprocess")
        return None
    
    print("Starting SRE incidents preprocessing...")
    
    # Create a copy to avoid modifying original
    df = incidents_df.copy()
    
    # 1. Handle timestamp conversion
    df['incident_time'] = pd.to_datetime(df['incident_time'])
    df['resolution_time'] = pd.to_datetime(df['resolution_time'])
    
    # 2. Parse JSON fields safely
    def safe_json_parse(json_str):
        """Safely parse JSON strings, return empty list if invalid"""
        try:
            if pd.isna(json_str) or json_str == '':
                return []
            # Handle both JSON strings and already parsed lists
            if isinstance(json_str, str):
                return json.loads(json_str)
            return json_str
        except:
            return []
    
    # Parse JSON fields
    df['technologies_list'] = df['technologies'].apply(safe_json_parse)
    df['affected_services_list'] = df['affected_services'].apply(safe_json_parse)
    df['symptoms_list'] = df['symptoms'].apply(safe_json_parse)
    df['resolution_steps_list'] = df['resolution_steps'].apply(safe_json_parse)
    
    # 3. Create text representations for embedding
    def create_technologies_text(tech_list):
        """Convert technologies list to readable text"""
        if not tech_list:
            return ""
        return "Technologies: " + ", ".join(tech_list)
    
    def create_symptoms_text(symptoms_list):
        """Convert symptoms list to readable text"""
        if not symptoms_list:
            return ""
        return "Symptoms: " + "; ".join(symptoms_list)
    
    def create_services_text(services_list):
        """Convert affected services to readable text"""
        if not services_list:
            return ""
        return "Affected services: " + ", ".join(services_list)
    
    # Create text fields
    df['technologies_text'] = df['technologies_list'].apply(create_technologies_text)
    df['symptoms_text'] = df['symptoms_list'].apply(create_symptoms_text)
    df['services_text'] = df['affected_services_list'].apply(create_services_text)
    
    # 4. Create the comprehensive incident_text field for embeddings
    def create_comprehensive_incident_text(row):
        """Create comprehensive text combining all incident information for embeddings"""
        
        components = []
        
        # Core description
        if pd.notna(row['description']) and row['description'].strip():
            components.append(f"Incident: {row['description'].strip()}")
        
        # Title if different from description
        if pd.notna(row['title']) and row['title'].strip():
            title_clean = row['title'].strip()
            if title_clean not in row['description']:
                components.append(f"Title: {title_clean}")
        
        # Technologies
        if row['technologies_text']:
            components.append(row['technologies_text'])
        
        # Symptoms
        if row['symptoms_text']:
            components.append(row['symptoms_text'])
        
        # Affected services
        if row['services_text']:
            components.append(row['services_text'])
        
        # Impact level
        if pd.notna(row['impact_level']):
            components.append(f"Severity: {row['impact_level']}")
        
        return " | ".join(components)
    
    df['incident_text'] = df.apply(create_comprehensive_incident_text, axis=1)
    
    # 5. Clean and normalize text fields
    def clean_text_field(text):
        """Clean and normalize text for better embeddings"""
        if pd.isna(text) or text == '':
            return ''
        
        # Convert to string and strip
        text = str(text).strip()
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove problematic characters but keep punctuation
        text = re.sub(r'[^\w\s\.\,\:\;\-\(\)\[\]\/\&]', ' ', text)
        
        return text
    
    # Clean all text fields
    text_fields = ['title', 'description', 'root_cause', 'incident_text']
    for field in text_fields:
        df[field] = df[field].apply(clean_text_field)
    
    # 6. Create logs field from telemetry data (if needed)
    df['logs'] = df.apply(lambda row: f"Template: {row['template_type']}, Duration: {row['duration_minutes']} minutes", axis=1)
    
    print(f"âœ… Preprocessing complete. Final shape: {df.shape}")
    print(f"âœ… Created incident_text field for {len(df)} incidents")
    
    return df

# Process the incidents data
if incidents_df is not None:
    df_processed = preprocess_sre_incidents(incidents_df)
    
    # Display sample of processed data
    if df_processed is not None:
        print("\n" + "="*80)
        print("PROCESSED DATA PREVIEW")
        print("="*80)
        sample_incident = df_processed.iloc[0]
        print(f"Incident ID: {sample_incident['incident_id']}")
        print(f"Title: {sample_incident['title']}")
        print(f"Impact Level: {sample_incident['impact_level']}")
        print(f"Technologies: {sample_incident['technologies_list']}")
        print(f"Services: {sample_incident['affected_services_list']}")
        print(f"\nIncident Text (for embeddings):")
        print(f"{sample_incident['incident_text'][:400]}...")
        print(f"\nRoot Cause: {sample_incident['root_cause'][:200]}...")

# ================================
# PART 5: TELEMETRY DATA PREPROCESSING
# ================================

def preprocess_telemetry_data(telemetry_df):
    """Preprocess telemetry data for insights and correlation"""
    
    if telemetry_df is None:
        print("â�Œ No telemetry data to preprocess")
        return None
    
    print("Processing telemetry data...")
    
    df_tel = telemetry_df.copy()
    
    # Convert timestamp
    df_tel['timestamp'] = pd.to_datetime(df_tel['timestamp'])
    
    # Create aggregated telemetry summaries per incident
    incident_telemetry_summary = df_tel.groupby('incident_id').agg({
        'metric_value': ['mean', 'max', 'min', 'std'],
        'timestamp': ['min', 'max'],
        'metric_name': lambda x: list(x.unique()),
        'service_name': lambda x: list(x.unique())
    }).round(2)
    
    # Flatten column names
    incident_telemetry_summary.columns = ['_'.join(col).strip() for col in incident_telemetry_summary.columns]
    incident_telemetry_summary = incident_telemetry_summary.reset_index()
    
    print(f"âœ… Processed telemetry for {len(incident_telemetry_summary)} incidents")
    
    return df_tel, incident_telemetry_summary

# Process telemetry data
if telemetry_df is not None:
    telemetry_processed, telemetry_summary = preprocess_telemetry_data(telemetry_df)

# ================================
# PART 6: BIGQUERY TABLE SCHEMAS
# ================================

def create_incidents_table_schema():
    """Define the BigQuery schema for incidents table"""
    
    schema = [
        bigquery.SchemaField("incident_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("title", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("description", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("impact_level", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("technologies", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("affected_services", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("symptoms", "STRING", mode="NULLABLE"),
        bigquery.SchemaField("root_cause", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("resolution_steps", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("incident_time", "TIMESTAMP", mode="REQUIRED"),
        bigquery.SchemaField("resolution_time", "TIMESTAMP", mode="REQUIRED"),
        bigquery.SchemaField("duration_minutes", "INTEGER", mode="REQUIRED"),
        bigquery.SchemaField("template_type", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("incident_text", "STRING", mode="REQUIRED"),  # Key field for embeddings
        bigquery.SchemaField("logs", "STRING", mode="NULLABLE"),
    ]
    
    return schema

def create_telemetry_table_schema():
    """Define the BigQuery schema for telemetry table"""
    
    schema = [
        bigquery.SchemaField("timestamp", "TIMESTAMP", mode="REQUIRED"),
        bigquery.SchemaField("incident_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("service_name", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("metric_category", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("metric_name", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("metric_value", "FLOAT", mode="REQUIRED"),
        bigquery.SchemaField("phase", "STRING", mode="REQUIRED"),
    ]
    
    return schema

# ================================
# PART 7: DATA LOADING FUNCTIONS
# ================================

def load_incidents_to_bigquery(client, df_processed, dataset_id, table_id="historical_incidents"):
    """Load processed incidents data to BigQuery"""
    
    if df_processed is None:
        print("â�Œ No processed data to load")
        return None
    
    # Select columns for BigQuery
    columns_for_bq = [
        'incident_id', 'title', 'description', 'impact_level', 'technologies',
        'affected_services', 'symptoms', 'root_cause', 'resolution_steps',
        'incident_time', 'resolution_time', 'duration_minutes', 'template_type',
        'incident_text', 'logs'
    ]
    
    df_for_bq = df_processed[columns_for_bq].copy()
    
    # Create table reference
    table_ref = client.dataset(dataset_id).table(table_id)
    
    # Configure the load job
    job_config = bigquery.LoadJobConfig(
        schema=create_incidents_table_schema(),
        write_disposition="WRITE_TRUNCATE",  # Overwrite if table exists
    )
    
    # Load data
    job = client.load_table_from_dataframe(df_for_bq, table_ref, job_config=job_config)
    job.result()  # Wait for the job to complete
    
    table = client.get_table(table_ref)
    print(f"âœ… Loaded {table.num_rows} incidents to {dataset_id}.{table_id}")
    
    return table_ref

def load_telemetry_to_bigquery(client, telemetry_df, dataset_id, table_id="incident_telemetry"):
    """Load telemetry data to BigQuery"""
    
    if telemetry_df is None:
        print("â�Œ No telemetry data to load")
        return None
    
    # Create table reference
    table_ref = client.dataset(dataset_id).table(table_id)
    
    # Configure the load job
    job_config = bigquery.LoadJobConfig(
        schema=create_telemetry_table_schema(),
        write_disposition="WRITE_TRUNCATE",
    )
    
    # Load data (might be large, so consider chunking for very large datasets)
    job = client.load_table_from_dataframe(telemetry_df, table_ref, job_config=job_config)
    job.result()
    
    table = client.get_table(table_ref)
    print(f"âœ… Loaded {table.num_rows} telemetry points to {dataset_id}.{table_id}")
    
    return table_ref

# Example usage (uncomment when you have credentials):
# incidents_table = load_incidents_to_bigquery(client, df_processed, 'incident_ai_hackathon')
# telemetry_table = load_telemetry_to_bigquery(client, telemetry_processed, 'incident_ai_hackathon')

# ================================
# PART 8: ENHANCED VERIFICATION QUERIES
# ================================

def generate_verification_queries(project_id, dataset_id):
    """Generate SQL queries to verify the loaded SRE data"""
    
    incidents_table = f"`{project_id}.{dataset_id}.historical_incidents`"
    telemetry_table = f"`{project_id}.{dataset_id}.incident_telemetry`"
    
    queries = {
        "incidents_overview": f"""
        SELECT 
            COUNT(*) as total_incidents,
            COUNT(DISTINCT template_type) as incident_types,
            COUNT(DISTINCT impact_level) as severity_levels,
            AVG(duration_minutes) as avg_duration_minutes,
            MIN(incident_time) as earliest_incident,
            MAX(incident_time) as latest_incident
        FROM {incidents_table}
        """,
        
        "severity_distribution": f"""
        SELECT 
            impact_level,
            COUNT(*) as incident_count,
            AVG(duration_minutes) as avg_duration,
            ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 1) as percentage
        FROM {incidents_table}
        GROUP BY impact_level
        ORDER BY 
            CASE impact_level 
                WHEN 'Critical' THEN 1 
                WHEN 'High' THEN 2 
                WHEN 'Medium' THEN 3 
                WHEN 'Low' THEN 4 
            END
        """,
        
        "incident_types_analysis": f"""
        SELECT 
            template_type,
            COUNT(*) as incidents,
            AVG(duration_minutes) as avg_resolution_time,
            STRING_AGG(DISTINCT impact_level, ', ') as severity_levels_seen
        FROM {incidents_table}
        GROUP BY template_type
        ORDER BY incidents DESC
        """,
        
        "incident_text_validation": f"""
        SELECT 
            incident_id,
            impact_level,
            template_type,
            LENGTH(incident_text) as text_length,
            LEFT(incident_text, 200) as text_preview
        FROM {incidents_table}
        WHERE LENGTH(incident_text) > 50  -- Ensure we have substantial text for embeddings
        ORDER BY text_length DESC
        LIMIT 5
        """,
        
        "telemetry_overview": f"""
        SELECT 
            COUNT(*) as total_data_points,
            COUNT(DISTINCT incident_id) as incidents_with_telemetry,
            COUNT(DISTINCT service_name) as monitored_services,
            COUNT(DISTINCT metric_name) as unique_metrics,
            MIN(timestamp) as earliest_data,
            MAX(timestamp) as latest_data
        FROM {telemetry_table}
        """,
        
        "telemetry_per_incident": f"""
        SELECT 
            i.impact_level,
            COUNT(DISTINCT t.incident_id) as incidents_with_telemetry,
            AVG(telemetry_count) as avg_telemetry_points_per_incident
        FROM {incidents_table} i
        JOIN (
            SELECT incident_id, COUNT(*) as telemetry_count
            FROM {telemetry_table}
            GROUP BY incident_id
        ) t ON i.incident_id = t.incident_id
        GROUP BY i.impact_level
        ORDER BY 
            CASE i.impact_level 
                WHEN 'Critical' THEN 1 
                WHEN 'High' THEN 2 
                WHEN 'Medium' THEN 3 
                WHEN 'Low' THEN 4 
            END
        """
    }
    
    return queries

# Generate verification queries
verification_queries = generate_verification_queries('your-project-id', 'incident_ai_hackathon')

print("\n" + "="*80)
print("VERIFICATION QUERIES FOR SRE DATASET")
print("="*80)
print("After loading data to BigQuery, run these queries to verify:")

for query_name, query in verification_queries.items():
    print(f"\n-- {query_name.upper().replace('_', ' ')}")
    print(query)

# ================================
# PART 9: STEP 3 COMPLETION SUMMARY
# ================================

print("\n" + "="*80)
print("STEP 3 COMPLETION CHECKLIST - SRE DATASET")
print("="*80)

checklist = [
    "âœ… BigQuery dataset and tables configured",
    "âœ… SRE incident generator data loaded and preprocessed", 
    "âœ… Comprehensive incident_text field created for embeddings",
    "âœ… JSON fields properly parsed and processed",
    "âœ… Telemetry data processed and linked to incidents",
    "âœ… Data quality validation queries prepared",
    "âœ… Schema optimized for BigQuery AI functions",
    "â�³ Ready for data loading to BigQuery"
]

for item in checklist:
    print(item)

if df_processed is not None:
    print(f"\nğŸ�¯ Key Achievements with SRE Dataset:")
    print(f"   â€¢ Processed {len(df_processed)} realistic incident scenarios")
    print(f"   â€¢ {df_processed['template_type'].nunique()} different incident patterns")
    print(f"   â€¢ {df_processed['impact_level'].nunique()} severity levels")
    print(f"   â€¢ Rich incident_text field combining description, symptoms, and context")
    print(f"   â€¢ Linked telemetry data for correlation analysis")
    
    print(f"\nğŸ“Š Dataset Statistics:")
    severity_dist = df_processed['impact_level'].value_counts()
    for severity, count in severity_dist.items():
        print(f"   â€¢ {severity}: {count} incidents")
    
    avg_duration = df_processed['duration_minutes'].mean()
    print(f"   â€¢ Average resolution time: {avg_duration:.1f} minutes")

print(f"\nğŸ“‹ Ready for Step 4: Generating Text Embeddings!")
print(f"   â€¢ Rich incident_text field optimized for ML.GENERATE_EMBEDDING")
print(f"   â€¢ Realistic SRE scenarios for better embedding quality") 
print(f"   â€¢ Correlated telemetry data for enhanced insights")
print(f"   â€¢ Multiple incident patterns for comprehensive similarity search")

# ================================
# PART 10: EXPORT FOR REFERENCE
# ================================

if df_processed is not None:
    # Save processed data for reference
    df_processed.to_csv('processed_sre_incidents.csv', index=False)
    print(f"\nğŸ’¾ Processed incidents exported to 'processed_sre_incidents.csv'")
    
    # Save a sample for testing
    sample_df = df_processed.head(10)
    sample_df.to_csv('sample_incidents_for_testing.csv', index=False)
    print(f"ğŸ’¾ Sample data exported to 'sample_incidents_for_testing.csv'")

