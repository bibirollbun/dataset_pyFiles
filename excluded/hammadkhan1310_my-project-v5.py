import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import random
import time
import json

# --- 1. Simulate a Message Bus (like Kafka) ---
# In a real system, these would be Kafka producer/consumer calls
def publish_message(topic, message):
    """Simulates publishing a message to a Kafka topic."""
    print(f"\n--- PUBLISHED to '{topic}': {json.dumps(message, indent=2)}")

def consume_message(topic):
    """Simulates consuming a message from a Kafka topic (very basic)."""
    # In a real system, this would be a persistent consumer loop
    # For this example, we'll just return a mock message if available
    pass # Real consumption logic would go here

# --- 2. APR Agent Class Definition ---
class APRAgent:
    def __init__(self, agent_id, input_topic, output_topic_anomalies, output_topic_insights):
        self.agent_id = agent_id
        self.input_topic = input_topic
        self.output_topic_anomalies = output_topic_anomalies
        self.output_topic_insights = output_topic_insights
        self.model = IsolationForest(contamination=0.05, random_state=42) # 5% assumed anomaly rate
        self.data_buffer = [] # Store recent data for training
        self.buffer_size = 100 # How many data points to keep for training
        self.is_trained = False
        print(f"[{self.agent_id}] Initialized.")

    def _train_model(self):
        """Trains or re-trains the Isolation Forest model with buffered data."""
        if len(self.data_buffer) >= self.buffer_size:
            df = pd.DataFrame(self.data_buffer)
            # Train only on numerical features relevant for anomaly detection
            features = df[['speed', 'volume']]
            self.model.fit(features)
            self.is_trained = True
            print(f"[{self.agent_id}] Model re-trained with {len(self.data_buffer)} data points.")
        else:
            print(f"[{self.agent_id}] Not enough data to train. Buffer size: {len(self.data_buffer)}/{self.buffer_size}")

    def process_data(self, data_point):
        """Processes an incoming data point, detects anomalies, and publishes insights."""
        print(f"[{self.agent_id}] Processing data: {data_point}")

        # Add data to buffer for re-training
        self.data_buffer.append(data_point)
        if len(self.data_buffer) > self.buffer_size:
            self.data_buffer.pop(0) # Remove oldest data if buffer is full

        # Re-train periodically or when buffer is full
        if len(self.data_buffer) == self.buffer_size and not self.is_trained:
            self._train_model()
        elif random.random() < 0.1 and self.is_trained: # Periodically re-train with new data
             self._train_model()

        if not self.is_trained:
            print(f"[{self.agent_id}] Skipping anomaly detection, model not trained yet.")
            return

        # Prepare data for prediction
        current_features = pd.DataFrame([data_point[['speed', 'volume']]])
        
        # Predict anomaly score (-1 for anomaly, 1 for normal)
        anomaly_prediction = self.model.predict(current_features)[0]
        anomaly_score = self.model.decision_function(current_features)[0]

        if anomaly_prediction == -1:
            anomaly_message = {
                "agent_id": self.agent_id,
                "timestamp": data_point['timestamp'],
                "location_id": data_point['location_id'],
                "type": "Traffic_Anomaly",
                "severity": "High" if anomaly_score < -0.15 else "Medium",
                "description": f"Unusual traffic pattern detected (speed: {data_point['speed']}, volume: {data_point['volume']}). Anomaly score: {anomaly_score:.2f}",
                "raw_data": data_point
            }
            publish_message(self.output_topic_anomalies, anomaly_message)
        else:
            # Publish normal insights if desired, or just log
            insight_message = {
                "agent_id": self.agent_id,
                "timestamp": data_point['timestamp'],
                "location_id": data_point['location_id'],
                "type": "Normal_Traffic_Flow",
                "description": f"Traffic flowing normally (speed: {data_point['speed']}, volume: {data_point['volume']}). Score: {anomaly_score:.2f}"
            }
            # publish_message(self.output_topic_insights, insight_message) # Uncomment to see normal flow insights
            print(f"[{self.agent_id}] Normal traffic, score: {anomaly_score:.2f}")

    def run(self):
        """Simulates the agent's continuous operation."""
        print(f"[{self.agent_id}] Starting to run...")
        # In a real system, this would be a loop consuming from Kafka
        # For this example, we'll wait for explicit data points
        pass

# --- 3. Simulate Data Ingestion (DIA Agent) ---
def generate_traffic_data(location_id, is_anomaly=False):
    """Generates a simulated traffic data point."""
    timestamp = time.time()
    speed = random.uniform(20, 70) # Normal speed range
    volume = random.randint(50, 300) # Normal volume range

    if is_anomaly:
        choice = random.choice(['slow_high_vol', 'fast_low_vol'])
        if choice == 'slow_high_vol':
            speed = random.uniform(5, 15) # Very slow
            volume = random.randint(400, 800) # Very high volume (congestion)
        else: # fast_low_vol
            speed = random.uniform(80, 120) # Very fast
            volume = random.randint(10, 40) # Very low volume (unusual emptiness)
        print(f"--- Generating ANOMALY for {location_id} ---")

    return {
        "timestamp": timestamp,
        "location_id": location_id,
        "speed": speed,
        "volume": volume
    }

# --- 4. Main Simulation Loop ---
if __name__ == "__main__":
    # Define topics (simulated Kafka topics)
    TRAFFIC_DATA_TOPIC = "traffic_sensor_data"
    TRAFFIC_ANOMALIES_TOPIC = "traffic_anomalies"
    TRAFFIC_INSIGHTS_TOPIC = "traffic_insights"

    # Instantiate an APR Agent
    apr_agent = APRAgent(
        agent_id="TrafficAPR_001",
        input_topic=TRAFFIC_DATA_TOPIC,
        output_topic_anomalies=TRAFFIC_ANOMALIES_TOPIC,
        output_topic_insights=TRAFFIC_INSIGHTS_TOPIC
    )

    # Simulate receiving data from a DIA agent or raw sensors
    print("\n--- Starting Data Simulation ---")
    data_points_to_simulate = []

    # Generate normal data for initial training
    for i in range(apr_agent.buffer_size):
        data_points_to_simulate.append(generate_traffic_data("loc_A", is_anomaly=False))
        time.sleep(0.1) # Simulate real-time delay

    # Introduce some normal data, then anomalies
    for i in range(10):
        data_points_to_simulate.append(generate_traffic_data("loc_A", is_anomaly=False))
        if i % 3 == 0: # Introduce an anomaly every few steps
            data_points_to_simulate.append(generate_traffic_data("loc_A", is_anomaly=True))
        time.sleep(0.1)

    # Now, process the simulated data through the APR Agent
    for data in data_points_to_simulate:
        apr_agent.process_data(data)
        time.sleep(0.5) # Simulate processing time

    print("\n--- Simulation Finished ---")
    




