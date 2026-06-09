# ----------- Agent Function Definitions -----------

def VideoStream():
    return "video_stream_output"

def ImagesAgent(data):
    return f"images_processed({data})"

def MCP(data):
    return f"mcp_output({data})"

def VisionAgent(data):
    return f"vision_output({data})"

def DepthAgent(data):
    return f"depth_output({data})"

def TrackerAgent(data):
    return f"tracking_output({data})"

def PredictorAgent(data):
    return f"prediction({data})"

def ForecasterAgent(data):
    return f"forecast_text_alerts({data})"

def EvaluatorAgent(data):
    return f"evaluation_metrics({data})"


# ----------- Pipeline Orchestration -----------

def run_pipeline():

    # Step 1: Video stream
    data = VideoStream()

    # Step 2: Image processing
    data = ImagesAgent(data)

    # Step 3: Parallel agents
    parallel_results = {
        "mcp": MCP(data),
        "vision": VisionAgent(data),
        "depth": DepthAgent(data),
        "tracking": TrackerAgent(data)
    }

    # Step 4: Sequential prediction
    prediction_output = PredictorAgent(parallel_results)

    # Step 5: Forecaster â†’ text alert
    alert_output = ForecasterAgent(prediction_output)

    # Step 6: Evaluation
    final_output = EvaluatorAgent(alert_output)

    return final_output


# ----------- Run Pipeline -----------

output = run_pipeline()
print(output)


{
"id": "uuid",
"from": "agent:vision",
"to": "agent:predictor",
"type": "EVENT",
"payload": {"timestamp": "...","frame_id":123, "objects":[...]} ,
"trace_id": "trace-uuid",
"ttl": 30
}


import os
class RedisMCP:
    def publish(self, to, message):
        print(f"[MCP] Publishing to {to}: {message}")

    def control(self, agent_id, cmd):
        print(f"[MCP] Controlling {agent_id} with command: {cmd}")
project_structure = {
    "accident-multiagent/app/orchestrator": ["main.py"],
    "accident-multiagent/app/agents": [
        "vision_agent.py",
        "predictor_agent.py",
        "llm_agent.py",
        "evaluator_agent.py"
    ],
    "accident-multiagent/app/tools": [
        "google_search.py",
        "code_exec.py",
        "carla_tool.py"
    ],
    "accident-multiagent/app/sessions": [
        "in_memory_session.py",
        "memory_bank.py"
    ],
    "accident-multiagent/app/mcp": [
        "redis_mcp.py"
    ],
    "accident-multiagent/app/observability": [
        "metrics.py",
        "tracing.py"
    ],
    "accident-multiagent/docker": [],
    "accident-multiagent/k8s": [],
}
# Create all directories and files
for folder, files in project_structure.items():
    os.makedirs(folder, exist_ok=True)
    for file in files:
        file_path = os.path.join(folder, file)
        if not os.path.exists(file_path):
            open(file_path, "w").close()  # create empty file

# Create README.md at root
open("accident-multiagent/README.md", "w").close()

print("Project structure created successfully!")


# app/mcp/redis_mcp.py
!pip install redis
import redis, json

class RedisMCP:
    def __init__(self, url='redis://localhost:6379/0'):
        # Initialize the Redis client connection
        self.r = redis.from_url(url)
    
    def publish(self, channel, message):
        # Publish a message to a specific Redis stream (channel)
        # message is dumped to JSON before being wrapped in the stream entry
        self.r.xadd(channel, {'msg': json.dumps(message)})

    def control(self, agent_id, cmd):
        # Send a control command to a specific agent's control stream
        ctl = {'type': 'CONTROL', 'cmd': cmd}
        self.r.xadd(f'agent_control:{agent_id}', {'msg': json.dumps(ctl)})

# Example Vision Agent section (as seen in the first image) likely continues below:
# 5.3 Example Vision Agent
# ...


# app/mcp/redis_mcp.py
import redis, json

class RedisMCP:
    def __init__(self, url='redis://localhost:6379/0'):
        self.r = redis.from_url(url)
    
    def publish(self, channel, message):
        self.r.xadd(channel, {'msg': json.dumps(message)})

    def control(self, agent_id, cmd):
        ctl = {'type': 'CONTROL', 'cmd': cmd}
        self.r.xadd(f'agent_control:{agent_id}', {'msg': json.dumps(ctl)})


# app/mcp/redis_mcp.py
import redis, json

class RedisMCP:
    def __init__(self, url='redis://localhost:6379/0'):
        # Initialize the Redis client connection
        self.r = redis.from_url(url)
    
    def publish(self, channel, message):
        """
        Publishes a message to a specific Redis stream (channel).
        """
        # message is dumped to JSON before being wrapped in the stream entry
        self.r.xadd(channel, {'msg': json.dumps(message)})

    def control(self, agent_id, cmd):
        """
        Sends a control command to a specific agent's control stream.
        """
        ctl = {'type': 'CONTROL', 'cmd': cmd}
        self.r.xadd(f'agent_control:{agent_id}', {'msg': json.dumps(ctl)})

if __name__ == '__main__':
    # Simple test block (requires running Redis server)
    try:
        test_mcp = RedisMCP()
        print("RedisMCP initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize RedisMCP. Ensure Redis is running: {e}")


import time

def predictor_agent_logic(objects_data):
    """
    Simulates the Predictor Agent's core logic: 
    1. Extracts velocities.
    2. Calculates a risk score based on the fastest object.
    3. Predicts an accident if the risk exceeds a threshold.
    
    Args:
        objects_data (list): A list of dictionaries, where each dict 
                             has an 'vx' key (velocity/speed).
                             e.g., [{'vx': 0.5}, {'vx': -2.1}]
    
    Returns:
        dict: The prediction result.
    """
    
    # --- 1. Calculate Risk Score ---
    
    # Get the absolute velocity (speed) of all objects. Use 0 if 'vx' is missing.
    # The risk is defined as the maximum speed observed.
    speeds = [abs(obj.get('vx', 0)) for obj in objects_data]
    
    if not speeds:
        risk_score = 0.0
    else:
        risk_score = max(speeds)
    
    # --- 2. Predict Accident ---
    
    RISK_THRESHOLD = 1.5  # Define the threshold for dangerous speed
    will_accident = risk_score > RISK_THRESHOLD
    
    # --- 3. Return Output ---
    
    output_message = {
        'prediction_time': time.strftime("%Y-%m-%d %H:%M:%S"),
        'will_accident': will_accident,
        'risk_score': risk_score,
        'trigger_agent': 'predictor_demo'
    }
    
    return output_message

# --- SIMULATION AND OUTPUT ---

## ğŸš— Simulation 1: Low Risk (No Accident Predicted)
low_risk_data = [
    {'id': 'c1', 'class': 'car', 'vx': 0.8},
    {'id': 'c2', 'class': 'bike', 'vx': -0.5}
]

print("## Simulation 1: Low Speed")
result_low_risk = predictor_agent_logic(low_risk_data)
print(result_low_risk)
print("-" * 30)

## ğŸ’¥ Simulation 2: High Risk (Accident Predicted)
high_risk_data = [
    {'id': 'c3', 'class': 'truck', 'vx': -2.5}, # High speed (-2.5)
    {'id': 'c4', 'class': 'pedestrian', 'vx': 0.2}
]

print("## Simulation 2: High Speed")
result_high_risk = predictor_agent_logic(high_risk_data)
print(result_high_risk)


!pip install redis
import openai
import os
import json

def summarize_alert_logic(alert_data):
    """
    Simulates the LLM Agent's core logic:
    1. Constructs a prompt based on an incoming alert.
    2. Sends the prompt to the OpenAI API (or other LLM).
    3. Returns the LLM's summarized recommendation.
    
    Args:
        alert_data (dict): Dictionary containing risk information.
    
    Returns:
        dict: The LLM's text response.
    """
    
    # ğŸš¨ Configuration Check
    # Ensure your API key is set in your environment variables 
    # OR set it directly here (NOT recommended for production)
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found in environment variables.")
        return {"text": "Configuration Error: API key missing."}
        
    openai.api_key = api_key
    
    # --- 1. Construct Prompt ---
    
    risk_score = alert_data.get('risk_score', 'unknown')
    
    prompt = (
        f"An accident is predicted with a risk score of {risk_score}. "
        "The current status is critical. Summarize this situation briefly (one sentence) "
        "and provide two clear, numbered recommended actions for the control system."
    )
    
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    # --- 2. Call LLM API ---
    try:
        # Use a reliable, cost-effective model for this task
        res = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages=messages,
            max_tokens=250,
            temperature=0.3 # Lower temperature for factual/actionable output
        )
        
        # --- 3. Return Response ---
        llm_response = res.choices[0].message.content.strip()
        
        return {
            'text': llm_response,
            'risk_score_used': risk_score
        }
        
    except openai.error.AuthenticationError:
        return {"text": "Authentication Error: Please check your OpenAI API key."}
    except Exception as e:
        return {"text": f"An unexpected error occurred: {e}"}

# --- SIMULATION AND OUTPUT ---

# Simulate an incoming alert from the Predictor Agent
incoming_alert = {
    'will_accident': True,
    'risk_score': 2.5,
    'trigger_agent': 'agent:predictor'
}

print("## ğŸ¤– LLM Agent Simulation Output")
final_recommendation = summarize_alert_logic(incoming_alert)

print(json.dumps(final_recommendation, indent=4))


import openai
import os
import json
import time

def summarize_alert_logic(alert_data):
    """
    Simulates the LLM Agent's core logic: 
    1. Constructs a prompt based on an incoming alert.
    2. Sends the prompt to the OpenAI API (or other LLM).
    3. Returns the LLM's summarized recommendation.
    
    Args:
        alert_data (dict): Dictionary containing risk information, e.g., {'risk_score': 2.5}.
    
    Returns:
        dict: The LLM's text response and metadata.
    """
    
    # ğŸš¨ Configuration Check
    # For execution, you MUST set the OPENAI_API_KEY environment variable.
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set.")
        return {"text": "Configuration Error: API key missing."}
        
    openai.api_key = api_key
    
    # --- 1. Construct Prompt ---
    risk_score = alert_data.get('risk_score', 'unknown')
    
    prompt = (
        f"An imminent accident is predicted with a risk score of {risk_score}. "
        "The current status is critical. Summarize this situation briefly (one sentence) "
        "and provide two clear, numbered recommended control actions for the traffic system."
    )
    
    messages = [{"role": "user", "content": prompt}]
    
    # --- 2. Call LLM API ---
    try:
        # Using gpt-3.5-turbo for a quick, effective response
        res = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages=messages,
            max_tokens=250,
            temperature=0.3
        )
        
        # --- 3. Return Response ---
        llm_response = res.choices[0].message.content.strip()
        
        return {
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'status': 'success',
            'llm_output': llm_response,
            'risk_score_used': risk_score
        }
        
    except openai.error.AuthenticationError:
        return {"text": "Authentication Error: Please check your OpenAI API key."}
    except Exception as e:
        return {"text": f"An unexpected error occurred: {e}"}

# --- SIMULATION AND OUTPUT ---

# Simulate an incoming alert from the Predictor Agent
incoming_alert = {
    'will_accident': True,
    'risk_score': 2.5,
    'trigger_agent': 'agent:predictor'
}

print("## ğŸ¤– LLM Agent Simulation Output")
print("Note: This requires a valid OPENAI_API_KEY environment variable.")
final_recommendation = summarize_alert_logic(incoming_alert)

print(json.dumps(final_recommendation, indent=4))


# agent control handler
def handle_control(msg):
    # Indent the main logic block (if/elif)
    if msg['cmd'] == 'PAUSE':
        # Indent the block inside the 'if'
        set_state('paused')
        save_checkpoint()
    elif msg['cmd'] == 'RESUME':
        # Indent the block inside the 'elif'
        set_state('running') # Assuming 'set_st' meant 'set_state'


{
  "from": "agent:vision",
  "to": "agent:predictor",
  "type": "EVENT",
  "payload": {
    "frame_id": 123,
    "timestamp": "2025-11-19T18:00:00Z",
    "objects": [
      {"id": 1, "class": "car", "bbox": [10, 10, 100, 100], "vx": -2.0},
      {"id": 2, "class": "pedestrian", "bbox": [200, 50, 220, 100], "vx": 0.0}
    ]
  },
  "trace_id": "trace-uuid",
  "ttl": 30
}


data = {
    "will_accident": True,
    "risk_score": 2.1,
    "predicted_tta_seconds": 7.5,
    "high_risk_objects": [
        {"id": 1, "class": "car", "bbox": [10, 10, 100, 100], "vx": -2.0}
    ]
}

print(data["will_accident"])  # True
print(data["high_risk_objects"][0]["class"])  # car



{
  "text": "Warning: A vehicle is approaching dangerously close to another car. Possible collision in ~7 seconds. Recommended action: slow down, increase distance, and prepare to brake."
}



{
  "metrics": {
    "frame_id": 123,
    "mTTA": 7.5,
    "precision": 1.0,
    "recall": 1.0,
    "false_alarms_per_minute": 0.0
  }
}



# accident_detection_demo.py

# ---------------------------
# Simulated Vision Agent Output
# ---------------------------
vision_output = {
    "from": "agent:vision",
    "to": "agent:predictor",
    "type": "EVENT",
    "payload": {
        "frame_id": 123,
        "timestamp": "2025-11-19T18:00:00Z",
        "objects": [
            {"id": 1, "class": "car", "bbox": [10, 10, 100, 100], "vx": -2.0},
            {"id": 2, "class": "pedestrian", "bbox": [200, 50, 220, 100], "vx": 0.0}
        ]
    },
    "trace_id": "trace-uuid",
    "ttl": 30
}

# ---------------------------
# Predictor Agent (Simulated)
# ---------------------------
def predictor(vision_payload):
    objects = vision_payload["objects"]
    # Simple heuristic: if any moving object has negative vx, consider high risk
    high_risk_objects = [obj for obj in objects if obj.get("vx", 0) < -1.0]
    will_accident = len(high_risk_objects) > 0
    risk_score = max([-obj.get("vx",0) for obj in high_risk_objects], default=0)
    predicted_tta_seconds = 7.5  # fixed for demo
    return {
        "will_accident": will_accident,
        "risk_score": risk_score,
        "predicted_tta_seconds": predicted_tta_seconds,
        "high_risk_objects": high_risk_objects
    }

predictor_output = predictor(vision_output["payload"])

# ---------------------------
# Evaluator Metrics (Simulated)
# ---------------------------
metrics_output = {
    "metrics": {
        "frame_id": vision_output["payload"]["frame_id"],
        "mTTA": predictor_output["predicted_tta_seconds"],
        "precision": 1.0,  # simulated
        "recall": 1.0,     # simulated
        "false_alarms_per_minute": 0.0
    }
}

# ---------------------------
# LLM Agent Alert Generation (Simulated)
# ---------------------------
def generate_alert(predictor_result):
    if predictor_result["will_accident"]:
        objects = predictor_result["high_risk_objects"]
        obj_classes = ", ".join([obj["class"] for obj in objects])
        tta = predictor_result["predicted_tta_seconds"]
        return {
            "text": f"Warning: An accident is predicted in approximately {tta} seconds involving {obj_classes}. "
                    f"The systemâ€™s predictions are highly accurate, with no false alarms detected. "
                    f"Recommended action: slow down immediately and prepare to brake."
        }
    else:
        return {"text": "No accident predicted for this frame."}

alert_output = generate_alert(predictor_output)

# ---------------------------
# Final Combined Output
# ---------------------------
final_output = {
    "prediction": predictor_output,
    "metrics": metrics_output,
    "alert": alert_output
}

# ---------------------------
# Print Final Output
# ---------------------------
import json
print(json.dumps(final_output, indent=4))


