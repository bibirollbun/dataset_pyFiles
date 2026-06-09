import random
import time
import uuid
import copy
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import pandas as pd

# --- MOCK GRAFANA ---
class MockGrafana:
    """
    Simulates a Grafana/Prometheus metrics API.
    Stores in-memory metrics for services.
    """
    def __init__(self):
        self.metrics_store = {
            "payment-service": {"cpu": 15.0, "memory": 40.0, "latency": 0.05, "error_rate": 0.0},
            "auth-service": {"cpu": 10.0, "memory": 30.0, "latency": 0.02, "error_rate": 0.0},
            "database": {"cpu": 25.0, "memory": 60.0, "latency": 0.01, "error_rate": 0.0},
        }
        self.anomalies = {}

    def get_metric(self, service_name: str, metric_name: str) -> float:
        if service_name not in self.metrics_store:
            raise ValueError(f"Service {service_name} not found")
        
        base_value = self.metrics_store[service_name].get(metric_name, 0.0)
        
        # Check if there's an active anomaly injection
        if service_name in self.anomalies and metric_name in self.anomalies[service_name]:
            anomaly = self.anomalies[service_name][metric_name]
            if isinstance(anomaly, dict) and anomaly.get("type") == "trend":
                # Update trend value
                anomaly["value"] += anomaly["slope"]
                return anomaly["value"]
            elif isinstance(anomaly, (int, float)):
                return float(anomaly)
        
        # Add noise
        noise = random.uniform(-0.1 * base_value, 0.1 * base_value)
        return max(0.0, base_value + noise)

    def get_all_metrics(self, service_name: str) -> Dict[str, float]:
        if service_name not in self.metrics_store:
            raise ValueError(f"Service {service_name} not found")
        
        return {
            k: self.get_metric(service_name, k)
            for k in self.metrics_store[service_name]
        }

    def inject_anomaly(self, service_name: str, metric_name: str, value: float):
        if service_name not in self.anomalies:
            self.anomalies[service_name] = {}
        self.anomalies[service_name][metric_name] = value
        print(f"[Grafana] Injected anomaly: {service_name} {metric_name} = {value}")

    def inject_trend(self, service_name: str, metric_name: str, slope: float):
        if service_name not in self.anomalies:
            self.anomalies[service_name] = {}
        current_val = self.metrics_store[service_name].get(metric_name, 0.0)
        self.anomalies[service_name][metric_name] = {"type": "trend", "slope": slope, "value": current_val}
        print(f"[Grafana] Injected trend: {service_name} {metric_name} slope={slope}")

    def get_metric_history(self, service_name: str, metric_name: str, window_size: int = 10) -> List[float]:
        current = self.get_metric(service_name, metric_name)
        history = []
        slope = 0
        if service_name in self.anomalies and metric_name in self.anomalies[service_name]:
            anomaly = self.anomalies[service_name][metric_name]
            if isinstance(anomaly, dict) and anomaly.get("type") == "trend":
                slope = anomaly["slope"]

        for i in range(window_size):
            val = current - (slope * i)
            noise = random.uniform(-0.05 * val, 0.05 * val) if val > 0 else 0
            history.insert(0, max(0.0, val + noise))
            
        return history

    def clear_anomalies(self):
        self.anomalies = {}
        print("[Grafana] Cleared all anomalies")

# --- MOCK INFRASTRUCTURE ---
class MockInfrastructure:
    """
    Simulates a Kubernetes/Docker environment.
    """
    def __init__(self):
        self.pods = {
            "payment-service": {"status": "Running", "replicas": 2, "version": "v1.2.0"},
            "auth-service": {"status": "Running", "replicas": 3, "version": "v1.1.5"},
            "database": {"status": "Running", "replicas": 1, "version": "postgres:14"},
        }
        self.logs = {
            "payment-service": ["INFO: Payment processed successfully", "INFO: Health check passed"],
            "auth-service": ["INFO: User logged in", "INFO: Token refreshed"],
            "database": ["INFO: Connection accepted", "INFO: Query executed"]
        }

    def get_pod_status(self, service_name: str) -> Dict:
        return self.pods.get(service_name, {"status": "Unknown"})

    def get_logs(self, service_name: str, lines: int = 10) -> List[str]:
        return self.logs.get(service_name, [])[-lines:]

    def restart_pod(self, service_name: str):
        print(f"[Infra] Restarting {service_name}...")
        self.pods[service_name]["status"] = "Restarting"
        time.sleep(0.5)
        self.pods[service_name]["status"] = "Running"
        self.logs[service_name].append("INFO: Service restarted successfully")
        print(f"[Infra] {service_name} is back online.")

    def scale_deployment(self, service_name: str, replicas: int):
        print(f"[Infra] Scaling {service_name} to {replicas} replicas...")
        self.pods[service_name]["replicas"] = replicas
        self.logs[service_name].append(f"INFO: Scaled to {replicas} replicas")

    def create_snapshot(self, service_name: str) -> str:
        snapshot_id = str(uuid.uuid4())[:8]
        if not hasattr(self, 'snapshots'):
            self.snapshots = {}
        self.snapshots[snapshot_id] = {
            "service": service_name,
            "state": copy.deepcopy(self.pods.get(service_name))
        }
        print(f"[Infra] Created snapshot {snapshot_id} for {service_name}")
        return snapshot_id

    def restore_snapshot(self, snapshot_id: str) -> bool:
        if not hasattr(self, 'snapshots') or snapshot_id not in self.snapshots:
            print(f"[Infra] Snapshot {snapshot_id} not found.")
            return False
        snapshot = self.snapshots[snapshot_id]
        service_name = snapshot["service"]
        self.pods[service_name] = snapshot["state"]
        print(f"[Infra] Restored {service_name} from snapshot {snapshot_id}")
        self.logs[service_name].append(f"WARN: Rolled back to snapshot {snapshot_id}")
        return True

    def inject_log_error(self, service_name: str, error_msg: str):
        if service_name in self.logs:
            self.logs[service_name].append(f"ERROR: {error_msg}")

# --- MOCK TICKETING ---
class MockTicketing:
    def __init__(self):
        self.tickets = {}

    def create_ticket(self, title: str, description: str, priority: str = "Medium") -> str:
        ticket_id = f"TICKET-{len(self.tickets) + 1001}"
        self.tickets[ticket_id] = {
            "id": ticket_id,
            "title": title,
            "description": description,
            "priority": priority,
            "status": "Open",
            "created_at": datetime.now().isoformat(),
            "comments": []
        }
        print(f"[Ticketing] Created ticket {ticket_id}: {title}")
        return ticket_id

    def update_ticket(self, ticket_id: str, status: str = None, comment: str = None):
        if ticket_id not in self.tickets:
            raise ValueError(f"Ticket {ticket_id} not found")
        if status:
            self.tickets[ticket_id]["status"] = status
            print(f"[Ticketing] Updated {ticket_id} status to {status}")
        if comment:
            self.tickets[ticket_id]["comments"].append({
                "timestamp": datetime.now().isoformat(),
                "text": comment
            })
            print(f"[Ticketing] Added comment to {ticket_id}: {comment}")
    
    def get_ticket(self, ticket_id: str) -> Dict:
        return self.tickets.get(ticket_id)


# --- MONITOR AGENT ---
class MonitorAgent:
    def __init__(self, grafana_client: MockGrafana):
        self.grafana = grafana_client
        self.thresholds = {
            "cpu": 80.0,
            "memory": 85.0,
            "latency": 0.5,
            "error_rate": 0.05
        }

    def check_health(self, service_name: str) -> List[str]:
        issues = []
        try:
            metrics = self.grafana.get_all_metrics(service_name)
            if metrics["cpu"] > self.thresholds["cpu"]:
                issues.append(f"High CPU usage: {metrics['cpu']:.1f}%")
            if metrics["memory"] > self.thresholds["memory"]:
                issues.append(f"High Memory usage: {metrics['memory']:.1f}%")
            if metrics["latency"] > self.thresholds["latency"]:
                issues.append(f"High Latency: {metrics['latency']:.3f}s")
            if metrics["error_rate"] > self.thresholds["error_rate"]:
                issues.append(f"High Error Rate: {metrics['error_rate']:.1f}%")
        except ValueError as e:
            print(f"[Monitor] Error checking {service_name}: {e}")
        return issues

# --- DIAGNOSER AGENT ---
class DiagnoserAgent:
    def __init__(self, infra_client: MockInfrastructure):
        self.infra = infra_client

    def diagnose(self, service_name: str, issues: List[str]) -> Dict:
        print(f"[Diagnoser] Analyzing {service_name} for issues: {issues}")
        logs = self.infra.get_logs(service_name, lines=20)
        log_text = "\n".join(logs)
        
        diagnosis = {
            "root_cause": "Unknown",
            "recommended_action": "escalate",
            "confidence": 0.0,
            "reasoning": "Could not determine cause from logs."
        }
        
        if any("Memory" in i for i in issues) or any("CPU" in i for i in issues):
            if "OutOfMemoryError" in log_text or "Kill process" in log_text:
                diagnosis = {
                    "root_cause": "Memory Leak / OOM",
                    "recommended_action": "restart_service",
                    "confidence": 0.95,
                    "reasoning": "Logs contain OOM errors and memory usage is high."
                }
            else:
                diagnosis = {
                    "root_cause": "High Traffic Load",
                    "recommended_action": "scale_up",
                    "confidence": 0.8,
                    "reasoning": "High resource usage without specific errors suggests load spike."
                }

        if any("Error Rate" in i for i in issues):
            if "Connection refused" in log_text or "Timeout" in log_text:
                diagnosis = {
                    "root_cause": "Dependency Failure",
                    "recommended_action": "restart_service",
                    "confidence": 0.85,
                    "reasoning": "Connection errors in logs indicate stuck connection pool or dependency issue."
                }
        
        print(f"[Diagnoser] Diagnosis: {diagnosis['root_cause']} -> {diagnosis['recommended_action']}")
        return diagnosis

# --- FORECASTER AGENT ---
class ForecasterAgent:
    def __init__(self, grafana_client: MockGrafana):
        self.grafana = grafana_client
        self.thresholds = {"cpu": 80.0, "memory": 85.0, "latency": 0.5}

    def forecast(self, service_name: str) -> Optional[Dict]:
        metrics = ["cpu", "memory", "latency"]
        for metric in metrics:
            history = self.grafana.get_metric_history(service_name, metric, window_size=10)
            if not history or len(history) < 2:
                continue
            
            slope = (history[-1] - history[0]) / len(history)
            if slope > 0.1:
                current_val = history[-1]
                limit = self.thresholds.get(metric, 100.0)
                remaining_headroom = limit - current_val
                if remaining_headroom <= 0:
                     continue
                steps_to_breach = remaining_headroom / slope
                if steps_to_breach < 20:
                    print(f"[Forecaster] PREDICTION: {service_name} {metric} will breach in {steps_to_breach:.1f} ticks (Slope: {slope:.2f})")
                    return {
                        "service": service_name,
                        "metric": metric,
                        "predicted_breach_in": steps_to_breach,
                        "current_value": current_val,
                        "slope": slope,
                        "risk_level": "High" if steps_to_breach < 10 else "Medium"
                    }
        return None

# --- PLANNER AGENT ---
class PlannerAgent:
    def create_plan(self, context: Dict) -> List[str]:
        plan = []
        if "predicted_breach_in" in context:
            metric = context.get("metric")
            print(f"[Planner] Generating PROACTIVE plan for {metric} trend...")
            plan.append("create_snapshot")
            if metric == "memory":
                plan.append("restart_service")
            elif metric == "cpu" or metric == "latency":
                plan.append("scale_up")
            plan.append("validate_health")
        elif "root_cause" in context:
            cause = context.get("root_cause")
            print(f"[Planner] Generating REACTIVE plan for {cause}...")
            plan.append("create_snapshot")
            if "Memory" in cause:
                plan.append("restart_service")
            elif "Load" in cause:
                plan.append("scale_up")
            else:
                plan.append("escalate")
            plan.append("validate_health")
        return plan

# --- VALIDATOR AGENT ---
class ValidatorAgent:
    def __init__(self, grafana_client: MockGrafana):
        self.grafana = grafana_client

    def validate(self, service_name: str) -> bool:
        try:
            metrics = self.grafana.get_all_metrics(service_name)
            if metrics["cpu"] > 90.0:
                print(f"[Validator] FAIL: CPU is {metrics['cpu']:.1f}%")
                return False
            if metrics["memory"] > 90.0:
                print(f"[Validator] FAIL: Memory is {metrics['memory']:.1f}%")
                return False
            if metrics["error_rate"] > 0.01:
                print(f"[Validator] FAIL: Error Rate is {metrics['error_rate']:.2f}%")
                return False
            print(f"[Validator] PASS: Service {service_name} is healthy.")
            return True
        except Exception as e:
            print(f"[Validator] Error during validation: {e}")
            return False

# --- SAFETY LAYER ---
class SafetyLayer:
    def __init__(self):
        self.allowed_actions = {"restart_service", "scale_up", "clear_cache", "escalate"}
        self.forbidden_actions = {"delete_database", "shutdown_cluster", "rm_rf_root"}

    def validate_action(self, action: str, service_name: str) -> bool:
        if action in self.forbidden_actions:
            print(f"[Safety] BLOCKED dangerous action: {action} on {service_name}")
            return False
        if action in self.allowed_actions:
            print(f"[Safety] Allowed action: {action} on {service_name}")
            return True
        print(f"[Safety] Unknown action blocked: {action}")
        return False

# --- EXECUTOR AGENT ---
class ExecutorAgent:
    def __init__(self, infra_client: MockInfrastructure, safety_layer: SafetyLayer, validator: ValidatorAgent):
        self.infra = infra_client
        self.safety = safety_layer
        self.validator = validator
        self.active_snapshot = None

    def execute_plan(self, service_name: str, plan: List[str]) -> bool:
        print(f"[Executor] Starting execution of plan: {plan}")
        for step in plan:
            success = self._execute_step(service_name, step)
            if not success:
                print(f"[Executor] Step {step} failed. Initiating ROLLBACK.")
                self._rollback(service_name)
                return False
        print("[Executor] Plan executed successfully.")
        return True

    def _execute_step(self, service_name: str, step: str) -> bool:
        if step == "create_snapshot":
            self.active_snapshot = self.infra.create_snapshot(service_name)
            return True
        if step == "validate_health":
            return self.validator.validate(service_name)
        if not self.safety.validate_action(step, service_name):
            return False
        if step == "restart_service":
            self.infra.restart_pod(service_name)
            return True
        elif step == "scale_up":
            status = self.infra.get_pod_status(service_name)
            current = status.get("replicas", 1)
            self.infra.scale_deployment(service_name, current + 1)
            return True
        elif step == "escalate":
            print("[Executor] Escalating to human.")
            return True
        return False

    def _rollback(self, service_name: str):
        if self.active_snapshot:
            print(f"[Executor] Rolling back to snapshot {self.active_snapshot}...")
            self.infra.restore_snapshot(self.active_snapshot)
            self.active_snapshot = None
        else:
            print("[Executor] No snapshot available for rollback!")

# --- AUDITOR AGENT ---
class AuditorAgent:
    def __init__(self, ticketing_client: MockTicketing):
        self.ticketing = ticketing_client

    def log_incident(self, service_name: str, issues: list) -> str:
        title = f"Incident: {service_name} - {', '.join(issues)}"
        description = f"Detected issues in {service_name}. Initiating automated resolution."
        ticket_id = self.ticketing.create_ticket(title, description, priority="High")
        return ticket_id

    def log_diagnosis(self, ticket_id: str, diagnosis: dict):
        comment = (
            f"Diagnosis Complete.\n"
            f"Root Cause: {diagnosis['root_cause']}\n"
            f"Reasoning: {diagnosis['reasoning']}\n"
            f"Recommended Action: {diagnosis['recommended_action']}"
        )
        self.ticketing.update_ticket(ticket_id, comment=comment)

    def log_action(self, ticket_id: str, action: str, success: bool):
        status = "Resolved" if success and action != "escalate" else "Escalated"
        if not success:
            status = "Failed"
        comment = f"Action Execution: {action}\nResult: {'Success' if success else 'Failed'}"
        self.ticketing.update_ticket(ticket_id, status=status, comment=comment)


def run_simulation_and_generate_data():
    print("=== Initializing Autonomous Reliability Command Agent ===\n")
    
    # 1. Setup
    grafana = MockGrafana()
    infra = MockInfrastructure()
    ticketing = MockTicketing()
    
    # Agents
    monitor = MonitorAgent(grafana)
    diagnoser = DiagnoserAgent(infra)
    forecaster = ForecasterAgent(grafana)
    planner = PlannerAgent()
    validator = ValidatorAgent(grafana)
    safety = SafetyLayer()
    executor = ExecutorAgent(infra, safety, validator)
    auditor = AuditorAgent(ticketing)
    
    service = "payment-service"
    data_rows = []
    start_time = datetime.now()
    
    # 2. Inject Trend
    print("--- 1. Injecting Latency Trend ---")
    grafana.inject_trend(service, "latency", 0.02)
    
    # 3. Simulation Loop (50 Ticks)
    for i in range(50):
        current_time = start_time + timedelta(seconds=i*10)
        metrics = grafana.get_all_metrics(service)
        
        row = {
            "timestamp": current_time,
            "tick": i,
            "service": service,
            "cpu": metrics["cpu"],
            "memory": metrics["memory"],
            "latency": metrics["latency"],
            "error_rate": metrics["error_rate"],
            "forecast_risk": "None",
            "action_taken": "None",
            "status": "Healthy"
        }
        
        # A. Forecast
        forecast = forecaster.forecast(service)
        if forecast:
            print(f"\n>>> PROACTIVE TRIGGER: {forecast['metric']} risk detected!")
            row["forecast_risk"] = f"High ({forecast['metric']})"
            plan = planner.create_plan(forecast)
            row["action_taken"] = f"Proactive: {plan}"
            
            success = executor.execute_plan(service, plan)
            if success:
                print(">>> MITIGATION SUCCESSFUL. Clearing anomalies.")
                grafana.clear_anomalies()
                row["status"] = "Mitigated"
        
        # B. Monitor (Reactive)
        issues = monitor.check_health(service)
        if issues and not forecast:
            print(f"\n>>> REACTIVE TRIGGER: {issues}")
            row["status"] = "Critical"
            ticket_id = auditor.log_incident(service, issues)
            diagnosis = diagnoser.diagnose(service, issues)
            auditor.log_diagnosis(ticket_id, diagnosis)
            
            plan = planner.create_plan(diagnosis)
            row["action_taken"] = f"Reactive: {plan}"
            success = executor.execute_plan(service, plan)
            auditor.log_action(ticket_id, str(plan), success)
            
            if success:
                print(">>> REACTIVE FIX SUCCESSFUL.")
                if "restart_service" in plan:
                     grafana.clear_anomalies()
        
        data_rows.append(row)
        # time.sleep(0.1) # Faster execution for notebook

    # Save Data
    df = pd.DataFrame(data_rows)
    df.to_csv('submission.csv', index=False)
    print(f"\nSuccessfully generated submission.csv with {len(df)} rows.")
    print(df.head())

run_simulation_and_generate_data()

