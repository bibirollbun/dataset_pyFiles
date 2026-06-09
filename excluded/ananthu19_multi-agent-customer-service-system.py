import json
import time
import logging
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum
import random
from abc import ABC, abstractmethod

print("Multi-Agent Customer Service System Initialized")


# ========== OBSERVABILITY SETUP ==========
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("CustomerServiceAgent")

class AgentMetrics:
    def __init__(self):
        self.requests_processed = 0
        self.average_response_time = 0
        self.errors_count = 0
    
    def record_request(self, processing_time: float):
        self.requests_processed += 1
        self.average_response_time = (
            (self.average_response_time * (self.requests_processed - 1) + processing_time) 
            / self.requests_processed
        )
    
    def record_error(self):
        self.errors_count += 1


# ========== SESSIONS & MEMORY ==========
class InMemorySessionService:
    def __init__(self):
        self.sessions: Dict[str, Dict] = {}
    
    def create_session(self, session_id: str, user_context: Dict = None):
        self.sessions[session_id] = {
            "created_at": datetime.now(),
            "user_context": user_context or {},
            "conversation_history": [],
            "state": "active"
        }
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        return self.sessions.get(session_id)
    
    def update_session(self, session_id: str, updates: Dict):
        if session_id in self.sessions:
            self.sessions[session_id].update(updates)
    
    def add_message(self, session_id: str, role: str, content: str):
        if session_id in self.sessions:
            self.sessions[session_id]["conversation_history"].append({
                "role": role,
                "content": content,
                "timestamp": datetime.now()
            })

class MemoryBank:
    def __init__(self):
        self.long_term_memory: Dict[str, List] = {}
        self.max_memory_entries = 100
    
    def store_interaction(self, user_id: str, interaction: Dict):
        if user_id not in self.long_term_memory:
            self.long_term_memory[user_id] = []
        
        self.long_term_memory[user_id].append({
            **interaction,
            "timestamp": datetime.now()
        })
        
        # Context compaction - keep only recent interactions
        if len(self.long_term_memory[user_id]) > self.max_memory_entries:
            self.long_term_memory[user_id] = self.long_term_memory[user_id][-self.max_memory_entries:]
    
    def get_user_history(self, user_id: str, limit: int = 10) -> List[Dict]:
        history = self.long_term_memory.get(user_id, [])
        return history[-limit:] if limit else history


# ========== TOOLS ==========
class Tool(ABC):
    @abstractmethod
    def execute(self, **kwargs) -> Dict:
        pass

class OrderLookupTool(Tool):
    """Custom tool for order lookup"""
    def execute(self, order_id: str) -> Dict:
        logger.info(f"Looking up order: {order_id}")
        # Simulate API call
        time.sleep(0.1)
        return {
            "status": "success",
            "order_id": order_id,
            "customer_name": "John Doe",
            "amount": 99.99,
            "status": "completed"
        }

class RefundCalculatorTool(Tool):
    """Custom tool for refund calculations"""
    def execute(self, order_amount: float, reason: str) -> Dict:
        logger.info(f"Calculating refund for amount: {order_amount}")
        # Simulate business logic
        refund_amount = order_amount * 0.8 if "defective" in reason.lower() else order_amount
        return {
            "status": "success",
            "original_amount": order_amount,
            "refund_amount": refund_amount,
            "currency": "USD"
        }

class BillingSystemTool(Tool):
    """OpenAPI-like tool for billing system integration"""
    def execute(self, customer_id: str, action: str, **kwargs) -> Dict:
        logger.info(f"Billing system action: {action} for customer: {customer_id}")
        time.sleep(0.2)
        
        if action == "get_invoice":
            return {
                "invoice_id": f"INV-{random.randint(1000, 9999)}",
                "amount": 149.99,
                "due_date": "2024-12-31",
                "status": "paid"
            }
        elif action == "cancel_subscription":
            return {
                "status": "cancelled",
                "cancellation_date": datetime.now().isoformat(),
                "refund_eligible": True
            }
        
        return {"status": "error", "message": "Unknown action"}


# ========== AGENTS ==========
class BaseAgent(ABC):
    def __init__(self, name: str):
        self.name = name
        self.metrics = AgentMetrics()
    
    @abstractmethod
    def process(self, input_data: Dict) -> Dict:
        pass

class IntentClassificationAgent(BaseAgent):
    """LLM-powered agent for intent classification"""
    def __init__(self):
        super().__init__("IntentClassificationAgent")
        self.intent_patterns = {
            "refund": ["refund", "return", "money back", "not satisfied"],
            "cancellation": ["cancel", "stop", "terminate", "end subscription"],
            "billing": ["invoice", "bill", "payment", "charge"],
            "technical_support": ["not working", "bug", "error", "technical issue"],
            "general_help": ["help", "support", "question", "assist"]
        }
    
    def process(self, input_data: Dict) -> Dict:
        start_time = time.time()
        try:
            message = input_data["message"].lower()
            session_id = input_data.get("session_id")
            
            # Simulate LLM processing
            detected_intents = []
            for intent, patterns in self.intent_patterns.items():
                if any(pattern in message for pattern in patterns):
                    detected_intents.append(intent)
            
            # Determine primary intent
            primary_intent = detected_intents[0] if detected_intents else "general_help"
            confidence = min(0.9, 0.3 + len(detected_intents) * 0.2)
            
            result = {
                "primary_intent": primary_intent,
                "detected_intents": detected_intents,
                "confidence": confidence,
                "requires_human": confidence < 0.5
            }
            
            processing_time = time.time() - start_time
            self.metrics.record_request(processing_time)
            logger.info(f"Intent classified: {primary_intent} (confidence: {confidence})")
            
            return result
            
        except Exception as e:
            self.metrics.record_error()
            logger.error(f"Intent classification error: {e}")
            return {"primary_intent": "general_help", "confidence": 0.1, "requires_human": True}

class RoutingAgent(BaseAgent):
    """Agent that routes requests to appropriate specialized agents"""
    def __init__(self):
        super().__init__("RoutingAgent")
        self.specialized_agents = {
            "refund": "RefundProcessingAgent",
            "cancellation": "CancellationAgent", 
            "billing": "BillingAgent",
            "technical_support": "TechnicalSupportAgent"
        }
    
    def process(self, input_data: Dict) -> Dict:
        intent_result = input_data["intent_result"]
        primary_intent = intent_result["primary_intent"]
        
        target_agent = self.specialized_agents.get(primary_intent, "GeneralSupportAgent")
        
        return {
            "target_agent": target_agent,
            "routing_reason": f"Routed to {target_agent} for {primary_intent}",
            "confidence": intent_result["confidence"]
        }

class RefundProcessingAgent(BaseAgent):
    """Specialized agent for refund processing with long-running operations"""
    def __init__(self, tools: Dict[str, Tool]):
        super().__init__("RefundProcessingAgent")
        self.tools = tools
        self.paused_operations: Dict[str, Dict] = {}
    
    def process(self, input_data: Dict) -> Dict:
        session_id = input_data["session_id"]
        message = input_data["message"]
        
        # Check if this is a continuation of paused operation
        if session_id in self.paused_operations:
            return self._resume_operation(session_id, input_data)
        
        return self._start_refund_process(session_id, message)
    
    def _start_refund_process(self, session_id: str, message: str) -> Dict:
        """Start refund process - could be paused waiting for user input"""
        logger.info(f"Starting refund process for session: {session_id}")
        
        # Extract order ID from message (simplified)
        order_id = None
        if "order" in message.lower():
            words = message.split()
            for i, word in enumerate(words):
                if word.lower() == "order" and i + 1 < len(words):
                    order_id = words[i + 1]
                    break
        
        if not order_id:
            # Pause operation until we get order ID
            self.paused_operations[session_id] = {
                "step": "awaiting_order_id",
                "context": {"message": message}
            }
            return {
                "action": "pause",
                "message": "I'd be happy to help with your refund. Could you please provide your order ID?",
                "next_step": "order_id_verification"
            }
        
        # Continue with order lookup
        order_info = self.tools["order_lookup"].execute(order_id=order_id)
        refund_calc = self.tools["refund_calculator"].execute(
            order_amount=order_info["amount"], 
            reason="customer request"
        )
        
        return {
            "action": "complete",
            "message": f"I've processed your refund request. Order {order_id} will be refunded ${refund_calc['refund_amount']}.",
            "refund_details": refund_calc,
            "next_step": None
        }
    
    def _resume_operation(self, session_id: str, input_data: Dict) -> Dict:
        """Resume a paused refund operation"""
        operation = self.paused_operations.pop(session_id, {})
        
        if operation["step"] == "awaiting_order_id":
            order_id = input_data["message"].strip()
            order_info = self.tools["order_lookup"].execute(order_id=order_id)
            refund_calc = self.tools["refund_calculator"].execute(
                order_amount=order_info["amount"], 
                reason="customer request"
            )
            
            return {
                "action": "complete", 
                "message": f"Thank you! I've processed your refund. ${refund_calc['refund_amount']} will be refunded.",
                "refund_details": refund_calc
            }
        
        return {"action": "error", "message": "Unable to resume operation"}

class BillingAgent(BaseAgent):
    """Specialized agent for billing inquiries"""
    def __init__(self, tools: Dict[str, Tool]):
        super().__init__("BillingAgent")
        self.tools = tools
    
    def process(self, input_data: Dict) -> Dict:
        message = input_data["message"].lower()
        session_id = input_data["session_id"]
        
        if "invoice" in message or "bill" in message:
            invoice_info = self.tools["billing_system"].execute(
                customer_id=session_id, 
                action="get_invoice"
            )
            return {
                "action": "complete",
                "message": f"Here's your invoice information: {invoice_info}",
                "data": invoice_info
            }
        
        return {
            "action": "complete",
            "message": "I can help with billing inquiries. Please ask about invoices, payments, or billing issues."
        }

class TechnicalSupportAgent(BaseAgent):
    """Agent for technical support issues"""
    def __init__(self):
        super().__init__("TechnicalSupportAgent")
    
    def process(self, input_data: Dict) -> Dict:
        message = input_data["message"]
        
        # Simulate technical issue analysis
        if "not working" in message.lower():
            return {
                "action": "escalate",
                "message": "I understand you're experiencing technical issues. Let me connect you with our technical team.",
                "priority": "high",
                "assigned_team": "technical_support"
            }
        
        return {
            "action": "complete",
            "message": "For technical support, please describe the issue you're experiencing in detail."
        }

class GeneralSupportAgent(BaseAgent):
    """Fallback agent for general support queries"""
    def __init__(self):
        super().__init__("GeneralSupportAgent")
    
    def process(self, input_data: Dict) -> Dict:
        return {
            "action": "complete",
            "message": "Thank you for reaching out. How can I assist you today?",
            "suggested_actions": ["Check order status", "Billing inquiry", "Technical support", "Cancel subscription"]
        }


# ========== COORDINATOR & A2A PROTOCOL ==========
class CustomerServiceCoordinator:
    """Main coordinator implementing A2A protocol"""
    def __init__(self):
        # Initialize tools
        self.tools = {
            "order_lookup": OrderLookupTool(),
            "refund_calculator": RefundCalculatorTool(),
            "billing_system": BillingSystemTool()
        }
        
        # Initialize agents (sequential and parallel capable)
        self.intent_agent = IntentClassificationAgent()
        self.routing_agent = RoutingAgent()
        self.specialized_agents = {
            "RefundProcessingAgent": RefundProcessingAgent(self.tools),
            "BillingAgent": BillingAgent(self.tools),
            "TechnicalSupportAgent": TechnicalSupportAgent(),
            "GeneralSupportAgent": GeneralSupportAgent()
        }
        
        # Session and memory management
        self.session_service = InMemorySessionService()
        self.memory_bank = MemoryBank()
        
        # Agent evaluation metrics
        self.evaluation_results = []
    
    def process_message(self, user_id: str, message: str) -> Dict:
        """Main entry point for processing customer messages"""
        start_time = time.time()
        
        # Session management
        session_id = f"user_{user_id}"
        if not self.session_service.get_session(session_id):
            self.session_service.create_session(session_id, {"user_id": user_id})
        
        # Store in conversation history
        self.session_service.add_message(session_id, "user", message)
        
        # SEQUENTIAL AGENTS: Intent -> Routing -> Specialized Agent
        try:
            # Step 1: Intent classification
            intent_result = self.intent_agent.process({
                "message": message,
                "session_id": session_id
            })
            
            # Step 2: Routing
            routing_result = self.routing_agent.process({
                "intent_result": intent_result,
                "session_id": session_id
            })
            
            # Step 3: Specialized agent processing
            target_agent_name = routing_result["target_agent"]
            specialized_agent = self.specialized_agents[target_agent_name]
            
            agent_result = specialized_agent.process({
                "message": message,
                "session_id": session_id,
                "intent_result": intent_result,
                "routing_result": routing_result
            })
            
            # Store agent response
            if "message" in agent_result:
                self.session_service.add_message(session_id, "agent", agent_result["message"])
            
            # Store in long-term memory
            self.memory_bank.store_interaction(user_id, {
                "message": message,
                "response": agent_result,
                "intent": intent_result["primary_intent"],
                "agent_used": target_agent_name
            })
            
            # Compile final response
            final_response = {
                "response": agent_result.get("message", "I apologize, but I encountered an error."),
                "session_id": session_id,
                "agent_chain": [
                    self.intent_agent.name,
                    self.routing_agent.name,
                    target_agent_name
                ],
                "processing_time": time.time() - start_time,
                "requires_human": intent_result.get("requires_human", False) or agent_result.get("action") == "escalate"
            }
            
            # Agent evaluation
            self._evaluate_agent_performance(user_id, intent_result, agent_result)
            
            logger.info(f"Request processed successfully in {final_response['processing_time']:.2f}s")
            return final_response
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return {
                "response": "I apologize, but I'm experiencing technical difficulties. Please try again later.",
                "error": str(e),
                "requires_human": True
            }
    
    def _evaluate_agent_performance(self, user_id: str, intent_result: Dict, agent_result: Dict):
        """Simple agent evaluation based on confidence and action taken"""
        evaluation = {
            "user_id": user_id,
            "timestamp": datetime.now(),
            "intent_confidence": intent_result.get("confidence", 0),
            "agent_action": agent_result.get("action", "unknown"),
            "successful": agent_result.get("action") in ["complete", "pause"]
        }
        self.evaluation_results.append(evaluation)
    
    def get_agent_metrics(self) -> Dict:
        """Get comprehensive metrics for observability"""
        metrics = {}
        
        # Add intent agent metrics
        metrics[self.intent_agent.name] = {
            "requests_processed": self.intent_agent.metrics.requests_processed,
            "average_response_time": self.intent_agent.metrics.average_response_time,
            "errors_count": self.intent_agent.metrics.errors_count
        }
        
        # Add routing agent metrics
        metrics[self.routing_agent.name] = {
            "requests_processed": self.routing_agent.metrics.requests_processed,
            "average_response_time": self.routing_agent.metrics.average_response_time,
            "errors_count": self.routing_agent.metrics.errors_count
        }
        
        # Add specialized agents metrics
        for agent_name, agent in self.specialized_agents.items():
            metrics[agent_name] = {
                "requests_processed": agent.metrics.requests_processed,
                "average_response_time": agent.metrics.average_response_time,
                "errors_count": agent.metrics.errors_count
            }
            
        return metrics


# ========== DEMONSTRATION ==========
def demo_multi_agent_system():
    """Demonstrate the multi-agent system with various scenarios"""
    print("ğŸš€ DEMONSTRATING MULTI-AGENT CUSTOMER SERVICE SYSTEM\n")
    
    coordinator = CustomerServiceCoordinator()
    test_cases = [
        {"user": "customer_123", "message": "I want a refund for my order"},
        {"user": "customer_123", "message": "Order 456789"},  # Continuation
        {"user": "customer_456", "message": "I need help with my invoice"},
        {"user": "customer_789", "message": "The application is not working properly"},
        {"user": "customer_999", "message": "Hello, I need some help"}
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"TEST CASE {i}: {test_case['message']}")
        print(f"{'='*60}")
        
        response = coordinator.process_message(test_case['user'], test_case['message'])
        
        print(f"USER: {test_case['message']}")
        print(f"AGENT: {response['response']}")
        print(f"Agent Chain: {', '.join(response['agent_chain'])}")
        print(f"Processing Time: {response['processing_time']:.2f}s")
        print(f"Requires Human: {response['requires_human']}")
    
    # Display metrics and observability data
    print(f"\n{'='*60}")
    print("ğŸ“Š OBSERVABILITY METRICS")
    print(f"{'='*60}")
    metrics = coordinator.get_agent_metrics()
    for agent, data in metrics.items():
        print(f"{agent}: {data['requests_processed']} requests, "
              f"avg {data['average_response_time']:.3f}s, "
              f"{data['errors_count']} errors")
    
    # Show memory bank usage
    print(f"\n{'='*60}")
    print("ğŸ’¾ MEMORY BANK SAMPLE")
    print(f"{'='*60}")
    user_history = coordinator.memory_bank.get_user_history("customer_123", limit=2)
    for interaction in user_history:
        print(f"Message: {interaction['message'][:50]}...")
        print(f"Intent: {interaction['intent']}, Agent: {interaction['agent_used']}")
        print("-" * 40)

# Run demonstration
if __name__ == "__main__":
    demo_multi_agent_system()


# ========== EXTENDED DEMONSTRATION ==========
def extended_demo_multi_agent_system():
    """Extended demonstration with more diverse scenarios"""
    print("ğŸš€ EXTENDED DEMONSTRATION - MULTI-AGENT CUSTOMER SERVICE SYSTEM\n")
    
    coordinator = CustomerServiceCoordinator()
    
    # More diverse test cases showing different features
    extended_test_cases = [
        # Scenario 1: Complex refund with missing information (pause/resume)
        {"user": "premium_user_001", "message": "I want to return my product and get my money back"},
        {"user": "premium_user_001", "message": "My order number is ORD-789123"},
        
        # Scenario 2: Billing inquiry with specific request
        {"user": "business_customer_002", "message": "Can you send me last month's invoice?"},
        
        # Scenario 3: Technical issue requiring escalation
        {"user": "mobile_user_003", "message": "The app keeps crashing when I try to checkout"},
        
        # Scenario 4: Multiple intents in one message
        {"user": "confused_customer_004", "message": "I need help with my bill and also want to cancel my subscription"},
        
        # Scenario 5: Low confidence intent (might require human)
        {"user": "vague_customer_005", "message": "Something is wrong with my account"},
        
        # Scenario 6: Order status inquiry (not in intent patterns - should go to general)
        {"user": "anxious_customer_006", "message": "Where is my order? It's been 5 days"},
        
        # Scenario 7: Complex technical issue
        {"user": "tech_savvy_007", "message": "Getting 404 errors on API endpoints and database connection timeouts"},
        
        # Scenario 8: Payment issue
        {"user": "payment_issue_008", "message": "My credit card was charged twice for the same invoice"},
    ]
    
    print("ğŸ“‹ TEST SCENARIOS EXECUTION")
    print("=" * 80)
    
    for i, test_case in enumerate(extended_test_cases, 1):
        print(f"\n{'â�¡ï¸� ' * 3} SCENARIO {i}: User '{test_case['user']}'")
        print(f"ğŸ’¬ USER INPUT: '{test_case['message']}'")
        print("-" * 80)
        
        start_time = time.time()
        response = coordinator.process_message(test_case['user'], test_case['message'])
        total_time = time.time() - start_time
        
        print(f"ğŸ¤– AGENT RESPONSE: {response['response']}")
        
        # Safely handle agent_chain with default value
        agent_chain = response.get('agent_chain', ['Unknown Agent'])
        print(f"ğŸ”— AGENT CHAIN: {' â†’ '.join(agent_chain)}")
        print(f"â�±ï¸�  PROCESSING TIME: {total_time:.3f}s")
        print(f"ğŸ‘¨â€�ğŸ’¼ REQUIRES HUMAN: {response.get('requires_human', False)}")
        
        # Show additional context for specific response types
        if "pause" in response.get('response', '').lower():
            print(f"â�¸ï¸�  OPERATION PAUSED: Waiting for user input")
        
        if "escalate" in response.get('response', '').lower():
            print(f"ğŸš¨ ESCALATION: {response.get('priority', 'medium')} priority to {response.get('assigned_team', 'support')}")
        
        # Show any additional data if present
        if 'refund_details' in response:
            refund_info = response['refund_details']
            print(f"ğŸ’° REFUND DETAILS: ${refund_info.get('refund_amount', 0)} (original: ${refund_info.get('original_amount', 0)})")
        
        if 'data' in response:
            print(f"ğŸ“Š DATA RETURNED: {response['data']}")
    
    # ========== ADVANCED OBSERVABILITY DEMONSTRATION ==========
    print(f"\n{'ğŸ“Š' * 20} ADVANCED OBSERVABILITY {'ğŸ“Š' * 20}")
    
    # Detailed metrics analysis
    metrics = coordinator.get_agent_metrics()
    
    print(f"\nğŸ“ˆ AGENT PERFORMANCE METRICS:")
    print("=" * 100)
    print(f"{'AGENT NAME':<25} | {'REQUESTS':<8} | {'AVG TIME':<10} | {'ERRORS':<6} | {'SUCCESS RATE':<12}")
    print("-" * 100)
    
    for agent_name, data in metrics.items():
        success_rate = ((data['requests_processed'] - data['errors_count']) / data['requests_processed'] * 100) if data['requests_processed'] > 0 else 0
        print(f"{agent_name:<25} | {data['requests_processed']:<8} | {data['average_response_time']:>8.3f}s | {data['errors_count']:<6} | {success_rate:>10.1f}%")
    
    # ========== MEMORY BANK ANALYSIS ==========
    print(f"\n{'ğŸ’¾' * 20} MEMORY BANK ANALYSIS {'ğŸ’¾' * 20}")
    
    sample_users = ["premium_user_001", "business_customer_002", "mobile_user_003"]
    
    for user_id in sample_users:
        history = coordinator.memory_bank.get_user_history(user_id)
        if history:
            print(f"\nğŸ“� USER {user_id} - INTERACTION HISTORY ({len(history)} interactions):")
            print("-" * 80)
            for j, interaction in enumerate(history[-3:], 1):  # Show last 3 interactions
                print(f"  {j}. Message: '{interaction['message'][:60]}...'")
                print(f"     Intent: {interaction['intent']} | Agent: {interaction['agent_used']}")
                print(f"     Time: {interaction['timestamp'].strftime('%H:%M:%S')}")
        else:
            print(f"\nğŸ“� USER {user_id} - No interaction history found")
    
    # ========== SESSION STATE ANALYSIS ==========
    print(f"\n{'ğŸ”�' * 20} SESSION STATE ANALYSIS {'ğŸ”�' * 20}")
    
    active_sessions = coordinator.session_service.sessions
    print(f"\nActive Sessions: {len(active_sessions)}")
    
    for session_id, session_data in list(active_sessions.items())[:3]:  # Show first 3 sessions
        print(f"\nğŸ�¯ SESSION: {session_id}")
        print(f"   Created: {session_data['created_at'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Messages: {len(session_data['conversation_history'])}")
        print(f"   State: {session_data['state']}")
        
        # Show last message
        if session_data['conversation_history']:
            last_msg = session_data['conversation_history'][-1]
            print(f"   Last: {last_msg['role']}: '{last_msg['content'][:50]}...'")
    
    # ========== AGENT EVALUATION INSIGHTS ==========
    print(f"\n{'ğŸ�¯' * 20} AGENT EVALUATION INSIGHTS {'ğŸ�¯' * 20}")
    
    evaluations = coordinator.evaluation_results
    if evaluations:
        successful_agents = sum(1 for e in evaluations if e.get('successful', False))
        avg_confidence = sum(e.get('intent_confidence', 0) for e in evaluations) / len(evaluations)
        human_escalations = sum(1 for e in evaluations if e.get('requires_human', False))
        
        print(f"\nğŸ“Š Overall Performance:")
        print(f"   Total Interactions: {len(evaluations)}")
        print(f"   Successful Resolutions: {successful_agents} ({successful_agents/len(evaluations)*100:.1f}%)")
        print(f"   Average Intent Confidence: {avg_confidence:.2f}")
        print(f"   Human Escalation Rate: {human_escalations/len(evaluations)*100:.1f}%")
    else:
        print(f"\nğŸ“Š No evaluation data available yet")
    
    # ========== LONG-RUNNING OPERATIONS STATUS ==========
    print(f"\n{'â�³' * 20} LONG-RUNNING OPERATIONS {'â�³' * 20}")
    
    refund_agent = coordinator.specialized_agents["RefundProcessingAgent"]
    if hasattr(refund_agent, 'paused_operations') and refund_agent.paused_operations:
        print(f"\nâ�¸ï¸�  PAUSED REFUND OPERATIONS: {len(refund_agent.paused_operations)}")
        for session_id, operation in refund_agent.paused_operations.items():
            print(f"   ğŸ“‹ {session_id}: Step '{operation.get('step', 'unknown')}' - Waiting for: {operation.get('context', {})}")
    else:
        print(f"\nâœ… No paused operations - all requests completed successfully!")
    
    # ========== CONTEXT COMPACTION DEMONSTRATION ==========
    print(f"\n{'ğŸ§¹' * 20} CONTEXT COMPACTION {'ğŸ§¹' * 20}")
    
    # Simulate adding many interactions to show compaction
    test_user = "high_volume_user_999"
    for i in range(150):  # More than max_memory_entries (100)
        coordinator.memory_bank.store_interaction(test_user, {
            "message": f"Test message {i}",
            "response": {"action": "complete"},
            "intent": "general_help",
            "agent_used": "GeneralSupportAgent"
        })
    
    final_history = coordinator.memory_bank.get_user_history(test_user)
    print(f"\nğŸ“š Context Compaction Example:")
    print(f"   Stored 150 interactions for user {test_user}")
    print(f"   After compaction: {len(final_history)} interactions kept")
    print(f"   Compaction rate: {(150 - len(final_history))/150*100:.1f}% reduction")
    
    # ========== TOOL USAGE STATISTICS ==========
    print(f"\n{'ğŸ› ï¸� ' * 20} TOOL USAGE STATISTICS {'ğŸ› ï¸� ' * 20}")
    
    tool_usage = {}
    for user_id, history in coordinator.memory_bank.long_term_memory.items():
        for interaction in history:
            agent_used = interaction.get('agent_used', '')
            if agent_used not in tool_usage:
                tool_usage[agent_used] = 0
            tool_usage[agent_used] += 1
    
    print(f"\nğŸ”§ Agent Usage Distribution:")
    for agent, count in sorted(tool_usage.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / sum(tool_usage.values())) * 100
        print(f"   {agent}: {count} times ({percentage:.1f}%)")
    
    print(f"\n{'ğŸ�‰' * 20} DEMONSTRATION COMPLETE {'ğŸ�‰' * 20}")

# Let's also fix the main coordinator to ensure agent_chain is always present
def fixed_process_message(self, user_id: str, message: str) -> Dict:
    """Main entry point for processing customer messages"""
    start_time = time.time()
    
    # Session management
    session_id = f"user_{user_id}"
    if not self.session_service.get_session(session_id):
        self.session_service.create_session(session_id, {"user_id": user_id})
    
    # Store in conversation history
    self.session_service.add_message(session_id, "user", message)
    
    # SEQUENTIAL AGENTS: Intent -> Routing -> Specialized Agent
    try:
        # Step 1: Intent classification
        intent_result = self.intent_agent.process({
            "message": message,
            "session_id": session_id
        })
        
        # Step 2: Routing
        routing_result = self.routing_agent.process({
            "intent_result": intent_result,
            "session_id": session_id
        })
        
        # Step 3: Specialized agent processing
        target_agent_name = routing_result["target_agent"]
        specialized_agent = self.specialized_agents[target_agent_name]
        
        agent_result = specialized_agent.process({
            "message": message,
            "session_id": session_id,
            "intent_result": intent_result,
            "routing_result": routing_result
        })
        
        # Store agent response
        if "message" in agent_result:
            self.session_service.add_message(session_id, "agent", agent_result["message"])
        
        # Store in long-term memory
        self.memory_bank.store_interaction(user_id, {
            "message": message,
            "response": agent_result,
            "intent": intent_result["primary_intent"],
            "agent_used": target_agent_name
        })
        
        # Ensure agent_chain is always present
        agent_chain = [
            self.intent_agent.name,
            self.routing_agent.name,
            target_agent_name
        ]
        
        # Compile final response with guaranteed fields
        final_response = {
            "response": agent_result.get("message", "I apologize, but I encountered an error."),
            "session_id": session_id,
            "agent_chain": agent_chain,
            "processing_time": time.time() - start_time,
            "requires_human": intent_result.get("requires_human", False) or agent_result.get("action") == "escalate"
        }
        
        # Copy additional fields from agent_result
        for key, value in agent_result.items():
            if key not in final_response:  # Don't overwrite existing keys
                final_response[key] = value
        
        # Agent evaluation
        self._evaluate_agent_performance(user_id, intent_result, agent_result)
        
        logger.info(f"Request processed successfully in {final_response['processing_time']:.2f}s")
        return final_response
        
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        return {
            "response": "I apologize, but I'm experiencing technical difficulties. Please try again later.",
            "error": str(e),
            "requires_human": True,
            "agent_chain": ["ErrorHandler"]
        }

# Replace the original method with the fixed one
CustomerServiceCoordinator.process_message = fixed_process_message

# Run the extended demonstration
if __name__ == "__main__":
    extended_demo_multi_agent_system()




