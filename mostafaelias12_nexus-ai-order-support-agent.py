
!pip install -q -U google-genai google-cloud-aiplatform google-adk gradio || true
print("âœ… NEXUS AI â€” Environment setup complete")



import os

try:
    from kaggle_secrets import UserSecretsClient
    os.environ["GOOGLE_API_KEY"] = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    print("âœ… API Key loaded from Kaggle Secrets")
except:
    print("âš ï¸� No API Key found â€” using local-only mode" if "GOOGLE_API_KEY" not in os.environ else "âœ… API Key loaded from environment")



import random, re, json, logging
from datetime import datetime, timedelta
from uuid import uuid4

random.seed(42)
ADK_AVAILABLE = GRADIO_AVAILABLE = False

try:
    from google.adk.agents import LlmAgent
    from google.adk.runners import InMemoryRunner
    ADK_AVAILABLE = True
    print("âœ… Google ADK available")
except:
    print("âš ï¸� Google ADK not available â€” using LocalRunner")

try:
    import gradio as gr
    GRADIO_AVAILABLE = True
    print("âœ… Gradio available")
except:
    print("âš ï¸� Gradio not available")

now_iso = lambda: datetime.utcnow().isoformat()
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger("NEXUS-AI")




# Order Database (ORD-01 to ORD-99)
order_db = {}
base_date = datetime.utcnow()
for i in range(1, 100):
    oid = f"ORD-{i:02d}"
    order_db[oid] = {
        "order_id": oid,
        "status": random.choice(["Shipped", "Processing", "Delivered", "Cancelled", "Pending"]),
        "item": random.choice(["Wireless Mouse", "Gaming Monitor", "Mechanical Keyboard", "USB-C Hub", "Laptop Stand"]),
        "delivery_date": (base_date + timedelta(days=random.randint(1, 30))).strftime("%Y-%m-%d")
    }

refund_db = {}
created_orders = {}

# Tool Functions
def check_order_status(order_id: str) -> dict:
    oid = str(order_id).strip().upper()
    if not re.match(r"^ORD-\d{2}$", oid):
        return {"error": "Invalid format. Use ORD-XX (01-99)"}
    order = order_db.get(oid)
    return order if order else {"error": f"{oid} not found"}

def batch_check_orders(order_ids) -> dict:
    tokens = re.findall(r"ORD-\d{2}", str(order_ids).upper()) if isinstance(order_ids, str) else [str(x).upper() for x in order_ids]
    return {t: check_order_status(t) for t in tokens}

def request_refund(order_id: str, reason: str = "") -> dict:
    oid = str(order_id).strip().upper()
    if not re.match(r"^ORD-\d{2}$", oid):
        return {"error": "Invalid order ID format"}
    ticket_id = f"RFD-{uuid4().hex[:8].upper()}"
    refund_db[ticket_id] = {"ticket_id": ticket_id, "order_id": oid, "reason": reason or "No reason", "status": "Pending"}
    return {"ticket_id": ticket_id, "message": f"Refund request created for {oid}", "status": "Pending"}

def create_order(item_name: str, eta_days: int = 7) -> dict:
    oid = f"ORD-{100 + len(created_orders) + 1}"
    delivery = (datetime.utcnow() + timedelta(days=eta_days)).strftime("%Y-%m-%d")
    created_orders[oid] = {"order_id": oid, "status": "Processing", "item": item_name, "delivery_date": delivery}
    return created_orders[oid]

tool_registry = {
    "check_order_status": check_order_status,
    "batch_check_orders": batch_check_orders,
    "request_refund": request_refund,
    "create_order": create_order
}

print(f"âœ… Database: {len(order_db)} orders | Tools: {list(tool_registry.keys())}")




class NexusAgent:
    """
    NEXUS AI - Intelligent Order Support Agent
    
    Features:
    - Session-based conversation memory
    - Natural language intent detection
    - Order ID extraction from text
    - Multi-tool routing (check, refund, create, cancel, help)
    - Context-aware responses
    - Conversation history tracking
    """
    
    def __init__(self, tools, memory_store=None):
        self.tools = tools
        self.memory_store = memory_store if memory_store is not None else {}
        
        # Intent detection patterns (expanded)
        self.intent_patterns = {
            "check": re.compile(r"\b(check|status|track|where|find|lookup|show|get|my order)\b", re.I),
            "refund": re.compile(r"\b(refund|return|money back|cancel and refund|want refund)\b", re.I),
            "create": re.compile(r"\b(order new|buy|create order|place order|new order|purchase)\b", re.I),
            "cancel": re.compile(r"\b(cancel|stop|abort|remove order)\b", re.I),
            "help": re.compile(r"\b(help|what can you|how to|support|assist|guide)\b", re.I),
            "greeting": re.compile(r"\b(hi|hello|hey|good morning|good afternoon|good evening|howdy)\b", re.I),
            "thanks": re.compile(r"\b(thanks|thank you|thx|appreciate|grateful)\b", re.I),
            "list": re.compile(r"\b(list|show all|all orders|my orders)\b", re.I)
        }
        
        # Response templates
        self.responses = {
            "greeting": "ğŸ‘‹ Hello! I'm **NEXUS AI**, your order support assistant.\n\n**I can help you with:**\nâ€¢ ğŸ“¦ Check order status â†’ *'Check ORD-05'*\nâ€¢ ğŸ’° Request refunds â†’ *'Refund for ORD-12'*\nâ€¢ ğŸ›’ Create orders â†’ *'Order a gaming mouse'*\nâ€¢ â�Œ Cancel orders â†’ *'Cancel ORD-08'*\n\nHow can I assist you today?",
            
            "thanks": "You're welcome! ğŸ˜Š Is there anything else I can help you with?",
            
            "help": "ğŸ“š **NEXUS AI Help Guide**\n\n**Available Commands:**\nâ€¢ `Check ORD-XX` â€” Get order status\nâ€¢ `Check ORD-05 and ORD-07` â€” Check multiple orders\nâ€¢ `Refund for ORD-XX` â€” Request a refund\nâ€¢ `Order a [item name]` â€” Create new order\nâ€¢ `Cancel ORD-XX` â€” Cancel an order\n\n**Tips:**\nâ€¢ Order IDs are in format ORD-01 to ORD-99\nâ€¢ You can check multiple orders at once\nâ€¢ Just type naturally, I'll understand!",
            
            "unknown": "ğŸ¤” I'm not sure what you mean. Try:\nâ€¢ *'Check ORD-05'* â€” Track order\nâ€¢ *'Refund ORD-12'* â€” Get refund\nâ€¢ *'Order a keyboard'* â€” New order\nâ€¢ *'Help'* â€” See all commands",
            
            "no_order_id": "ğŸ’¡ Please provide the order ID (e.g., ORD-05) so I can help you.",
            
            "list_sample": "ğŸ“‹ **Sample Orders Available:**\nORD-01, ORD-05, ORD-10, ORD-15, ORD-20, ORD-25...\n\n*Try: 'Check ORD-05' to see details*"
        }
    
    def _extract_order_ids(self, text):
        """Extract all ORD-XX patterns from text"""
        return re.findall(r"ORD-\d{2}", text.upper())
    
    def _detect_intent(self, text):
        """Detect user intent from message"""
        for intent, pattern in self.intent_patterns.items():
            if pattern.search(text):
                return intent
        # Fallback: if order ID present, assume check
        if re.search(r"ORD-\d{2}", text.upper()):
            return "check"
        return "unknown"
    
    def _remember(self, session_id, role, text):
        """Store conversation in memory"""
        session = self.memory_store.setdefault(session_id, {
            "history": [],
            "created_at": now_iso(),
            "last_intent": None,
            "last_order_ids": []
        })
        session["history"].append({
            "role": role,
            "text": text,
            "time": now_iso()
        })
    
    def _get_context(self, session_id):
        """Get previous context for follow-up handling"""
        session = self.memory_store.get(session_id, {})
        return {
            "last_intent": session.get("last_intent"),
            "last_order_ids": session.get("last_order_ids", []),
            "history_count": len(session.get("history", []))
        }
    
    def _update_context(self, session_id, intent, order_ids):
        """Update session context"""
        if session_id in self.memory_store:
            self.memory_store[session_id]["last_intent"] = intent
            self.memory_store[session_id]["last_order_ids"] = order_ids
    
    def _handle_check(self, ids):
        """Handle order status check"""
        if not ids:
            return {"text": self.responses["no_order_id"], "tool": None, "result": None}
        
        batch_result = self.tools["batch_check_orders"](ids)
        lines = ["ğŸ“¦ **Order Status Report:**\n"]
        
        for oid in ids:
            r = batch_result.get(oid, {"error": "No response"})
            if "error" in r:
                lines.append(f"â�Œ **{oid}**: {r['error']}")
            else:
                status_emoji = {"Shipped": "ğŸšš", "Delivered": "âœ…", "Processing": "â�³", "Cancelled": "â�Œ", "Pending": "ğŸ•�"}.get(r['status'], "ğŸ“¦")
                lines.append(f"{status_emoji} **{oid}**")
                lines.append(f"   â€¢ Status: {r['status']}")
                lines.append(f"   â€¢ Item: {r['item']}")
                lines.append(f"   â€¢ ETA: {r['delivery_date']}\n")
        
        return {"text": "\n".join(lines), "tool": "batch_check_orders", "result": batch_result}
    
    def _handle_refund(self, ids, user_text):
        """Handle refund requests"""
        if not ids:
            return {"text": "ğŸ’° I can help with refunds!\n\n" + self.responses["no_order_id"], "tool": None, "result": None}
        
        # Extract reason if provided
        reason_match = re.search(r"(?:because|reason|due to)\s+(.+)", user_text, re.I)
        reason = reason_match.group(1).strip()[:200] if reason_match else "Requested via chat"
        
        results = {}
        lines = ["ğŸ’° **Refund Request Results:**\n"]
        
        for oid in ids:
            res = self.tools["request_refund"](oid, reason=reason)
            results[oid] = res
            if "error" in res:
                lines.append(f"â�Œ **{oid}**: {res['error']}")
            else:
                lines.append(f"âœ… **{oid}**: Ticket `{res['ticket_id']}` created")
                lines.append(f"   â€¢ Status: {res['status']}\n")
        
        return {"text": "\n".join(lines), "tool": "request_refund", "result": results}
    
    def _handle_create(self, user_text):
        """Handle order creation"""
        # Extract item name
        patterns = [
            r"(?:order|buy|purchase|create order|place order|new order)\s+(?:a\s+|an\s+)?(.+)",
            r"(?:i want|i need|get me)\s+(?:a\s+|an\s+)?(.+)"
        ]
        
        item = "Unknown Item"
        for pattern in patterns:
            match = re.search(pattern, user_text, re.I)
            if match:
                item = match.group(1).strip()[:100]
                # Clean up common trailing words
                item = re.sub(r"\s+(please|pls|asap|now|today)$", "", item, flags=re.I)
                break
        
        created = self.tools["create_order"](item_name=item, eta_days=7)
        
        resp = f"""ğŸ›’ **Order Created Successfully!**

ğŸ“¦ **Order ID:** `{created['order_id']}`
ğŸ“� **Item:** {created['item']}
ğŸ“… **Estimated Delivery:** {created['delivery_date']}
â�³ **Status:** {created['status']}

*Track your order anytime with: 'Check {created['order_id']}'*"""
        
        return {"text": resp, "tool": "create_order", "result": created}
    
    def _handle_cancel(self, ids):
        """Handle order cancellation (demo - marks as cancelled)"""
        if not ids:
            return {"text": "â�Œ To cancel an order, please provide the order ID.\n\n" + self.responses["no_order_id"], "tool": None, "result": None}
        
        lines = ["â�Œ **Cancellation Results:**\n"]
        results = {}
        
        for oid in ids:
            # Check if order exists first
            order_info = self.tools["check_order_status"](oid)
            if "error" in order_info:
                lines.append(f"â�Œ **{oid}**: {order_info['error']}")
                results[oid] = order_info
            elif order_info.get("status") == "Delivered":
                lines.append(f"â�Œ **{oid}**: Cannot cancel - already delivered")
                results[oid] = {"error": "Already delivered"}
            elif order_info.get("status") == "Cancelled":
                lines.append(f"âš ï¸� **{oid}**: Already cancelled")
                results[oid] = {"status": "Already cancelled"}
            else:
                # Demo: Create refund as cancellation
                refund = self.tools["request_refund"](oid, reason="Order cancelled by customer")
                lines.append(f"âœ… **{oid}**: Cancelled successfully")
                lines.append(f"   â€¢ Refund Ticket: `{refund['ticket_id']}`\n")
                results[oid] = {"status": "Cancelled", "refund": refund}
        
        return {"text": "\n".join(lines), "tool": "cancel", "result": results}
    
    def run(self, session_id, user_text):
        """Main agent execution"""
        logger.info(f"[NEXUS] Session: {session_id} | Input: '{user_text[:50]}...'")
        
        # Store user message
        self._remember(session_id, "user", user_text)
        
        # Get context and detect intent
        context = self._get_context(session_id)
        intent = self._detect_intent(user_text)
        ids = self._extract_order_ids(user_text)
        
        # Update context
        self._update_context(session_id, intent, ids)
        
        # Route to handlers
        if intent == "greeting":
            result = {"text": self.responses["greeting"], "tool": None, "result": None}
        
        elif intent == "thanks":
            result = {"text": self.responses["thanks"], "tool": None, "result": None}
        
        elif intent == "help":
            result = {"text": self.responses["help"], "tool": None, "result": None}
        
        elif intent == "list":
            result = {"text": self.responses["list_sample"], "tool": None, "result": None}
        
        elif intent == "create":
            result = self._handle_create(user_text)
        
        elif intent == "refund":
            result = self._handle_refund(ids, user_text)
        
        elif intent == "cancel":
            result = self._handle_cancel(ids)
        
        elif intent == "check":
            result = self._handle_check(ids)
        
        else:
            # Unknown - check if there's an order ID anyway
            if ids:
                result = self._handle_check(ids)
            else:
                result = {"text": self.responses["unknown"], "tool": None, "result": None}
        
        # Store response
        self._remember(session_id, "assistant", result["text"])
        
        return result
    
    def get_session_stats(self, session_id):
        """Get session statistics"""
        session = self.memory_store.get(session_id, {})
        history = session.get("history", [])
        return {
            "total_messages": len(history),
            "user_messages": len([h for h in history if h["role"] == "user"]),
            "agent_responses": len([h for h in history if h["role"] == "assistant"]),
            "created_at": session.get("created_at", "N/A"),
            "last_intent": session.get("last_intent", "N/A")
        }

# Initialize agent
agent = NexusAgent(tool_registry, memory_store={})
print("âœ… NEXUS AI Agent initialized with enhanced features:")
print("   â€¢ Memory & context tracking")
print("   â€¢ 8 intent types (check, refund, create, cancel, help, greeting, thanks, list)")
print("   â€¢ Smart order ID extraction")
print("   â€¢ Formatted responses with emojis")




print("="*50 + "\n  NEXUS AI â€” Enhanced Test Suite\n" + "="*50)

tests = [
    "Hi there!",
    "What can you do?",
    "Check ORD-05",
    "Check ORD-05 and ORD-07",
    "I want a refund for ORD-08 because it arrived damaged",
    "Cancel ORD-12",
    "Order a mechanical keyboard",
    "Show my orders",
    "Thanks!",
    "Random gibberish text",
]

for t in tests:
    print(f"\nğŸ‘¤ USER: {t}")
    result = agent.run("test_session", t)
    print(f"ğŸ¤– NEXUS:\n{result['text']}")
    print("-"*50)

# Show session stats
print("\nğŸ“Š Session Stats:")
stats = agent.get_session_stats("test_session")
for k, v in stats.items():
    print(f"   â€¢ {k}: {v}")




if GRADIO_AVAILABLE:
    
    def gradio_chat(user_text, session_id="user_001"):
        """Process message and return response"""
        if not user_text.strip():
            return "ğŸ’¡ Please type a message to get started!"
        out = agent.run(session_id, user_text)
        return out["text"]
    
    def get_stats(session_id):
        """Get session statistics"""
        stats = agent.get_session_stats(session_id)
        return f"ğŸ“Š Messages: {stats['total_messages']} | Last intent: {stats['last_intent']}"
    
    def clear_session(session_id):
        """Clear session memory"""
        if session_id in agent.memory_store:
            agent.memory_store[session_id] = {
                "history": [],
                "created_at": now_iso(),
                "last_intent": None,
                "last_order_ids": []
            }
        return "", "Session cleared! Start fresh.", "ğŸ“Š Messages: 0 | Last intent: N/A"
    
    with gr.Blocks() as demo:
        
        # Header
        gr.Markdown("""
        # ğŸ”® NEXUS AI â€” Order Support Agent
        
        **Your intelligent assistant for order management**
        """)
        
        # Session Settings (collapsible)
        with gr.Accordion("âš™ï¸� Session Settings", open=False):
            with gr.Row():
                sid = gr.Textbox(
                    value="user_001", 
                    label="Session ID",
                    info="Change to simulate different users",
                    scale=2
                )
                stats_display = gr.Textbox(
                    value="ğŸ“Š Messages: 0 | Last intent: N/A",
                    label="Session Stats",
                    interactive=False,
                    scale=2
                )
            clear_btn = gr.Button("ğŸ—‘ï¸� Clear Session", size="sm")
        
        # Input Box
        chat_in = gr.Textbox(
            placeholder="Type anything... (e.g., 'Hi', 'Check ORD-05', 'Help')",
            label="ğŸ’¬ Your Message",
            lines=2,
            max_lines=6
        )
        
        # Send Button
        btn = gr.Button("ğŸ“¤ Send Message", variant="primary")
        
        # Output Box
        chat_out = gr.Textbox(
            label="ğŸ¤– NEXUS AI Response",
            lines=6,
            max_lines=20,
            interactive=False
        )
        
        # Quick Action Buttons
        gr.Markdown("---\n**ğŸš€ Quick Actions:**")
        
        with gr.Row():
            b_hi = gr.Button("ğŸ‘‹ Hi", size="sm")
            b_help = gr.Button("â�“ Help", size="sm")
            b_list = gr.Button("ğŸ“‹ List Orders", size="sm")
        
        with gr.Row():
            b_check = gr.Button("ğŸ“¦ Check ORD-05", size="sm")
            b_multi = gr.Button("ğŸ“¦ Check ORD-05 & ORD-07", size="sm")
            b_status = gr.Button("ğŸ“¦ Track ORD-12", size="sm")
        
        with gr.Row():
            b_refund = gr.Button("ğŸ’° Refund ORD-08", size="sm")
            b_cancel = gr.Button("â�Œ Cancel ORD-10", size="sm")
            b_create = gr.Button("ğŸ›’ Order Keyboard", size="sm")
        
        with gr.Row():
            b_thanks = gr.Button("ğŸ™� Thanks", size="sm")
            b_mouse = gr.Button("ğŸ›’ Order Mouse", size="sm")
            b_monitor = gr.Button("ğŸ›’ Order Monitor", size="sm")
        
        # Footer
        gr.Markdown("""
        ---
        *NEXUS AI â€” Intelligent Order Support Agent*  
        *Built with Python & Gradio*
        """)
        
        # â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�
        # EVENT HANDLERS
        # â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�
        
        def send_and_update(session_id, message):
            """Send message and update stats"""
            response = gradio_chat(message, session_id)
            stats = get_stats(session_id)
            return response, stats
        
        # Main send button
        btn.click(
            fn=send_and_update,
            inputs=[sid, chat_in],
            outputs=[chat_out, stats_display]
        )
        
        # Enter key to send
        chat_in.submit(
            fn=send_and_update,
            inputs=[sid, chat_in],
            outputs=[chat_out, stats_display]
        )
        
        # Clear session
        clear_btn.click(
            fn=clear_session,
            inputs=[sid],
            outputs=[chat_in, chat_out, stats_display]
        )
        
        # Quick action buttons - Row 1 (Greetings/Help)
        b_hi.click(lambda s: send_and_update(s, "Hi there!"), inputs=[sid], outputs=[chat_out, stats_display])
        b_help.click(lambda s: send_and_update(s, "Help"), inputs=[sid], outputs=[chat_out, stats_display])
        b_list.click(lambda s: send_and_update(s, "Show my orders"), inputs=[sid], outputs=[chat_out, stats_display])
        
        # Quick action buttons - Row 2 (Check Orders)
        b_check.click(lambda s: send_and_update(s, "Check ORD-05"), inputs=[sid], outputs=[chat_out, stats_display])
        b_multi.click(lambda s: send_and_update(s, "Check ORD-05 and ORD-07"), inputs=[sid], outputs=[chat_out, stats_display])
        b_status.click(lambda s: send_and_update(s, "Track ORD-12"), inputs=[sid], outputs=[chat_out, stats_display])
        
        # Quick action buttons - Row 3 (Refund/Cancel)
        b_refund.click(lambda s: send_and_update(s, "I want a refund for ORD-08"), inputs=[sid], outputs=[chat_out, stats_display])
        b_cancel.click(lambda s: send_and_update(s, "Cancel ORD-10"), inputs=[sid], outputs=[chat_out, stats_display])
        b_create.click(lambda s: send_and_update(s, "Order a mechanical keyboard"), inputs=[sid], outputs=[chat_out, stats_display])
        
        # Quick action buttons - Row 4 (More)
        b_thanks.click(lambda s: send_and_update(s, "Thanks!"), inputs=[sid], outputs=[chat_out, stats_display])
        b_mouse.click(lambda s: send_and_update(s, "Order a wireless mouse"), inputs=[sid], outputs=[chat_out, stats_display])
        b_monitor.click(lambda s: send_and_update(s, "Order a gaming monitor"), inputs=[sid], outputs=[chat_out, stats_display])
    
    print("ğŸš€ Launching NEXUS AI Interface...")
    demo.launch(share=True, inline=True)

else:
    print("âš ï¸� Gradio not available. Use: agent.run('session', 'Your message')")

