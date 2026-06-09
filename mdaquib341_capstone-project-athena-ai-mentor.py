# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Cell 1: Install required packages
!pip install google-adk kaggle-secrets ipywidgets plotly

print("âœ… Required packages installed successfully!")


# Cell 2: Import libraries and setup authentication
import os
import asyncio
from kaggle_secrets import UserSecretsClient
from IPython.display import display, HTML, Markdown
import ipywidgets as widgets
from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.tools import google_search
from google.genai import types
from jupyter_server.serverapp import list_running_servers

print("ğŸ“š All required libraries imported successfully!")


# Cell 3: Setup Google API Key with enhanced styling
def setup_authentication():
    try:
        GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
        os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
        
        success_html = """
        <div style="padding: 15px; border: 2px solid #4CAF50; border-radius: 8px; background-color: #f0f9f0; margin: 20px 0;">
            <div style="font-family: sans-serif; color: #333;">
                <strong style="color: #2E7D32;">âœ… Authentication Successful!</strong>
                <p style="margin: 10px 0 0 0; color: #555;">Gemini API key has been securely configured.</p>
            </div>
        </div>
        """
        display(HTML(success_html))
        return True
        
    except Exception as e:
        error_html = f"""
        <div style="padding: 15px; border: 2px solid #f44336; border-radius: 8px; background-color: #ffebee; margin: 20px 0;">
            <div style="font-family: sans-serif; color: #333;">
                <strong style="color: #C62828;">ğŸ”‘ Authentication Error</strong>
                <p style="margin: 10px 0 0 0; color: #555;">
                    Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets.<br>
                    <strong>Error Details:</strong> {e}
                </p>
            </div>
        </div>
        """
        display(HTML(error_html))
        return False

# Run authentication
auth_success = setup_authentication()


# Cell 4: Configure retry settings and create Athena Agent
retry_config = types.HttpRetryOptions(
    attempts=5,
    exp_base=7,
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504]
)

# Create Athena Agent
athena_agent = Agent(
    name="Athena_Research_Assistant",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=retry_config
    ),
    description="An intelligent research assistant that provides detailed, well-researched answers using Google Search.",
    instruction="""
    You are Athena, an advanced research assistant. Your capabilities include:
    
    ğŸ”� **Research & Analysis**: Use Google Search to find current, accurate information
    ğŸ“š **Comprehensive Answers**: Provide detailed, well-structured responses
    ğŸ’¡ **Critical Thinking**: Analyze information from multiple perspectives
    ğŸ�¯ **Problem Solving**: Break down complex questions into manageable parts
    ğŸ“Š **Structured Output**: Organize information clearly with headings and bullet points
    
    Always:
    - Verify information using Google Search when needed
    - Cite sources when available
    - Provide balanced perspectives on controversial topics
    - Break down complex concepts into understandable parts
    - Offer practical applications and next steps
    """,
    tools=[google_search],
)

display(HTML("""
<div style="padding: 15px; border: 2px solid #2196F3; border-radius: 8px; background-color: #e3f2fd; margin: 20px 0;">
    <div style="font-family: sans-serif; color: #333;">
        <strong style="color: #1976D2;">ğŸ¤– Athena Agent Created Successfully!</strong>
        <p style="margin: 10px 0 0 0; color: #555;">
            Your intelligent research assistant is ready to help with comprehensive, well-researched answers.
        </p>
    </div>
</div>
"""))


# Cell 5: Initialize the runner
runner = InMemoryRunner(agent=athena_agent)

display(HTML("""
<div style="padding: 15px; border: 2px solid #9C27B0; border-radius: 8px; background-color: #f3e5f5; margin: 20px 0;">
    <div style="font-family: sans-serif; color: #333;">
        <strong style="color: #7B1FA2;">ğŸš€ Runner Initialized!</strong>
        <p style="margin: 10px 0 0 0; color: #555;">
            Athena is now ready to process your queries with enhanced capabilities.
        </p>
    </div>
</div>
"""))


# Cell 6: Create an ULTRA-ATTRACTIVE chat interface
class AthenaChatInterface:
    def __init__(self, runner):
        self.runner = runner
        self.chat_history = []
        self.message_count = 0
        self.setup_interface()
    
    def setup_interface(self):
        # Enhanced Header with animated gradient
        header_html = """
        <div style="background: linear-gradient(135deg, 
                    #667eea 0%, #764ba2 25%, #f093fb 50%, #f5576c 75%, #4facfe 100%);
                    background-size: 400% 400%;
                    animation: gradientShift 15s ease infinite;
                    padding: 25px; border-radius: 15px; color: white; text-align: center;
                    margin-bottom: 25px; box-shadow: 0 8px 32px rgba(0,0,0,0.2);
                    border: 1px solid rgba(255,255,255,0.2);">
            <style>
                @keyframes gradientShift {
                    0% { background-position: 0% 50% }
                    50% { background-position: 100% 50% }
                    100% { background-position: 0% 50% }
                }
                @keyframes fadeIn {
                    from { opacity: 0; transform: translateY(10px); }
                    to { opacity: 1; transform: translateY(0); }
                }
                .message-animation {
                    animation: fadeIn 0.5s ease-out;
                }
                .typing-dots {
                    display: inline-block;
                }
                .typing-dots::after {
                    content: '...';
                    animation: typing 1.5s infinite;
                }
                @keyframes typing {
                    0%, 20% { content: '.'; }
                    40% { content: '..'; }
                    60%, 100% { content: '...'; }
                }
            </style>
            <h1 style="margin: 0; font-family: 'Segoe UI', sans-serif; font-size: 2.5em; 
                       text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">ğŸ�›ï¸� Athena AI Mentor</h1>
            <p style="margin: 8px 0 0 0; opacity: 0.95; font-size: 1.1em; font-weight: 300;">
                Your Intelligent Learning & Career Companion
            </p>
            <div style="margin-top: 15px; display: flex; justify-content: center; gap: 15px; flex-wrap: wrap;">
                <span style="background: rgba(255,255,255,0.2); padding: 5px 12px; border-radius: 20px; 
                            font-size: 0.9em; backdrop-filter: blur(10px);">ğŸ¤– AI-Powered</span>
                <span style="background: rgba(255,255,255,0.2); padding: 5px 12px; border-radius: 20px; 
                            font-size: 0.9em; backdrop-filter: blur(10px);">ğŸ”� Real-time Research</span>
                <span style="background: rgba(255,255,255,0.2); padding: 5px 12px; border-radius: 20px; 
                            font-size: 0.9em; backdrop-filter: blur(10px);">ğŸ�¯ Personalized Guidance</span>
            </div>
        </div>
        """
        display(HTML(header_html))
        
        # Stats Dashboard
        stats_html = """
        <div style="background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); 
                    padding: 15px; border-radius: 12px; margin: 20px 0; 
                    border: 1px solid #e1e8ed; box-shadow: 0 4px 15px rgba(0,0,0,0.1);">
            <div style="display: flex; justify-content: space-around; text-align: center; flex-wrap: wrap;">
                <div style="padding: 10px;">
                    <div style="font-size: 1.8em; font-weight: bold; color: #667eea;">0</div>
                    <div style="font-size: 0.9em; color: #666;">Messages</div>
                </div>
                <div style="padding: 10px;">
                    <div style="font-size: 1.8em; font-weight: bold; color: #764ba2;">ğŸ�¯</div>
                    <div style="font-size: 0.9em; color: #666;">Active</div>
                </div>
                <div style="padding: 10px;">
                    <div style="font-size: 1.8em; font-weight: bold; color: #f093fb;">ğŸš€</div>
                    <div style="font-size: 0.9em; color: #666;">Ready</div>
                </div>
            </div>
        </div>
        """
        display(HTML(stats_html))
        
        # Enhanced Chat output area with glass morphism effect
        self.output = widgets.Output(layout={
            'border': '1px solid rgba(255,255,255,0.2)',
            'height': '500px', 
            'overflow_y': 'auto',
            'padding': '15px',
            'background': 'linear-gradient(135deg, rgba(255,255,255,0.9) 0%, rgba(245,247,250,0.9) 100%)',
            'border_radius': '15px',
            'box_shadow': '0 8px 32px rgba(0,0,0,0.1)',
            'backdrop_filter': 'blur(10px)'
        })
        
        # Enhanced Input area
        self.input_text = widgets.Textarea(
            value='',
            placeholder='ğŸ’­ Ask your AI mentor anything about career guidance, learning strategies, or project help... (Press Shift+Enter to send)',
            layout=widgets.Layout(
                width='100%', 
                height='100px',
                border='2px solid #e1e8ed',
                border_radius='12px',
                padding='15px',
                font_size='14px',
                background='rgba(255,255,255,0.8)'
            )
        )
        
        # Enhanced Buttons with hover effects
        self.send_button = widgets.Button(
            description="ğŸš€ Send to Mentor",
            button_style='success',
            icon='rocket',
            layout=widgets.Layout(
                width='180px', 
                height='45px',
                margin='5px',
                border_radius='25px',
                font_weight='bold'
            ),
            style={'button_color': '#667eea', 'font_weight': 'bold'}
        )
        
        self.clear_button = widgets.Button(
            description="ğŸ—‘ï¸� Clear Chat",
            button_style='warning',
            icon='trash',
            layout=widgets.Layout(
                width='150px', 
                height='45px',
                margin='5px',
                border_radius='25px'
            )
        )
        
        self.export_button = widgets.Button(
            description="ğŸ“¥ Export Chat",
            button_style='info',
            icon='download',
            layout=widgets.Layout(
                width='150px', 
                height='45px',
                margin='5px',
                border_radius='25px'
            )
        )
        
        # Button events
        self.send_button.on_click(self.send_message)
        self.clear_button.on_click(self.clear_chat)
        self.export_button.on_click(self.export_chat)
        self.input_text.observe(self.handle_enter, names='value')
        
        # Display widgets with better layout
        buttons_layout = widgets.HBox([
            self.send_button, 
            self.clear_button, 
            self.export_button
        ], layout=widgets.Layout(
            justify_content='center',
            margin='15px 0'
        ))
        
        display(self.output)
        display(self.input_text)
        display(buttons_layout)
        
        # Welcome message
        self.add_welcome_message()
    
    def add_welcome_message(self):
        welcome_msg = """
        <div class="message-animation" style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; padding: 20px; margin: 15px 0; 
            border-radius: 20px; border-left: 5px solid #4facfe;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
            position: relative;
        ">
            <div style="position: absolute; top: 10px; right: 15px; font-size: 1.5em;">ğŸ�›ï¸�</div>
            <strong style="font-size: 1.1em;">Athena AI Mentor:</strong> 
            <div style="margin-top: 10px; font-size: 1.05em;">
                Welcome! I'm your personal AI mentor, here to guide your learning and career journey.
            </div>
            <div style="margin-top: 15px; display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px;">
                <div style="background: rgba(255,255,255,0.2); padding: 8px; border-radius: 8px;">
                    <span style="font-size: 1.2em;">ğŸ�¯</span> Career Guidance
                </div>
                <div style="background: rgba(255,255,255,0.2); padding: 8px; border-radius: 8px;">
                    <span style="font-size: 1.2em;">ğŸ“š</span> Learning Support
                </div>
                <div style="background: rgba(255,255,255,0.2); padding: 8px; border-radius: 8px;">
                    <span style="font-size: 1.2em;">ğŸ’¡</span> Project Mentoring
                </div>
                <div style="background: rgba(255,255,255,0.2); padding: 8px; border-radius: 8px;">
                    <span style="font-size: 1.2em;">ğŸš€</span> Skill Development
                </div>
            </div>
            <div style="margin-top: 15px; padding: 12px; background: rgba(255,255,255,0.1); 
                        border-radius: 10px; border-left: 3px solid #4facfe;">
                <strong>ğŸ’¡ Pro Tip:</strong> Be specific about your goals for the most personalized advice!
            </div>
        </div>
        """
        with self.output:
            display(HTML(welcome_msg))
    
    def update_stats(self):
        """Update the message statistics"""
        stats_html = f"""
        <script>
            document.querySelector('[style*="f5f7fa"]').innerHTML = `
                <div style="display: flex; justify-content: space-around; text-align: center; flex-wrap: wrap;">
                    <div style="padding: 10px;">
                        <div style="font-size: 1.8em; font-weight: bold; color: #667eea;">{self.message_count}</div>
                        <div style="font-size: 0.9em; color: #666;">Messages</div>
                    </div>
                    <div style="padding: 10px;">
                        <div style="font-size: 1.8em; font-weight: bold; color: #764ba2;">ğŸ�¯</div>
                        <div style="font-size: 0.9em; color: #666;">Active</div>
                    </div>
                    <div style="padding: 10px;">
                        <div style="font-size: 1.8em; font-weight: bold; color: #f093fb;">ğŸš€</div>
                        <div style="font-size: 0.9em; color: #666;">Ready</div>
                    </div>
                </div>
            `;
        </script>
        """
        display(HTML(stats_html))
    
    def handle_enter(self, change):
        if change['new'] and change['new'].endswith('\n\n'):
            self.input_text.value = change['new'].rstrip('\n')
            self.send_message(None)
    
    async def process_query(self, query):
        try:
            response = await self.runner.run_debug(query)
            return response
        except Exception as e:
            return f"â�Œ I encountered an error while processing your request: {str(e)}\n\nPlease try again or rephrase your question."
    
    def send_message(self, b):
        query = self.input_text.value.strip()
        if not query:
            return
        
        self.message_count += 1
        self.update_stats()
            
        # Add user message to chat with enhanced styling
        user_html = f"""
        <div class="message-animation" style="
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            color: white; padding: 15px 20px; margin: 15px 0; 
            border-radius: 20px 20px 5px 20px; max-width: 75%; 
            float: right; clear: both; box-shadow: 0 4px 15px rgba(79, 172, 254, 0.3);
            position: relative;
        ">
            <div style="position: absolute; top: -8px; right: -8px; background: #667eea; 
                        border-radius: 50%; width: 25px; height: 25px; display: flex; 
                        align-items: center; justify-content: center; font-size: 0.8em;">
                ğŸ‘¤
            </div>
            <strong>You:</strong><br>
            <div style="margin-top: 5px; font-size: 1.05em;">{query}</div>
            <div style="font-size: 0.8em; opacity: 0.8; text-align: right; margin-top: 8px;">
                {self.get_current_time()}
            </div>
        </div>
        """
        
        with self.output:
            display(HTML(user_html))
        
        # Clear input
        self.input_text.value = ''
        
        # Show enhanced typing indicator
        thinking_html = """
        <div class="message-animation" style="
            background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
            padding: 15px 20px; margin: 15px 0; border-radius: 20px 20px 20px 5px; 
            max-width: 60%; float: left; clear: both; box-shadow: 0 4px 15px rgba(168, 237, 234, 0.3);
            border-left: 4px solid #667eea;
        ">
            <strong>ğŸ�›ï¸� Athena:</strong> 
            <span class="typing-dots" style="color: #666; font-style: italic;">
                Analyzing your query and researching
            </span>
            <div style="margin-top: 8px; font-size: 0.8em; color: #666;">
                ğŸ”� Searching knowledge â€¢ ğŸ’¡ Processing insights â€¢ ğŸ�¯ Preparing guidance
            </div>
        </div>
        """
        
        with self.output:
            thinking_display = display(HTML(thinking_html), display_id=True)
        
        # Process query
        async def get_response():
            response = await self.process_query(query)
            
            # Enhanced response formatting
            response_html = f"""
            <div class="message-animation" style="
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white; padding: 20px; margin: 15px 0; 
                border-radius: 20px 20px 20px 5px; max-width: 85%; 
                float: left; clear: both; box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
                position: relative; border-left: 5px solid #4facfe;
            ">
                <div style="position: absolute; top: -10px; left: -10px; background: #ff6b6b; 
                            border-radius: 50%; width: 30px; height: 30px; display: flex; 
                            align-items: center; justify-content: center; font-size: 0.9em;
                            box-shadow: 0 2px 8px rgba(255, 107, 107, 0.4);">
                    ğŸ�›ï¸�
                </div>
                <strong style="font-size: 1.1em;">Athena AI Mentor:</strong><br>
                <div style="margin-top: 10px; font-size: 1.05em; line-height: 1.5;">
                    {response}
                </div>
                <div style="margin-top: 15px; padding-top: 10px; border-top: 1px solid rgba(255,255,255,0.3); 
                            font-size: 0.8em; opacity: 0.8; display: flex; justify-content: space-between;">
                    <span>âœ… Response Generated</span>
                    <span>{self.get_current_time()}</span>
                </div>
            </div>
            <div style="clear: both;"></div>
            """
            
            # Update display
            thinking_display.update(HTML(response_html))
            
            # Auto-scroll to bottom
            self.scroll_to_bottom()
        
        # Run async function
        asyncio.run(get_response())
    
    def get_current_time(self):
        from datetime import datetime
        return datetime.now().strftime("%H:%M:%S")
    
    def scroll_to_bottom(self):
        """Scroll chat to bottom"""
        display(HTML("""
        <script>
            setTimeout(() => {
                const output = document.querySelector('[data-mime-type="application/vnd.jupyter.widget-view+json"]');
                if (output) {
                    output.scrollTop = output.scrollHeight;
                }
            }, 100);
        </script>
        """))
    
    def clear_chat(self, b):
        self.message_count = 0
        self.update_stats()
        self.output.clear_output()
        self.add_welcome_message()
        display(HTML("""
        <div style="background: #ff6b6b; color: white; padding: 10px; border-radius: 8px; 
                    margin: 10px 0; text-align: center; font-weight: bold;">
            ğŸ—‘ï¸� Chat history cleared
        </div>
        """))
    
    def export_chat(self, b):
        """Export chat history (placeholder functionality)"""
        display(HTML("""
        <div style="background: #4ecdc4; color: white; padding: 12px; border-radius: 8px; 
                    margin: 10px 0; text-align: center; font-weight: bold;">
            ğŸ“¥ Export feature coming soon! Your chat history will be downloadable in the next update.
        </div>
        """))

# Initialize the enhanced chat interface
display(HTML("""
<div style="text-align: center; margin: 20px 0;">
    <div style="font-size: 1.2em; color: #667eea; font-weight: bold;">
        Initializing Enhanced AI Mentor Interface...
    </div>
    <div style="margin-top: 10px;">
        <span style="display: inline-block; width: 10px; height: 10px; background: #667eea; 
                   border-radius: 50%; margin: 0 2px; animation: pulse 1.5s infinite;"></span>
        <span style="display: inline-block; width: 10px; height: 10px; background: #764ba2; 
                   border-radius: 50%; margin: 0 2px; animation: pulse 1.5s infinite 0.2s;"></span>
        <span style="display: inline-block; width: 10px; height: 10px; background: #f093fb; 
                   border-radius: 50%; margin: 0 2px; animation: pulse 1.5s infinite 0.4s;"></span>
    </div>
</div>
"""))

chat_interface = AthenaChatInterface(runner)


# Cell 7: Create ULTRA-ATTRACTIVE quick action buttons for common queries
def create_quick_actions(chat_interface):
    # Enhanced Header with animated background
    quick_actions_html = """
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 25%, #f093fb 50%, #f5576c 75%, #4facfe 100%);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
        padding: 25px; 
        border-radius: 20px; 
        margin: 30px 0; 
        box-shadow: 0 12px 40px rgba(102, 126, 234, 0.25);
        border: 1px solid rgba(255,255,255,0.3);
        backdrop-filter: blur(10px);
        position: relative;
        overflow: hidden;
    ">
        <div style="position: absolute; top: 0; left: 0; right: 0; bottom: 0; 
                    background: rgba(255,255,255,0.1); backdrop-filter: blur(5px);"></div>
        <div style="position: relative; z-index: 2; text-align: center; color: white;">
            <h2 style="margin: 0 0 10px 0; font-family: 'Segoe UI', sans-serif; 
                      font-size: 2.2em; text-shadow: 2px 2px 8px rgba(0,0,0,0.3);">
                ğŸš€ Quick Mentor Actions
            </h2>
            <p style="margin: 0; font-size: 1.2em; opacity: 0.95; font-weight: 300;
                     text-shadow: 1px 1px 4px rgba(0,0,0,0.2);">
                Instant guidance for your learning and career journey
            </p>
        </div>
        <div style="position: absolute; bottom: 10px; right: 20px; font-size: 3em; 
                   opacity: 0.1; z-index: 1;">âš¡</div>
    </div>
    
    <div style="background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); 
                padding: 20px; border-radius: 15px; margin: 20px 0;
                box-shadow: 0 6px 25px rgba(0,0,0,0.1); border: 1px solid rgba(255,255,255,0.5);">
        <div style="text-align: center; margin-bottom: 15px;">
            <h3 style="color: #667eea; margin: 0 0 10px 0; font-size: 1.4em;">
                ğŸ’¡ One-Click Mentor Guidance
            </h3>
            <p style="color: #666; margin: 0; font-size: 1.05em;">
                Tap any button below for instant AI-powered mentorship on common topics
            </p>
        </div>
    </div>
    """
    display(HTML(quick_actions_html))
    
    # Enhanced quick actions with mentor focus
    quick_queries = [
        {
            "emoji": "ğŸ�¯",
            "title": "Career Path Guidance",
            "query": "Can you help me plan my career path in data science and artificial intelligence?",
            "color": "#667eea",
            "description": "Get personalized career roadmap"
        },
        {
            "emoji": "ğŸ“š",
            "title": "Study Strategy Support",
            "query": "I need help developing effective learning strategies for machine learning and AI concepts.",
            "color": "#764ba2",
            "description": "Optimize your learning approach"
        },
        {
            "emoji": "ğŸ’¡",
            "title": "Project Mentoring",
            "query": "How should I approach my capstone project in AI and what are the best practices?",
            "color": "#f093fb",
            "description": "Expert project guidance"
        },
        {
            "emoji": "ğŸš€",
            "title": "Skill Development",
            "query": "What are the most important skills I should focus on to advance in AI and data science careers?",
            "color": "#f5576c",
            "description": "Build in-demand skills"
        },
        {
            "emoji": "ğŸ”�",
            "title": "Industry Research",
            "query": "What are the latest trends and job market insights in artificial intelligence and machine learning?",
            "color": "#4facfe",
            "description": "Stay current with industry"
        },
        {
            "emoji": "ğŸ“Š",
            "title": "Portfolio Building",
            "query": "How can I build an impressive portfolio for data science and AI job applications?",
            "color": "#43e97b",
            "description": "Showcase your abilities"
        },
        {
            "emoji": "ğŸ�“",
            "title": "Learning Resources",
            "query": "What are the best online courses, books, and resources for learning AI and data science?",
            "color": "#ff9a9e",
            "description": "Discover top learning materials"
        },
        {
            "emoji": "ğŸ’¼",
            "title": "Interview Preparation",
            "query": "How should I prepare for technical interviews in data science and AI roles?",
            "color": "#a8edea",
            "description": "Ace your job interviews"
        }
    ]
    
    # Create enhanced buttons with proper styling
    buttons = []
    for action in quick_queries:
        # Create button with proper HTML description
        btn = widgets.Button(
            description=f"{action['emoji']} {action['title']}",
            tooltip=action['query'],
            layout=widgets.Layout(
                width='280px',
                height='80px',
                margin='8px',
                border_radius='15px',
                border=f'3px solid {action["color"]}40'
            )
        )
        
        # Apply proper button styling using the style attribute
        btn.style.button_color = '#ffffff'
        btn.style.font_weight = 'bold'
        btn.style.text_color = '#333333'
        
        btn.query = action['query']
        btn.emoji = action['emoji']
        btn.color = action['color']
        btn.on_click(lambda b: chat_interface.input_text.set_value(b.query))
        buttons.append(btn)
    
    # Add custom CSS for hover effects
    display(HTML(f"""
    <style>
        .quick-action-button {{
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1) !important;
        }}
        .quick-action-button:hover {{
            transform: translateY(-3px) !important;
            box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3) !important;
        }}
        {''.join([f'''
        .button-{i} {{
            background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%) !important;
            border: 3px solid {action['color']}40 !important;
        }}
        .button-{i}:hover {{
            background: linear-gradient(135deg, {action['color']}15 0%, {action['color']}30 100%) !important;
            border: 3px solid {action['color']}80 !important;
        }}
        ''' for i, action in enumerate(quick_queries)])}
    </style>
    """))
    
    # Apply CSS classes to buttons
    for i, btn in enumerate(buttons):
        btn.add_class(f"quick-action-button button-{i}")
    
    # Enhanced grid layout with better spacing
    grid = widgets.GridBox(
        buttons,
        layout=widgets.Layout(
            grid_template_columns="repeat(auto-fit, minmax(280px, 1fr))",
            gap="20px",
            padding="20px",
            justify_content="center",
            align_items="center"
        )
    )
    
    # Container with enhanced styling
    grid_container = widgets.VBox([
        widgets.HTML("""
        <div style="text-align: center; margin: 10px 0 20px 0;">
            <div style="display: inline-flex; align-items: center; background: rgba(102, 126, 234, 0.1); 
                       padding: 8px 16px; border-radius: 20px; color: #667eea; font-weight: 500;">
                <span style="margin-right: 8px;">ğŸ”½</span>
                Click any card below to instantly ask your mentor
            </div>
        </div>
        """),
        grid
    ], layout=widgets.Layout(
        width='100%',
        padding='10px'
    ))
    
    display(grid_container)
    
    # Add detailed descriptions for each action
    actions_details_html = """
    <div style="
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 25px; 
        border-radius: 15px; 
        margin: 20px 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    ">
        <h4 style="text-align: center; color: #667eea; margin: 0 0 20px 0; font-size: 1.3em;">
            ğŸ“‹ Quick Actions Overview
        </h4>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 15px;">
    """
    
    for action in quick_queries:
        actions_details_html += f"""
            <div style="
                background: white; 
                padding: 15px; 
                border-radius: 10px; 
                border-left: 4px solid {action['color']};
                box-shadow: 0 2px 8px rgba(0,0,0,0.05);
            ">
                <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
                    <span style="font-size: 1.5em;">{action['emoji']}</span>
                    <strong style="color: #333; font-size: 1.1em;">{action['title']}</strong>
                </div>
                <div style="color: #666; font-size: 0.95em; line-height: 1.4;">
                    {action['description']}
                </div>
            </div>
        """
    
    actions_details_html += """
        </div>
    </div>
    """
    display(HTML(actions_details_html))
    
    # Footer with usage tips
    footer_html = """
    <div style="
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 20px; 
        border-radius: 15px; 
        margin: 30px 0 10px 0;
        text-align: center;
        box-shadow: 0 6px 20px rgba(168, 237, 234, 0.3);
        border: 1px solid rgba(255,255,255,0.5);
    ">
        <div style="display: flex; align-items: center; justify-content: center; gap: 15px; flex-wrap: wrap;">
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="background: #667eea; color: white; border-radius: 50%; 
                           width: 25px; height: 25px; display: flex; align-items: center; 
                           justify-content: center; font-size: 0.8em;">ğŸ’¡</span>
                <span style="color: #666; font-weight: 500;">Quick Start</span>
            </div>
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="background: #764ba2; color: white; border-radius: 50%; 
                           width: 25px; height: 25px; display: flex; align-items: center; 
                           justify-content: center; font-size: 0.8em;">ğŸ�¯</span>
                <span style="color: #666; font-weight: 500;">Focused Topics</span>
            </div>
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="background: #f5576c; color: white; border-radius: 50%; 
                           width: 25px; height: 25px; display: flex; align-items: center; 
                           justify-content: center; font-size: 0.8em;">ğŸš€</span>
                <span style="color: #666; font-weight: 500;">Instant Results</span>
            </div>
        </div>
        <div style="margin-top: 15px; color: #666; font-size: 0.95em;">
            <strong>Pro Tip:</strong> Use these quick actions as starting points, then ask follow-up questions for personalized advice!
        </div>
    </div>
    """
    display(HTML(footer_html))

# Display loading animation before showing quick actions
display(HTML("""
<div style="text-align: center; margin: 20px 0; padding: 30px;">
    <div style="font-size: 1.3em; color: #667eea; font-weight: bold; margin-bottom: 15px;">
        ğŸ�¯ Loading Mentor Quick Actions...
    </div>
    <div style="display: flex; justify-content: center; gap: 8px;">
        <div style="width: 12px; height: 12px; background: #667eea; border-radius: 50%; 
                  animation: bounce 1.4s infinite ease-in-out;"></div>
        <div style="width: 12px; height: 12px; background: #764ba2; border-radius: 50%; 
                  animation: bounce 1.4s infinite ease-in-out 0.2s;"></div>
        <div style="width: 12px; height: 12px; background: #f093fb; border-radius: 50%; 
                  animation: bounce 1.4s infinite ease-in-out 0.4s;"></div>
    </div>
    <style>
        @keyframes bounce {
            0%, 80%, 100% { transform: scale(0.8); opacity: 0.5; }
            40% { transform: scale(1.2); opacity: 1; }
        }
    </style>
</div>
"""))

create_quick_actions(chat_interface)


# Cell 8: Enhanced ADK Web UI setup function
def get_adk_proxy_url():
    PROXY_HOST = "https://kkb-production.jupyter-proxy.kaggle.net"
    ADK_PORT = "8000"

    servers = list(list_running_servers())
    if not servers:
        raise Exception("No running Jupyter servers found.")

    baseURL = servers[0]["base_url"]

    try:
        path_parts = baseURL.split("/")
        kernel = path_parts[2]
        token = path_parts[3]
    except IndexError:
        raise Exception(f"Could not parse kernel/token from base URL: {baseURL}")

    url_prefix = f"/k/{kernel}/{token}/proxy/proxy/{ADK_PORT}"
    url = f"{PROXY_HOST}{url_prefix}"

    styled_html = f"""
    <div style="padding: 20px; border: 2px solid #FF6B35; border-radius: 10px; background: linear-gradient(135deg, #fff8e1 0%, #ffecb3 100%); margin: 20px 0;">
        <div style="font-family: 'Arial', sans-serif; margin-bottom: 15px; color: #333;">
            <h3 style="color: #FF6B35; margin: 0 0 10px 0;">ğŸŒ� ADK Web Interface</h3>
            <div style="background: #fff3e0; padding: 15px; border-radius: 6px; border-left: 4px solid #FF6B35;">
                <strong>âš ï¸� Action Required:</strong> Follow these steps to launch the ADK Web UI
                <ol style="margin: 10px 0 0 0; padding-left: 20px;">
                    <li style="margin-bottom: 8px;"><strong>Run the next cell</strong> to start the ADK web server</li>
                    <li style="margin-bottom: 8px;">Wait for the cell to show "Running" status</li>
                    <li style="margin-bottom: 8px;"><strong>Return here and click the button below</strong> to open the interface</li>
                </ol>
            </div>
        </div>
        <a href='{url}' target='_blank' style="
            display: inline-block; background: linear-gradient(135deg, #FF6B35 0%, #F7931E 100%); 
            color: white; padding: 12px 25px; text-decoration: none; border-radius: 25px; 
            font-family: sans-serif; font-weight: 600; font-size: 14px;
            box-shadow: 0 4px 15px rgba(255, 107, 53, 0.3); transition: all 0.3s ease;">
            ğŸš€ Launch ADK Web UI (After starting server) â†—
        </a>
    </div>
    """

    display(HTML(styled_html))
    return url_prefix

# Display the ADK interface info
url_prefix = get_adk_proxy_url()


# Cell 9: Start the ADK web server
print("ğŸš€ Starting ADK Web Server...")
print("This cell will remain running while the server is active.")
print("Once you see 'Running' status, return to the previous cell and click the launch button.")

!adk web --url_prefix {url_prefix}


# Cell 10: Create a sample agent with enhanced styling
def create_sample_agent():
    display(HTML("""
    <div style="padding: 15px; border: 2px solid #17a2b8; border-radius: 8px; background-color: #e8f4f8; margin: 20px 0;">
        <div style="font-family: sans-serif; color: #333;">
            <strong style="color: #138496;">ğŸ› ï¸� Creating Sample Agent...</strong>
            <p style="margin: 10px 0 0 0; color: #555;">
                This will create a sample agent configuration for advanced testing.
            </p>
        </div>
    </div>
    """))
    
    try:
        !adk create sample-agent --model gemini-2.5-flash-lite --api_key $GOOGLE_API_KEY
        display(HTML("""
        <div style="padding: 15px; border: 2px solid #28a745; border-radius: 8px; background-color: #f0f9f0; margin: 20px 0;">
            <div style="font-family: sans-serif; color: #333;">
                <strong style="color: #2E7D32;">âœ… Sample Agent Created Successfully!</strong>
                <p style="margin: 10px 0 0 0; color: #555;">
                    You can now explore advanced agent configurations.
                </p>
            </div>
        </div>
        """))
    except Exception as e:
        display(HTML(f"""
        <div style="padding: 15px; border: 2px solid #dc3545; border-radius: 8px; background-color: #f8d7da; margin: 20px 0;">
            <div style="font-family: sans-serif; color: #333;">
                <strong style="color: #721c24;">â�Œ Error Creating Sample Agent</strong>
                <p style="margin: 10px 0 0 0; color: #555;">{e}</p>
            </div>
        </div>
        """))

create_sample_agent()


# Cell 11: Test the Athena Agent with sample queries
async def test_athena_agent():
    test_queries = [
        "What is Google's Agent Development Kit and what are its main features?",
        "How can I improve my machine learning model's performance?",
        "What are the latest advancements in natural language processing?"
    ]
    
    display(HTML("""
    <div style="padding: 15px; border: 2px solid #6f42c1; border-radius: 8px; background-color: #eae4f2; margin: 20px 0;">
        <div style="font-family: sans-serif; color: #333;">
            <strong style="color: #6f42c1;">ğŸ§ª Testing Athena Agent...</strong>
            <p style="margin: 10px 0 0 0; color: #555;">
                Running sample queries to demonstrate capabilities.
            </p>
        </div>
    </div>
    """))
    
    for i, query in enumerate(test_queries, 1):
        display(HTML(f"""
        <div style="background: #007bff; color: white; padding: 10px 15px; margin: 10px 0; 
                    border-radius: 8px; border-left: 4px solid #0056b3;">
            <strong>Test Query {i}:</strong> {query}
        </div>
        """))
        
        try:
            response = await runner.run_debug(query)
            display(HTML(f"""
            <div style="background: #28a745; color: white; padding: 15px; margin: 10px 0; 
                        border-radius: 8px; border-left: 4px solid #1e7e34;">
                <strong>ğŸ�›ï¸� Athena Response:</strong><br>{response}
            </div>
            """))
        except Exception as e:
            display(HTML(f"""
            <div style="background: #dc3545; color: white; padding: 15px; margin: 10px 0; 
                        border-radius: 8px; border-left: 4px solid #c82333;">
                <strong>â�Œ Error:</strong> {str(e)}
            </div>
            """))

# Run the test
await test_athena_agent()


# Cell 12: Display completion summary
completion_html = """
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            padding: 30px; border-radius: 15px; color: white; text-align: center;
            margin: 20px 0; box-shadow: 0 8px 25px rgba(0,0,0,0.15);">
    <h1 style="margin: 0 0 15px 0; font-family: 'Arial', sans-serif;">ğŸ�‰ Athena Agent Setup Complete!</h1>
    
    <div style="background: rgba(255,255,255,0.1); padding: 20px; border-radius: 10px; margin: 20px 0;">
        <h3 style="margin-top: 0;">âœ… What's Ready:</h3>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; text-align: left;">
            <div>ğŸ¤– <strong>Intelligent Agent</strong> - Powered by Gemini 2.5 Flash Lite</div>
            <div>ğŸ”� <strong>Web Search</strong> - Real-time information retrieval</div>
            <div>ğŸ’¬ <strong>Chat Interface</strong> - Beautiful interactive UI</div>
            <div>ğŸŒ� <strong>Web UI</strong> - Advanced ADK interface</div>
            <div>ğŸš€ <strong>Quick Actions</strong> - Common research queries</div>
            <div>âš¡ <strong>Error Handling</strong> - Robust retry mechanisms</div>
        </div>
    </div>
    
    <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px;">
        <h4 style="margin: 0 0 10px 0;">Next Steps:</h4>
        <p style="margin: 0; opacity: 0.9;">
            Use the chat interface above to start researching, or launch the ADK Web UI for advanced features!
        </p>
    </div>
</div>
"""

display(HTML(completion_html))

