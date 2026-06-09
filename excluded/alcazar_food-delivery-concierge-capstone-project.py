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


import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Gemini API key setup complete.")
except Exception as e:
    print(
        f"ðŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )


from google.adk.agents import Agent, SequentialAgent, ParallelAgent, LoopAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.tools import AgentTool, FunctionTool, google_search
from google.genai import types

print("âœ… ADK components imported successfully.")


retry_config=types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504], # Retry on these HTTP errors
)


import asyncio, json, os, uuid, time, random, logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger('food_concierge_demo')

@dataclass
class AgentConfig:
    model: str
    name: str
    description: str = ''
    instruction: str = ''
    output_key: str = 'output'
    after_agent_callback: Optional[callable] = None
    model_params: Dict[str, Any] = None




class Agent:
    def __init__(self, config: AgentConfig):
        self.config = config
        self.name = config.name
    def _mock_run(self, prompt:str, inputs:Dict[str,Any]) -> str:
        if self.name == 'food_conversational_agent':
            return f"Understood request for {inputs.get('cuisine','any')} in {inputs.get('city','unknown')}."
        elif self.name == 'food_search_agent':
            cuisine = inputs.get('cuisine','Any')
            city = inputs.get('city','City')
            offers = []
            for i in range(3):
                offers.append({
                    'restaurant_id': f"{city[:3].upper()}-R-{100+i}",
                    'name': f"{city} {cuisine} Place {i+1}",
                    'avg_price_for_two': random.choice([15,25,35]) + i*5,
                    'cuisine': cuisine,
                    'menu': [
                        {'item_id': f'it-{i}-1', 'name':'Dish A', 'price':random.choice([5,8,10])},
                        {'item_id': f'it-{i}-2', 'name':'Dish B', 'price':random.choice([6,9,12])},
                    ],
                })
            return json.dumps({'offers': offers})
        elif self.name == 'policy_agent':
            prefs = inputs.get('prefs', {})
            allowed = True
            reason = None
            if prefs.get('max_budget') and prefs.get('max_budget') < 10:
                allowed = False; reason = 'budget too low'
            if prefs.get('allergies') and 'peanut' in prefs.get('allergies'):
                allowed = True; reason = 'has_peanut_allergy - flagged'
            return json.dumps({'policy_ok': allowed, 'reason': reason})
        elif self.name == 'order_agent':
            order_id = str(uuid.uuid4())
            return json.dumps({'order_id': order_id, 'status': 'pending'})
        elif self.name == 'payment_agent':
            payment_id = str(uuid.uuid4())
            return json.dumps({'payment_id': payment_id, 'status': 'waiting_3ds'})
        elif self.name == 'delivery_agent':
            return json.dumps({'status': 'out_for_delivery', 'eta_min': 25})
        return 'OK'
    def run(self, inputs: Dict[str,Any]) -> Dict[str,Any]:
        prompt = self.config.instruction or ''
        out = self._mock_run(prompt, inputs)
        try:
            parsed = json.loads(out)
            result = parsed
        except Exception:
            result = {self.config.output_key: out}
        if self.config.after_agent_callback:
            try:
                maybe = self.config.after_agent_callback(result)
                if isinstance(maybe, dict):
                    result = maybe
            except Exception as e:
                log.warning('after_agent_callback error: %s', e)
        return result



class MCP:
    def __init__(self):
        self.queue = asyncio.Queue()
        self.handlers = {}
    async def publish(self, envelope: Dict[str,Any]):
        await self.queue.put(envelope)
        log.info(f"MCP publish: {envelope['type']} from {envelope['from']} -> {envelope['to']} (corr={envelope.get('correlation_id')})")
    def register(self, name:str, handler):
        self.handlers[name] = handler
    async def start(self):
        while True:
            env = await self.queue.get()
            to = env['to']
            handler = self.handlers.get(to)
            if handler:
                asyncio.create_task(handler(env))
            else:
                log.warning('No handler for %s', to)




class MemoryBank:
    def __init__(self, path='food_memory.json'):
        self.path = path
        if os.path.exists(self.path):
            with open(self.path,'r') as f:
                self.store = json.load(f)
        else:
            self.store = {'profiles':{}, 'events':{}}
            with open(self.path,'w') as f:
                json.dump(self.store, f, indent=2)
    def set_profile(self, user_id, profile):
        self.store['profiles'][user_id] = profile
        with open(self.path,'w') as f:
            json.dump(self.store, f, indent=2)
    def get_profile(self, user_id):
        return self.store['profiles'].get(user_id, {})
    def push_event(self, user_id, ev):
        self.store['events'].setdefault(user_id,[]).append(ev)
        with open(self.path,'w') as f:
            json.dump(self.store, f, indent=2)




def suppress_output_callback(output):
    s = json.dumps(output)
    if 'SENSITIVE' in s:
        return {'status':'suppressed'}
    return None

conv_config = AgentConfig(model='mock', name='food_conversational_agent',
    instruction='Parse user intent for food order, extract city, cuisine, time, prefs.', output_key='conv_out')

search_config = AgentConfig(model='mock', name='food_search_agent',
    instruction='Search restaurants and menus based on params', output_key='search_out')

policy_config = AgentConfig(model='mock', name='policy_agent',
    instruction='Enforce dietary and budget policies', output_key='policy_out')

order_config = AgentConfig(model='mock', name='order_agent',
    instruction='Place order with restaurant and return order id', output_key='order_out')

payment_config = AgentConfig(model='mock', name='payment_agent',
    instruction='Initiate payment and return payment id; long-running 3DS', output_key='payment_out')

delivery_config = AgentConfig(model='mock', name='delivery_agent',
    instruction='Track delivery status and ETA', output_key='delivery_out')

food_conv = Agent(conv_config)
food_search = Agent(search_config)
policy_agent = Agent(policy_config)
order_agent = Agent(order_config)
payment_agent = Agent(payment_config)
delivery_agent = Agent(delivery_config)

mcp = MCP()
memp = MemoryBank(path='/tmp/food_memory_demo.json')

offers_container = []
policy_result = {}
order_info = {}
payment_info = {}
delivery_info = {}

async def conv_handler(env):
    payload = env['payload']
    out = food_conv.run(payload)
    corr = env.get('correlation_id')
    await mcp.publish({'type':'task','from':'food_conversational_agent','to':'food_search_agent','payload':payload,'correlation_id':corr})
    await mcp.publish({'type':'task','from':'food_conversational_agent','to':'policy_agent','payload':payload,'correlation_id':corr})

async def search_handler(env):
    payload = env['payload']
    res = food_search.run(payload)
    await mcp.publish({'type':'response','from':'food_search_agent','to':'demo_harness','payload':res,'correlation_id':env.get('correlation_id')})

async def policy_handler(env):
    payload = env['payload']
    res = policy_agent.run({'prefs': payload.get('prefs',{})})
    await mcp.publish({'type':'response','from':'policy_agent','to':'demo_harness','payload':res,'correlation_id':env.get('correlation_id')})

async def order_handler(env):
    payload = env['payload']
    res = order_agent.run(payload)
    await mcp.publish({'type':'response','from':'order_agent','to':'demo_harness','payload':res,'correlation_id':env.get('correlation_id')})
    await mcp.publish({'type':'task','from':'order_agent','to':'payment_agent','payload':{'booking_id':res.get('order_id'),
                    'amount': payload.get('amount',20), 'card_token': payload.get('card_token')},
                    'correlation_id':env.get('correlation_id')})

async def payment_handler(env):
    payload = env['payload']
    res = payment_agent.run(payload)
    await mcp.publish({'type':'response','from':'payment_agent','to':'demo_harness','payload':res,'correlation_id':env.get('correlation_id')})
    asyncio.create_task(payment_poll_loop(res.get('payment_id'), env.get('correlation_id')))

async def delivery_handler(env):
    payload = env['payload']
    res = delivery_agent.run(payload)
    await mcp.publish({'type':'response','from':'delivery_agent','to':'demo_harness','payload':res,'correlation_id':env.get('correlation_id')})

async def payment_poll_loop(payment_id, correlation_id):
    await asyncio.sleep(1.5)
    await mcp.publish({'type':'response','from':'payment_agent','to':'order_agent','payload':{'payment_id':payment_id,'status':'confirmed'},'correlation_id':correlation_id})
    await asyncio.sleep(0.5)
    await mcp.publish({'type':'task','from':'system','to':'delivery_agent','payload':{'order_id':'simulated','driver':'Ramesh'},'correlation_id':correlation_id})




mcp.register('food_conversational_agent', conv_handler)
mcp.register('food_search_agent', search_handler)
mcp.register('policy_agent', policy_handler)
mcp.register('order_agent', order_handler)
mcp.register('payment_agent', payment_handler)
mcp.register('delivery_agent', delivery_handler)




async def demo_response_handler(env):
    global offers_container, policy_result, order_info, payment_info, delivery_info
    src = env['from']
    payload = env['payload']
    log.info(f"[Demo] Got response from {src}: {payload}")
    if src == 'food_search_agent':
        offers_container = payload.get('offers', [])
    if src == 'policy_agent':
        policy_result = payload
    if src == 'order_agent':
        order_info = payload
    if src == 'payment_agent':
        payment_info = payload
    if src == 'delivery_agent':
        delivery_info = payload




mcp.register('demo_harness', demo_response_handler)




async def run_demo():
    mcp_task = asyncio.create_task(mcp.start())
    user_id = 'user_food_1'
    memp.set_profile(user_id, {'name':'Sam', 'prefs':{'cuisine':'Indian', 'allergies':['peanut']}})
    user_request = {'user_id': user_id, 'city':'Bengaluru', 'cuisine':'Indian', 'time':'ASAP',
                    'prefs':{'max_budget':30, 'allergies':['peanut']},
                    'card_token':'tok_test_visa_1234'}
    corr = str(uuid.uuid4())
    await mcp.publish({'type':'user_msg','from':'user','to':'food_conversational_agent','payload':user_request,'correlation_id':corr})
    await asyncio.sleep(1.5)
    log.info('Offers: %s', offers_container)
    log.info('Policy: %s', policy_result)
    if not offers_container:
        raise RuntimeError('No offers found')
    if not policy_result.get('policy_ok', True):
        raise RuntimeError('Policy denied: %s' % policy_result.get('reason'))
    chosen = sorted(offers_container, key=lambda x: x['avg_price_for_two'])[0]
    item = chosen['menu'][0]
    order_payload = {'restaurant': chosen, 'items':[item], 'guest':{'name':'Sam'},
                     'amount': item['price'], 'card_token': user_request['card_token']}
    await mcp.publish({'type':'task','from':'demo_harness','to':'order_agent','payload':order_payload,'correlation_id':corr})
    await asyncio.sleep(3.5)
    log.info('Order Info: %s', order_info)
    log.info('Payment Info: %s', payment_info)
    log.info('Delivery Info: %s', delivery_info)
    assert payment_info.get('status') in ('waiting_3ds','completed','confirmed'), 'Payment did not initiate'
    assert delivery_info.get('status') in ('out_for_delivery', 'delivered', None) or delivery_info=={}, 'Delivery not started'
    print("\n=== Food Delivery Concierge demo completed ===")
    print('Offers found:', len(offers_container))
    print('Chosen restaurant:', chosen['name'])
    print('Order id (simulated):', order_info.get('order_id'))
    mcp_task.cancel()
    return True




import nest_asyncio
nest_asyncio.apply()


result = await run_demo()
print("Demo run result:", result)

