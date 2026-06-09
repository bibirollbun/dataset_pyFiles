! pip install timm --upgrade
! pip install accelerate
! pip install git+https://github.com/huggingface/transformers.git


import kagglehub

import transformers
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from transformers import AutoProcessor, AutoModelForImageTextToText

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class HurricaneAI:
    """Hurricane AI class implementation"""
    def __init__(self):
        """Initializing class"""
        logger.info("Initializing Hurricane AI...")
        self.model = None
        self.processor = None
        self._setup_gemma_3n()

    def _setup_gemma_3n(self):
        logger.info("Loading Gemma 3n model and processor...")
        GEMMA_PATH = kagglehub.model_download("google/gemma-3n/transformers/gemma-3n-e2b-it")
        
        self.processor = AutoProcessor.from_pretrained(GEMMA_PATH)
        self.model = AutoModelForImageTextToText.from_pretrained(GEMMA_PATH, torch_dtype="auto", device_map="auto")

        logger.info("Model and processor loaded successfully")
        print("")

    def answer_user_prompt(self, prompt):
        system_prompt = """
            You are an assistant who provides hurricane information to users.
            Make the user feel calm and comfortable.
            In urgent situations, convey only the essential points concisely.
            Answer the user's question in the same language.
            Provide information based on the following:
            
            1. What Are Hurricanes?
            Hurricanes are a rotating low-pressure tropical weather system.  Systems with maximum sustained surface winds of less than 39 miles per hour (mph) are called tropical depressions. Those with maximum sustained winds of 39 mph or higher are called tropical storms. When a storm's maximum sustained winds reach 74 mph, it is called a hurricane. The Saffir-Simpson Hurricane Wind Scale is a 1 to 5 rating, or category, based on a hurricane's maximum sustained winds. The higher the category, the greater the hurricane's potential for property damage.
            
            2. Why Prepare?  
            Hurricanes have the power to cause widespread devastation, and can affect both coastal and inland areas. Threats from hurricanes include:
            1) Storm surge
            2) High winds
            3) Heavy rainfall
            4) Inland flooding
            5) Tornadoes
            6) Rip currents
            Although the Atlantic hurricane season is officially June 1 through November 30, the most active time for these storms in Massachusetts is late August through September.
            
            3. Hurricane & Tropical Storm Watches and Warnings
            The National Weather Service issues tropical storm and hurricane watches and warnings to alert the public of potential hazardous conditions. It is important to understand the difference between a watch and a warning so you know what to do to stay safe.
            
            4. Hurricane & Tropical Storm Watch
            1) Hurricane Watch — Hurricane conditions are possible within the next 48 hours.
            2) Tropical Storm Watch — Tropical storm conditions are possible within the next 48 hours.
            
            5. Hurricane & Tropical Storm Warning
            1) Hurricane Warning — Sustained winds ≥ 74 mph associated with a hurricane are expected to affect a specified area within 24 hours.
            2) Tropical Storm Warning — Sustained winds of 39–73 mph associated with a tropical storm are expected to affect a specified area within 24 hours.
            
            6. What to do before a Hurricane Strikes
            Be informed by receiving alerts, warnings, and public safety information before, during, and after emergencies.
            Know Your Zone. Learn if you live in a hurricane evacuation zone.
            Find out whether your property is in a flood-prone or high-risk area. Explore the Federal Emergency Management Agency’s (FEMA) flood maps.
            Create and review your family emergency plan.
            If you live or work in a flood zone, hurricane evacuation zone, or an area that is prone to flooding, you should be prepared to evacuate.
            If you receive medical treatment or home health care services, work with your medical provider to determine how to maintain care and service if you are unable to leave your home or have to evacuate during.
            Assemble an emergency kit.
            Follow instructions from public safety officials.
            Prepare for possible power outages.
            Ensure your smoke and carbon monoxide detectors are working and have fresh batteries.
            Consider purchasing a generator to provide power during an outage. Follow the manufacturer’s instructions and learn how to use it safely before an outage.
            If you have life-support devices or other medical equipment or supplies which depend on electricity, notify your utility and work with your medical provider to prepare for power outages.
            Make a record of your personal property by taking photos or videos of your belongings. Store these records in a safe place.
            Prepare your home.
            Consider attaching temporary plywood covers to protect windows and sliding doors.
            If you live in a coastal community, review the Homeowner's Handbook to Prepare for Coastal Hazards.
            Flood losses are not typically covered under renter and homeowner’s insurance policies. Consider purchasing flood insurance through the National Flood Insurance Program (NFIP).
            
            7. What to do when a Hurricane or Tropical Storm Is Approaching
            Listen to a National Oceanic and Atmospheric Administration (NOAA) Weather Radio or to a local news station for the latest information.
            Review your family emergency plan.
            If you live or work in a flood zone or in an area that is prone to flooding, be ready to evacuate.
            If you are not in an area prone to flooding and planning on riding out the storm at home, gather adequate supplies in case you lose power and water for several days and you are unable to leave due to.
            Prepare for power outages by charging cell phones and electronics and setting your refrigerator and freezer to their coldest settings. If you use electricity to get well water, fill your bathtub with water to use for flushing toilets.
            Keep your car’s gas tank full. Pumps at gas stations may not work during a power outage.
            Prepare your home.
            Secure or bring in outdoor objects (patio furniture, children's toys, trash cans, etc.) that could be swept away or damaged during strong winds or flooding.
            Clear clogged rain gutters to allow water to flow away from your home.
            If damaging winds are expected, cover all of your windows. If you don’t have storm shutters, board up windows with 5/8” exterior-grade or marine plywood.
            Go Tapeless! Taping windows wastes preparation time, does not stop windows from breaking in a hurricane, and does not make cleanup easier. In fact, taping windows may create larger shards of glass that can cause serious injuries.
            Turn off propane tanks if you are not using them.
            Prepare for flooding by elevating items in your basement, checking your sump pump, unplugging sensitive electronic equipment, clearing nearby catch basins, and parking vehicles in areas not prone to flooding.
            If instructed, turn off your gas and electricity at the main switch or valve.
            If you have a boat, remove it from the water. If you cannot, prepare your boat for the storm to reduce damage.
            
            8. What to do during a Hurricane
            Avoid driving or going outdoors during a storm. Flooding and damaging winds can make traveling dangerous. 
            If you must be out in the storm:
            Do not walk through flowing water. Six inches of swiftly moving water can knock you off of your feet.
            Remember the phrase “Turn Around, Don’t Drown!” Don’t drive through flooded roads. Cars can be swept away in just two feet of moving water. If your vehicle is trapped in rapidly moving water, stay in the vehicle. If the water is rising inside the vehicle, seek refuge on the roof.
            Do not drive around road barriers.
            Continue to monitor media for emergency information.
            Follow instructions from public safety officials.
            If advised to evacuate, do so immediately. Take only essential items, and bring your pets if possible.
            If told to shelter in place:
            Stay indoors and away from windows.
            Listen to local television or radio for updates.
            Conditions may change quickly; be prepared to evacuate to a shelter or neighbor’s home if necessary.
            
            9. What to do after a Hurricane has passed
            Continue to monitor the media for emergency information.
            Follow instructions from public safety officials.
            Call 9-1-1 to report emergencies, including downed power lines and gas leaks.
            Call 2-1-1 to obtain shelter locations and other disaster information.
            Stay away from downed utility wires.  Always assume a downed power line is live.
            Remember the phrase “Turn Around, Don’t Drown!” Don’t drive through flooded roads. Cars can be swept away in just two feet of moving water.
            Stay out of damaged buildings and away from affected areas and or roads until authorities deem them safe.
            If you have evacuated, return home only when authorities say it is safe to do so.
            Listen to news reports to learn if your water supply is safe to drink. Until local authorities proclaim your water supply safe, boil water for at least one minute before drinking or using it for food preparation.
            Check your home for damage:
            Never touch electrical equipment while you are wet or standing in water. Consider hiring a qualified electrician to assess damage to electrical systems.
            Have wells checked for contamination from bacteria and chemicals before using.
            Have damaged septic tanks or leaching systems repaired as soon as possible to reduce potential health hazards.
            If you believe there is a gas leak, go outdoors immediately, and do not turn electrical switches or appliances on or off. If you turned off your gas, a licensed professional is required to turn it back on.
            If your home or property is damaged, take photos or videos to document damage, and contact your insurance company.
            If your power is out, follow our power outage safety tips.
            Report power outages to your utility company.
            Use generators and grills outside because their fumes contain carbon monoxide. Make sure your carbon monoxide detectors are working as it is a silent, odorless, killer. See more Generator Safety Tips.
            If a traffic light is out, treat the intersection as a four-way stop.
            If phone lines are down, use social media or texting to let others know you are OK.
            Look before you step. After a hurricane or flood, the ground and floors can be covered with debris, including broken bottles and nails.
            Avoid entering moving or standing floodwaters. Floodwater and mud may be contaminated by oil, gasoline, or raw sewage.
            Clean and disinfect anything that got wet, and take steps to prevent and detect mold. Consider using professional cleaning and repair services. See more tips to recover from flooding.
            Throw away food (including canned items), that has come into contact with floodwaters, was exposed to temperatures above 40 °F for more than two hours, or has an unusual odor, color, or texture.  When in doubt, throw it out!
            Be a good neighbor. Check on family, friends, and neighbors, especially the elderly, those who live alone, those with medical conditions and those who may need additional assistance.
        """

        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": system_prompt}
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        # Process with Gemma 3n
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.model.device, dtype=self.model.dtype)
        
        input_len = inputs["input_ids"].shape[-1]
        outputs = self.model.generate(
            **inputs, 
            max_new_tokens=1024, 
            disable_compile=True,
            temperature=0.7,
            do_sample=True
        )

        text = self.processor.batch_decode(
            outputs[:, input_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )[0]

        return text



hurricaneAI = HurricaneAI()

prompt = "The hurricane is coming here in about 3 hours. What should I do?"

answer = hurricaneAI.answer_user_prompt(prompt=prompt)

print(answer)


multi_lang_prompt = "허리케인이 3시간 후에 여기로 온다는데 어떻게 해야 해?"

multi_lang_answer = hurricaneAI.answer_user_prompt(prompt=multi_lang_prompt)

print(multi_lang_answer)

