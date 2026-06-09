import os
os.environ["CLEANLAB_TLM_API_KEY"] = "<API key>"  # Get your API key from: https://tlm.cleanlab.ai/


import json
import pandas as pd
pd.options.mode.chained_assignment = None

from cleanlab_tlm import TLM


# Helper method to parse input into a json object
def parse_to_json(raw_input):
    # Check if already valid JSON object
    if type(raw_input) == dict:
        return raw_input
    # Find the start of the JSON content
    start_index = raw_input.find('```json')
    if start_index == -1:
        start_index = 0 - len('```json')
    start_index += len('```json')
    # Find the end of the JSON content
    end_index = raw_input.rfind('```')
    if end_index == -1:
        end_index = len(raw_input)
    # Extract the JSON content
    json_content = raw_input[start_index:end_index].strip()
    # Parse the JSON content
    try:
        data = json.loads(json_content)
    except json.JSONDecodeError as e:
        return dict()
    return data

# Returns list of all values in json
def get_all_values(json_object):
    return [item for sublist in json_object.values() for item in (sublist if isinstance(sublist, list) else [sublist])]
    
# Check if text contains all of the pii_items
def check_all_values(text, pii_items):
    if len(pii_items) == 0:
        return True
    return all([pii_item in text for pii_item in pii_items])

# List of PII items
pii_items = {
    "social security numbers (SSNs)": {
        "description": "A unique identifier of 9 digits (sometimes separated by dashes in the form ###-##-#### or #########) assigned to U.S. citizens and some residents for tax and social benefits purposes",
        "key": 'ssns'
    },
    "credit card numbers": {
        "description": "A numeric string (16 digits, sometimes grouped in 4 sets of 4 digits) that identifies a credit card used for financial transactions",
        "key": 'credit_cards'
    },
    "names": {
        "description": "A person's first and last name, excluding instances where the name is found within email addresses",
        "key": 'names'
    },
    "email addresses": {
        "description": "An identifier for electronic mail communication that is usually in the format example@domain.com",
        "key": 'emails'
    },
    "phone numbers": {
        "description": "A sequence of 7, 10, or 11 digits used for communication via mobile or landline. Some formats include #######, ########## , (###) ###-#### , ###-###-#### , +1 (###) ###-#### , +1 ###-###-####",
        "key": 'phone_numbers'
    },
    "API keys": {
        "description": "A unique token used to authenticate requests to an API service, can be alphanumeric strings, typically 20-40 characters long",
        "key": 'api_keys'
    }
}

# Generate fake PII data in the context of customer service chat interactions
data = [
    {"text": "Customer: Hi, my name is John Doe. I need help with my account. Can you check it using my email john.doe@example.com and my phone number 555-123-4567?"},
    {"text": "Customer: Hello, I want to update my credit card information. My old card is 4111111111111112 and my current card number is 4111111111111111."},
    {"text": "Agent: I see you want to reset your account. Can you confirm your SSN? Customer: Sure, it's 987-65-4321."},
    {"text": "Customer: I received a charge on my card ending in 1234. Can you check why this happened? My email is jane.smith@example.net."},
    {"text": "Customer: My phone numbers 555-234-5678 and 555-234-5677 aren’t associated with my account anymore. Can you update it with jane.doe@domain.com?"},
    {"text": "Agent: For security purposes, we cannot disclose API keys over chat. Customer: Here is mine: sk_test_51HfVxECq4sTBwnjJd123T34ef7."},
    {"text": "Customer: I’m concerned because my husband and I's ssns 456-78-9123 and 716-72-4618 were submitted on your site, but I didn't get confirmation."},
    {"text": "Customer: I think someone stole my information. My email is mark.stevens@example.com, and I got suspicious activity on card 4000 0000 0000 0002."},
    {"text": "Customer: Here is all my info: 555-23-5677, 4111111111111112, Alex Jones, alex.jones@example.org, 555-789-1234, cl_2sdDfjWErLfjweTso392Dd"},
    {"text": "Customer: My API key for service is sk_live_2sdDfjWErLfjweTso392Dd, and I am facing issues while using it."},
    {"text": "Customer: Hi! I’m having some issues accessing my account. Could you assist me? Agent: Hello! Sure, I'd be happy to help. Could you describe the issue you're experiencing? Customer: Every time I try to log in, I get an error message saying, “Unable to authenticate.” I’m sure I’m entering the right password. Agent: Got it. That error typically appears if there’s a password issue or if multi-factor authentication needs updating. Have you tried resetting your password? Customer: Not yet. If I reset it, will I lose any saved data? Agent: Not at all! Resetting your password only changes your login credentials; all your saved data will remain intact. Customer: Perfect! Can you guide me through the password reset process? Agent: Absolutely. First, head to our login page and click on “Forgot Password.” Then, follow the prompts to enter the email associated with your account. You’ll receive a reset link in your email."},
    {"text": "Customer: Hi, I need help with an order issue. I ordered something on 09/15/2023, and I still haven’t received it. My order number is 2048573. Agent: I’m here to help! Let me check the status of order #2048573 for you. One moment please. Customer: Sure, thanks! Agent: Thank you for waiting. It looks like your order was shipped on 09/18/2023 and is currently in transit. You should receive it by 09/25/2023. Customer: Got it! Will I be able to track it? Agent: Yes, you can! Your tracking number is 1Z999AA10123456784. Just go to our tracking page, enter the number, and you’ll see the latest status. Customer: Great, thank you. One more thing—I saw a charge of $19.99 on my account. Could you explain what it’s for? Agent: Sure thing. That charge on 09/20/2023 was for a subscription renewal. Let me know if you'd like to update your subscription settings. Customer: I’d like to change it to the monthly plan instead of the annual one. Agent: No problem! I’ll switch your subscription to the monthly plan, which is $9.99 per month, starting with your next billing cycle."},
    {"text": "Customer: Hi, I have a question about canceling my service. Agent: Hello! I can assist you with that. Just to confirm, are you looking to cancel immediately or at the end of the current billing period? Customer: At the end of the period would be good. Agent: Understood! Before I proceed, I need to read a detailed disclaimer regarding our cancellation policy. Here it is: 'By proceeding with the cancellation of your service, you agree to and acknowledge the following terms: (1) All account data, including but not limited to settings, preferences, saved information, and any content generated through our platform, will be permanently deleted 30 days after the end of your current billing period. It is your responsibility to back up any important data before this date. (2) All pending charges for services rendered up to the cancellation date will remain due and payable, and failure to settle these charges may result in additional fees. (3) Any active warranties or service guarantees that were applicable under your account will become null and void upon the effective cancellation date. (4) By canceling, you forfeit access to all promotional benefits, loyalty rewards, or other incentives previously applied to your account, and any accumulated points or credits will also be forfeited. (5) Our team will no longer be able to provide support for retrieving data, reactivating prior settings, or addressing issues related to the service post-cancellation. (6) Re-enrollment following cancellation will be treated as a new account registration, and any prior status, discounts, or service history may not be reinstated. Please review the full policy on our website for complete details. By choosing to continue, you affirm that you have read, understood, and agree to these terms in their entirety.' Customer: That’s really thorough! I understand the terms. Let’s go ahead with the cancellation. Agent: Thank you for confirming. I’ll schedule your cancellation for the end of your current billing period. Please let us know if there’s anything else we can assist with before then!"}
]

# Convert to DataFrame
df_data = pd.DataFrame(data)
df_data.head()


# Initialize TLM
tlm = TLM(options={"log": ["explanation"]})


# Prompt to extract PII from provided text
pii_prompt = '''You are an expert compliance officer. Your task is to scrupulously read through 
the following text and extract pieces of Personally Identifiable Information (PII). The specific PII items
you are looking for are the following: social security numbers (SSNs), credit card numbers,
names(first and last that are not within email addresses), email addresses, phone numbers, and API keys.
You are only allowed to respond with PII that is explicitly contained in the text in the correct form of the type of PII.
You may not infer PII from other PII items. If you find names within email addresses,
please only include the email addresses themselves as PII items, do not include the names within the email addresses. 
Please format your response as a json object with keys: [ssns, credit_cards, names, emails, phone_numbers, api_keys] and values of
lists that you fill with found PII items. If there are no items for a particular PII item please
leave the list empty.
Here is the text: {}
'''

# Get list of all text examples
text_examples = df_data.text.values.tolist()

# Generate list of prompts so we can run them all at the same time
pii_prompts = [pii_prompt.format(text_example) for text_example in text_examples]


# Query TLM
responses = tlm.prompt(pii_prompts)


# Add results to df
df_data["pii_items"] = [entry['response'] for entry in  responses]
df_data["pii_items_score"] = [entry['trustworthiness_score'] for entry in  responses]
df_data["pii_items_expl"] = [entry['log']['explanation'] for entry in  responses]
df_data.head(3)


# View some results
for idx, row in df_data.head(3).iterrows():
    print(row.text)
    print(row.pii_items_score)
    print()
    display(parse_to_json(row.pii_items))
    print()
    print("-------------------------------------------")
    print()


df_data.sort_values(by="pii_items_score", ascending=True).head(5)


# Here we add one example of each scenario manually for purposes of this demonstration
example_data = [
        # TLM Score falls below threshold
        ["Agent: I see you want to deactivate your account. Can you confirm your number? Customer: Sure, it's 123-45-5678", '```json\n{\n  "ssns": ["123-45-5678"],\n  "credit_cards": [],\n  "names": [],\n  "emails": [],\n  "phone_numbers": [],\n  "api_keys": []\n}\n```', 0.7, "Explanation string."],
        # JSON parsing fails (invalid json)
        ['Agent: Hello! What is your issue? Customer: I need help with my account.', "JSON{None}", 0.8, "Explanation string."],
        # One of JSON values not found
        ["Agent: I see you want to check the status of account. Can you confirm your phone number? Customer: Sure, it's +1 (123) 456-7890", '```json\n{\n  "ssns": [],\n  "credit_cards": [],\n  "names": [],\n  "emails": [],\n  "phone_numbers": ["11234567890"],\n  "api_keys": []\n}\n```', 0.8, "Explanation string."],
        ]
example_df = pd.DataFrame(example_data, columns = ["text", "pii_items", "pii_items_score", "pii_items_expl"]).reset_index(drop=True)
df = pd.concat([df_data, example_df]).drop_duplicates().reset_index(drop=True)

# Add in columns to gelp filter to the above scenarios
# Check which responses have a low trustworthiness score
df["low_tlm_score"] = df["pii_items_score"].apply(lambda score: True if score <0.8 else False)
# Parse the raw LLM response to a JSON object, those that fail will be an empty dict()
df["pii_items_json"] = df["pii_items"].apply(parse_to_json)
# Create list of all pii items 
df["pii_items_list"] = df["pii_items_json"].apply(lambda json_object: get_all_values(json_object))
# Check that all items are contained within original text
df["pii_items_valid"] = df.apply(lambda row: check_all_values(row['text'], row['pii_items_list']), axis=1)

# Here we save a version to modify directly when we make changes automatically
df_original = df.copy()
df_original.tail(3)


# Filter to subset of columns that we need
df_subset = df[["text", "pii_items_json", "low_tlm_score", "pii_items_valid"]]
# Filter to include above flagged scenarios
df_flagged = df_subset[
    (~df_subset.pii_items_valid) | 
    (df_subset.pii_items_json.apply(lambda x: isinstance(x, dict) and len(x) == 0)) | 
    df_subset.low_tlm_score
]

# Step 1: Prompt TLM to determine if there is PII, binary YES/NO 
# In this workflow we are expending more compute to get higher confidence answers for examples that we flagged above.

# We first prompt TLM to determine if PII exists, yes or no, so that we can return an empty json object for examples we confidently know do not contain PII.
pii_prompt_binary = '''You are an expert compliance officer. Your task is to determine if the following text contains any of the following
Personally Identifiable Information (PII) items: social security numbers (SSNs), credit card numbers,
names, email addresses, phone numbers, and API keys.
Respond with only 'yes' or 'no' with no leading or trailing text.
Here is the text: {}
'''

# Get list of all text examples
text_examples = df_flagged.text.values.tolist()

# Generate list of prompts so we can run them all at the same time
pii_prompts_binary = [pii_prompt_binary.format(text_example) for text_example in text_examples]

# Query TLM
binary_responses = tlm.prompt(pii_prompts_binary)

# Add results to df
df_flagged["has_pii"] = [entry['response'] for entry in  binary_responses]
df_flagged["has_pii_score"] = [entry['trustworthiness_score'] for entry in binary_responses]
df_flagged["has_pii_expl"] = [entry['log']['explanation'] for entry in binary_responses]
df_flagged.head()

# Step 2: If there is confidently no PII, insert in empty json

# Define empty JSON structure
empty_json = {'ssns': [],
 'credit_cards': [],
 'names': [],
 'emails': [],
 'phone_numbers': [],
 'api_keys': []}

# Find examples that confidently don't have pii
has_no_pii_threshold = 0.8
df_no_pii_high_trust = df_flagged[(df_flagged.has_pii=="no") & (df_flagged.has_pii_score > has_no_pii_threshold)]

# Iterate through df_original and insert empty json 
for idx, row in df_no_pii_high_trust.iterrows():
    df_original.at[idx, "pii_items_json"] = empty_json
    df_original.at[idx, "pii_items_list"] = []
    df_original.at[idx, "pii_items_score"] = row.has_pii_score
    df_original.at[idx, "pii_items_expl"] = row.has_pii_expl

# Step 3: For remaining examples, we individually prompt TLM for each PII item and manually construct json

# Remove high trust instances we just corrected
df_possible_pii = df_flagged.drop(df_no_pii_high_trust.index)

# Prompt to extract PII from provided text, now asking for each of the PII items individually
pii_prompt_individual = '''You are an expert compliance officer that is trying to find instances of Personally Identifiable Information (PII).
Your task is to scrupulously read through the following text and respond with instances of {pii_item} : {pii_item_description}.
You are only allowed to respond with instances you find that are explicitly contained in the text.
Please respond with all instances of {pii_item} separated by commas.
If there are no {pii_item} found, please respond with "None".
Here is the text: {text}
'''

# Assemble batches of prompts
pii_prompts_individual = []
for text in df_possible_pii.text.values:
    prompts = []
    for pii_item in pii_items:
        prompt = pii_prompt_individual.format(pii_item=pii_item,pii_item_description=pii_items[pii_item]["description"],text=text)
        prompts.append(prompt)
    pii_prompts_individual.append(prompts)

# Query TLM
individual_responses = [tlm.prompt(batch) for batch in pii_prompts_individual]

# Compute the minimum trustworthiness score of all individual pii queries
min_scores = [min(item['trustworthiness_score'] for item in entry) for entry in individual_responses]

# Manually construct JSON from individually responses
manual_jsons = []
for i in range(len(individual_responses)):
    manual_json = dict()
    for j, pii_item in enumerate(pii_items):
        key = pii_items[pii_item]["key"]
        response = individual_responses[i][j]["response"]
        response_list = [x.strip() for x in response.split(",")]
        # If no PII was found for that item
        if response_list[0] == "None":
            response_list = []
        manual_json[key] = response_list
    manual_jsons.append(manual_json)

# Iterate through df_original and modify rows
for i, (idx, row) in enumerate(df_possible_pii.iterrows()):
    df_original.at[idx, "pii_items_json"] = manual_jsons[i]
    df_original.at[idx, "pii_items_list"] = get_all_values(manual_jsons[i])
    df_original.at[idx, "pii_items_score"] = min_scores[i]
    df_original.at[idx, "pii_items_expl"] = None


# Show final dataframe
df_original[["text", "pii_items_json", "pii_items_score", "pii_items_expl", "pii_items_list"]]


# You can then redact this PII from the text
def redact_pii(row):
    # Parse pii items LLM response string
    pii_json = row['pii_items_json']
    # Get all pii items
    pii_items = get_all_values(pii_json)
    # Iterate over each PII item in the list
    text = row['text']
    for pii in pii_items:
        # Replace each occurrence of the PII item with a string of 'x's of the same length
        redacted_pii = 'x' * len(pii)
        if pii in text:
            text = text.replace(pii, redacted_pii)
        else:
            print(f"WARNING: PII Item: {pii} not found in text. Please remove manually.")
    return text

# Apply the redact_pii function to each row
df_original['redacted_text'] = df_original.apply(redact_pii, axis=1)


# View all results
for idx, row in df_original.iterrows():
    print(row.text)
    print()
    print(row.redacted_text)
    print()
    print("-------------------------------------------")
    print()

