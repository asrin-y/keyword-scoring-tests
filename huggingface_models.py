import requests
from time import sleep
import sys

HF_KEY = "YOUR_API_KEY"

headers = {"Authorization": f"Bearer {HF_KEY}"}

# if torch.backends.mps.is_available():
#     device = torch.device("mps")
# else:
#     device = torch.device("cpu")
#     print("Mps is not available, using cpu instead")

def query(API_URL, payload):
    response = requests.post(API_URL, headers=headers, json=payload)

    #parse the model name from the API_URL
    model_name = API_URL.split("/")[-1]

    while response.status_code == 503:
        print(f"{model_name} is loading or rate limit has been reached, waiting 10 seconds")
        print(f"Error: {response.json()['error']}")
        sleep(10)
        response = requests.post(API_URL, headers=headers, json=payload)

    if response.status_code != 200:
        print(f"Error {response.status_code} in {model_name} API call")
        sys.exit(1)

    return response.json()
    
def classify_keywords(text_content, labels):
    bart_API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"
    deberta_API_URL = "https://api-inference.huggingface.co/models/MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"

    try:
        if not text_content.strip():
            return {}
    except:
        return {}

    text_content = translate_to_english(text_content)

    if not text_content.strip():
        return {}

    return_dict = {}

    for model_type, API_URL in [("bart", bart_API_URL), ("deberta", deberta_API_URL)]:
        output = ""

        # Divide labels into sublists with a maximum of 10 items each
        sublists = [labels[i:i+10] for i in range(0, len(labels), 10)]

        for sublist in sublists:
            # threshold = 1 / len(sublist)
            payload = {
                "inputs": text_content,
                "parameters": {
                    "candidate_labels": sublist
                }
            }
            response = query(API_URL, payload)

            zipped_response = zip(response['labels'], response['scores'])

            for label, score in zipped_response:
                # if float(score) > threshold:
                output = output + f"{label}: {score}\n"
            output = output + "-------------------\n"
        
        return_dict[model_type] = output

    return return_dict



def detect_language(text):
    API_URL = "https://api-inference.huggingface.co/models/papluca/xlm-roberta-base-language-detection"

    # Prepare the payload for the API request
    payload = {
        "inputs": text,
    }

    # Make the API request  
    response = query(API_URL, payload)

    return response[0][0]["label"]



def translate_to_english(text):
    API_URL = "https://api-inference.huggingface.co/models/facebook/mbart-large-50-many-to-one-mmt"
    src_lang = detect_language(text)

    if src_lang == "en":
        return text

    src_lang = map_language_code(src_lang)

    if not src_lang:
        return ""

    # Prepare the payload for the API request
    payload = {
        "inputs": text,
    }

    # Make the API request
    response = query(API_URL, payload)

    # Extract the translated text from the response
    translated_text = response[0]["generated_text"]

    return translated_text

def map_language_code(two_char_code):
    language_map = {
        "ar": "ar_AR",  # Arabic
        "de": "de_DE",  # German
        "en": "en_XX",  # English
        "es": "es_XX",  # Spanish
        "fr": "fr_XX",  # French
        "hi": "hi_IN",  # Hindi
        "it": "it_IT",  # Italian
        "ja": "ja_XX",  # Japanese
        "nl": "nl_XX",  # Dutch
        "pl": "pl_PL",  # Polish
        "pt": "pt_XX",  # Portuguese
        "ru": "ru_RU",  # Russian
        "sw": "sw_KE",  # Swahili
        "th": "th_TH",  # Thai
        "tr": "tr_TR",  # Turkish
        "ur": "ur_PK",  # Urdu
        "vi": "vi_VN",  # Vietnamese
        "zh": "zh_CN",  # Chinese
    }

    return language_map.get(two_char_code)