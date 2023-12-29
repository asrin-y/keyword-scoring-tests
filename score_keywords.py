import pandas as pd
from tqdm import tqdm
from huggingface_models import *
from tqdm import tqdm
import sys
import os

llm_system_prompts = [
'''
    You are a helpful assistant and your role is to understand and explain what the company does
    by using the company website index text. Your output should be keywords or small sentences of the company’s product/services. 
    Use maximum 250 characters while summarizing it. Focus on what the company's products/services are.
''',
'''
    You are a helpful assistant and your role is to understand and explain what the company does
    by using the company website index text. Your output should be a list of keywords or small sentences.
    Use maximum 250 characters while summarizing it. Focus on what the company's products/services are.
'''
]


keywords_list = [
    ["Machinery for filtering or purifying liquids", "Machinery for filtering or purifying gas", "Agricultural Processing Machinery", "Agricultural and Farming Heavy Equipment", "Agricultural Machinery"],
    ["Machinery and apparatus for filtering or purifying liquids or gases", "Agricultural and Farming Machinery", "NOT Agricultual and Farming Machinery", "NOT Machinery and apparatus for filtering or purifying liquids or gases"],
    ["metal filtering equipment", "agricultural machine", "harvesting maching", "metal sprayer frames", "threshing machinery", "agricultural processing machines", "farming heavy equipments", "industrial gas filteing frames", "large filter housing"],
    ["Soil preparation or cultivation machinery (plows, harrows, seeders, transplanters).",
    "Harvesting or threshing machinery; mowers for lawns, parks or sports-grounds; machines for cleaning, sorting or grading eggs, fruit or other agricultural produce.",
    "Other agricultural, horticultural, forestry, poultry-keeping or bee-keeping machinery, including germination plant fitted with mechanical or thermal equipment; poultry incubators and brooders.",
    "Milking machines and dairy machinery.",
    "Mechanical appliances for projecting, dispersing or spraying liquids or powders; such as agricultural sprayers.",
    "Machines for cleaning, sorting or grading seed, grain or dried leguminous vegetables; machinery used in the milling industry for the working of cereals or dried leguminous vegetables, other than farm-type machinery.",
    "Wheeled tractors used in agriculture and forestry.",
    "Machinery and apparatus for filtering or purifying liquids or gases."]
]
 

for keywords in keywords_list:
    if len(keywords) > 10:
        print(f"There are {len(keywords)} keywords in one of the lists. The maximum number of keywords per list is 10.")
        print(f"List: {keywords}")
        sys.exit(1)
        

file_name = input("Enter file name: ")

while not os.path.isfile(file_name):
    print("File not found")
    file_name = input("Enter file name: ")

file_extension = os.path.splitext(file_name)[1]

if file_extension == '.csv':
    df = pd.read_csv(file_name)
elif file_extension == '.xlsx':
    df = pd.read_excel(file_name)
else:
    print("Invalid file format. Only CSV and XLSX files are supported.")
    sys.exit(1)

# Column names input with 'none' check
seo_descr_column_name = input("Enter SEO Description column name (type 'none' to skip, press Enter for default 'SEO Description', or type 'none' to skip): ").strip() or "SEO Description"
short_descr_column_name = input("Enter Short Description column name (type 'none' to skip, press Enter for default 'Short Description', or type 'none' to skip): ").strip() or "Short Description"
website_url_column_name = input("Enter Website URL column name (type 'none' to skip, press Enter for default 'Website', or type 'none' to skip): ").strip() or "Website"

company_count = input("Enter number of companies to process (press Enter to process all): ")
if not company_count.strip():
    company_count = len(df)

# Conditional checks and data processing based on column names
seo_descriptions = []
short_descriptions = []
website_urls = []

if seo_descr_column_name.lower() != 'none':
    try:
        df[seo_descr_column_name] = df[seo_descr_column_name].fillna('')
        seo_descriptions = df[seo_descr_column_name].tolist()[:int(company_count)]
    except KeyError:
        print(f"Invalid column name: {seo_descr_column_name}")
        sys.exit(1)
else:
    seo_descriptions = [''] * int(company_count)

if short_descr_column_name.lower() != 'none':
    try:
        df[short_descr_column_name] = df[short_descr_column_name].fillna('')
        short_descriptions = df[short_descr_column_name].tolist()[:int(company_count)]
    except KeyError:
        print(f"Invalid column name: {short_descr_column_name}")
        sys.exit(1)
else:
    short_descriptions = [''] * int(company_count)

if website_url_column_name.lower() != 'none':
    try:
        df[website_url_column_name] = df[website_url_column_name].fillna('')
        website_urls = df[website_url_column_name].tolist()[:int(company_count)]
    except KeyError:
        print(f"Invalid column name: {website_url_column_name}")
        sys.exit(1)
else:
    website_urls = [''] * int(company_count)

# Rest of the processing
texts_to_process = [f"{seo_desc} {short_desc}" for seo_desc, short_desc in zip(seo_descriptions, short_descriptions)]

deberta_scores_seo_and_short_desc = []
deberta_scores_webpage = []
webpage_texts = []
llm_webpage_summaries = []
webpage_scores = []

for text, url in tqdm(zip(texts_to_process, website_urls), total=int(company_count)):
    debarta_score = classify_keywords(text, keywords_list)

    if debarta_score:
        deberta_scores_seo_and_short_desc.append(debarta_score["deberta"])
    else:
        deberta_scores_seo_and_short_desc.append("")

    webpage_text = get_page_text(url)
    webpage_texts.append(webpage_text)

    llm_responses = []
    if webpage_text.strip():
        for llm_system_prompt in llm_system_prompts:
            llm_response = prompt_llm(llm_system_prompt, webpage_text)
            llm_responses.append(llm_response)

        llm_webpage_summaries.append(llm_responses)

    else:
        llm_webpage_summaries.append([""] * len(llm_system_prompts))
        llm_responses = [""] * len(llm_system_prompts)

    deberta_scores = []

    for llm_response in llm_responses:
        if llm_response.strip():
            scores = classify_keywords(llm_response, keywords_list)
            deberta_score = scores["deberta"]
            deberta_scores.append(deberta_score)
        else:
            deberta_scores.append("")

    deberta_scores_webpage.append(deberta_scores)

df = pd.DataFrame()

if seo_descr_column_name.lower() != 'none' or short_descr_column_name.lower() != 'none':
    df["Scored Text"] = [""] + texts_to_process
    df["Score of SEO Description and Short Description"] = [""] + deberta_scores_seo_and_short_desc
df["Webpage Text"] = [""] + webpage_texts
for i, llm_system_prompt in enumerate(llm_system_prompts):
    df[f"LLM Webpage Summary {i+1}"] = [llm_system_prompt] + [llm_responses[i] for llm_responses in llm_webpage_summaries]
    df[f"Score of LLM Webpage Summary {i+1}"] = [""] + [deberta_scores[i] for deberta_scores in deberta_scores_webpage]

# Save the output file with an incremented number if it already exists
output_file = "scored.csv"
count = 1
while os.path.isfile(output_file):
    output_file = f"scored({count}).csv"
    count += 1

print(f"Saving output to {output_file}")
df.to_csv(output_file, index=False)
