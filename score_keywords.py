import pandas as pd
from tqdm import tqdm
from huggingface_models import *
from tqdm import tqdm
import sys
import os

zephyr_system_prompt = '''
    You are a helpful assistant and your role is to understand and explain what the company does 
    BY USING THE COMPANY WEBSITE INDEX TEXT. Use a maximum of 250 characters while summarizing it. 
    Focus on what the company's products/services are. Give me the result in keywords or small sentences.
'''

keywords_list = [
    ["Machinery for filtering or purifying liquids", "Machinery for filtering or purifying gas", "Agricultural Processing Machinery", "Agricultural and Farming Heavy Equipment", "Agricultural Machinery"],
    ["Machinery and apparatus for filtering or purifying liquids or gases", "Agricultural and Farming Machinery", "NOT Agricultual and Farming Machinery", "NOT Machinery and apparatus for filtering or purifying liquids or gases"],
    ["metal filtering equipment", "agricultural machine", "harvesting maching", "metal sprayer frames", "threshing machinery", "agricultural processing machines", "farming heavy equipments", "industrial gas filteing frames", "large filter housing"]
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

df = pd.read_csv(file_name)

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
zephyr_webpage_summaries = []
webpage_scores = []

for text, url in tqdm(zip(texts_to_process, website_urls), total=int(company_count)):
    debarta_score = classify_keywords(text, keywords_list)

    if debarta_score:
        deberta_scores_seo_and_short_desc.append(debarta_score["deberta"])
    else:
        deberta_scores_seo_and_short_desc.append("")

    webpage_text = get_page_text(url)
    webpage_texts.append(webpage_text)

    zephyr_response = ""
    if webpage_text.strip():
        zephyr_response = prompt_zephyr(zephyr_system_prompt, webpage_text)

    zephyr_webpage_summaries.append(zephyr_response)

    deberta_score = ""
    if zephyr_response.strip():
        scores = classify_keywords(zephyr_response, keywords_list)
        deberta_score = scores["deberta"]

    deberta_scores_webpage.append(deberta_score)

df = pd.DataFrame()

df["Scored Text"] = texts_to_process
df["Score of SEO Description and Short Description"] = deberta_scores_seo_and_short_desc
df["Webpage Text"] = webpage_texts
df["Zephyr Webpage Summary"] = zephyr_webpage_summaries
df["Score of Webpage Summary"] = deberta_scores_webpage

# Save the output file with an incremented number if it already exists
output_file = "scored.csv"
count = 1
while os.path.isfile(output_file):
    output_file = f"scored({count}).csv"
    count += 1

print(f"Saving output to {output_file}")
df.to_csv(output_file, index=False)
