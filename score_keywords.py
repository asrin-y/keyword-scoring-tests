import pandas as pd
from tqdm import tqdm
from huggingface_models import *
from tqdm import tqdm
import sys
import os
import traceback

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
    ["plant nursery", "commercial plant nursery", "seedlings", "wholesale plant nursery", "floral nursery", "horticulture construction", "horticulture producers", "Gardening Center"],
    ["Plant Nursery", "NOT Plant Nursery"],
    ["Garden Designer", "Landscaper", "Plant Nursery (Grower)", "Horticulture and Farm Supplies", "Plant Retailer"]
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
seo_descr_column_name = input("Enter SEO Description column name (type 'none' to skip, press Enter for default 'SEO Description'): ").strip() or "SEO Description"
short_descr_column_name = input("Enter Short Description column name (type 'none' to skip, press Enter for default 'Short Description'): ").strip() or "Short Description"
website_url_column_name = input("Enter Website URL column name (type 'none' to skip, press Enter for default 'Website'): ").strip() or "Website"

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

try:

    for text, url in tqdm(zip(texts_to_process, website_urls), total=int(company_count)):
        try:
            deberta_score_seo_short_desc = classify_keywords(text, keywords_list)
        except:
            traceback.print_exc()
            print("An error occured, continuing...")
            deberta_score_seo_short_desc = {}

        try:
            webpage_text = get_page_text(url)
        except:
            traceback.print_exc()
            print("An error occured, continuing...")
            webpage_text = ""

        llm_responses = []
        if webpage_text.strip():
            for llm_system_prompt in llm_system_prompts:
                try:
                    llm_response = prompt_llm(llm_system_prompt, webpage_text)
                except:
                    traceback.print_exc()
                    print("An error occured, continuing...")
                    llm_response = ""
                llm_responses.append(llm_response)
        else:
            llm_responses = [""] * len(llm_system_prompts)

        deberta_scores = []

        for llm_response in llm_responses:
            if llm_response.strip():
                try:
                    scores = classify_keywords(llm_response, keywords_list)
                except:
                    traceback.print_exc()
                    print("An error occured, continuing...")
                    scores = {}
                if scores:
                    deberta_score = scores["deberta"]
                else:
                    deberta_score = ""
                deberta_scores.append(deberta_score)
            else:
                deberta_scores.append("")

        # Append the all gathered data to the lists

        if deberta_score_seo_short_desc:
            deberta_scores_seo_and_short_desc.append(deberta_score_seo_short_desc["deberta"])
        else:
            deberta_scores_seo_and_short_desc.append("")

        deberta_scores_webpage.append(deberta_scores)
        
        webpage_texts.append(webpage_text)

        if webpage_text.strip():
            llm_webpage_summaries.append(llm_responses)

        else:
            llm_webpage_summaries.append([""] * len(llm_system_prompts))


except KeyboardInterrupt:
    print("Keyboard Interrupt. Exiting...")
    sys.exit(1)

except Exception:
    traceback.print_exc()

finally:
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
