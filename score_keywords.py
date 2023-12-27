import pandas as pd
from tqdm import tqdm
from huggingface_models import *
from tqdm import tqdm
import sys
import os

zephyr_system_prompt = '''
    You are an assistant that helps people find information about companies. I will give you all text in company's webpage
    and you will give me a summary of the company and keywords that describe it. You response should be approximately 50 words long.
    Give short and concise answers.
'''

labels = [
    "Baby Apparel",
    "Home Textiles",
    "Towels & Bathrobes",
    "Knitwear",
    "Woven Fabrics",
    "Children's Clothing",
    "Sportswear & Activewear",
    "Women's Apparel",
    "Men's Apparel",
    "Workwear & Uniforms",
    "Fashion Accessories",
    "Footwear",
    "Outerwear",
    "Undergarments",
    "Ethnic & Traditional Wear",
    "Formal Wear",
    "Casual Wear",
    "Sleepwear & Loungewear",
    "Bridal & Ceremonial",
    "Organic & Sustainable Textiles",
    "Textile Raw Materials",
    "Dyeing & Printing",
    "Textile Machinery",
    "Technical Textiles",
    "Home Decor",
    "Fabric Care & Maintenance",
    "Handcrafted Textiles",
    "Leather Goods",
    "Outdoor & Adventure Gear",
    "Corporate & Promotional Apparel"]

file_name = input("Enter file name: ")

while not os.path.isfile(file_name):
    print("File not found")
    file_name = input("Enter file name: ")

df = pd.read_csv(file_name)

seo_descr_column_name = input("Enter SEO Description column name (press Enter to use default 'SEO Description'): ") or "SEO Description"
short_descr_column_name = input("Enter Short Description column name (press Enter to use default 'Short Description'): ") or "Short Description"
website_url_column_name = input("Enter Website URL column name (press Enter to use default 'Website'): ") or "Website"

company_count = input("Enter number of companies to process (press Enter to process all): ")
if not company_count.strip():
    company_count = len(df)

try:
    df[seo_descr_column_name] = df[seo_descr_column_name].fillna('')
    seo_descriptions = df[seo_descr_column_name].tolist()[:int(company_count)]
except:
    print(f"Invalid column name: {seo_descr_column_name}")
    sys.exit(1)
try:
    df[short_descr_column_name] = df[short_descr_column_name].fillna('')
    short_descriptions = df[short_descr_column_name].tolist()[:int(company_count)]
except:
    print(f"Invalid column name: {short_descr_column_name}")
    sys.exit(1)
try:
    df[website_url_column_name] = df[website_url_column_name].fillna('')
    website_urls = df[website_url_column_name].tolist()[:int(company_count)]
except:
    print(f"Invalid column name: {website_url_column_name}")
    sys.exit(1)

texts_to_process = [f"{seo_desc} {short_desc}" for seo_desc, short_desc in zip(seo_descriptions, short_descriptions)]

deberta_scores_seo_and_short_desc = []
deberta_scores_webpage = []
webpage_texts = []
zephyr_webpage_summaries = []
webpage_scores = []

for text, url in tqdm(zip(texts_to_process, website_urls), total=int(company_count)):
    debarta_score = classify_keywords(text, labels)

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
        scores = classify_keywords(zephyr_response, labels)
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
