import pandas as pd
from tqdm import tqdm
from huggingface_models import *
from tqdm import tqdm
import sys
import os

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
company_count = input("Enter number of companies to process (press Enter to process all): ")
if not company_count.strip():
    company_count = len(df)

try:
    seo_descriptions = df[seo_descr_column_name].tolist()[:int(company_count)]
except:
    print(f"Invalid column name: {seo_descr_column_name}")
    sys.exit(1)
try: 
    short_descriptions = df[short_descr_column_name].tolist()[:int(company_count)]
except:
    print(f"Invalid column name: {short_descr_column_name}")
    sys.exit(1)

texts_to_process = [f"{seo_desc} {short_desc}" for seo_desc, short_desc in zip(seo_descriptions, short_descriptions)]

bart_scores = []
deberta_scores = []

for text in tqdm(texts_to_process):

    score = classify_keywords(text, labels)

    if not score:
        bart_scores.append("")
        deberta_scores.append("")
        continue

    bart_scores.append(score["bart"])
    deberta_scores.append(score["deberta"])

df = pd.DataFrame()

df["Scored Text"] = texts_to_process
df["BART Score"] = bart_scores
df["DeBERTa Score"] = deberta_scores

# Save the output file with an incremented number if it already exists
output_file = "scored.csv"
count = 1
while os.path.isfile(output_file):
    output_file = f"scored({count}).csv"
    count += 1

print(f"Saving output to {output_file}")
df.to_csv(output_file, index=False)
