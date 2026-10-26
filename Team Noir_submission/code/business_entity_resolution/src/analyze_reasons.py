import pandas as pd
import numpy as np
from pathlib import Path
import json

print("Loading false negative sample...")
fn_df = pd.read_csv("fn_sample.csv", dtype=str)

# Extract unique IDs needed
needed_ids = set(fn_df['s1_id']).union(set(fn_df['cand_id']))
print(f"Need to lookup {len(needed_ids)} unique entities...")

# Load raw datasets to get the text
raw_dir = Path("/home/rover/Amazon ML challenge - Copy/_student_resource/student_resource/dataset/train")
entity_dict = {}

cols = ['entity_id', 'business_name', 'business_address']
for fname in ["train_source1.tsv", "train_source2.tsv", "train_source3.tsv"]:
    print(f"Scanning {fname} for strings...")
    df = pd.read_csv(raw_dir / fname, sep='\t', usecols=cols, dtype=str)
    
    # Filter only needed rows to save memory/time
    mask = df['entity_id'].isin(needed_ids)
    df_filtered = df[mask]
    
    for _, row in df_filtered.iterrows():
        name = str(row['business_name']) if pd.notna(row['business_name']) else ""
        addr = str(row['business_address']) if pd.notna(row['business_address']) else ""
        entity_dict[row['entity_id']] = (name, addr)

def get_bigrams(text):
    text = str(text).lower()
    return set([text[i:i+2] for i in range(len(text)-1)]) if len(text) >= 2 else set()

def jaccard(s1, s2):
    if not s1 or not s2: return 0.0
    u = len(s1.union(s2))
    return len(s1.intersection(s2)) / u if u > 0 else 0.0

def has_foreign(text):
    import re
    return bool(re.search(r'[^\x00-\x7F]', text))

reasons = {
    "Missing Data (Null String)": 0,
    "Language/Script Mismatch": 0,
    "Completely Different (Jaccard < 0.2)": 0,
    "Heavy Typos (Jaccard 0.2 - 0.4)": 0,
    "Near Miss (Jaccard > 0.4)": 0
}

print("Analyzing failure reasons...")
for _, row in fn_df.iterrows():
    s1, cand = row['s1_id'], row['cand_id']
    t1 = entity_dict.get(s1, ("", ""))
    t2 = entity_dict.get(cand, ("", ""))
    
    str1 = (t1[0] + " " + t1[1]).strip()
    str2 = (t2[0] + " " + t2[1]).strip()
    
    if not str1 or not str2:
        reasons["Missing Data (Null String)"] += 1
        continue
        
    if has_foreign(str1) != has_foreign(str2):
        reasons["Language/Script Mismatch"] += 1
        continue
        
    b1 = get_bigrams(str1)
    b2 = get_bigrams(str2)
    sim = jaccard(b1, b2)
    
    if sim < 0.2:
        reasons["Completely Different (Jaccard < 0.2)"] += 1
    elif sim < 0.4:
        reasons["Heavy Typos (Jaccard 0.2 - 0.4)"] += 1
    else:
        reasons["Near Miss (Jaccard > 0.4)"] += 1

total = sum(reasons.values())
results = {k: round(v/total * 100, 2) for k, v in reasons.items()}

with open('fn_analysis.json', 'w') as f:
    json.dump(results, f)

print(json.dumps(results, indent=2))
