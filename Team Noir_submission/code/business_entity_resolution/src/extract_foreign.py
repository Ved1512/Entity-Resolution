import pandas as pd
from pathlib import Path

raw_dir = Path("/home/rover/Amazon ML challenge - Copy/_student_resource/student_resource/dataset/train")
output_dir = Path("../../output")
output_dir.mkdir(parents=True, exist_ok=True)
out_file = output_dir / "foreign_businesses.tsv"

print(f"Extracting non-English businesses to {out_file}...")

# Write header
header = ['entity_id', 'business_name', 'business_address', 'country', 'source_file']
with open(out_file, 'w', encoding='utf-8') as f:
    f.write('\t'.join(header) + '\n')

# Source 1 was confirmed to be 100% English, so we only need to scan 2 and 3.
files_to_check = ["train_source2.tsv", "train_source3.tsv"]
cols = ['entity_id', 'business_name', 'business_address', 'country']
total_extracted = 0

for fname in files_to_check:
    print(f"Scanning {fname}...")
    # Load raw dataset
    df = pd.read_csv(raw_dir / fname, sep='\t', usecols=cols, dtype=str)
    
    names = df['business_name'].fillna("")
    addresses = df['business_address'].fillna("")
    
    # Regex: anything outside standard ASCII (0-127) is flagged as foreign script
    mask_name = names.str.contains(r'[^\x00-\x7F]', regex=True)
    mask_addr = addresses.str.contains(r'[^\x00-\x7F]', regex=True)
    
    # Find rows where EITHER the name or the address contains a foreign character
    foreign_mask = mask_name | mask_addr
    
    foreign_df = df[foreign_mask].copy()
    foreign_df['source_file'] = fname
    
    # Append safely to disk to preserve memory
    foreign_df.to_csv(out_file, sep='\t', index=False, header=False, mode='a')
    
    found = len(foreign_df)
    total_extracted += found
    print(f"  Found {found} foreign entries.")
    
    del df, names, addresses, mask_name, mask_addr, foreign_mask, foreign_df

print(f"\nTotal foreign businesses safely extracted to disk: {total_extracted}")
print("Done!")
