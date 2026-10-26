import pandas as pd
from pathlib import Path

raw_dir = Path("/home/rover/Amazon ML challenge - Copy/_student_resource/student_resource/dataset/train")

for fname in ["train_source1.tsv", "train_source2.tsv", "train_source3.tsv"]:
    file_path = raw_dir / fname
    print(f"Analyzing {fname}...")
    df = pd.read_csv(file_path, sep='\t', usecols=['business_name', 'business_address'], dtype=str)
    
    total = len(df)
    
    # Check how many rows have absolutely NO english letters
    no_english_name = ~df['business_name'].fillna("").str.contains(r'[a-zA-Z]', regex=True)
    no_english_addr = ~df['business_address'].fillna("").str.contains(r'[a-zA-Z]', regex=True)
    
    # Check how many rows contain non-ASCII characters (Foreign scripts like Arabic, Chinese, etc.)
    # We ignore standard ASCII 0-127.
    foreign_name = df['business_name'].fillna("").str.contains(r'[^\x00-\x7F]', regex=True)
    foreign_addr = df['business_address'].fillna("").str.contains(r'[^\x00-\x7F]', regex=True)
    
    print(f"Total Rows: {total}")
    print(f"Names with NO English letters: {no_english_name.sum()} ({no_english_name.sum()/total*100:.2f}%)")
    print(f"Addresses with NO English letters: {no_english_addr.sum()} ({no_english_addr.sum()/total*100:.2f}%)")
    print(f"Names containing Foreign/Non-ASCII characters: {foreign_name.sum()} ({foreign_name.sum()/total*100:.2f}%)")
    print(f"Addresses containing Foreign/Non-ASCII characters: {foreign_addr.sum()} ({foreign_addr.sum()/total*100:.2f}%)")
