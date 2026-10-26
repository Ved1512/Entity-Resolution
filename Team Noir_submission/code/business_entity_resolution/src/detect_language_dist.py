import pandas as pd
from langdetect import detect, DetectorFactory
from collections import Counter
import time

# Ensure deterministic language detection
DetectorFactory.seed = 0

print("Loading foreign businesses dataset...")
file_path = "../../output/foreign_businesses.tsv"
df = pd.read_csv(file_path, sep='\t', dtype=str)

# Combine name and address to give langdetect more context
df['text'] = df['business_name'].fillna('') + " " + df['business_address'].fillna('')

# We take a representative random sample of 25,000 rows to calculate the distribution 
# (running langdetect on all 2 million rows would take hours).
sample_size = min(25000, len(df))
print(f"Randomly sampling {sample_size} rows for statistical language distribution...")
sample_df = df.sample(n=sample_size, random_state=42)

lang_counts = Counter()
start = time.time()

print("Detecting languages...")
for i, text in enumerate(sample_df['text']):
    text = text.strip()
    if len(text) > 2:
        try:
            lang = detect(text)
            lang_counts[lang] += 1
        except:
            lang_counts['unknown'] += 1
    else:
        lang_counts['unknown'] += 1
        
    if (i+1) % 5000 == 0:
        print(f"  Processed {i+1}/{sample_size}...")

print(f"\nDetection finished in {time.time() - start:.2f} seconds.")

total_detected = sum(lang_counts.values())

print("\n--- ESTIMATED LANGUAGE DISTRIBUTION (Foreign Subset) ---")
for lang, count in lang_counts.most_common(15):
    percentage = (count / total_detected) * 100
    print(f"{lang.upper()}: {percentage:.2f}%")
