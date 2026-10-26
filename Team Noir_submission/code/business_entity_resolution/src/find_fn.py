import pandas as pd
import random
from pathlib import Path
from collections import defaultdict
import time

print("Loading Ground Truth into memory...")
gt_path = Path("/home/rover/Amazon ML challenge - Copy/_student_resource/student_resource/dataset/train/train_ground_truth.tsv")
pred_path = Path("/home/rover/Amazon ML challenge - Copy/Team Noir_submission/code/output/candidate_pairs.tsv")

gt_dict = defaultdict(set)
total_gt = 0

with open(gt_path, 'r', encoding='utf-8') as f:
    next(f)
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) == 2:
            s1 = parts[0]
            cands = parts[1].split(',')
            for c in cands:
                gt_dict[s1].add(c)
                total_gt += 1

print(f"Total Ground Truth Pairs: {total_gt}")

print("Streaming predictions from disk and removing found items...")
start = time.time()
with open(pred_path, 'r', encoding='utf-8') as f:
    next(f) # skip header
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) == 2:
            s1 = parts[0]
            if s1 in gt_dict:
                cands = parts[1].split(',')
                # Fast set removal
                gt_dict[s1].difference_update(cands)

print(f"Streaming finished in {time.time() - start:.2f} seconds.")

# Collect remaining items as False Negatives
false_negatives = []
for s1, missed_cands in gt_dict.items():
    for c in missed_cands:
        false_negatives.append([s1, c])

print(f"Total False Negatives (Missed Matches): {len(false_negatives)}")
print(f"Recall: {100 * (1 - len(false_negatives)/total_gt):.2f}%")

sample_size = min(25000, len(false_negatives))
fn_sample = random.sample(false_negatives, sample_size)

out_sample_path = Path("fn_sample.csv")
pd.DataFrame(fn_sample, columns=["s1_id", "cand_id"]).to_csv(out_sample_path, index=False)
print(f"Saved {sample_size} False Negatives to fn_sample.csv for deeper string analysis.")
