import time
from pathlib import Path

# --- RECALL CALCULATION ---
gt_path = Path("/home/rover/Amazon ML challenge - Copy/_student_resource/student_resource/dataset/train/train_ground_truth.tsv")
pred_path = Path("/home/rover/Amazon ML challenge - Copy/Team Noir_submission/code/output/candidate_pairs.tsv")

print(f"\nEvaluating Recall against {gt_path.name}...")
eval_start = time.time()

# 1. Load ground truth into memory (it's only 127 MB, perfectly safe)
print("Loading ground truth...")
gt_dict = {}
with open(gt_path, 'r', encoding='utf-8') as f:
    header = f.readline()
    for line in f:
        line = line.strip()
        if not line: continue
        parts = line.split('\t')
        s1 = parts[0]
        if len(parts) > 1 and parts[1]:
            matches = set(parts[1].split(','))
        else:
            matches = set()
        if matches:
            gt_dict[s1] = matches 

total_true = sum(len(matches) for matches in gt_dict.values())
total_found = 0

print(f"Total True Matches to find: {total_true}")
print("Streaming predictions from disk (saves memory)...")

# 2. Stream predictions line by line to calculate intersection
with open(pred_path, 'r', encoding='utf-8') as f:
    header = f.readline()
    for line in f:
        line = line.strip()
        if not line: continue
        
        parts = line.split('\t')
        s1 = parts[0]
        
        # We only care if this S1 is in our ground truth
        if s1 in gt_dict:
            true_matches = gt_dict[s1]
            if len(parts) > 1 and parts[1]:
                cands = parts[1].split(',')
                # Check intersection manually to avoid creating massive sets
                for c in cands:
                    if c in true_matches:
                        total_found += 1

recall = (total_found / total_true) * 100 if total_true > 0 else 0
print(f"Total Ground Truth Matches: {total_true}")
print(f"Total Found Matches:        {total_found}")
print(f"Blocking Recall:            {recall:.2f}%")
print(f"Evaluation Time:            {time.time() - eval_start:.2f} seconds")
