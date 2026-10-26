import pandas as pd
from pathlib import Path
current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parent.parent.parent.parent

# Combine with the path to the target tsv file
file_path = root_dir / "_student_resource" / "student_resource" / "dataset" / "train" / "Preprocessed"/"train_source1_clean.tsv"

# Read the file
df = pd.read_csv(file_path, sep="\t")
print(df.columns)