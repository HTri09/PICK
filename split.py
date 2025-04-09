import os
import shutil
import random
from pathlib import Path
import pandas as pd

# Define paths
data_root = Path("my_data")
output_root = Path("split_data")
train_ratio = 0.8  # Ratio of training data

# Ensure output directories exist
train_dir = output_root / "train"
test_dir = output_root / "test"
for subdir in [train_dir, test_dir]:
    (subdir / "boxes_and_transcripts").mkdir(parents=True, exist_ok=True)
    (subdir / "images").mkdir(parents=True, exist_ok=True)
    (subdir / "entities").mkdir(parents=True, exist_ok=True)

# Get all file basenames (without extensions) from the boxes_and_transcripts folder
boxes_and_transcripts_folder = data_root / "boxes_and_transcripts"
all_files = [f.stem for f in boxes_and_transcripts_folder.glob("*.csv")]

# Shuffle and split the dataset
random.shuffle(all_files)
split_index = int(len(all_files) * train_ratio)
train_files = all_files[:split_index]
test_files = all_files[split_index:]

# Helper function to copy files
def copy_files(file_list, dest_dir):
    for file in file_list:
        # Copy boxes_and_transcripts
        src = boxes_and_transcripts_folder / f"{file}.csv"
        dest = dest_dir / "boxes_and_transcripts" / f"{file}.csv"
        shutil.copy(src, dest)

        # Copy images
        image_src = next((data_root / "images").glob(f"{file}.*"), None)
        if image_src:
            image_dest = dest_dir / "images" / image_src.name
            shutil.copy(image_src, image_dest)

        # Copy entities
        entity_src = data_root / "entities" / f"{file}.txt"
        if entity_src.exists():
            entity_dest = dest_dir / "entities" / f"{file}.txt"
            shutil.copy(entity_src, entity_dest)

# Copy training and testing files
copy_files(train_files, train_dir)
copy_files(test_files, test_dir)

# Create files_name.csv for train and test
def create_files_name_csv(file_list, dest_dir):
    data = [{"index": i, "document_class": "product_packaging", "file_name": file} for i, file in enumerate(file_list)]
    df = pd.DataFrame(data)
    df.to_csv(dest_dir / "train_samples_list.csv", index=False, header=False)

create_files_name_csv(train_files, train_dir)
create_files_name_csv(test_files, test_dir)

print(f"Dataset split completed. Training and testing data saved in {output_root}.")