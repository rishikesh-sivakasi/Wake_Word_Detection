import os
import subprocess
import numpy as np

DATASET_IN = "/home/arz-1029/wake_word_detection/python_training_&_feature_extraction/dataset/test"
DATASET_OUT = "/home/arz-1029/wake_word_detection/python_training_&_feature_extraction/cpp_dataset/test"
os.makedirs(DATASET_OUT, exist_ok=True)

# 1. Count the files first so we know how much work we have
total_files = 0
for folder in ["negative", "positive"]:
    in_folder = os.path.join(DATASET_IN, folder)
    if os.path.exists(in_folder):
        total_files += len([f for f in os.listdir(in_folder) if f.endswith(".wav")])

print(f"1. Converting {total_files} WAVs to C++ CSVs...")
print("   (This takes a few minutes because SD card writing is slow)\n")

# 2. Convert and track progress
processed = 0
for folder in ["negative", "positive"]:
    in_folder = os.path.join(DATASET_IN, folder)
    out_folder = os.path.join(DATASET_OUT, folder)
    os.makedirs(out_folder, exist_ok=True)
    
    if not os.path.exists(in_folder): continue
    
    for file in os.listdir(in_folder):
        if file.endswith(".wav"):
            in_path = os.path.join(in_folder, file)
            out_path = os.path.join(out_folder, file.replace(".wav", ".csv"))
            
            # Run the C++ extractor
            subprocess.run(["./dataset_extractor", in_path, out_path])
            
            # Update the UI on the same line
            processed += 1
            if processed % 10 == 0 or processed == total_files:
                print(f"\r   Progress: {processed} / {total_files} files converted...", end="", flush=True)

print("\n\n2. Calculating True C++ Magic Numbers...")
all_mfccs = []
for root, _, files in os.walk(DATASET_OUT):
    for file in files:
        if file.endswith(".csv"):
            mfcc = np.loadtxt(os.path.join(root, file), delimiter=',')
            all_mfccs.append(mfcc)

stacked = np.hstack(all_mfccs)
print("\n=== YOUR ULTIMATE MAGIC NUMBERS ===")
print(f"GLOBAL_MEAN = {np.mean(stacked)}f;")
print(f"GLOBAL_STD  = {np.std(stacked)}f;")
print("===================================\n")