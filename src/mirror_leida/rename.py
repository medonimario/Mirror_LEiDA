import os
import re

# Define the root directory
root_dir = "data/raw"
folders = ["PPT1", "PPT2"]

# Function to rename files
def rename_files(folder):
    folder_path = os.path.join(root_dir, folder)
    if not os.path.exists(folder_path):
        print(f"Folder {folder_path} does not exist.")
        return

    for filename in os.listdir(folder_path):
        old_path = os.path.join(folder_path, filename)
        if not os.path.isfile(old_path):
            continue

        # Remove "_AverageReref"
        new_name = filename.replace("_AverageReref", "")

        if folder == "PPT1":
            # Change numbering from 01, 02, 03 to 101, 102, 103
            new_name = re.sub(r"^s_(\d{2})_", lambda m: f"s_1{m.group(1)}_", new_name)

        elif folder == "PPT2":
            # Ensure underscore after number
            new_name = re.sub(r"^s_(\d{3})([A-Za-z])", r"s_\1_\2", new_name)

        new_path = os.path.join(folder_path, new_name)
        
        # Rename the file
        os.rename(old_path, new_path)
        print(f"Renamed: {filename} -> {new_name}")

# Process both folders
for folder in folders:
    rename_files(folder)

print("Renaming complete!")
