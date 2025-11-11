import os
import shutil
import kagglehub

print("Checking for dataset in KaggleHub cache...")
path = kagglehub.dataset_download("pinstripezebra/google-streetview-top-50-us-cities")
print("Dataset ready at:", path)

SRC_DIR = os.path.join(path, "Images")
DEST_DIR = os.path.abspath("data")  

os.makedirs(DEST_DIR, exist_ok=True)
print(f"Copying images (recursively) into {DEST_DIR} ...")

count = 0
for root, dirs, files in os.walk(SRC_DIR):
    for file in files:
        if file.lower().endswith(".jpg"):
            src = os.path.join(root, file)
            dst = os.path.join(DEST_DIR, file)
            if not os.path.exists(dst):  
                shutil.copy2(src, dst)
                count += 1
print(f"Copied {count} images into {DEST_DIR}")

def organize_by_city():
    moved = 0
    for file in os.listdir(DEST_DIR):
        if not file.lower().endswith(".jpg"):
            continue
        city = file.split("_")[0].strip()
        city_folder = os.path.join(DEST_DIR, city)
        os.makedirs(city_folder, exist_ok=True)
        src = os.path.join(DEST_DIR, file)
        dst = os.path.join(city_folder, file)
        shutil.move(src, dst)
        moved += 1
        print(f"Moved {file} → {city_folder}")
    print(f"Sorted {moved} images into city folders.")

if __name__ == "__main__":
    organize_by_city()
    print("All done! Dataset organized successfully in ./data/")
