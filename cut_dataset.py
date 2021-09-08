import glob
import os


disease_dirs = glob.glob("split/ptb-images-2-cropped/*/*")

for d in disease_dirs:
    all_files = sorted(glob.glob(f"{d}/*"))
    print(len(all_files))
    # Remove the first 1000 files we do not want to delete
    to_delete = all_files[10000:]
    for f in to_delete:
        os.remove(f)
