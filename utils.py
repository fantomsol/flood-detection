import numpy as np
import os
import fnmatch
from PIL import Image

## remove "noise" pictures from directory data_dir
def purgeNoise(data_dir):
  print("Indexing files...")
  rm_paths = []
  cats = ["flood_label", "vh", "vv", "water_body_label"]
  for setname in os.listdir(data_dir):
    idx = []
    tiles = os.path.join(data_dir, setname, "tiles")
    categories = [os.path.join(tiles, c) for c in cats]
    files = os.listdir(categories[1])
    files = fnmatch.filter(files, "*.png")
    
    for i, f in enumerate(files):
      path = os.path.join(categories[1], f)
      img = Image.open(path)
      if np.median(img) > 254:
        idx.append(i)
    print(categories[1] + ": " + f"{len(idx)}" + " of " + f"{len(files)}" + " noise.")
    
    for cat in categories:
      files = os.listdir(cat)
      files = fnmatch.filter(files, "*.png")
      for i in idx:
        rm_paths.append(os.path.join(cat, files[i]))
  
  print(f"Found {len(rm_paths)} images treated as noise.")
  prompt = input("Would you like to delete these?(y/n)")
  
  if prompt == "y":
    print("Deleting...")
    for f in rm_paths:
      if os.path.exists(f): os.remove(f)
      else: print("ajaj")
    print("Done.")
  else:
    print("Abort.")
  
