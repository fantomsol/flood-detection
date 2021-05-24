import numpy as np
import os
import fnmatch
from PIL import Image

def _blacklist(img):
  ### write your logic here to filter out images. 
  ### Since this is a blacklist, True -> removed
  ### This method should not be called by the user.
  median = np.median(img)
  return median > 254 or median < 1
  
def filter(data_dir, category):
  ### filter function. is called to filter the images in "data_dir"
  ### based on condition in "_blacklist" applied to the images of "category".
  ### Category should be one of "flood_label", "vh", "vv" or "water_body_label"]
  ###
  ### NOTE that this function deletes images from your file system. Back up your data
  ### if you don't want to download it again to restore filtering.
  
  cats = ["flood_label", "vh", "vv", "water_body_label"]
  if not category in cats:
    print("Invadlid category. Please enter one of the following:")
    print(cats)
    return -1
  print("Indexing files...")
  rm_paths = []
  for setname in os.listdir(data_dir):
    idx = []
    tiles = os.path.join(data_dir, setname, "tiles")
    cc = os.path.join(tiles, category)
    files = os.listdir(cc)
    files = fnmatch.filter(files, "*.png")
    
    for i, f in enumerate(files):
      path = os.path.join(cc, f)
      img = Image.open(path)
      if _blacklist(img):
        idx.append(i)
    print(cc + ": " + f"{len(idx)}" + " of " + f"{len(files)}" + " noise.")
    
    categories = [os.path.join(tiles, c) for c in cats]
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

