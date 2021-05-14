# -*- coding: utf-8 -*-
"""
Created on Wed May 12 13:47:26 2021

@author: david
"""

import numpy as np
import os
import fnmatch
import random

from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img

class DataSplitter:
  def __init__(self, data_dir, batch_size, validation_split):
    self.data_dir = data_dir
    self.batch_size = batch_size
    set_paths = {"vv": [],
                 "vh": [],
                 "flood_label": [] }
    
    # organize image path names into dictionary structure
    for subdir, dirs, files in os.walk(data_dir):
      category = os.path.basename(subdir)
      if category in set_paths.keys(): 
        files = fnmatch.filter(files, '*.png')
        for f in files:
          set_paths[category].append(os.path.join(subdir, f))
    
    # random shuffle and split
    n_imgs = len(set_paths['vv'])
    idx = list(range(n_imgs));
    random.shuffle(idx);
    
    n_val = int(validation_split * n_imgs)
    idx_tr = idx[:-n_val]
    idx_val = idx[-n_val:]
    
    self.set_paths_tr = {"vv": [set_paths["vv"][i] for i in idx_tr],
                         "vh": [set_paths["vh"][i] for i in idx_tr],
                         "flood_label": [set_paths["flood_label"][i] for i in idx_tr]}
    self.set_paths_val = {"vv": [set_paths["vv"][i] for i in idx_val],
                          "vh": [set_paths["vh"][i] for i in idx_val],
                          "flood_label": [set_paths["flood_label"][i] for i in idx_val]}
                          
  def getBatchGenerators(self):
    return BatchGenerator(self.set_paths_tr, self.batch_size), \
           BatchGenerator(self.set_paths_val, self.batch_size)
  

# Subclassed from the Keras API. Feeds model.fit batches of images that are
# converted into normalized [0,1) float32 arrays

# Updated to use augmentation for flipping UD / LR. Idea is we keep an index list 
# 4x the size of the actual file list, and let overflow correspond to the transformations

class BatchGenerator(keras.utils.Sequence):
  def __init__(self, set_paths, batch_size):
    self.length = len(set_paths['vv'])
    self.set_paths = set_paths
    self.batch_size = batch_size
    self.idxlist = np.arange(self.length*4)
    
  def __len__(self):
    return int(self.length*4 / self.batch_size)
  
  def __getitem__(self, batch_idx):
    x = np.zeros((self.batch_size, 256, 256, 2), dtype='float32')
    y = np.zeros((self.batch_size, 256, 256, 1), dtype='float32')
    idx_a = batch_idx * self.batch_size
    idx_b = idx_a + self.batch_size
    idx_batch = self.idxlist[idx_a : idx_b] #indices to fetch for this batch
    
    for i, k in enumerate(idx_batch):
      k_arr = k // 4
      vv_i = load_img(self.set_paths['vv'][k_arr], color_mode='grayscale') 
      vh_i = load_img(self.set_paths['vh'][k_arr], color_mode='grayscale')
      fl_i = load_img(self.set_paths['flood_label'][k_arr], color_mode='grayscale')
      x[i, :, :, 0] = np.asarray(vv_i) / 255
      x[i, :, :, 1] = np.asarray(vh_i) / 255
      y[i, ::] = np.expand_dims(fl_i, 2) / 255
      
      if k % 4 == 1:
        x = np.flip(x, 1)
        y = np.flip(y, 1)
        
      elif k % 4 == 2:
        x = np.flip(x, 2)
        y = np.flip(y, 2)
        
      elif k % 4 == 3:
        x = np.flip(np.flip(x, 1), 2)
        y = np.flip(np.flip(y, 1), 2)
        
    return x, y
  
  def on_epoch_end(self):
    random.shuffle(self.idxlist)
    

## in case we needed an iterator, which it turned out we didn't

# def __iter__(self):
#   for i in range(int(self.length / self.batch_size + 1)):
#     yield self.__getitem__(i)

# class BGIterator:
#   def __init__(self, batchGenerator):
#     self._batchGenerator = batchGenerator
#     self._index = 0
  
#   def __next__(self):
#     if self._index < self._batchGenerator.length / self._batchGenerator.batch_size:
#       return self._batchGenerator.__getitem__(self._index)
#     raise StopIteration
