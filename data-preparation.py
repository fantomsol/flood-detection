# -*- coding: utf-8 -*-
"""
Created on Wed May  5 12:13:11 2021

@author: david
"""

import os
import fnmatch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def makeNpy():
  cwd = os.getcwd()
  rootdir = cwd + "/train_raw/train/"
  for setname in os.listdir(rootdir):
    print('Working on ' + setname)
    setdir = rootdir + setname + '/tiles/'
    for subdir, dirs, files in os.walk(setdir):
      category = os.path.basename(subdir) #flood_label, vh, vv or water_body_label
      files = fnmatch.filter(files, '*.png')
      n_files = len(files)
      arr = np.empty((n_files, 256, 256, 1), dtype='uint8')
      for i, file in enumerate(files):
        path = os.path.join(subdir, file)
        image = Image.open(path)
        rgb = np.asarray(image)
        grey = np.sum(rgb, axis=2).reshape((1, 256, 256, 1)) / 3
        arr[i, :, :, :] = grey
      if files:
        np.save(cwd + "/new_data/" + setname + "_" + category + ".npy", arr)

def makeNpz():
  cwd = os.getcwd()
  rootdir = cwd + "/new_data/"
  files = [f for f in os.listdir(rootdir)]  
  
  flood_label = np.empty((0,256,256,1), dtype='uint8')
  vv = np.empty((0,256,256,1), dtype='uint8')
  vh = np.empty((0,256,256,1), dtype='uint8')
  water_body_label = np.empty((0,256,256,1), dtype='uint8')
  
  for file in fnmatch.filter(files, '*flood_label*'):
    arr = np.load(rootdir + file)
    flood_label = np.append(flood_label, arr, axis=0)
  for file in fnmatch.filter(files, '*vv*'):
    arr = np.load(rootdir + file)
    vv = np.append(vv, arr, axis=0)
  for file in fnmatch.filter(files, '*vh*'):
    arr = np.load(rootdir + file)
    vh = np.append(vh, arr, axis=0)
  for file in fnmatch.filter(files, '*water_body_label*'):
    arr = np.load(rootdir + file)
    water_body_label = np.append(water_body_label, arr, axis=0)
  
  np.savez_compressed('data.npz', vv=vv, vh=vh, flood_label=flood_label, water_body_label=water_body_label)

def countPoints():
  cwd = os.getcwd()
  rootdir = cwd + "/new_data/"
  files = [f for f in os.listdir(rootdir)]
  n_points = 0
  for file in files:
    arr = np.load(rootdir + file)
    n_points += arr.shape[0]
  print(n_points)

dataset = np.load('data.npz')
vv = dataset['vv']
vh = dataset['vh']
fl = dataset['flood_label']
wl = dataset['water_body_label']

def displayRandomPic():
  idx = np.random.randint(0, vv.shape[0])
  
  im_vv = Image.fromarray(np.concatenate((vv[idx, ::], vv[idx, ::], vv[idx, ::]), axis=-1))
  im_vh = Image.fromarray(np.concatenate((vh[idx, ::], vh[idx, ::], vh[idx, ::]), axis=-1))
  im_fl = Image.fromarray(np.concatenate((fl[idx, ::], fl[idx, ::], fl[idx, ::]), axis=-1))
  im_wl = Image.fromarray(np.concatenate((wl[idx, ::], wl[idx, ::], wl[idx, ::]), axis=-1))
  
  print(f"Displaying images with index {idx}")
  fig, axs = plt.subplots(2, 2)
  axs[0, 0].imshow(im_vv)
  axs[1, 0].imshow(im_vh)
  axs[0, 1].imshow(im_fl)
  axs[1, 1].imshow(im_wl)
  for ax in axs.ravel(): ax.axis('off')
  
  