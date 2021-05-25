#!/usr/bin/env python3

import numpy as np
import os
import fnmatch
from PIL import Image
import pathlib
from pathlib import Path
import natsort
import PIL

## remove "noise" pictures from directory data_dir
## only call remove_noise function on train/train
## Not very foolproof, if run more than ones the filenames will be messed up!

def remove_noise(data_dir):
    path = pathlib.Path(data_dir)

    areas = [p for p in path.iterdir() if p.is_dir()]
    # print(areas)
    for area in areas:
        # area = Path('/home/lumo/Documents/adv-ml/train/train/bangladesh_20170314t115609')
        print("Processing directory: " + str(area))
        tiles = area / 'tiles'
        rename_vv_vh(tiles)

        vh = tiles/ 'vh'
        pics = list(vh.glob('*.png'))
        pics = np.array(natsort.natsorted(pics))
        noise_args = np.array([i for i, pic in enumerate(pics) if is_noise_pic(pic)])
        print('Number of noise pics is ' + f'{np.size(noise_args)}' + " out of " + f'{len(pics)}')
        print('Removing files...')

        subdirs = [x for x in tiles.iterdir() if x.is_dir()]
        for path in subdirs:
            pics = list(path.glob('*.png'))
            pics = np.array(natsort.natsorted(pics))
            rem_pics = pics[noise_args]
            for pic in rem_pics:
                pic.unlink()
    print("Done.")

def rename_vv_vh(path):
    subdirs = [x for x in path.iterdir() if x.is_dir()]
    for path in subdirs:
        pics = list(path.glob('*.png'))
        if 'vv' in str(path) or 'vh' in str(path):
            for pic in pics:
                old_name = pic.stem
                directory = pic.parent
                new_name = str(old_name)[:-3] + pic.suffix
                pic.rename(pathlib.Path(directory, new_name))

def is_noise_pic(pic):
    return np.median(np.asarray(PIL.Image.open(pic))) >= 255 or np.median(np.asarray(PIL.Image.open(pic))) == 0
