import numpy as np
import pandas as pd
import math
import random
import os
from tqdm import tqdm
tqdm.pandas()
import time
import copy
import gc
import json
# visualization
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
# Sklearn
from sklearn.model_selection import StratifiedKFold, KFold, StratifiedGroupKFold
import itertools
# PyTorch 
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import SegformerForSemanticSegmentation, get_cosine_schedule_with_warmup,Mask2FormerForUniversalSegmentation
import timm
from datetime import datetime
# Albumentations for augmentations
import albumentations as A
from albumentations.pytorch import ToTensorV2

import tifffile as tiff

import warnings
warnings.filterwarnings("ignore")

# For descriptive error messages
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

class CFG:
    seed          = 1
    model_name    = 'nvidia/segformer-b0-finetuned-ade-512-512'
    train_bs      = 16
    valid_bs      = train_bs * 2
    img_size      = 512
    total_epoch   = 100
    learning_rate = 1e-4
    weight_decay  = 1e-6
    warmup_epochs = 0
    num_fold      = 5
    num_classes   = 1
    device        = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    PATH          = '/kaggle/input/hubmap-hacking-the-human-vasculature/train/'
    JSON_FILE     = "/kaggle/input/hubmap-hacking-the-human-vasculature/polygons.jsonl"

def set_seed(seed = 42):
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    # print('> SEEDING DONE')

def init_logger(log_file='train12.log'):
    from logging import getLogger, INFO, FileHandler,  Formatter,  StreamHandler
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger

set_seed(CFG.seed) 



def coordinates_to_masks(coordinates, shape):
    masks = []
    for coord in coordinates:
        mask = np.zeros(shape, dtype=np.uint8)
        cv2.fillPoly(mask, [np.array(coord)], 1)
        masks.append(mask)
    return masks

def binary_mask_to_rle(binary_mask):
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(itertools.groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))
    return rle
with open(f"/storage/tungnx/hubmap-hacking-the-human-vasculature/polygons.jsonl", "r") as json_file:
    json_list = list(json_file)
# print(json_list)    
tiles_data = {}
for json_str in tqdm(json_list, total=len(json_list)):
    json_data = json.loads(json_str)
    tiles_data[json_data['id']] = json_data['annotations']

def get_no_of_blood_vessel(img_id):
    mask = np.zeros((512, 512), dtype=np.float32)
    cnt = 0
    for annot in tiles_data[img_id]:
        if annot['type'] != "blood_vessel": continue
        for cord in annot['coordinates']:
            row, col = np.array([i[1] for i in cord]), np.asarray([i[0] for i in cord])
            mask[row, col] = 1
            cnt += 1
    return cnt
tile_meta = pd.read_csv(f"/storage/tungnx/hubmap-hacking-the-human-vasculature/tile_meta.csv")
df = tile_meta.query('dataset < 2').reset_index(drop=True)
df['no_of_bv'] = df['id'].apply(get_no_of_blood_vessel)
df = df.query('no_of_bv > 0').reset_index(drop=True)

skf = StratifiedKFold(n_splits=CFG.num_fold, random_state=CFG.seed, shuffle=True)
df['fold'] = -1
for f, (train_idx, test_idx) in enumerate(skf.split(df, df['source_wsi'])):
    df.loc[test_idx, 'fold'] = f


# print(df.groupby('fold')['source_wsi'].value_counts())

def get_image_path(image_id):
    return f"/storage/tungnx/hubmap-hacking-the-human-vasculature/train/{image_id}.tif"
df['image_path'] = ""
df['height'] = 512
df['width'] = 512
df['image_path'] = df['id'].apply(get_image_path)

df1 = tile_meta.query('dataset ==2').reset_index(drop=True)
df1['no_of_bv'] = df1['id'].apply(get_no_of_blood_vessel)
df1 = df1.query('no_of_bv > 0').reset_index(drop=True)

df1['image_path'] = ""
df1['height'] = 512
df1['width'] = 512
df1['image_path'] = df1['id'].apply(get_image_path)
df1['fold'] = -1
def get_annotations(df):
    cats = [{"id": 1, "name": "blood_vessels"}]
    annotations = []
    images = []
    obj_count = 1
    index = 1
    for idx, row in tqdm(df.iterrows(), total=len(df)):

        images.append(
            {
                "id": index,
                "width": row.width,
                "height": row.height,
                "file_name": row.image_path,
            }
        )
    #     if row.dataset ==2:
    #         for annot in tiles_data[row.id]:
    #             if annot['type'] != "blood_vessel": 
    #                 continue
    #             for cord in annot['coordinates']:
    #                 # print("Cord: ", cord)
    # #                 print(cord)
    #                 # mask = coordinates_to_masks([cord], (512, 512))[0]
    #                 # mask = mask.astype(np.uint8)
    #                 # kernel = np.ones(shape=(1, 1), dtype=np.uint8)
    #                 # mask = cv2.dilate(mask, kernel, 1)
    #                 # ys, xs = np.where(mask)
    #                 # x1, x2 = min(xs), max(xs)
    #                 # y1, y2 = min(ys), max(ys)

    #                 # rle = binary_mask_to_rle(mask)
    #                 xmin = xmax = cord[0][0]
    #                 ymin = ymax = cord[0][1]

    #                 for point in cord:
    #                     x, y = point
    #                     if x < xmin:
    #                         xmin = x
    #                     elif x > xmax:
    #                         xmax = x
    #                     if y < ymin:
    #                         ymin = y
    #                     elif y > ymax:
    #                         ymax = y
    #                 # xmin = int(np.min(cord[:, 0]))
    #                 # xmax = int(np.max(cord[:, 0]))
    #                 # ymin = int(np.min(cord[:, 1]))
    #                 # ymax = int(np.max(cord[:, 1]))

    #                 data_anno = {
    #                     "image_id": index,
    #                     "id": obj_count,
    #                     "category_id": 1,
    #                     "segmentation": [[x for c in cord for x in c]],
    #                     "area": (xmax - xmin) * (ymax - ymin),
    #                     "bbox": [xmin, ymin, (xmax - xmin), (ymax - ymin)],
    #                     "iscrowd": 0,
    #                 }
    #                 annotations.append(data_anno)

    #                 obj_count += 1
        # else:
        for annot in tiles_data[row.id]:
            if annot['type'] == "blood_vessel": 
                for cord in annot['coordinates']:
    #                 print(cord)
                    xmin = xmax = cord[0][0]
                    ymin = ymax = cord[0][1]

                    for point in cord:
                        x, y = point
                        if x < xmin:
                            xmin = x
                        elif x > xmax:
                            xmax = x
                        if y < ymin:
                            ymin = y
                        elif y > ymax:
                            ymax = y
    #                 xmin = int(np.min(cord[:, 0]))
    #                 xmax = int(np.max(cord[:, 0]))
    #                 ymin = int(np.min(cord[:, 1]))
    #                 ymax = int(np.max(cord[:, 1]))

                    data_anno = {
                        "image_id": index,
                        "id": obj_count,
                        "category_id": 1,
                        "segmentation": [[x for c in cord for x in c]],
                        "area": (xmax - xmin) * (ymax - ymin),
                        "bbox": [xmin, ymin, (xmax - xmin), (ymax - ymin)],
                        "iscrowd": 0,
                    }
                    annotations.append(data_anno)

                    obj_count += 1

        index+=1
    return {"categories": cats, "images": images, "annotations": annotations}
fold = 2
print("Fold", fold)
# train_df = df
train_df = df[df['fold']!=fold]
# final_df = pd.concat([df[df['fold']!=fold], df1], axis = 0).reset_index(drop=True)
# for f, (train_idx, test_idx) in enumerate(skf.split(final_df, final_df['source_wsi'])):
#     final_df.loc[test_idx, 'fold'] = f
# train_df = final_df[final_df['fold']!=fold]
train_json = get_annotations(train_df)
valid_json = get_annotations(df[df['fold']==fold])
with open(f"/storage/tungnx/train_fold_{fold}.json", "w+", encoding="utf-8") as f:
    json.dump(train_json, f, ensure_ascii=True, indent=4)
# with open(f"/storage/tungnx/valid_fold_{fold}.json", "w+", encoding="utf-8") as f:
#     json.dump(valid_json, f, ensure_ascii=True, indent=4)