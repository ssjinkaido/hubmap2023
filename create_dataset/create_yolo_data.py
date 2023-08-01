import shutil
import os
import pandas as pd
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt

from pathlib import Path
from glob import glob
from collections import defaultdict
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from IPython.display import Image as show_image
from sklearn.model_selection import StratifiedKFold, KFold, StratifiedGroupKFold
# import ultralytics
# from ultralytics import YOLO
import json
import torch

# ultralytics.checks()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

IMAGE_SIZE = 512
BATCH_SIZE = 2
EPOCHS = 10

print(device)

def mkdir_yolo_data(train_path, val_path):
    """
    make yolo data's directories
    
    parameters
    ----------
    train_path: str
        path for training data
    val_path: str
        path for validation data
    
    returns
    ----------
    train_image_path: str
        path for images of training data
    train_label_path: str
        path for labels of trainingdata
    val_image_path: str
        path for images of validation data
    val_label_path: str
        path for labels of validation data
    """
    train_image_path = Path(f'{train_path}/images')
    train_label_path = Path(f'{train_path}/labels')
    val_image_path = Path(f'{val_path}/images')
    val_label_path = Path(f'{val_path}/labels')
    
    train_image_path.mkdir(parents=True, exist_ok=True)
    train_label_path.mkdir(parents=True, exist_ok=True)
    val_image_path.mkdir(parents=True, exist_ok=True)
    val_label_path.mkdir(parents=True, exist_ok=True)
    
    return train_image_path, train_label_path, val_image_path, val_label_path


with open(f"hubmap-hacking-the-human-vasculature/polygons.jsonl", "r") as json_file:
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

tile_meta = pd.read_csv(f"hubmap-hacking-the-human-vasculature/tile_meta.csv")
df = tile_meta.query('dataset < 2').reset_index(drop=True)
df['no_of_bv'] = df['id'].apply(get_no_of_blood_vessel)
df = df.query('no_of_bv > 0').reset_index(drop=True)

skf = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)
df['fold'] = -1

for fold, (train_idx, test_idx) in enumerate(skf.split(df, df['source_wsi'])):
    df.loc[test_idx, 'fold'] = fold


def get_image_path(image_id):
    return f"/storage/tungnx/hubmap-hacking-the-human-vasculature/train/{image_id}.tif"
df['image_path'] = ""
df['height'] = 512
df['width'] = 512
df['image_path'] = df['id'].apply(get_image_path)

df1 = tile_meta.query('dataset == 2').reset_index(drop=True)
df1['no_of_bv'] = df1['id'].apply(get_no_of_blood_vessel)
df1 = df1.query('no_of_bv > 0').reset_index(drop=True)

skf = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)
df1['fold'] = -1

for fold, (train_idx, test_idx) in enumerate(skf.split(df1, df1['source_wsi'])):
    df1.loc[test_idx, 'fold'] = fold

def get_image_path(image_id):
    return f"/storage/tungnx/hubmap-hacking-the-human-vasculature/train/{image_id}.tif"
df['image_path'] = ""
df['height'] = 512
df['width'] = 512
df['image_path'] = df['id'].apply(get_image_path)

df1['image_path'] = ""
df1['height'] = 512
df1['width'] = 512
df1['image_path'] = df1['id'].apply(get_image_path)
fold = 3
# train0 = df[df['fold']!=fold]
train0 = pd.concat([df[df['fold']!=fold], df1], axis = 0).reset_index(drop=True)
valid0 = df[df['fold']==fold]

# File path settings
BASE_DIR = Path('/storage/tungnx/hubmap-hacking-the-human-vasculature')

test_paths = glob(f'{BASE_DIR}/test/*')
polygons_path = f'{BASE_DIR}/polygons.jsonl'

yolo_train_path = f'/storage/tungnx/datasets/train_ds12_{fold}'
yolo_val_path = f'/storage/tungnx/datasets/val{fold}'

train_image_path, train_label_path, val_image_path, val_label_path = mkdir_yolo_data(yolo_train_path, yolo_val_path)
print(train_image_path)
print(train_label_path)
print(val_image_path)
print(val_label_path)

def create_vessel_annotations(polygons_path):
    """
    Create annotations set which have blood_vessel label.
    
    parameters
    ----------
    polygons_path: str
        path of polygons.jsonl
    
    returns
    ----------
    annotations_dict: dict {key=id, value=coordinates}
        annotations dict with key id and value coordinates of blood_vessel
    """
    # load polygons data
    polygons = pd.read_json(polygons_path, orient='records', lines=True)
    
    # extract blood_vessel annotation
    annotations_dict = defaultdict(list)
    for idx, row in polygons.iterrows():
        id_ = row['id']
        annotations = row['annotations']
        for annotation in annotations:
            if annotation['type'] == 'blood_vessel':
                annotations_dict[id_].append(annotation['coordinates'])
    
    return annotations_dict

def create_label_file(id_, coordinates, path):
    """
    Create label txt file for yolo v8
    
    parameters
    ----------
    id_: str
        label id
    coordinates: list
        coordinates of blood_vessel
    path: str
        path for saving label txt file
    """
    label_txt = ''
    for coordinate in coordinates:
        label_txt += '0 '
        # Normalize
        coor_array = np.array(coordinate[0]).astype(float)
        coor_array /= float(IMAGE_SIZE)
        # transform to str
        coor_list = list(coor_array.reshape(-1).astype(str))
        coor_str = ' '.join(coor_list)
        # add string to label txt
        label_txt += f'{coor_str}\n'
    
    # Write labels to txt file
    with open(f'{path}/{id_}.txt', 'w') as f:
        f.write(label_txt)
        
def prepare_yolo_dataset(train_df, valid_df,
        annotaions_dict, train_image_path, train_label_path, 
        val_image_path, val_label_path):
    """
    Prepare yolo dataset with images and labels
    
    parameters
    ----------
    annotations_dict: dict {key=id, value=coordinates}
        annotations dict with key id and value coordinates of blood_vessel
    train_image_path: str
        path for images of training data
    train_label_path: str
        path for labels of trainingdata
    val_image_path: str
        path for images of validation data
    val_label_path: str
        path for labels of validation data
    """
    # ids = list(annotations_dict.keys())
    
    # # train test split
    # indices = [i for i in range(len(ids))]
    # train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=1234)
    train_ids = train_df['id'].values.tolist()
    valid_ids = valid_df['id'].values.tolist()
    # Training data
    for id_ in tqdm(train_ids):
        # id_ = ids[index]
        
        # create label txt file
        create_label_file(id_, annotations_dict[id_], train_label_path)
        # copy tif image file to yolo directory
        source_file = f'{BASE_DIR}/train/{id_}.tif'
        shutil.copy2(source_file, train_image_path)
    
    # Validation data
    for id_ in tqdm(valid_ids):
        # id_ = ids[index]
        
        # create label txt file
        create_label_file(id_, annotations_dict[id_], val_label_path)
        # copy tif image file to yolo directory
        source_file = f'{BASE_DIR}/train/{id_}.tif'
        shutil.copy2(source_file, val_image_path)
    





# Create annotations dict with key=id and value=coordinates
annotations_dict = create_vessel_annotations(polygons_path)
# Prepare dataset for yolo training
prepare_yolo_dataset(
    train0, valid0,
    annotations_dict, train_image_path, train_label_path,
    val_image_path, val_label_path
)

# yaml_content = f'''
# train: /storage/tungnx/train/images
# val: /storage/tungnx/val/images

# names:
#     0: blood_vessel
# '''

# yaml_file = '/storage/tungnx/data.yaml'

# with open(yaml_file, 'w') as f:
#     f.write(yaml_content)

# model = YOLO('yolov8n-seg.pt')