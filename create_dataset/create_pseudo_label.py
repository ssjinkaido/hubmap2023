import numpy as np
import pandas as pd
import os
from tqdm import tqdm
tqdm.pandas()
import json
# visualization
import cv2
from sklearn.model_selection import StratifiedKFold
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from mmdet.apis import inference_detector, init_detector
from mmcv import Config
import mmcv
# PyTorch 
import itertools
import tifffile as tiff
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
import warnings
warnings.filterwarnings("ignore")
from numba import jit
# For descriptive error messages
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"



@jit(nopython=True)
def bb_intersection_over_union(A, B):
    xA = max(A[0], B[0])
    yA = max(A[1], B[1])
    xB = min(A[2], B[2])
    yB = min(A[3], B[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)

    if interArea == 0:
        return 0.0

    # compute the area of both the prediction and ground-truth rectangles
    boxAArea = (A[2] - A[0]) * (A[3] - A[1])
    boxBArea = (B[2] - B[0]) * (B[3] - B[1])

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def prefilter_boxes(boxes, scores, labels, weights, thr):
    # Create dict with boxes stored by its label
    new_boxes = dict()
    for t in range(len(boxes)):

        if len(boxes[t]) != len(scores[t]):
            print('Error. Length of boxes arrays not equal to length of scores array: {} != {}'.format(len(boxes[t]),
                                                                                                       len(scores[t])))
            exit()

        if len(boxes[t]) != len(labels[t]):
            print('Error. Length of boxes arrays not equal to length of labels array: {} != {}'.format(len(boxes[t]),
                                                                                                       len(labels[t])))
            exit()

        for j in range(len(boxes[t])):
            score = scores[t][j]
            if score < thr:
                continue
            label = int(labels[t][j])
            box_part = boxes[t][j]
            x1 = float(box_part[0])
            y1 = float(box_part[1])
            x2 = float(box_part[2])
            y2 = float(box_part[3])

            # Box data checks
            if x2 < x1:
                warnings.warn('X2 < X1 value in box. Swap them.')
                x1, x2 = x2, x1
            if y2 < y1:
                warnings.warn('Y2 < Y1 value in box. Swap them.')
                y1, y2 = y2, y1
            if x1 < 0:
                warnings.warn('X1 < 0 in box. Set it to 0.')
                x1 = 0
            if x1 > 1:
                warnings.warn('X1 > 1 in box. Set it to 1. Check that you normalize boxes in [0, 1] range.')
                x1 = 1
            if x2 < 0:
                warnings.warn('X2 < 0 in box. Set it to 0.')
                x2 = 0
            if x2 > 1:
                warnings.warn('X2 > 1 in box. Set it to 1. Check that you normalize boxes in [0, 1] range.')
                x2 = 1
            if y1 < 0:
                warnings.warn('Y1 < 0 in box. Set it to 0.')
                y1 = 0
            if y1 > 1:
                warnings.warn('Y1 > 1 in box. Set it to 1. Check that you normalize boxes in [0, 1] range.')
                y1 = 1
            if y2 < 0:
                warnings.warn('Y2 < 0 in box. Set it to 0.')
                y2 = 0
            if y2 > 1:
                warnings.warn('Y2 > 1 in box. Set it to 1. Check that you normalize boxes in [0, 1] range.')
                y2 = 1
            if (x2 - x1) * (y2 - y1) == 0.0:
                warnings.warn("Zero area box skipped: {}.".format(box_part))
                continue

            b = [int(label), float(score) * weights[t], x1, y1, x2, y2]
            if label not in new_boxes:
                new_boxes[label] = []
            new_boxes[label].append(b)

    # Sort each list in dict by score and transform it to numpy array
    for k in new_boxes:
        current_boxes = np.array(new_boxes[k])
        new_boxes[k] = current_boxes[current_boxes[:, 1].argsort()[::-1]]

    return new_boxes


def get_weighted_box(boxes):
    """
    Create weighted box for set of boxes
    :param boxes: set of boxes to fuse 
    :return: weighted box
    """

    box = np.zeros(6, dtype=np.float32)
    best_box = boxes[0]
    conf = 0
    for b in boxes:
        iou = bb_intersection_over_union(b[2:], best_box[2:])
        weight = b[1] * iou
        box[2:] += (weight * b[2:])
        conf += weight
    box[0] = best_box[0]
    box[1] = best_box[1]
    box[2:] /= conf
    return box


def find_matching_box(boxes_list, new_box, match_iou):
    best_iou = match_iou
    best_index = -1
    for i in range(len(boxes_list)):
        box = boxes_list[i]
        if box[0] != new_box[0]:
            continue
        iou = bb_intersection_over_union(box[2:], new_box[2:])
        if iou > best_iou:
            best_index = i
            best_iou = iou

    return best_index, best_iou
def non_maximum_weighted(boxes_list, scores_list, labels_list, weights=None, iou_thr=0.55, skip_box_thr=0.0):
    '''
    :param boxes_list: list of boxes predictions from each model, each box is 4 numbers. 
    It has 3 dimensions (models_number, model_preds, 4)
    Order of boxes: x1, y1, x2, y2. We expect float normalized coordinates [0; 1]
    :param scores_list: list of scores for each model 
    :param labels_list: list of labels for each model
    :param weights: list of weights for each model. Default: None, which means weight == 1 for each model
    :param iou_thr: IoU value for boxes to be a match
    :param skip_box_thr: exclude boxes with score lower than this variable  
    
    :return: boxes: boxes coordinates (Order of boxes: x1, y1, x2, y2). 
    :return: scores: confidence scores
    :return: labels: boxes labels
    '''

    if weights is None:
        weights = np.ones(len(boxes_list))
    if len(weights) != len(boxes_list):
        print('Warning: incorrect number of weights {}. Must be: {}. Set weights equal to 1.'.format(len(weights), len(boxes_list)))
        weights = np.ones(len(boxes_list))
    weights = np.array(weights) / max(weights)
    # for i in range(len(weights)):
    #     scores_list[i] = (np.array(scores_list[i]) * weights[i])

    filtered_boxes = prefilter_boxes(boxes_list, scores_list, labels_list, weights, skip_box_thr)
    if len(filtered_boxes) == 0:
        return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,))

    overall_boxes = []
    for label in filtered_boxes:
        boxes = filtered_boxes[label]
        new_boxes = []
        main_boxes = []

        # Clusterize boxes
        for j in range(0, len(boxes)):
            index, best_iou = find_matching_box(main_boxes, boxes[j], iou_thr)
            if index != -1:
                new_boxes[index].append(boxes[j].copy())
            else:
                new_boxes.append([boxes[j].copy()])
                main_boxes.append(boxes[j].copy())

        weighted_boxes = []
        for j in range(0, len(new_boxes)):
            box = get_weighted_box(new_boxes[j])
            weighted_boxes.append(box.copy())

        overall_boxes.append(np.array(weighted_boxes))

    overall_boxes = np.concatenate(overall_boxes, axis=0)
    overall_boxes = overall_boxes[overall_boxes[:, 1].argsort()[::-1]]
    boxes = overall_boxes[:, 2:]
    scores = overall_boxes[:, 1]
    labels = overall_boxes[:, 0]
    return boxes, scores, labels

def nms_predictions(classes, scores, bboxes, masks, 
                    iou_th=.3, shape=(512, 512)):
    he, wd = shape[0], shape[1]
    boxes_list = [[x[0] / wd, x[1] / he, x[2] / wd, x[3] / he]
                  for x in bboxes]
    scores_list = [x for x in scores]
    labels_list = [x for x in classes]
    nms_bboxes, nms_scores, nms_classes = non_maximum_weighted([boxes_list], [scores_list], [labels_list], weights=None,iou_thr=iou_th)
    nms_masks = []
    for s in nms_scores:
        nms_masks.append(masks[scores.index(s)])
    nms_scores, nms_classes, nms_masks = zip(*sorted(zip(nms_scores, nms_classes, nms_masks), reverse=True))
    return nms_classes, nms_scores, nms_masks, nms_bboxes
with open(f"/home/tungnx/hubmap-hacking-the-human-vasculature/cleaned_polygons.jsonl", "r") as json_file:
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
tile_meta = pd.read_csv(f"/home/tungnx/hubmap-hacking-the-human-vasculature/tile_meta.csv")
df = tile_meta.query('dataset < 2').reset_index(drop=True)
skf = StratifiedKFold(n_splits=5, random_state=1, shuffle=True)
df['fold'] = -1

for fold, (train_idx, test_idx) in enumerate(skf.split(df, df['source_wsi'])):
    df.loc[test_idx, 'fold'] = fold

df1 = tile_meta.query('dataset ==2').reset_index(drop=True)

def get_image_path(image_id):
    return f"/home/tungnx/hubmap-hacking-the-human-vasculature/train/{image_id}.tif"
df['image_path'] = ""
df['height'] = 512
df['width'] = 512
df['image_path'] = df['id'].apply(get_image_path)
df1['image_path'] = ""
df1['height'] = 512
df1['width'] = 512
df1['image_path'] = df1['id'].apply(get_image_path)
df2 = tile_meta.query('dataset ==3').reset_index(drop=True)
df2['image_path'] = ""
df2['height'] = 512
df2['width'] = 512
df2['image_path'] = df2['id'].apply(get_image_path)
fold = 0
train_df = pd.concat([df[df['fold']!=fold], df2], axis = 0).reset_index(drop=True)
# print(train_df.iloc[400])
cfg = Config.fromfile('/home/tungnx/models_config/r101_config_ds12_fold0.py')
model = init_detector(cfg, '/home/tungnx/models/r101_ds12_fold0_best_segm_mAP_epoch_11.pth', device='cuda:0')

cfg1 = Config.fromfile('/home/tungnx/models_config/r101_config_ds12_fold1.py')
model1 = init_detector(cfg1, '/home/tungnx/models/r101_ds12_fold1_best_segm_mAP_epoch_14.pth', device='cuda:0')

cfg2 = Config.fromfile('/home/tungnx/models_config/r101_config_ds12_fold2.py')
model2 = init_detector(cfg2, '/home/tungnx/models/r101_ds12_fold2_best_segm_mAP_epoch_11.pth', device='cuda:0')


all_models = [model, model1, model2]
cats = [{"id": 0, "name": "blood_vessels"}]
annotations = []
images = []
obj_count = 0

for idx, row in tqdm(df2.iterrows(), total=len(df2)):
    filename = row.image_path

    images.append(
        {
            "id": row.id,
            "file_name": row.image_path,
            "width": row.width,
            "height": row.height,
        }
    )
    if row.dataset != 3:
        for annot in tiles_data[row.id]:
            if annot['type'] != "blood_vessel": 
                continue
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

                data_anno = {
                    "image_id": row.id,
                    "id": obj_count,
                    "category_id": 0,
                    "bbox": [xmin, ymin, (xmax - xmin), (ymax - ymin)],
                    "area": (xmax - xmin) * (ymax - ymin),
                    "segmentation": [[x for c in cord for x in c]],
                    "iscrowd": 0,
                }
                annotations.append(data_anno)

                obj_count += 1
    else:
        print(row.image_path)
        classes_nms = []
        scores_nms = []
        bboxes_nms = []
        masks_nms = []
        image = mmcv.imread(row.image_path)
        for step, model in enumerate(all_models):
        
            result = inference_detector(model, image)
            
            for j, classe in enumerate(result[0]):
                bbs = classe
                sgs = result[1][j] #change
                
                for bb, mask in zip(bbs,sgs):
    #                 print(sg.shape)
                    box = bb[:4]
                    # print("Box:", box)
                    cnf = bb[4]
                    if cnf> 0.5:
                        mask = np.array(mask,dtype=np.uint8)  
                        masks_nms.append(mask)
                        scores_nms.extend([cnf])
                        bboxes_nms.extend([box.tolist()])
        classes_nms = [0] * len(masks_nms)
        if len(scores_nms) ==0 or len(masks_nms) ==0 or len(bboxes_nms) ==0:
            continue
        classes_nms, scores_nms, masks_nms, bboxes_nms = nms_predictions(classes_nms, scores_nms, bboxes_nms, masks_nms, iou_th = 0.5) 
        used = np.zeros((512, 512), dtype=int)
        print(len(scores_nms))
        for _mask, _box, cnf in zip(masks_nms, bboxes_nms, scores_nms):
            _box = _box*512
            # print(_box)
            xmin = int(_box[0])
            ymin = int(_box[1])
            xmax = int(_box[2])
            ymax = int(_box[3])
            if np.sum(_mask == 1) < 100:
                continue
            mask = _mask *(1-used)
            used += mask
            rle = binary_mask_to_rle(mask)
            data_anno = {
                    "image_id": row.id,
                    "id": obj_count,
                    "category_id": 0,
                    "bbox": [xmin, ymin, (xmax - xmin), (ymax - ymin)],
                    "area": (xmax - xmin) * (ymax - ymin),
                    "segmentation": rle,
                    "iscrowd": 0,
                }
            annotations.append(data_anno)

            obj_count += 1

train_json = {"categories": cats, "images": images, "annotations": annotations}
with open(f"/home/tungnx/train_ds3.json", "w+", encoding="utf-8") as f:
    json.dump(train_json, f, ensure_ascii=True, indent=4)