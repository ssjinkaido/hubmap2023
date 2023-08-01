import json, cv2, numpy as np, itertools, random, pandas as pd
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
import mmcv
import matplotlib.pyplot as plt
from pathlib import Path
from copy import deepcopy
dataDir = Path("hubmap-hacking-the-human-vasculature/train")
annFile = Path("/home/tungnx/train_fold_0.json")
coco = COCO(annFile)

imgIds = coco.getImgIds()
print(len(imgIds))
imgs = coco.loadImgs(random.sample(imgIds, 3))

imgs = coco.loadImgs(imgIds[-3:])
fig, axs = plt.subplots(len(imgs), 2, figsize=(10, 5*len(imgs)))

for img, ax_row in zip(imgs, axs):
    ax = ax_row[0]  # Access the first axis in each row
    I = mmcv.imread(dataDir / img["file_name"])
    annIds = coco.getAnnIds(imgIds=[img["id"]])
    anns = coco.loadAnns(annIds)
    ax.imshow(I)

    ax = ax_row[1]  # Access the second axis in each row
    ax.imshow(I)
    plt.sca(ax)
    coco.showAnns(anns, draw_bbox=True)