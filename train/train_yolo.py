import os
import torch
import sys
sys.path.append("/storage/tungnx/ultralytics-main")
import ultralytics
from ultralytics import YOLO
ultralytics.checks()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
os.environ['WANDB_DISABLED'] = 'true'
IMAGE_SIZE = 1536
BATCH_SIZE = 4
EPOCHS = 80
model = YOLO('yolov8m-seg.pt')
yaml_file = '/storage/tungnx/data.yaml'
results = model.train(
    batch=BATCH_SIZE,
    device=0,
    data=yaml_file,
    # epochs=EPOCHS,
    imgsz=IMAGE_SIZE,
    # optimizer='AdamW',
    # task='segment',
    # lr0 = 5e-5,
    # lrf = 1e-8,
    # weight_decay = 5e-2,
    # flipud = 0.5,
    # degrees = 10,
    # translate = 0.1,
    # copy_paste = 0.1,
    # mixup = 0.3,
    # workers = 6,

)