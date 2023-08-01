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
import segmentation_models_pytorch as smp
# Sklearn
from sklearn.model_selection import StratifiedKFold,StratifiedGroupKFold
import timm
# PyTorch 
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import SegformerForSemanticSegmentation, get_cosine_schedule_with_warmup,Mask2FormerForUniversalSegmentation, OneFormerForUniversalSegmentation,  OneFormerProcessor
import torch.nn.functional as F
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
    model_name    = 'nvidia/segformer-b4-finetuned-ade-512-512'
    train_bs      = 8
    valid_bs      = train_bs * 2
    img_size      = 512
    total_epoch   = 60
    learning_rate = 1e-4
    weight_decay  = 1e-6
    warmup_epochs = 6
    num_fold      = 5
    num_classes   = 1
    device        = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    PATH          = 'hubmap-hacking-the-human-vasculature/train/'
    JSON_FILE     = "hubmap-hacking-the-human-vasculature/polygons.jsonl"

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

LOGGER = init_logger()
now = datetime.now()
datetime_now = now.strftime("%m/%d/%Y, %H:%M:%S")
LOGGER.info(f"Date :{datetime_now}")
LOGGER.info(f"Seed: {CFG.seed}")
LOGGER.info(f"Train bs: {CFG.train_bs}")
LOGGER.info(f"Valid bs: {CFG.valid_bs}")
LOGGER.info(f"Learning rate: {CFG.learning_rate}")
LOGGER.info(f"Weight decay: {CFG.weight_decay}")
LOGGER.info(f"Warmup epochs: {CFG.warmup_epochs}")
LOGGER.info(f"Model name: {CFG.model_name}")
set_seed(CFG.seed) 

with open(f"hubmap-hacking-the-human-vasculature/polygons.jsonl", "r") as json_file:
    json_list = list(json_file)
    
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
df = tile_meta.query('dataset < 3').reset_index(drop=True)
df['no_of_bv'] = df['id'].apply(get_no_of_blood_vessel)
df = df.query('no_of_bv > 0').reset_index(drop=True)

skf = StratifiedKFold(n_splits=CFG.num_fold, random_state=CFG.seed, shuffle=True)
df['fold'] = -1

for fold, (train_idx, test_idx) in enumerate(skf.split(df, df['source_wsi'])):
    df.loc[test_idx, 'fold'] = fold

print(df.groupby('fold')['source_wsi'].value_counts())

# df = pd.read_csv('hubmap-hacking-the-human-vasculature/tile_meta.csv')
# print("Unique id:",len(df['id'].unique()))
# df['row'] = df.index
# df["fold"] = -1
# skf = StratifiedKFold(n_splits=CFG.num_fold, shuffle=True, random_state=CFG.seed)
# for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['source_wsi'])):
#     df.loc[val_idx, 'fold'] = fold

# LOGGER.info(len(df))
data_transforms = {
    "train": A.Compose([
        # A.Resize(1024, 1024),
        A.RandomRotate90(p = 0.5),
        A.HorizontalFlip(always_apply=False, p=0.5),
        A.VerticalFlip(always_apply=False, p=0.5),
        A.Transpose(p = 0.5),
        A.OneOf([
            A.RandomBrightnessContrast(always_apply=False, p=0.5, brightness_limit=(-0.1, 0.1), contrast_limit=(-0.2, 0.2), brightness_by_max=True),
            A.HueSaturationValue(always_apply=False, p=0.5, hue_shift_limit=(-30, 30), sat_shift_limit=(-90, 20), val_shift_limit=(-20, 20)),
            A.RandomGamma(always_apply=False, p=0.5, gamma_limit=(50, 200), eps=None),
        ], p=0.5),
        A.OneOf([
            A.ChannelShuffle(always_apply=False, p=0.1),
            A.ColorJitter(always_apply=False, p=0.1, brightness=[0.8, 1.2], contrast=[0.8, 1.2], saturation=[0.8, 1.2], hue=[-0.2, 0.2]),
            A.FancyPCA(always_apply=False, p=0.1, alpha=0.1),
            A.CLAHE(always_apply=False, p=0.1, clip_limit=(1, 4.0), tile_grid_size=(8, 8)),
        ], p=0.2),
        A.OneOf([
            A.OpticalDistortion(always_apply=False, p=0.3, distort_limit=(-0.05, 0.05), shift_limit=(-0.05, 0.05), interpolation=1, border_mode=4, value=None, mask_value=None),
            A.GridDistortion(always_apply=False, p=0.3, num_steps=5, distort_limit=(-0.3, 0.3), interpolation=1, border_mode=4, value=None, mask_value=None, normalized=False),
            A.ElasticTransform(always_apply=False, p=0.1, alpha=1, sigma=50, alpha_affine=50, interpolation=1, border_mode=4, value=None, mask_value=None, approximate=False, same_dxdy=False),
        ], p=0.5),
        A.OneOf([
            A.Blur(always_apply=False, p=1.0, blur_limit=(3, 7)),
            A.GaussNoise(always_apply=False, p=1.0, var_limit=(10.0, 50.0), per_channel=True, mean=0),
            A.MultiplicativeNoise(always_apply=False, p=1.0, multiplier=(0.9, 1.1), per_channel=True, elementwise=True),
        ], p=0.1),
        # A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.2, rotate_limit= 15,
        #                                 interpolation=cv2.INTER_LINEAR, border_mode=0, p=0.9),
        # A.CoarseDropout(max_holes=8, max_height=CFG.img_size//20, max_width=CFG.img_size//20,
        #                 min_holes=5, fill_value=0, mask_fill_value=0, p=0.5),
        A.Normalize(),
        ToTensorV2()
        ], p=1.0),
    
    "valid": A.Compose([
        # A.Resize(1024, 1024),
        A.Normalize(),
        ToTensorV2()
        ], p=1.0)
}

LOGGER.info(f"train transform{data_transforms['train']}")

class HuBMAPDataset(Dataset):
    def __init__(self, df, image_dir, transforms=None, mode='train'):
        # with open(labels_file, 'r') as json_file:
        #     self.json_labels = [json.loads(line) for line in json_file]
        self.image_dir = image_dir
        self.transforms = transforms
        self.df = df
        self.mode = mode
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # Load image
        row = self.df.iloc[idx]
        image_id = row['id']
        image_path = os.path.join(self.image_dir, f"{image_id}.tif")
        
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        # image = np.pad(image, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='constant')
        # Initialize mask
        mask = np.zeros((512, 512), dtype=np.float32)
        # Process annotations 
        for annot in tiles_data[image_id]:
            if annot['type'] == "blood_vessel":
                coordinates = np.array(annot['coordinates'])
                coordinates = coordinates.reshape(-1, 1, 2)
                # Draw the polygon on the mask
                cv2.fillPoly(mask, [coordinates], 1)
                # print(np.unique(mask))
                # for cord in annot['coordinates']:
                #     row, col = np.array([i[1] for i in cord]), np.asarray([i[0] for i in cord])
                #     mask[row, col] = 1
            # elif annot['type'] == 'glomerulus':
            #     for cord in annot['coordinates']:
            #         row, col = np.array([i[1] for i in cord]), np.asarray([i[0] for i in cord])
            #         mask[1,row, col] = 1
        # if self.mode=='train':
        #     mask = np.pad(mask, ((pad_size, pad_size), (pad_size, pad_size)), mode='constant')
        
        # mask = torch.from_numpy(mask)
        # num_classes = 2  # there are 2 classes: blood vessel and glomerulus
        # print(mask.shape)
        # mask= F.one_hot(mask.long(), num_classes).permute(2, 0, 1).float()
        # print(mask.shape)
        if self.transforms:
            data = self.transforms(image=image, mask=mask)
            image, mask = data['image'], torch.unsqueeze(data['mask'], 0)
            # image = np.transpose(image, (2, 0, 1))
            # mask = np.transpose(mask, (2, 0, 1))
            # mask = mask.copy()
            # mask = np.transpose(mask, (2, 0, 1))
        return image , mask
    
class HubMapModel(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        # self.backbone = smp.Unet('efficientnet-b5', encoder_weights = 'imagenet', in_channels = 3, classes = 1)
        # self.backbone = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-small-ade-semantic")
        # self.backbone = timm.create_model('coat_lite_medium.in1k', pretrained=True)
        # self.backbone = OneFormerForUniversalSegmentation.from_pretrained("shi-labs/oneformer_ade20k_swin_tiny")
        self.backbone = SegformerForSemanticSegmentation.from_pretrained(backbone)
        # self.backbone = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-tiny-ade-semantic")

        self.logit = nn.Sequential(
            nn.Conv2d(150, 1, kernel_size=1),
            # nn.Upsample(scale_factor = 4, mode='bilinear', align_corners=False),
        )
        self.mixing = nn.Parameter(torch.tensor(0.5))
        self.scale_factor = 4

    def forward(self, x):
        output = self.backbone(x)
        x = self.logit(output['logits'])
        output = self.mixing * F.interpolate(
            x, scale_factor=self.scale_factor, mode="bilinear", align_corners=False
        ) + (1 - self.mixing) * F.interpolate(
            x, scale_factor=self.scale_factor, mode="nearest"
        )
        # print(output['transformer_decoder_last_hidden_state'].shape)
        # output = self.logit(output['logits'])
        # output = self.logit(output['masks_queries_logits'])
        return output
    

JaccardLoss = smp.losses.JaccardLoss(mode='binary')
DiceLoss    = smp.losses.DiceLoss(mode='binary')
BCELoss     = smp.losses.SoftBCEWithLogitsLoss()
# LovaszLoss  = smp.losses.LovaszLoss(mode='binary', per_image=False)
FocalLoss = smp.losses.FocalLoss(mode='binary')
TverskyLoss = smp.losses.TverskyLoss(mode='binary', log_loss=False)
# class DiceLoss(nn.Module):
#     def __init__(self, weight=None, size_average=True):
#         super(DiceLoss, self).__init__()

#     def forward(self, inputs, targets, smooth=1):
        
#         #comment out if your model contains a sigmoid or equivalent activation layer
#         inputs = F.sigmoid(inputs)       
        
#         #flatten label and prediction tensors
#         inputs = inputs.view(-1)
#         targets = targets.view(-1)
        
#         intersection = (inputs * targets).sum()                            
#         dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
#         return 1 - dice

# class IoULoss(nn.Module):
#     def __init__(self, weight=None, size_average=True):
#         super(IoULoss, self).__init__()

#     def forward(self, inputs, targets, smooth=1):
        
#         #comment out if your model contains a sigmoid or equivalent activation layer
#         inputs = F.sigmoid(inputs)       
        
#         #flatten label and prediction tensors
#         inputs = inputs.view(-1)
#         targets = targets.view(-1)
        
#         #intersection is equivalent to True Positive count
#         #union is the mutually inclusive area of all labels & predictions 
#         intersection = (inputs * targets).sum()
#         total = (inputs + targets).sum()
#         union = total - intersection 
        
#         IoU = (intersection + smooth)/(union + smooth)
                
#         return 1 - IoU

def dice_coef(y_true, y_pred, dim=(2,3), epsilon=0.001):
    y_true = y_true.to(torch.float32)
    # y_pred = (y_pred>thr).to(torch.float32)
    inter = (y_true*y_pred).sum(dim=dim)
    den = y_true.sum(dim=dim) + y_pred.sum(dim=dim)
    dice = ((2*inter+epsilon)/(den+epsilon)).mean(dim=(1,0))
    return dice

def iou_coef(y_true, y_pred, dim=(2,3), epsilon=0.001):
    y_true = y_true.to(torch.float32)
    # y_pred = (y_pred>thr).to(torch.float32)
    inter = (y_true*y_pred).sum(dim=dim)
    union = (y_true + y_pred - y_true*y_pred).sum(dim=dim)
    iou = ((inter+epsilon)/(union+epsilon)).mean(dim=(1,0))
    return iou

def criterion(y_pred, y_true):
    return (JaccardLoss(y_pred, y_true) + DiceLoss(y_pred, y_true) + BCELoss(y_pred, y_true) + FocalLoss(y_pred, y_true))/4


def iou_torch(mask1, mask2):
    intersection = torch.logical_and(mask1, mask2).sum().float()
    union = torch.logical_or(mask1, mask2).sum().float()
    return intersection / union if union > 0 else 0

def average_precision_torch(mask, pred_masks, iou_threshold=0.6):
    num_masks = mask.shape[0]
    num_pred_masks = pred_masks.shape[0]
    ious = torch.zeros((num_masks, num_pred_masks))

    for i in range(num_masks):
        for j in range(num_pred_masks):
            ious[i, j] = iou_torch(mask[i, 0], pred_masks[j, 0])

    # Sort the pairs based on IoU scores in descending order
    ious, indices = torch.sort(ious.view(-1), descending=True)

    # Calculate TP, FP, and FN
    tp = (ious >= iou_threshold).float()
    fp = 1 - tp
    fn = num_masks - tp.sum()

    # Calculate the precision and recall values
    precision = torch.cumsum(tp, dim=0) / (torch.cumsum(tp, dim=0) + torch.cumsum(fp, dim=0))
    recall = torch.cumsum(tp, dim=0) / (torch.cumsum(tp, dim=0) + fn)

    # Compute the average precision using the precision-recall curve
    ap = (precision[:-1] * (recall[1:] - recall[:-1])).sum()
    return ap.item()
    # num_masks = mask.shape[0]
    # ious = torch.zeros((num_masks, num_masks))

    # for i in range(num_masks):
    #     for j in range(num_masks):
    #         ious[i, j] = iou_torch(mask[i, 0], pred_masks[j, 0])

    # tp = 0
    # fp = 0
    # fn = 0

    # for i in range(num_masks):
    #     max_iou = ious[i].max()
    #     if max_iou >= iou_threshold:
    #         tp += 1
    #         pred_idx = ious[i].argmax()
    #         ious[:, pred_idx] = 0
    #     else:
    #         fn += 1

    # fp = num_masks - tp
    # precision = tp / (tp + fp + fn)

    # return precision
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))

def train_fn(train_loader, model, criterion, optimizer, scheduler, device):
    torch.cuda.empty_cache()
    gc.collect()
    losses = AverageMeter()
    model.train()
    global_step = 0
    scaler = GradScaler()
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc='Train')
    for step, (images, masks) in pbar:
        optimizer.zero_grad()
        images = images.to(device, dtype=torch.float)
        masks = masks.to(device)
        batch_size = images.size(0)
        with autocast():
            outputs = model(images)
            # print(outputs.shape)
            # print(masks.shape)
            loss = criterion(outputs, masks)
            # print(loss)
        losses.update(loss.item(), batch_size)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        global_step += 1
        scheduler.step()
            # measure elapsed time
        torch.cuda.empty_cache()
        gc.collect()
        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix(train_loss=f'{losses.avg:0.4f}',
                        lr=f'{current_lr:0.7f}',
                        gpu_mem=f'{mem:0.2f} GB')

    return losses.avg

def valid_fn(val_dataloader, model, criterion, device, thr):
    losses = AverageMeter()
    ap = AverageMeter()
    model.eval()
    val_scores = []
    pbar = tqdm(enumerate(val_dataloader), total=len(val_dataloader), desc='Val')
    for step, (images, masks) in pbar:
        images = images.to(device, dtype=torch.float)
        masks = masks.to(device)
        batch_size = images.size(0)
        with torch.no_grad():
            outputs = model(images)
            # pad_size = 64

            # # Remove the padded pixels
            # outputs = outputs[:, :, pad_size:-pad_size, pad_size:-pad_size]
            loss = criterion(outputs, masks)
        losses.update(loss.item(), batch_size)
        y_pred = nn.Sigmoid()(outputs)
        pred_masks = (y_pred>thr).bool()
        y_pred = (y_pred>thr).to(torch.float32)
        valid_ap = average_precision_torch(masks, pred_masks, 0.6)
        ap.update(valid_ap, batch_size)
        val_dice = dice_coef(masks, y_pred).cpu().detach().numpy()
        val_jaccard = iou_coef(masks, y_pred).cpu().detach().numpy()
        val_scores.append([val_dice, val_jaccard])
        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
        pbar.set_postfix(eval_loss=f'{losses.avg:0.4f}',
                        gpu_mem=f'{mem:0.2f} GB',
                        valid_ap = f'{ap.avg:.5f}')
        torch.cuda.empty_cache()
        gc.collect()
    val_scores  = np.mean(val_scores, axis=0)
    return losses.avg, val_scores, ap.avg

if __name__ == '__main__':

    set_seed(CFG.seed)
    gc.collect()
    torch.cuda.empty_cache()
    
    for fold in [0]:
        LOGGER.info(f"Fold: {fold}/{CFG.num_fold}")
        train_df = df[df['fold']!=fold].reset_index(drop=True)
        valid_df = df[df['fold']==fold].reset_index(drop=True)
        train_dataset = HuBMAPDataset(train_df, CFG.PATH, transforms=data_transforms['train'], mode='train')
        valid_dataset = HuBMAPDataset(valid_df, CFG.PATH, transforms=data_transforms['valid'], mode='valid')
        train_loader = DataLoader(train_dataset, batch_size = CFG.train_bs,
                                            num_workers=10, shuffle=True, pin_memory=True, drop_last=True)

        valid_loader = DataLoader(valid_dataset, batch_size = CFG.valid_bs, 
                                            num_workers=10, shuffle=False, pin_memory=True, drop_last=False)
        LEN_DL_TRAIN = len(train_loader)
        model = HubMapModel(CFG.model_name).to(CFG.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.learning_rate, weight_decay = CFG.weight_decay)  
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps = LEN_DL_TRAIN* CFG.warmup_epochs, num_training_steps = LEN_DL_TRAIN*CFG.total_epoch)
        best_metric = 0
        best_thr = 0
        start = time.time()
        for epoch in range(CFG.total_epoch):
            LOGGER.info(f"Epoch: {epoch+1}/{CFG.total_epoch}")
            loss_train = train_fn(train_loader, model, criterion, optimizer, scheduler, CFG.device)
            for thr in np.arange(0.2, 0.51, 0.05):
                loss_valid, valid_scores, valid_ap = valid_fn(valid_loader, model, criterion, CFG.device, thr)
                LOGGER.info(f"Train loss: {loss_train:.5f}, Valid loss: {loss_valid:.5f}")
                valid_dice, valid_jaccard = valid_scores
                LOGGER.info(f'Valid Dice: {valid_dice:0.4f} | Valid Jaccard: {valid_jaccard:0.4f}')
                LOGGER.info(f'Valid ap: {valid_ap:.4f}')
                if valid_jaccard > best_metric:
                    LOGGER.info(f"Model improve: {best_metric:.4f} -> {valid_jaccard:.4f}")
                    LOGGER.info(f"Best threshold: {thr:.3f}")
                    best_metric = valid_jaccard
                    best_thr = thr

            # if valid_jaccard >best_metric:
            #     LOGGER.info(f"Model improve: {best_metric:.4f} -> {valid_jaccard:.4f}")
            #     best_metric = valid_jaccard
            if best_metric > 0.54:
                state = {'epoch': epoch+1, 'state_dict': model.state_dict()}
                path = f'fold_{fold}_model_epoch_{epoch+1}_{best_metric:.6f}.pth'
                torch.save(state, path)

        end = time.time()
        time_elapsed = end - start
        LOGGER.info('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
            time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
        LOGGER.info("Best Score: {:.4f}".format(best_metric))