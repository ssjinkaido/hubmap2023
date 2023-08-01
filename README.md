# Hupmap2023
## Training strategy
- Train only on dataset 1 (Dont have enough GPU and Time to train on perform pseudo labeling on dataset 3 and retrain).
- Light augmentation (HFlip, VFlip, ShiftScaleRotate, HSV, RandomBrightnessConstrast, RGBshift).
- 12 epochs, AdamW, lr = 0.00005.
- Try YoloV8 but not use it due to competition restriction.
## Model used
- Use DetectoRS + Resnet 101, image size 1536x1536.
- Training time is around 1 hour.
## Inference 
- Ensemble 5 models + SWA + Non maximum weighed with threshold 0.7.
- Fill smale hole in mask.
- No dilation in mask.

## Future Work
- Should have performed pl and retrain on dataset3.
- Should migrate to mmdet 3 to use stronger models and augmentations.
## Results 
| Models | Public LB | Private LB |
| -------- | -------- | -------- |
| SWA 5 R101 + 1 R50 | 0.504 (36/1064) | 0.527 (18/1064)|
