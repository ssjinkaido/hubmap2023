import torch

out_file = '/home/tungnx/swa_ds1_fold0.pth' 
iteration = [
    '/home/tungnx/mmdetection-2.26.0/work_dirs/hubmap/1class_ds1/detectors_r50/fold0/final/epoch_8.pth',
    # '/home/tungnx/mmdetection-2.26.0/work_dirs/hubmap/1class_ds1/detectors_r50/fold0/final/epoch_9.pth',
    '/home/tungnx/mmdetection-2.26.0/work_dirs/hubmap/1class_ds1/detectors_r50/fold0/final/epoch_10.pth',
    '/home/tungnx/mmdetection-2.26.0/work_dirs/hubmap/1class_ds1/detectors_r50/fold0/final/epoch_11.pth',
]
a1 = 1/3
a2 = 1/3
a3 = 1/3
a4 = 1/3
state_dict = None
for i in iteration:
    f = i
    f = torch.load(f, map_location=lambda storage, loc: storage)
    meta = f['meta']
    print(i)
    if state_dict is None:
        state_dict = f['state_dict']
        key = list(f['state_dict'].keys())
        for k in key:
            state_dict[k] = f['state_dict'][k]*a1
    # elif i=='/home/tungnx/mmdetection-2.26.0/work_dirs/hubmap/1class_ds1/detectors_r50/fold0/final/epoch_9.pth':
    #     key = list(f['state_dict'].keys())
    #     for k in key:
    #         state_dict[k] = state_dict[k] + a2*f['state_dict'][k]

    elif i=='/home/tungnx/mmdetection-2.26.0/work_dirs/hubmap/1class_ds1/detectors_r50/fold0/final/epoch_10.pth':
        key = list(f['state_dict'].keys())
        for k in key:
            state_dict[k] = state_dict[k] + a3*f['state_dict'][k]

    elif i=='/home/tungnx/mmdetection-2.26.0/work_dirs/hubmap/1class_ds1/detectors_r50/fold0/final/epoch_11.pth':
        key = list(f['state_dict'].keys())
        for k in key:
            state_dict[k] = state_dict[k] + a4*f['state_dict'][k]

torch.save({'state_dict': state_dict, 'meta':meta}, out_file)