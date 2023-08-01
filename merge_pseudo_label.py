import json
with open('/home/tungnx/train_ds3_r50.json', 'r') as f:
    ds3 = json.load(f)
with open('/home/tungnx/train_fold_0_ds12.json', 'r') as f:
    ds0 = json.load(f)
cats = [{"id": 0, "name": "blood_vessels"}]
images3 = ds3['images']
for i in images3:
    i['file_name'] = i['file_name'].replace('home/tungnx/', '/home/tungnx/')
images0 = ds0['images']
merge_images = images3 + images0

anno3 = ds3['annotations']
anno0 = ds0['annotations']
annotations = []
for i in anno0:
    i['id'] +=43404
final_anno = anno3 + anno0
print(len(final_anno))
# print(anno0[12])
print(len(anno3))
print(ds3['categories'])
print(len(merge_images))

train_json = {"categories": cats, "images": merge_images, "annotations": final_anno}
with open(f"/home/tungnx/train_r50_ds123_fold_0.json", "w+", encoding="utf-8") as f:
    json.dump(train_json, f, ensure_ascii=True, indent=4)