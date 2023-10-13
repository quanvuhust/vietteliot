import json, cv2, numpy as np, itertools, random, pandas as pd
from skimage import io
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm.auto import tqdm
from sklearn import model_selection
from copy import deepcopy
import cv2
import os
import pickle
from scipy.spatial import distance

feature_list = pickle.load(open('dino-all-feature-list.pickle','rb'))
filenames = pickle.load(open('dino-all-filenames.pickle','rb'))

feature_map = {}
for filename, feature in zip(filenames, feature_list):
    feature_map[filename] = feature

train_df = pd.read_csv("train/groundtruth.csv")
warmup_train_df = pd.read_csv("warmup_train/groundtruth.csv")
warmup_train_df = warmup_train_df.drop(warmup_train_df.columns[0], axis=1)
print(warmup_train_df.head())
df = pd.concat([train_df, warmup_train_df])
print(df.head())
coco_data = {"info": {}, "licenses": [], "categories": [], "images": [], "annotations": []}
categories = []
for index, row in df.iterrows():
    class_id = row["class_id"]
    if class_id == 15:
        continue
    if class_id > 15:
        class_id = class_id - 1
    if class_id > 11:
        class_id = class_id - 1
    class_name = row["class_name"]

    if class_name not in categories:
        categories.append(class_name)
        coco_data["categories"].append({"id": class_id, "name": class_name})

image_ids = {}
bbox_dup_check = {}
feature_dup_check = set()
index = 0
c_dup_bbox = 0
c_dup_feature = 0
c_remove = 0
large_class = {0: "assam macaque", 3: "chinese serow", 4: "roosevelt's muntjac", 6: "wild boar", 10: "giant muntjac",
   19: "grey peacock-pheasant", 22: "stump-tailed macaque", 23: "red-shanked douc", 24: "pig-tailed macaque",
   25: "sambar", 28: "red muntjac"
}
clean_filenames = []
clean_feature_list = []
for _, row in tqdm(df.iterrows()):
    image_id = row["image_id"]

    class_id = row["class_id"]
    if class_id == 15:
        continue
    x_min,y_min,x_max,y_max = row["x_min"], row["y_min"], row["x_max"], row["y_max"]
    if (class_id, x_min,y_min,x_max,y_max) in bbox_dup_check.keys():
        # print("Duplicate bbox: ", image_id, bbox_dup_check[(class_id, x_min,y_min,x_max,y_max)])
        c_dup_bbox += 1
        continue
    else:
        bbox_dup_check[(class_id, x_min,y_min,x_max,y_max)] = image_id
    
    if x_max - x_min <= 10:
        continue

    if y_max - y_min <= 10:
        continue

    if (x_max - x_min <= 20 or y_max - y_min <= 20) and class_id in large_class.keys():
        c_remove += 1
        continue
    filename = os.path.join("train/images", str(image_id) + ".jpg")
    feature = feature_map[filename]
    flag = False
    for old_feature, old_imageid in zip(clean_feature_list, clean_filenames):
        if image_id != old_imageid:
            if(image_id, old_imageid) not in feature_dup_check:
                if distance.cosine(feature, old_feature) <= 0.05:
                    # print("Duplicate feature: ", image_id, old_imageid)
                    c_dup_feature += 1
                    feature_dup_check.add((image_id, old_imageid))
                    feature_dup_check.add((old_imageid, image_id))
                    flag = True
                    break
    if flag == True:
        continue
    clean_filenames.append(image_id)
    clean_feature_list.append(feature_map[filename])
    if class_id > 15:
        class_id = class_id - 1
    if class_id > 11:
        class_id = class_id - 1
    class_name = row["class_name"]

    if image_id not in image_ids.keys():
        image = cv2.imread(os.path.join("train/images", image_id + ".jpg"))
        h, w, _ = image.shape
        image_ids[image_id] = (h, w)
    else:
        h, w = image_ids[image_id]

    image_info = {"id": image_id, "file_name": image_id + ".jpg", "height": h, "width": w}
    
    coco_data["images"].append(image_info)
    annotation_info = {
        "id": index,
        "image_id": image_id,
        "category_id": class_id,
        "bbox": [int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)],
        "bbox_mode": 1,
        "iscrowd": 0,
        "area": int(x_max - x_min)*int(y_max - y_min),
    }
    coco_data["annotations"].append(annotation_info)
    index += 1
print("Total dup bbox: ", c_dup_bbox)
print("Total remove: ", c_remove)
print("Total dup feature: ", c_dup_feature)
print("Total image: ", len(coco_data["images"]))
output_file_path = "train_clean.json"
with open(output_file_path, "w", encoding="utf-8") as output_file:
    json.dump(coco_data, output_file, ensure_ascii=True, indent=4)