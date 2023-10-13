import json, cv2, numpy as np, itertools, random, pandas as pd
from skimage import io
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm.auto import tqdm
from sklearn import model_selection
from copy import deepcopy
import cv2
import os

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
dup_check = {}
index = 0
c_dup = 0
for _, row in df.iterrows():
    image_id = row["image_id"]

    class_id = row["class_id"]
    if class_id == 15:
        continue
    x_min,y_min,x_max,y_max = row["x_min"], row["y_min"], row["x_max"], row["y_max"]
    if (class_id, x_min,y_min,x_max,y_max) in dup_check.keys():
        print("Duplicate ", image_id, dup_check[(class_id, x_min,y_min,x_max,y_max)])
        c_dup += 1
        continue
    else:
        dup_check[(class_id, x_min,y_min,x_max,y_max)] = image_id

    if class_id > 15:
        class_id = class_id - 1
    if class_id > 11:
        class_id = class_id - 1
    class_name = row["class_name"]
    
    if x_max - x_min <= 10:
        continue
    if y_max - y_min <= 10:
        continue
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
print("Total dup: ", c_dup)
output_file_path = "train.json"
with open(output_file_path, "w", encoding="utf-8") as output_file:
    json.dump(coco_data, output_file, ensure_ascii=True, indent=4)