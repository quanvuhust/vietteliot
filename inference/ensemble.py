import pandas as pd
import numpy as np
from ensemble_boxes import *
import cv2
from tqdm import tqdm
import os

size_df = pd.read_csv("sizes.csv")
sizes = {}
for index, row in size_df.iterrows():
    image_id = row['image_id']
    h = row['h']
    w = row['w']
    sizes[image_id] = (h, w)


results = [
    "results_epoch8_noTTA.csv",
    "results_finetune_epoch3_TTA.csv",
    "results_epoch7_ema_TTA.csv",
    "results_epoch7_ema_noTTA.csv",
    "results_epoch7_ema_492_noTTA.csv",
    "results_epoch7_ema_564_noTTA.csv",
    "results_epoch2_ema_TTA.csv",
    "results_epoch2_ema_564_noTTA.csv",
    "results_epoch2_ema_noTTA.csv",
]


dfs = []
for result_file in results:
    df = pd.read_csv(result_file)
    dfs.append(df)

images = []

for df in dfs:
    images.append({})
    for index, row in df.iterrows():
        image_id = row['image_id']
        class_id = row['class_id']
        confidence_score = row['confidence_score']
        x_min = row['x_min']
        y_min = row['y_min']
        x_max = row['x_max']
        y_max = row['y_max']
        if image_id not in images[-1].keys():
            images[-1][image_id] = {}
            images[-1][image_id]['boxes_list'] = []
            images[-1][image_id]['scores_list'] = []
            images[-1][image_id]['labels_list'] = []
        images[-1][image_id]['boxes_list'].append([x_min, y_min, x_max, y_max])
        images[-1][image_id]['scores_list'].append(confidence_score)
        images[-1][image_id]['labels_list'].append(class_id)


thresh = 0.1
iou_thr = 0.7
image_id_csv = []
class_id_csv = []
confidence_score_csv = [] 
x_min_csv = []
y_min_csv = []
x_max_csv = []
y_max_csv = []
for image_id in tqdm(images[0].keys()):
    boxes_list = []
    scores_list = []
    labels_list = []

    h, w = sizes[image_id]
    for m in range(len(results)):
        boxes_list.append([])
        scores_list.append([])
        labels_list.append([])
        for i in range(len(images[m][image_id]['boxes_list'])):
            bbox = images[m][image_id]['boxes_list'][i]
            class_id = images[m][image_id]['labels_list'][i]
            confidence_score = images[m][image_id]['scores_list'][i]
            if class_id == 15:
                continue
            if confidence_score < thresh:
                continue
            boxes_list[-1].append([bbox[0]/w, bbox[1]/h, bbox[2]/w, bbox[3]/h])
            scores_list[-1].append(confidence_score)
            labels_list[-1].append(class_id)
    flag_noempty = False
    for m in range(len(results)):
        if len(boxes_list[m]) > 0:
            flag_noempty = True
            break
    if flag_noempty == False:
        image_id_csv.append(image_id)
        class_id_csv.append(15)
        confidence_score_csv.append(1.0)
        x_min_csv.append(0)
        y_min_csv.append(0)
        x_max_csv.append(0)
        y_max_csv.append(0)
    else:
        nms_boxes_list, nms_scores_list, nms_labels_list = [], [], []
        weights = []
        for m in range(len(boxes_list)):
            if len(boxes_list[m]) == 0:
                continue
            boxes, scores, labels = nms([boxes_list[m]], [scores_list[m]], [labels_list[m]], weights=None, iou_thr=0.5)
            nms_boxes_list.append(boxes)
            nms_scores_list.append(scores)
            nms_labels_list.append(labels)
            weights.append(1)
        boxes, scores, labels = weighted_boxes_fusion(nms_boxes_list, nms_scores_list, nms_labels_list, weights=weights, iou_thr=iou_thr)
        box_labels = []
        for t in range(len(boxes)):
            box_labels.append([int(boxes[t][0]*w), int(boxes[t][1]*h), int(boxes[t][2]*w), int(boxes[t][3]*h), labels[t], scores[t]])
        nms_bboxes = np.array(box_labels)   
        scores = np.array(scores) 
        for t in range(nms_bboxes.shape[0]):
            x_min = int(nms_bboxes[t][0])
            y_min = int(nms_bboxes[t][1])
            x_max = int(nms_bboxes[t][2])
            y_max = int(nms_bboxes[t][3])
            x_min = max(x_min, 0); y_min = max(y_min, 0)
            x_max = max(x_max, 0); y_max = max(y_max, 0)
            x_min = min(x_min, w); y_min = min(y_min, h)
            x_max = min(x_max, w); y_max = min(y_max, h)
            if x_min >= x_max or y_min >= y_max:
                continue
            image_id_csv.append(image_id)
            class_id_csv.append(int(nms_bboxes[t][4]))
            confidence_score_csv.append(nms_bboxes[t][5])
            x_min_csv.append(x_min)
            y_min_csv.append(y_min)
            x_max_csv.append(x_max)
            y_max_csv.append(y_max)

image_id_csv = np.array(image_id_csv)
class_id_csv = np.array(class_id_csv)
confidence_score_csv = np.array(confidence_score_csv)
x_min_csv = np.array(x_min_csv)
y_min_csv = np.array(y_min_csv)
x_max_csv = np.array(x_max_csv)
y_max_csv = np.array(y_max_csv)

image_id_csv = np.expand_dims(image_id_csv, 1)
class_id_csv = np.expand_dims(class_id_csv, 1)
confidence_score_csv = np.expand_dims(confidence_score_csv, 1)
x_min_csv = np.expand_dims(x_min_csv, 1)
y_min_csv = np.expand_dims(y_min_csv, 1)
x_max_csv = np.expand_dims(x_max_csv, 1)
y_max_csv = np.expand_dims(y_max_csv, 1)

df = pd.DataFrame(np.concatenate((image_id_csv, class_id_csv, confidence_score_csv, x_min_csv, y_min_csv, x_max_csv, y_max_csv), axis=1), columns=["image_id", "class_id", "confidence_score", "x_min", "y_min", "x_max", "y_max"])
df.to_csv('ensemble_results.csv', index=False)