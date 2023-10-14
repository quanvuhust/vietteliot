import os, glob
import sys
import json
from PIL import Image

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import argparse

import sys
sys.path.append("../train/det")

from mmdet.apis import init_detector, inference_detector,show_result_pyplot, set_random_seed
import mmcv_custom  
import mmdet_custom 
import torch

from mmcv import Config


parser = argparse.ArgumentParser(description="test code")
parser.add_argument("--exp", type=str, default="exp_0")
args = parser.parse_args()

f = open(os.path.join('configs', "{}.json".format(args.exp)))
configs = json.load(f)
f.close()

intern_config_files = [
    configs['config_path']
                      ]
intern_weight_files = [
    configs['model_path']
]
intern_image_sizes = [
    [(configs['image_scale_h'], configs['image_scale_w'])]
                     ]

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
intern_models = []
for weight_file, config_file, image_size in zip(intern_weight_files, intern_config_files, intern_image_sizes):
    cfg = Config.fromfile(config_file)
    cfg.data.test.pipeline[1].img_scale = image_size
    cfg.seed = 42
    set_random_seed(42, deterministic=True)
    model = init_detector(cfg, weight_file, device=device)
    intern_models.append(model)

all_imgs = glob.glob('private_test/images/*.jpg')

import mmcv
import matplotlib.pyplot as plt
from ensemble_boxes import *
from tqdm import tqdm
k = 0
batch_size = 5
img_batch = []
image_id_batch = []
image_id_csv = []
class_id_csv = []
confidence_score_csv = [] 
x_min_csv = []
y_min_csv = []
x_max_csv = []
y_max_csv = []
for i in tqdm(range(len(all_imgs))):
    image_id = all_imgs[i].split("/")[-1].replace(".jpg", "")
    image = cv2.imread(all_imgs[i])
    image = np.ascontiguousarray(image)
    if configs['is_hflip'] == True:
        image = cv2.flip(image, 1)
    img_batch.append(image)
    image_id_batch.append(image_id)

    if len(img_batch) == batch_size or i == len(all_imgs) - 1:
        pred_model = inference_detector(intern_models[0], img_batch)
            
        for j in range(0, len(pred_model)):
            h, w, _ = img_batch[j].shape
            image_id = image_id_batch[j]
            pred = pred_model[j]

            for class_id in range(len(pred)):
                for result in pred[class_id]:
                    new_class_id = class_id
                    if class_id >= 14:
                        new_class_id += 2
                    elif class_id >= 11:
                        new_class_id += 1
                    
                    image_id_csv.append(image_id)
                    class_id_csv.append(int(new_class_id))
                    confidence_score_csv.append(result[4])
                    if configs['is_hflip'] == True:
                        x_min_csv.append(w-result[0])
                    else:
                        x_min_csv.append(result[0])
                    y_min_csv.append(result[1])
                    if configs['is_hflip'] == True:
                        x_max_csv.append(w-result[2])
                    else:
                        x_max_csv.append(result[2])
                    y_max_csv.append(result[3])

        del img_batch
        img_batch = []
        image_id_batch = []

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
df.to_csv(configs['result_path'], index=False)