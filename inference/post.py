import pandas as pd
import numpy as np
df = pd.read_csv("ensemble_results.csv")

# image_id,class_id,confidence_score,x_min,y_min,x_max,y_max
image_id_csv = []
class_id_csv = []
confidence_score_csv = [] 
x_min_csv = []
y_min_csv = []
x_max_csv = []
y_max_csv = []

images = {}
large_objects = {6: "wild boar"}

for index, row in df.iterrows():
    image_id = row['image_id']
    if image_id not in images.keys():
        images[image_id] = []
    class_id = row['class_id']
    confidence_score = row['confidence_score']
    x_min = int(row['x_min'])
    y_min = int(row['y_min'])
    x_max = int(row['x_max'])
    y_max = int(row['y_max'])

    images[image_id].append((class_id, confidence_score, (x_min, y_min, x_max, y_max)))


for image_id in images.keys():
    images[image_id].sort(key=lambda a: a[1], reverse=True)

for image_id in images.keys():
    max_conf = 0
    total_box = 0
    bboxes = {}
    for class_id, confidence_score, (x_min, y_min, x_max, y_max) in images[image_id]:
        
        flag = None
        for k in bboxes.keys():
            if abs(x_min - k[0]) <= 5 and abs(y_min - k[1]) <= 5 and abs(x_max - k[2]) <= 5 and abs(y_max - k[3]) <= 5:
                flag = k
                break
        if flag != None:
            bboxes[flag].append((confidence_score, class_id))
        else:
            total_box += 1
            bboxes[(x_min, y_min, x_max, y_max)] = []
            bboxes[(x_min, y_min, x_max, y_max)].append((confidence_score, class_id))
        if confidence_score > max_conf:
            max_conf = confidence_score
    for k in bboxes.keys():
        bboxes[k].sort(key=lambda a: a[0], reverse=True)   
        

    for (x_min, y_min, x_max, y_max) in bboxes.keys():
        for t in range(len(bboxes[(x_min, y_min, x_max, y_max)])):
            confidence_score, class_id = bboxes[(x_min, y_min, x_max, y_max)][t]
            if class_id == 15:
                continue

            if max_conf < 0.35 and total_box >=8 and confidence_score <= 0.08:
                continue

            if class_id == 10:
                image_id_csv.append(image_id)
                class_id_csv.append(28)
                confidence_score_csv.append(3/4*confidence_score)
                x_min_csv.append(x_min)
                y_min_csv.append(y_min)
                x_max_csv.append(x_max)
                y_max_csv.append(y_max)
            image_id_csv.append(image_id)
            class_id_csv.append(class_id)
            confidence_score_csv.append(confidence_score)
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
df.to_csv('results.csv', index=False)