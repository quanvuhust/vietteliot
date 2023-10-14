import pandas as pd
import glob
import cv2
import numpy as np
from tqdm import tqdm

all_imgs = glob.glob('private_test/images/*.jpg')
h_list = []
w_list = []
imageid_list = []
for img_path in tqdm(all_imgs):
    image_id = img_path.split("/")[-1].replace(".jpg", "")
    image = cv2.imread(img_path)
    h, w, _ = image.shape
    imageid_list.append(image_id)
    h_list.append(h)
    w_list.append(w)

imageid_list = np.array(imageid_list)
h_list = np.array(h_list)
w_list = np.array(w_list)

imageid_list = np.expand_dims(imageid_list, 1)
h_list = np.expand_dims(h_list, 1)
w_list = np.expand_dims(w_list, 1)

df = pd.DataFrame(np.concatenate((imageid_list, h_list, w_list), axis=1), columns=["image_id", "h", "w"])
df.to_csv('sizes.csv', index=False)