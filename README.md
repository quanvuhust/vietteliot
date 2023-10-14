# vietteliot
 VHAC 2023 - AI for IoT
 # Cài đặt môi trường
# Cài đặt môi trường
 CUDA drive 12.0\
 Docker images: nvcr.io/nvidia/pytorch:23.08-py3
```
docker pull nvcr.io/nvidia/pytorch:23.08-py3
```
* Install pytorch nightly cho cuda 12.0:
```
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121
```
* Install mmcv-1.5.0 từ source:
```
curl https://codeload.github.com/open-mmlab/mmcv/zip/refs/tags/v1.5.0 -o mmcv-1.5.0.zip
unzip -qq mmcv-1.5.0.zip
cd mmcv-1.5.0
```
* Vào file setup.py sửa tất cụm từ "-std=c++14" thành "-std=c++17", sau đó bắt đầu install mmcv-1.5.0:
```
export MMCV_WITH_OPS=1
export FORCE_CUDA=1
pip install -e . -v
```
* Install requirements cho internimage:
```
pip install timm==0.6.11 mmdet==2.28.1
pip install opencv-python termcolor yacs pyyaml scipy
```
* Compile CUDA operators cho internimage:
```
cd vietteliot/train/det/ops_dcnv3
sh ./make.sh
```
# Huấn luyện mô hình
* Tải dữ liệu public train và warmup train, lưu vào thư mục process_data:
* Extract dinov2 embedding của tất cả các ảnh, lọc bỏ ảnh trùng lặp, tạo coco annotations dùng để huấn luyện cho 2 pha:
```
cd process_data
./process_data.sh
```
* Cấu trúc thư mục huấn luyện:
```
process_data/
    ├── train
    │   ├── images
    │   ├── groundtruth.csv
    ├── warmup_train
    │   ├── images
    │   ├── groundtruth.csv
    ├── dino-all-feature-list.pickle
    ├── dino-all-filenames.pickle
    ├── train.json
    ├── train_clean.json
```
* Huấn luyện pha 1
** Huấn luyện 3 epoch đầu, sau khi hoàn thành 3 epoch đầu thì ngắt training
```
cd train/det
python train.py --config configs/custom/config_0.py --deterministic
```
** Loại bỏ optimizer state_dict ra khỏi weight:
```
cp convert.py ../workdirs/
cd ../workdirs/
python convert.py
```
** Huấn luyện 9 epoch cuối
```
python train.py --config configs/custom/config_1.py --deterministic

```
* Huấn luyện pha 2
```
cd train/det
python train.py --config configs/custom/config_finetune.py --deterministic
cp convert.py ../workdirs_finetune/
cd ../workdirs_finetune/
python convert.py
```
* Finetune cho giai đoạn private test
```
cd train/det
python train.py --config configs/custom/config_finetune_ema.py --deterministic
cp convert.py ../workdirs_finetune_ema/
cd ../workdirs_finetune_ema/
python convert.py
```
# Inference
* Tải weights và config về, lưu vào thư mục weights
* Cấu trúc thư mục test:
```
test/
    ├── configs
    │   ├── exp_0.json
    │   ├── ...
    ├── private_test
    │   ├── images
    ├── weights
    │   ├── ...
    ├── ...
```
* Infer single model
```
cd test
python test.py --exp exp_0
python test.py --exp exp_1
python test.py --exp exp_2
python test.py --exp exp_3
python test.py --exp exp_4
python test.py --exp exp_5
python test.py --exp exp_6
python test.py --exp exp_7
python test.py --exp exp_8
```
* Lưu lại kích thước h, w của các ảnh trong tập private xuống ổ cứng
```
python create_sizes.py
```
* Ensemble model
```
python ensemble.py
```
* Postprocess để lọc bớt box để có thể submit
```
python post.py
```
* Inference notebook: https://www.kaggle.com/code/huyduong7101/vhac2023-ocr-inference
* Weights: https://www.kaggle.com/datasets/quan0095/tablerecognitionmodel
