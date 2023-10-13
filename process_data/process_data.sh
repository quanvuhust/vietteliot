cp -r warmup_train/images/*.jpg train/images
python create_coco_anno.py
python extract_embedding.py
python create_coco_anno_clean.py
