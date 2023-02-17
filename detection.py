#%%
import torch
import numpy as np
from module.mmdetection.mmdet.apis import init_detector, inference_detector

# %%
mm_path = "./module/mmdetection"
config_file = f"{mm_path}/yolov3_mobilenetv2_320_300e_coco.py"
checkpoint_file = f"{mm_path}/yolov3_mobilenetv2_320_300e_coco_20210719_215349-d18dff72.pth"
model = init_detector(config_file, checkpoint_file, device="cpu")
model = init_detector(config_file, checkpoint_file, device="mps")
inference_detector(model, f"{mm_path}/demo/demo.jpg")

next(model.parameters()).device

