#%%
import torch
import numpy as np
from module.mmdetection.mmdet.apis import init_detector, inference_detector

# %%
mm_path = "./module/mmdetection"
config_file = f"{mm_path}/yolov3_mobilenetv2_320_300e_coco.py"
checkpoint_file = f"{mm_path}/yolov3_mobilenetv2_320_300e_coco_20210719_215349-d18dff72.pth"

model = init_detector(config_file, checkpoint_file)
model = init_detector(config_file, checkpoint_file, device="cpu")
next(model.parameters())
import time
current = time.time()
result = inference_detector(model, f"{mm_path}/demo/demo.jpg")
result[2]
len(result)
result_img = model.show_result(f"{mm_path}/demo/demo.jpg", result)
from torchvision import transforms
transforms.ToPILImage()(result_img)
print(time.time() - current)
model.show_result()
next(model.parameters()).device

#%%
