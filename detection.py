#%%
import torch
import numpy as np
from module.mmdetection.mmdet.apis import init_detector, inference_detector

# %%
mm_path = "./module/mmdetection"

# config_file = f"{mm_path}/yolov3_mobilenetv2_320_300e_coco.py"
# checkpoint_file = f"{mm_path}/yolov3_mobilenetv2_320_300e_coco_20210719_215349-d18dff72.pth"

config_file = f"{mm_path}/retinanet_r101_fpn_1x_coco.py"
checkpoint_file = f"{mm_path}/retinanet_r101_fpn_1x_coco_20200130-7a93545f.pth"

config_file = f"{mm_path}/faster_rcnn_r50_fpn_1x_coco.py"
checkpoint_file = f"{mm_path}/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth"

model = init_detector(config_file, checkpoint_file)

result = inference_detector(model, f"{mm_path}/demo/demo.jpg")
result_img = model.show_result(f"{mm_path}/demo/demo.jpg", result, show=True)


#%%
from mmdet.datasets import replace_ImageToTensor
from mmdet.datasets.pipelines import Compose
from mmcv.parallel import collate, scatter

imgs = f"{mm_path}/demo/demo.jpg"

if isinstance(imgs, (list, tuple)):
    is_batch = True
else:
    imgs = [imgs]
    is_batch = False

cfg = model.cfg

device = next(model.parameters()).device  # model device

if isinstance(imgs[0], np.ndarray):
    cfg = cfg.copy()
    # set loading pipeline type
    cfg.data.test.pipeline[0].type = 'LoadImageFromWebcam'

cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
test_pipeline = Compose(cfg.data.test.pipeline)

datas = []
for img in imgs:
    # prepare data
    if isinstance(img, np.ndarray):
        # directly add img
        data = dict(img=img)
    else:
        # add information into dict
        data = dict(img_info=dict(filename=img), img_prefix=None)
    # build the data pipeline
    data = test_pipeline(data)
    datas.append(data)

data = collate(datas, samples_per_gpu=len(imgs))
# just get the actual data from DataContainer
data['img_metas'] = [img_metas.data[0] for img_metas in data['img_metas']]
data['img'] = [img.data[0] for img in data['img']]
if next(model.parameters()).is_cuda:
    # scatter to specified GPU
    data = scatter(data, [device])[0]
else:
    for m in model.modules():
        assert not isinstance(
            m, RoIPool
        ), 'CPU inference with RoIPool is not supported currently.'

cfg["model"]["test_cfg"]["nms"] = None

#%% proposal region extraction
anchor_cfg = model.cfg.model.bbox_head.anchor_generator
anchor_type = anchor_cfg.type
octave_base_scale = anchor_cfg.octave_base_scale
scales_per_octave = anchor_cfg.scales_per_octave
ratios = anchor_cfg.ratios
strides = anchor_cfg.strides

from mmdet.core import AnchorGenerator
anchor_gerator = AnchorGenerator(strides=strides,
                                 ratios=ratios,
                                 octave_base_scale=octave_base_scale,
                                 scales_per_octave=scales_per_octave
                                 )

class Backbone(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.backbone = model.backbone
        self.neck = model.neck
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        
        return x

SubModel = Backbone(model)
necks = SubModel(img)
feature_map_shapes = [tuple(neck.shape[-2:]) for neck in necks]

anchors = anchor_gerator.grid_anchors(feature_map_shapes, device="cuda")

img_size = torch.tensor([img.shape[-2], img.shape[-1]])
normalize_factor = torch.tile(img_size, (2,)).to(device)

anchors = [anchor / normalize_factor for anchor in  anchors]

# %%
result = model(return_loss=False, rescale=True, **data)

#%%
video = mmcv.VideoReader("./module/mmdetection/mmdet/video.mp4")
os.getcwd()


