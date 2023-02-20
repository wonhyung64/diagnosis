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

model = init_detector(config_file, checkpoint_file)
# model = init_detector(config_file, checkpoint_file, device="cpu")

result = inference_detector(model, f"{mm_path}/demo/demo.jpg")
result_img = model.show_result(f"{mm_path}/demo/demo.jpg", result)
result_img.shape


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

img = data["img"][0]


# %%
result = model(return_loss=False, rescale=True, **data)

# %%
class tmp(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        # self.sub_model = torch.nn.Sequential([model.backbone, model.neck.lateral_convs, model.neck.fpn_convs[:2]])
        self.backbone = model.backbone
        self.neck = model.neck
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        
        return x
neck[0].shape
torch.max(neck[0])
SubModel = tmp(model)
neck = SubModel(img)
img[0].shape
len(neck)
# %%
tmp(img[0])
model.neck.lateral_convs
#%%
sub_model(img[0])


sub_model.add_modules(, )


modeaa

neck[4].shape

octave_base_scale=4
scales_per_octave=3
ratios=[0.5, 1.0, 2.0]
strides=[8, 16, 32, 64, 128]
from module.mmdetection.mmdet.core import AnchorGenerator
anchor_gerator = AnchorGenerator(strides=strides,
                                 ratios=ratios,
                                 octave_base_scale=octave_base_scale,
                                 scales_per_octave=scales_per_octave
                                 )
self
anchors = anchor_gerator.grid_anchors([(100, 152), (50, 76), (25, 38), (13, 19), (7, 10)], device="cuda")
torch.max(anchors[4])
anchors[0].shape
anchors[1].shape
anchors[2].shape
anchors[3].shape
anchors[4].shape
427 / 8
torch.max(anchors[0][:,2] - anchors[0][:,0])
torch.max(anchors[1][:,2] - anchors[1][:,0])
torch.max(anchors[2][:,2] - anchors[2][:,0])
torch.max(anchors[3][:,2] - anchors[3][:,0])

100 * 8
152 * 8
50 * 16
76 * 16

from mmdet.core import AnchorGenerator
self = AnchorGenerator([16], [1.], [1.], [9])
all_anchors = self.grid_priors([(2, 2)], device='cpu')
print(all_anchors)


self = AnchorGenerator([16, 32], [1.], [1.], [9, 18])
all_anchors = self.grid_priors([(2, 2), (1, 1)], device='cpu')
print(all_anchors)
[tensor([[-4.5000, -4.5000,  4.5000,  4.5000],
        [11.5000, -4.5000, 20.5000,  4.5000],
        [-4.5000, 11.5000,  4.5000, 20.5000],
        [11.5000, 11.5000, 20.5000, 20.5000]]), \
tensor([[-9., -9., 9., 9.]])]
