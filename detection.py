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
help(inference_detector)
model(a["img"], a["img_metas"], return_loss=False)
type(a["img"])
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
data
a
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


model.cfg
#%%
cfg = model.cfg

# replace the ${key} with the value of cfg.key
cfg = replace_cfg_vals(cfg)
# update data root according to MMDET_DATASETS
update_data_root(cfg)
cfg = compat_cfg(cfg)
# set multi-process settings
setup_multi_processes(cfg)
# set cudnn_benchmark
if cfg.get('cudnn_benchmark', False):
    torch.backends.cudnn.benchmark = True

if 'pretrained' in cfg.model:
    cfg.model.pretrained = None
elif 'init_cfg' in cfg.model.backbone:
    cfg.model.backbone.init_cfg = None

if cfg.model.get('neck'):
    if isinstance(cfg.model.neck, list):
        for neck_cfg in cfg.model.neck:
            if neck_cfg.get('rfp_backbone'):
                if neck_cfg.rfp_backbone.get('pretrained'):
                    neck_cfg.rfp_backbone.pretrained = None
    elif cfg.model.neck.get('rfp_backbone'):
        if cfg.model.neck.rfp_backbone.get('pretrained'):
            cfg.model.neck.rfp_backbone.pretrained = None


test_dataloader_default_args = dict(
    samples_per_gpu=1, workers_per_gpu=1, dist=False, shuffle=False)
    # samples_per_gpu=1, workers_per_gpu=2, dist=distributed, shuffle=False)

# in case the test dataset is concatenated
if isinstance(cfg.data.test, dict):
    cfg.data.test.test_mode = True
    if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
        # Replace 'ImageToTensor' to 'DefaultFormatBundle'
        cfg.data.test.pipeline = replace_ImageToTensor(
            cfg.data.test.pipeline)
elif isinstance(cfg.data.test, list):
    for ds_cfg in cfg.data.test:
        ds_cfg.test_mode = True
    if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
        for ds_cfg in cfg.data.test:
            ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

test_loader_cfg = {
    **test_dataloader_default_args,
    **cfg.data.get('test_dataloader', {})
}
a
rank, _ = get_dist_info()

# build the dataloader
dataset = build_dataset(cfg.data.test)
dataset.evaluate()
data_loader = build_dataloader(dataset, **test_loader_cfg)
a = next(iter(data_loader))
cfg.data.test
os.chdir("module/mmdetection")
to_tensor(a["img"])
help(data_loader)
data_loader.__getitem__
data_loader.__len__()
from mmdet.datasets import to_tensor
import mmdet
mmdet.datasets.to_tensor()
model.backbone.norm_eval
data
a
result = inference_detector(model, a)

for i, data in enumerate(data_loader):break
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)
        model.eval()

help(type(a["img"]))
a["img"].__init__
a["img"]


#%%
# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import time
import warnings

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmdet.apis import multi_gpu_test, single_gpu_test
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector
from mmdet.utils import (build_ddp, build_dp, compat_cfg, get_device,
                         replace_cfg_vals, rfnext_init_model,
                         setup_multi_processes, update_data_root)


args = argparse.Namespace(
    config = "faster_rcnn_r50_fpn_1x_coco.py",
    checkpoint = "faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth",
    out=None,
    eval="bbox",
    format_only=None,
    show=None,
    show_dir=None,
    gpu_ids = None,
    gpu_id = 0,
    show_score_thr = 0.3,
    launcher="none",
    local_rank=0,
    cfg_options=None,
    work_dir = None,
    fuse_conv_bn = None,
    gpu_collect = None,
    tmpdir = None,
    options = {"classwise":True},
    eval_options = None
)


if 'LOCAL_RANK' not in os.environ:
    os.environ['LOCAL_RANK'] = str(args.local_rank)

if args.options and args.eval_options:
    raise ValueError(
        '--options and --eval-options cannot be both '
        'specified, --options is deprecated in favor of --eval-options')
if args.options:
    warnings.warn('--options is deprecated in favor of --eval-options')
    args.eval_options = args.options


assert args.out or args.eval or args.format_only or args.show \
    or args.show_dir, \
    ('Please specify at least one operation (save/eval/format/show the '
        'results / save the results) with the argument "--out", "--eval"'
        ', "--format-only", "--show" or "--show-dir"')

if args.eval and args.format_only:
    raise ValueError('--eval and --format_only cannot be both specified')

if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
    raise ValueError('The output file must be a pkl file.')

cfg = Config.fromfile(args.config)

# replace the ${key} with the value of cfg.key
cfg = replace_cfg_vals(cfg)

# update data root according to MMDET_DATASETS
update_data_root(cfg)

if args.cfg_options is not None:
    cfg.merge_from_dict(args.cfg_options)

cfg = compat_cfg(cfg)

# set multi-process settings
setup_multi_processes(cfg)

# set cudnn_benchmark
if cfg.get('cudnn_benchmark', False):
    torch.backends.cudnn.benchmark = True

if 'pretrained' in cfg.model:
    cfg.model.pretrained = None
elif 'init_cfg' in cfg.model.backbone:
    cfg.model.backbone.init_cfg = None

if cfg.model.get('neck'):
    if isinstance(cfg.model.neck, list):
        for neck_cfg in cfg.model.neck:
            if neck_cfg.get('rfp_backbone'):
                if neck_cfg.rfp_backbone.get('pretrained'):
                    neck_cfg.rfp_backbone.pretrained = None
    elif cfg.model.neck.get('rfp_backbone'):
        if cfg.model.neck.rfp_backbone.get('pretrained'):
            cfg.model.neck.rfp_backbone.pretrained = None

if args.gpu_ids is not None:
    cfg.gpu_ids = args.gpu_ids[0:1]
    warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                    'Because we only support single GPU mode in '
                    'non-distributed testing. Use the first GPU '
                    'in `gpu_ids` now.')
else:
    cfg.gpu_ids = [args.gpu_id]
cfg.device = get_device()
# init distributed env first, since logger depends on the dist info.
if args.launcher == 'none':
    distributed = False
else:
    distributed = True
    init_dist(args.launcher, **cfg.dist_params)

test_dataloader_default_args = dict(
    samples_per_gpu=1, workers_per_gpu=2, dist=distributed, shuffle=False)

# in case the test dataset is concatenated
# if isinstance(cfg.data.test, dict):
#     cfg.data.test.test_mode = True
#     if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
#         # Replace 'ImageToTensor' to 'DefaultFormatBundle'
#         cfg.data.test.pipeline = replace_ImageToTensor(
#             cfg.data.test.pipeline)
# elif isinstance(cfg.data.test, list):
#     for ds_cfg in cfg.data.test:
#         ds_cfg.test_mode = True
#     if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
#         for ds_cfg in cfg.data.test:
#             ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

# test_loader_cfg = {
#     **test_dataloader_default_args,
#     **cfg.data.get('test_dataloader', {})
# }

# rank, _ = get_dist_info()
# # allows not to create
# if args.work_dir is not None and rank == 0:
#     mmcv.mkdir_or_exist(osp.abspath(args.work_dir))
#     timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
#     json_file = osp.join(args.work_dir, f'eval_{timestamp}.json')

# # build the dataloader
# dataset = build_dataset(cfg.data.test)
# data_loader = build_dataloader(dataset, **test_loader_cfg)

if isinstance(cfg.data.train, dict):
    cfg.data.train.test_mode = True
    if cfg.data.train_dataloader.get('samples_per_gpu', 1) > 1:
        # Replace 'ImageToTensor' to 'DefaultFormatBundle'
        cfg.data.train.pipeline = replace_ImageToTensor(
            cfg.data.train.pipeline)

test_loader_cfg = {
    **test_dataloader_default_args,
    **cfg.data.get('test_dataloader', {})
}

rank, _ = get_dist_info()
# allows not to create
if args.work_dir is not None and rank == 0:
    mmcv.mkdir_or_exist(osp.abspath(args.work_dir))
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    json_file = osp.join(args.work_dir, f'eval_{timestamp}.json')

# build the dataloader
dataset = build_dataset(cfg.data.train)
data_loader = build_dataloader(dataset, **test_loader_cfg)

# build the model and load checkpoint
cfg.model.train_cfg = None
model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
# init rfnext if 'RFSearchHook' is defined in cfg
rfnext_init_model(model, cfg=cfg)
fp16_cfg = cfg.get('fp16', None)
if fp16_cfg is None and cfg.get('device', None) == 'npu':
    fp16_cfg = dict(loss_scale='dynamic')
if fp16_cfg is not None:
    wrap_fp16_model(model)
checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
if args.fuse_conv_bn:
    model = fuse_conv_bn(model)
# old versions did not save class info in checkpoints, this walkaround is
# for backward compatibility
if 'CLASSES' in checkpoint.get('meta', {}):
    model.CLASSES = checkpoint['meta']['CLASSES']
else:
    model.CLASSES = dataset.CLASSES

if not distributed:
    model = build_dp(model, cfg.device, device_ids=cfg.gpu_ids)
    outputs = single_gpu_test(model, data_loader, args.show, args.show_dir,
                                args.show_score_thr)
else:
    model = build_ddp(
        model,
        cfg.device,
        device_ids=[int(os.environ['LOCAL_RANK'])],
        broadcast_buffers=False)
os.chdir("./module/mmdetection")
#%%
import os.path as osp
import pickle
import shutil
import tempfile
import time

import mmcv
import torch
import torch.distributed as dist
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info

from mmdet.core import encode_mask_results

def single_gpu_test(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    show_score_thr=0.3):
    model.eval()
    results = []
    dataset = data_loader.dataset
    PALETTE = getattr(dataset, 'PALETTE', None)
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):break
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

        batch_size = len(result)
        if show or out_dir:
            if batch_size == 1 and isinstance(data['img'][0], torch.Tensor):
                img_tensor = data['img'][0]
            else:
                img_tensor = data['img'][0].data[0]
            img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)

            for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                else:
                    out_file = None

                model.module.show_result(
                    img_show,
                    result[i],
                    bbox_color=PALETTE,
                    text_color=PALETTE,
                    mask_color=PALETTE,
                    show=show,
                    out_file=out_file,
                    score_thr=show_score_thr)

        # encode mask results
        if isinstance(result[0], tuple):
            result = [(bbox_results, encode_mask_results(mask_results))
                      for bbox_results, mask_results in result]
        # This logic is only used in panoptic segmentation test.
        elif isinstance(result[0], dict) and 'ins_results' in result[0]:
            for j in range(len(result)):
                bbox_results, mask_results = result[j]['ins_results']
                result[j]['ins_results'] = (bbox_results,
                                            encode_mask_results(mask_results))

        results.extend(result)

        for _ in range(batch_size):
            prog_bar.update()
    return results
