#%%
import os
import time
import argparse
import warnings
import torch
import mmcv

import numpy as np
import os.path as osp

from mmcv import Config
from mmcv.cnn import fuse_conv_bn
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
# from mmdet.apis import init_detector
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector
from mmdet.utils import (build_dp, compat_cfg, get_device,
                         replace_cfg_vals, rfnext_init_model,
                         setup_multi_processes, update_data_root)
from mmdet.core import (encode_mask_results, AnchorGenerator)


class FeatureExtractor(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.backbone = model.backbone
        self.neck = model.neck
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        
        return x


def build_model_datasets(args, split, path, diagnosis=False):
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

    rank, _ = get_dist_info()
    # allows not to create
    if args.work_dir is not None and rank == 0:
        mmcv.mkdir_or_exist(osp.abspath(args.work_dir))
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        json_file = osp.join(args.work_dir, f'eval_{timestamp}.json')

    
    cfg.data.test.ann_file = f'{path}/data/coco/annotations/instances_{split}2017.json'
    cfg.data.test.img_prefix = f'{path}/data/coco/{split}2017/'

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(dataset, **test_loader_cfg)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    if diagnosis: 
        cfg.model.test_cfg.nms = None
        cfg.model.test_cfg.score_thr = 1e-3
    else:
        cfg.model.test_cfg.score_thr = 0.3

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

    return model, data_loader, cfg


def predict_dataset(model, data_loader,):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        if i == 100: break
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        
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
        prog_bar.update()
    
    return results, dataset


# %% ARGS
path = "./module/mmdetection"

# config_file = f"{mm_path}/yolov3_mobilenetv2_320_300e_coco.py"
# checkpoint_file = f"{mm_path}/yolov3_mobilenetv2_320_300e_coco_20210719_215349-d18dff72.pth"
config_file = f"{path}/retinanet_r101_fpn_1x_coco.py"
checkpoint_file = f"{path}/retinanet_r101_fpn_1x_coco_20200130-7a93545f.pth"
# config_file = f"{mm_path}/faster_rcnn_r50_fpn_1x_coco.py"
# checkpoint_file = f"{mm_path}/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth"

args = argparse.Namespace(
    config = config_file,
    checkpoint = checkpoint_file,
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

#%% CONFIG ASSIGN
model, data_loader, cfg = build_model_datasets(args, "train", path)

#%% ANCHOR LOADER
anchor_cfg = cfg.model.bbox_head.anchor_generator
anchor_type = anchor_cfg.type
octave_base_scale = anchor_cfg.octave_base_scale
scales_per_octave = anchor_cfg.scales_per_octave
ratios = anchor_cfg.ratios
strides = anchor_cfg.strides

anchor_gerator = AnchorGenerator(strides=strides,
                                 ratios=ratios,
                                 octave_base_scale=octave_base_scale,
                                 scales_per_octave=scales_per_octave
                                 )

results, dataset = predict_dataset(model, data_loader)

#%%
for i, result in enumerate(results):break
    img_metas = dataset[i]["img_metas"][0].data
    ori_shape = img_metas["ori_shape"][:2]
    pad_shape = img_metas["pad_shape"][:2]

    box_norm_factor = torch.tile(torch.tensor([ori_shape[1], ori_shape[0]]), (2,))
    anchor_norm_factor = torch.tile(torch.tensor([pad_shape[1], pad_shape[0]]), (2,))

    ann = dataset.get_ann_info(i)
    gt_label = ann["labels"]
    gt_box = ann["bboxes"]
    gt_box = torch.tensor(gt_box) / box_norm_factor

    feature_map_shapes = [(np.ceil(pad_shape[0] / stride), np.ceil(pad_shape[1] / stride)) for stride in strides]
    anchors = anchor_gerator.grid_anchors(feature_map_shapes, device="cpu")
    anchors = [anchor / anchor_norm_factor for anchor in  anchors]

    for c in np.unique(gt_label):break
        result_per_class = result[c] / np.array(box_norm_factor.tolist() + [1])
        gt_per_class = gt_box[gt_label == c]






# %%
