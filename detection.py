#%%
import os
import time
import argparse
import warnings
import torch
import mmcv

import numpy as np
import pandas as pd
import os.path as osp

from tqdm import tqdm
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


def find_fn(results, dataset, cfg, iou_thr=0.5):

    print("FIND FALSE NEGATIVEs")    
    gt_fns = []
    for i, result in enumerate(tqdm(results)):
        img_metas = dataset[i]["img_metas"][0].data
        ori_shape = img_metas["ori_shape"][:2]

        box_norm_factor = torch.tile(torch.tensor([ori_shape[1], ori_shape[0]]), (2,))

        ann = dataset.get_ann_info(i)
        gt_label = ann["labels"]
        if gt_label.shape[0] == 0:
            gt_fns.append(torch.tensor([]).to(cfg.device))
            continue
        gt_label = torch.tensor(gt_label).to(cfg.device)
        gt_box = ann["bboxes"]
        gt_box = (torch.tensor(gt_box) / box_norm_factor).to(cfg.device)

        gt_fn = torch.ones_like(gt_label)
        for c in torch.unique(gt_label):
            result_per_class = result[c]
            if result_per_class.shape[0] == 0:
                continue

            result_per_class = torch.tensor(result_per_class / np.array(box_norm_factor.tolist() + [1])).to(cfg.device)
            gt_per_class = gt_box[gt_label == c]

            iou = compute_iou(result_per_class, gt_per_class)
            max_iou = torch.max(iou, dim=0).values
            tp = (max_iou >= iou_thr).type(torch.LongTensor).to(cfg.device)

            gt_indices = (gt_label == c).nonzero().squeeze()
            gt_tp = torch.zeros_like(gt_label).scatter(dim=0, index=gt_indices, src=tp)
            gt_fn -= gt_tp

        gt_fns.append(gt_fn)

    return gt_fns


class RoIExtractor(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.backbone = model.backbone
        self.neck = model.neck
        self.rpn = model.rpn_head
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.rpn(x)
        
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
    # cfg.model.train_cfg = None
    if diagnosis: 
        cfg.model.test_cfg.nms = None
        cfg.model.test_cfg.score_thr = 1e-3
        cfg.model.test_cfg.max_per_img = None
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
        # if i == 100: break
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


def compute_iou(result_per_class, gt_per_class):
    box_y1, box_x1, box_y2, box_x2, _ = result_per_class.tensor_split(5, dim=1)
    gt_y1, gt_x1, gt_y2, gt_x2 = gt_per_class.tensor_split(4, dim=1)

    box_area = (box_y2 - box_y1) * (box_x2 - box_x1)
    gt_area = (gt_y2 - gt_y1) * (gt_x2 - gt_x1)

    x_top = torch.maximum(box_x1, gt_x1.transpose(0, 1))
    y_top = torch.maximum(box_y1, gt_y1.transpose(0, 1))
    x_bottom = torch.minimum(box_x2, gt_x2.transpose(0, 1))
    y_bottom = torch.minimum(box_y2, gt_y2.transpose(0, 1))

    inter_area = torch.maximum(x_bottom - x_top, torch.tensor(0)) * torch.max(y_bottom - y_top, torch.tensor(0))
    union_area = box_area + torch.transpose(gt_area, 1, 0) - inter_area
    iou = inter_area / union_area

    return iou


def build_args(config_file, checkpoint_file):
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
    
    return args


def false_neg_mechanism(results, dataset, gt_fns, anchor_generator, cfg, iou_thr=0.5):
    fn_mechanism = []
    print("FALSE NEGATIVE MECHANISM")
    for i, result in enumerate(tqdm(results)):
        gt_fn_bool = gt_fns[i].type(torch.BoolTensor).to(cfg.device)
        img_metas = dataset[i]["img_metas"][0].data
        ori_shape = img_metas["ori_shape"][:2]
        pad_shape = img_metas["pad_shape"][:2]

        box_norm_factor = torch.tile(torch.tensor([ori_shape[1], ori_shape[0]]), (2,)).to(cfg.device)
        anchor_norm_factor = torch.tile(torch.tensor([pad_shape[1], pad_shape[0]]), (2,)).to(cfg.device)

        ann = dataset.get_ann_info(i)
        gt_label = torch.tensor(ann["labels"]).to(cfg.device)
        gt_box = torch.tensor(ann["bboxes"]).to(cfg.device) / box_norm_factor

        strides = cfg.model.bbox_head.anchor_generator.strides
        feature_map_shapes = [(np.ceil(pad_shape[0] / stride), np.ceil(pad_shape[1] / stride)) for stride in strides]
        anchors = anchor_generator.grid_anchors(feature_map_shapes, device=cfg.device)
        anchor_box = torch.concat(anchors, dim=0) / anchor_norm_factor

        iou = compute_iou(anchor_box, gt_box)
        pos_bool = iou >= cfg.model.train_cfg.assigner.pos_iou_thr
        neg_bool = iou < cfg.model.train_cfg.assigner.neg_iou_thr
        pos_num = torch.where(pos_bool, 1, 0).sum(dim=0)
        neg_num = torch.where(neg_bool, 1, 0).sum(dim=0)
        fg_bg_ratio = pos_num / neg_num

        iou_mean = (torch.where(pos_bool, iou, 0).sum(dim=0) / pos_num).nan_to_num(0)
        iou_std = (torch.where(pos_bool, iou - iou_mean, 0).square().sum(dim=0) / (pos_num - 1)).sqrt().nan_to_num(0)
        gt_property = torch.stack([
            fg_bg_ratio, iou_mean, iou_std
        ]).T

        fn_box = gt_box[gt_fn_bool]
        tp_box = gt_box[~gt_fn_bool]

        fn_label = gt_label[gt_fn_bool]
        tp_label = gt_label[~gt_fn_bool]

        fn_property = gt_property[gt_fn_bool]
        tp_property = gt_property[~gt_fn_bool]

        if fn_label.shape[0] == 0:
            continue

        total_box = torch.tensor(np.vstack(result)[:, :4]).to(cfg.device) / box_norm_factor
        fn_box = fn_box 

        iou = compute_iou(total_box, fn_box)
        max_iou = torch.max(iou, dim=0).values
        cls_fn = (max_iou >= iou_thr).to(cfg.device)


        iou = compute_iou(anchor_box, fn_box[~cls_fn])
        max_iou = torch.max(iou, dim=0).values

        box_fn = (max_iou >= iou_thr)
        box_fn_indices = (~cls_fn).nonzero().squeeze()
        reg_fn = torch.zeros_like(cls_fn).scatter(dim=0, index=box_fn_indices, src=box_fn)

        rpn_fn = torch.logical_and(~reg_fn, ~cls_fn)
        
        cls_fn_box = fn_box[cls_fn]
        reg_fn_box = fn_box[reg_fn]
        rpn_fn_box = fn_box[rpn_fn]

        tp_label = torch.unsqueeze(tp_label, -1)
        cls_fn_label = torch.unsqueeze(fn_label[cls_fn], -1)
        reg_fn_label = torch.unsqueeze(fn_label[reg_fn], -1)
        rpn_fn_label = torch.unsqueeze(fn_label[rpn_fn], -1)

        cls_fn_property = fn_property[cls_fn]
        reg_fn_property = fn_property[reg_fn]
        rpn_fn_property = fn_property[rpn_fn]


        fn_mechanism_per_img = torch.cat([
            torch.cat((torch.zeros_like(tp_label), tp_box, tp_label, tp_property), dim=-1),
            torch.cat((torch.ones_like(cls_fn_label) * 1, cls_fn_box, cls_fn_label, cls_fn_property), dim=-1),
            torch.cat((torch.ones_like(reg_fn_label) * 2, reg_fn_box, reg_fn_label, reg_fn_property), dim=-1),
            torch.cat((torch.ones_like(rpn_fn_label) * 3, rpn_fn_box, rpn_fn_label, rpn_fn_property), dim=-1)
            ], dim=0)
        
        fn_mechanism += fn_mechanism_per_img.tolist()
    
    return fn_mechanism
        

# %% ARGS
path = "./module/mmdetection"

parser = argparse.ArgumentParser()

parser.add_argument('--config', type=str)
parser.add_argument('--checkpoint', type=str)
args = parser.parse_args()

config_file = f"{path}/{args.config}"
checkpoint_file = f"{path}/{args.checkpoint}"
# config_file = f"{path}/retinanet_r101_fpn_1x_coco.py"
# checkpoint_file = f"{path}/retinanet_r101_fpn_1x_coco_20200130-7a93545f.pth"
# config_file = f"{path}/configs/resnet_strikes_back/retinanet_r50_fpn_rsb-pretrain_1x_coco.py"
# checkpoint_file = f"{path}/retinanet_r50_fpn_rsb-pretrain_1x_coco_20220113_175432-bd24aae9.pth"
# config_file = f"{path}/retinanet_x101_32x4d_fpn_1x_coco.py"
# checkpoint_file = f"{path}/retinanet_x101_32x4d_fpn_1x_coco_20200130-5c8b7ec4.pth"
# config_file = f'{path}/yolov3_mobilenetv2_320_300e_coco.py'
# checkpoint_file = f'{path}/yolov3_mobilenetv2_320_300e_coco_20210719_215349-d18dff72.pth'
# import sys
# import subprocess
# if not os.path.exists(config_file):
#     subprocess.check_call([sys.executable, "-m", "mim", "download", "mmdet", "--config" f"{config_file.split('/')[-1].split('.')[0]}", "--dest", "."])
#%% CONFIG ASSIGN
args = build_args(config_file, checkpoint_file)
model, data_loader, cfg = build_model_datasets(args, "train", path)
results, dataset = predict_dataset(model, data_loader)

#%%
gt_fns = find_fn(results, dataset, cfg)

del results
# %%
args = build_args(config_file, checkpoint_file)
model, data_loader, cfg = build_model_datasets(args, "test", path, diagnosis=True)
results, dataset = predict_dataset(model, data_loader)

#%%
anchor_cfg = cfg.model.bbox_head.anchor_generator
anchor_type = anchor_cfg.type
octave_base_scale = anchor_cfg.octave_base_scale
scales_per_octave = anchor_cfg.scales_per_octave
ratios = anchor_cfg.ratios
strides = anchor_cfg.strides

anchor_generator = AnchorGenerator(strides=strides,
                                 ratios=ratios,
                                 octave_base_scale=octave_base_scale,
                                 scales_per_octave=scales_per_octave
                                 )

fn_mechanism = false_neg_mechanism(results, dataset, gt_fns, anchor_generator, cfg)


# %%
fn_df = pd.DataFrame(fn_mechanism, columns=[
    "type", "box_x1", "box_y1", "box_x2", "box_y2", "label", "fg_bg_ratio", "iou_mean", "iou_std"
    ])
fn_df["area"] = fn_df.apply(lambda x: (x["box_y2"] - x["box_y1"]) * (x["box_x2"] - x["box_x1"]), axis=1)
fn_df["ratio"] = fn_df.apply(lambda x: (x["box_y2"] - x["box_y1"]) / (x["box_x2"] - x["box_x1"]), axis=1)
fn_df["ctr_x"] = fn_df["box_y2"] - fn_df["box_y1"]
fn_df["ctr_y"] = fn_df["box_x2"] - fn_df["box_x1"]

file_name = config_file.split("/")[-1].split(".")[0]
fn_df.to_csv(f"test_{file_name}.csv", index=False)
# pd.read_csv(f"{file_name}.csv")
# fn_df["type"].value_counts()
#%%
'''
cfg.model.test_cfg.rpn
model
sub_model = RoIExtractor(model)
img = torch.unsqueeze(dataset[0]["img"][0], 0)
sub = sub_model(img)
feat = feat_model(img)
len(sub)
len(feat)

len(sub[0])

len(sub[1])

sub[0][2].shape
sub[0][2][0,0].shape
sub[1][2].shape
feat[2].shape

model
model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
cfg.model.test_cfg
tmp = sub[1][1][0][:, 0, 0]
torch.reshape(tmp, [3, 4])
len(sub)
model
cfg.model["bbox_coder"]
cfg.model.rpn_head.bbox_coder
model.rpn_head
cfg.model.roi_extractor
cfg.model.roi_head.bbox_head.bbox_coder
cfg.model.rpn_head.bbox_coder
'''