#!/bin/bash

# virutal environment directory
ENV=/home1/wonhyung64/anaconda3/envs/openmmlab/bin/python

# file directory of multiple execution source ".sh"
RUN_SRC=./run_gpu1_src.sh
# RUN_SRC=./run_gpu2_src.sh
# RUN_SRC=./run_hgx_src.sh
# RUN_SRC=./run_cpu_src.sh

# file directory of experiment ".py"
EXECUTION_FILE=/home1/wonhyung64/Github/diagnosis/module/mmdetection/tools/train.py

# default prefix of job name
DEFAULT_NAME=experiment

# python argparse source for experiments
experiments=(
# "./module/mmdetection/config_files/retinanet_swin-t-p4-w7_fpn_1x_coco.py --seed 0"
# "./module/mmdetection/config_files/no_flip_retinanet_swin-t-p4-w7_fpn_1x_coco.py --seed 0"
# "./module/mmdetection/config_files/retinanet_r50_fpn_rsb-pretrain_1x_coco.py --seed 0"
# "./module/mmdetection/config_files/no_flip_retinanet_r50_fpn_rsb-pretrain_1x_coco.py --seed 0"
# "./module/mmdetection/config_files/fcos_r50_fpn_rsb_gn-head_1x_coco.py --seed 0"
# "./module/mmdetection/config_files/no_flip_fcos_r50_fpn_rsb_gn-head_1x_coco.py --seed 0"
# "./module/mmdetection/config_files/fcos_swin-t-p4-w7_fpn_gn-head_1x_coco.py --seed 0"
# "./module/mmdetection/config_files/no_flip_fcos_swin-t-p4-w7_fpn_gn-head_1x_coco.py --seed 0"
# "./module/mmdetection/config_files/retinanet_swin-t-p4-w7_fpn_1x_coco.py --seed 2" nan
# "./module/mmdetection/config_files/no_flip_retinanet_swin-t-p4-w7_fpn_1x_coco.py --seed 2" nan
# "./module/mmdetection/config_files/retinanet_swin-t-p4-w7_fpn_1x_coco.py --seed 4" 
# "./module/mmdetection/config_files/no_flip_retinanet_swin-t-p4-w7_fpn_1x_coco.py --seed 4" nan
# "./module/mmdetection/config_files/retinanet_swin-t-p4-w7_fpn_1x_coco.py --seed 5" 
# "./module/mmdetection/config_files/no_flip_retinanet_swin-t-p4-w7_fpn_1x_coco.py --seed 5" nan
# "./module/mmdetection/config_files/retinanet_swin-t-p4-w7_fpn_1x_coco.py --seed 6"
# "./module/mmdetection/config_files/no_flip_retinanet_swin-t-p4-w7_fpn_1x_coco.py --seed 6" nan
# "./module/mmdetection/config_files/retinanet_swin-t-p4-w7_fpn_1x_coco.py --seed 7"
# "./module/mmdetection/config_files/no_flip_retinanet_swin-t-p4-w7_fpn_1x_coco.py --seed 7" nan
# "./module/mmdetection/config_files/retinanet_swin-t-p4-w7_fpn_1x_coco.py --seed 8" nan
# "./module/mmdetection/config_files/no_flip_retinanet_swin-t-p4-w7_fpn_1x_coco.py --seed 8"
# "./module/mmdetection/config_files/retinanet_swin-t-p4-w7_fpn_1x_coco.py --seed 42"
"./module/mmdetection/config_files/no_flip_retinanet_swin-t-p4-w7_fpn_1x_coco.py --seed 42"
)

for index in ${!experiments[*]}; do
    sbatch --job-name=$DEFAULT_NAME$index $RUN_SRC $ENV $EXECUTION_FILE ${experiments[$index]}
done
