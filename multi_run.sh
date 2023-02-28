#!/bin/bash

# virutal environment directory
ENV=/home1/wonhyung64/anaconda3/envs/openmmlab/bin/python

# file directory of multiple execution source ".sh"
RUN_SRC=./run_src.sh

# file directory of experiment ".py"
EXECUTION_FILE=/home1/wonhyung64/Github/diagnosis/detection.py

# default prefix of job name
DEFAULT_NAME=experiment

# python argparse source for experiments
experiments=(
"--config=retinanet_r18_fpn_1x_coco.py --checkpoint=retinanet_r18_fpn_1x_coco_20220407_171055-614fd399.pth"
"--config=retinanet_r50_fpn_1x_coco.py --checkpoint=retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth"
"--config=retinanet_x101_64x4d_fpn_1x_coco.py --checkpoint=retinanet_x101_64x4d_fpn_1x_coco_20200130-366f5af1.pth"
)

for index in ${!experiments[*]}; do
    sbatch --job-name=$DEFAULT_NAME$index $RUN_SRC $ENV $EXECUTION_FILE ${experiments[$index]}
done
