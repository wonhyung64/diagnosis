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
# "./module/mmdetection/r18_flip.py"
# "./module/mmdetection/r18_flip.py"
# "./module/mmdetection/r18_flip.py"
# "./module/mmdetection/r18_flip.py"
# "./module/mmdetection/r18_flip.py"
# "./module/mmdetection/r50_flip.py"
# "./module/mmdetection/r50_flip.py"
# "./module/mmdetection/r50_flip.py"
# "./module/mmdetection/r50_flip.py"
# "./module/mmdetection/r50_flip.py"
"./module/mmdetection/x101_flip.py --seed 0"
"./module/mmdetection/x101_flip.py --seed 0"
"./module/mmdetection/x101_flip.py --seed 0"
"./module/mmdetection/x101_flip.py --seed 0"
"./module/mmdetection/x101_flip.py --seed 0"


)

for index in ${!experiments[*]}; do
    sbatch --job-name=$DEFAULT_NAME$index $RUN_SRC $ENV $EXECUTION_FILE ${experiments[$index]}
done
