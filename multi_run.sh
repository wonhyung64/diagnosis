#!/bin/bash

# virutal environment directory
ENV=/home1/wonhyung64/anaconda3/envs/openmmlab/bin/python

# file directory of multiple execution source ".sh"
RUN_SRC=./run_src.sh

# file directory of experiment ".py"
EXECUTION_FILE=/home1/wonhyung64/Github/diagnosis/test.py

# default prefix of job name
DEFAULT_NAME=experiment

# python argparse source for experiments
experiments=(
"--e=12 --b=33 --l=0.1"
"--e=12 --b=33 --l=0.1"
"--b=44 --l=0.2"
"--e=14"
)

for index in ${!experiments[*]}; do
    sbatch --job-name=$DEFAULT_NAME$index $RUN_SRC $ENV $EXECUTION_FILE ${experiments[$index]}
done
