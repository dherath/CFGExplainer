#!/bin/bash

# the following params worked the best:
declare -a batch_size="32" #"32" # always keep it 1, GPU cannot handle it
declare -a path="data"
# "../../y_datasets/YANCFG_sample1"
declare -a hiddens="1024-512-128"
declare -a elr="0.00001" # "0.00001"
declare -a model_name="classifier_lynxv2_"
declare -a data_name="yancfg_complete_sample2"
declare -a eepochs="300"
declare -a expname="ep300_b32_elr00001_" #"MLP2v2_ep300_b32_elr00005_sciflow_CFGExplainer_"
# declare -a disbale_tqdm="True"

CUDA_VISIBLE_DEVICES=0 python exp_train_CFGExplainer.py $batch_size $path $hiddens $elr $model_name $data_name $eepochs $expname> ./trace/training_sample2_CFGExplainer.txt 2>&1 &
# > ./trace/test2_sciflow_b32_elr00001_newMLP2version.txt 2>&1 &
