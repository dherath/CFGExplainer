#!/bin/bash

declare -a batch_size="10"
declare -a path="../../../y_datasets/YANCFG_sample2"
#"../../y_datasets/YANCFG_sample1"
declare -a hiddens="1024-512-128"
declare -a lr="0.0001"
declare -a model_name="classifier_lynxv2_"
declare -a data_name="yancfg_complete_sample2"
declare -a epochs="50"
#declare -a disbale_tqdm="True"

CUDA_VISIBLE_DEVICES=1 python exp_train_GCNClassifier.py $batch_size $path $hiddens $lr $model_name $data_name $epochs > ./trace/training_sample2_classifier.txt 2>&1 &

