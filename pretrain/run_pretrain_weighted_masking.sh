#!/usr/bin/env bash


python main.py \
--exp_name=spark_in224_bs128_weighted_masking \
--exp_dir=/spark_models \
--data_path=/jvrielink/AIML_rot_corrected \
--model=resnet50 \
--bs=128 \
--ep=301 \
--dataloader_workers=12 \
--mask=0.6 \
--weighted_masking=True \
--input_size=224


