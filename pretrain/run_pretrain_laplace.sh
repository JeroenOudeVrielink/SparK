#!/usr/bin/env bash


python main.py \
--exp_name=spark_in448_bs32_ep400_laplace_recon \
--exp_dir=/spark_models \
--data_path=/jvrielink/AIML_rot_corrected \
--model=resnet50 \
--bs=32 \
--ep=401 \
--dataloader_workers=12 \
--mask=0.75 \
--laplace_recon=True \
--input_size=448


