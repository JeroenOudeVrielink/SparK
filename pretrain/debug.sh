
python main.py \
--exp_name=debug \
--exp_dir=/spark_models \
--data_path=/jvrielink/AIML_rot_corrected \
--model=resnet50 \
--bs=256 \
--ep=6 \
--dataloader_workers=14 \
--annotations_file=annotations/img_paths_mini.csv \
--model_ckpt_freq=2


