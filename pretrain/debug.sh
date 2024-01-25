
python main.py \
--exp_name=debug \
--exp_dir=debug \
--data_path=/mnt/sdb1/Data_remote/AIML_rot_corrected \
--model=resnet50 \
--bs=4 \
--ep=6 \
--dataloader_workers=4 \
--annotations_file=/mnt/sdb1/Data_remote/AIML_rot_corrected/annotations/img_paths_mini.csv \
--model_ckpt_freq=2 \
--input_size=384


