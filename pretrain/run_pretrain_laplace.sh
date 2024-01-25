
python main.py \
--exp_name=test \
--exp_dir=/spark_models \
--data_path=/jvrielink/AIML_rot_corrected \
--model=resnet50 \
--bs=256 \
--ep=300 \
--dataloader_workers=14 \
--mask=0.75 \
--laplace_recon=True \
--input_size=448


