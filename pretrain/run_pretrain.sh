

#base_lr = 2e-4 * (4096 / bs)

python main.py \
--exp_name=debug \
--exp_dir=/media/jvrielink/9958194f-e772-4a20-b156-50f2ac51f106/spark_data \
--data_path=/home/jvrielink/AIML_rot_corrected \
--model=resnet50 \
--bs=128 \
--base_lr=0.0064 \
--ep=400
