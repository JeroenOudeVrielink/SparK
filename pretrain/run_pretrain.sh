
# python main.py \
# --exp_name=spark_base_params_bs64_ep400 \
# --exp_dir=jvrielink/data/spark_models \
# --data_path=/jvrielink/AIML_rot_corrected \
# --model=resnet50 \
# --bs=64 \
# --ep=400

python main.py \
--exp_name=debug \
--exp_dir=jvrielink/data/spark_models \
--data_path=/jvrielink/AIML_rot_corrected \
--model=resnet50 \
--bs=512 \
--ep=2


