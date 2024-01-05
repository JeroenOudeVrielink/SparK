

#base_lr = 2e-4 * (4096 / bs)

python main.py \
--exp_name=debug \
--data_path=/media/jvrielink/9958194f-e772-4a20-b156-50f2ac51f106/spark_data \
--model=resnet50 \
--bs=256 \
--base_lr=0.0032 \
--ep=400
