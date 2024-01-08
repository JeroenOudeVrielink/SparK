# docker run --gpus all -v $pwd -w /code -it pytorch-test
docker run \
-it \
--rm \
-v $(pwd):/code \
--gpus '"device=1"' \
--mount type=bind,src=/home/jvrielink,target=/jvrielink \
--mount type=bind,src=/home/jvrielink/data/spark_models,target=/spark_models \
--shm-size 64G \
jvrielink/pytorch_spark_new