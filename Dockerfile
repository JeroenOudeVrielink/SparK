FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

WORKDIR /code

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

COPY ./requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

ENV WANDB_API_KEY="e2acc047b3e4f05edb42a802cb361ef41eacb607"

COPY . ./code

ENV PYTHONPATH "${PYTHONPATH}:/code"


