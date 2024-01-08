FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime

WORKDIR /code

COPY ./requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

ENV WANDB_API_KEY="e2acc047b3e4f05edb42a802cb361ef41eacb607"

COPY . ./code

ENV PYTHONPATH "${PYTHONPATH}:/code"


