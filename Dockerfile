#FROM nvcr.io/nvidia/pytorch:24.01-py3
FROM python:3.12

WORKDIR /workspace/garak
COPY . /workspace/garak

RUN python -m pip install -r requirements.txt

