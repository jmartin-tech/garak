FROM python:3.12

WORKDIR /workspace/garak
COPY . /workspace/garak
# TODO: add cleanup steps to remove repository only files

RUN python -m pip install .

