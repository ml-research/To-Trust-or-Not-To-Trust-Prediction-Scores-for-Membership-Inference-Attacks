# get the cuda 10.1 ubuntu docker image
FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-devel

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# set the working directory and copy everything to the docker file
WORKDIR ./confidences
COPY ./requirements.txt ./

RUN apt-get update && apt-get -y upgrade && apt-get -y install git nano build-essential cmake libboost-all-dev libgtk-3-dev git-lfs
RUN pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
