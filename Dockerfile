FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

LABEL author="Stefan Podgorski"

RUN apt-get update -y && \
    apt-get install -y swig


RUN apt-get install -y git

# get the repo
RUN git clone https://github.com/podgorki/rl_demos.git

# install the requirements
RUN pip install -r rl_demos/requirements.txt

WORKDIR rl_demos/
