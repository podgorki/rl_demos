FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime@sha256:1e26efd426b0fecbfe7cf3d3ae5003fada6ac5a76eddc1e042857f5d049605ee

LABEL author="Stefan Podgorski" \
      description="A container for running reinforcement learning demos." \
      version='0.1'

RUN apt-get update -y && \
    apt-get install -y build-essential python-dev swig python-pygame git

RUN git clone https://github.com/pybox2d/pybox2d

WORKDIR "pybox2d"

RUN python setup.py build

RUN python setup.py install

# get the repo
RUN git clone --single-branch --branch main https://github.com/podgorki/rl_demos.git

# install the requirements
RUN pip install -r rl_demos/requirements.txt

WORKDIR  "rl_demos/demos"
