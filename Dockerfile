FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

LABEL author="Stefan Podgorski" \
      description="A container for running reinforcement learning demos." \
      version='0.1'

RUN apt-get update -y && \
    apt-get install -y build-essential python-dev swig python-pygame git

# get the repo
RUN git clone --single-branch --branch add-dockerfile https://github.com/podgorki/rl_demos.git

# install the requirements
RUN pip install -r rl_demos/requirements.txt

CMD ["python", "rl_demos/demos/demo_discrete.py"]