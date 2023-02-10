This is a repo for demonstrating policy proximal optimization (PPO) on a few openai gym reinforcement learning (RL) 
environments at The Australian Institute for Machine Learning (AIML) https://www.adelaide.edu.au/aiml/

The code can run in its own environment but so that it is reproducible it is recommended to use the docker image.

# Install

There are two ways to use this repo, 1. create a venv and run it or 2. use the docker container (recommended)

## Conda Env

Install swig: `sudo apt-get install swig`

* Create and environment: `conda env create -f environment.yml`
* Activate it: `conda activate rl_demos`


## Docker

* Install Docker https://docs.docker.com/get-docker/ 
* Build the image `rl_demos/demos/docker-compose build`

# Usage

To run: `python rl_demos/demos/rl_demo --gym_env {environment name} --demo_type {discrete/continuous}`

i.e. to run the Acrobot: ```python rl_demos/demos/rl_demo --gym_env Acrobot-v1 --demo_type discrete```


To change demos, edit the relevant config file and change the type i.e. `Acrobot-v1`  

(Docker) 
To run:
1. First clone this repo
2. Build the dockerfile
3. run `docker-compose up`

Different environments can be run by changing the command args in the docker-compose.yml

Note: To kill the demo hit 'q'

# Environments

## Discrete PPO

Working demos:
* Acrobot-v1: https://www.gymlibrary.dev/environments/classic_control/acrobot/
* CartPole-v1: https://www.gymlibrary.dev/environments/classic_control/cart_pole/ 
* MountainCar-v0: https://www.gymlibrary.dev/environments/classic_control/mountain_car/


## Continuous PPO

Working demos: 
* BipedalWalker-v3: https://www.gymlibrary.dev/environments/box2d/bipedal_walker/
* LunarLanderContinuous-v2: https://www.gymlibrary.dev/environments/box2d/lunar_lander/
* MountainCarContinuous-v0: https://www.gymlibrary.dev/environments/classic_control/mountain_car_continuous/
