# Install

apt-get install swig


# Discrete PPO

Working demos:
* Acrobot-v1
* CartPole-v1 
* MountainCar-v0


# Continuous PPO

Working demos: 
* BipedalWalker-v3
* LunarLanderContinuous-v2
* MountainCarContinuous-v0

# Usage

To run:
1. First clone this repo
2. Create an environment (python 3.10)
3. Install pytorch
4. Install requirements
5. run the demo you want i.e. `python demos/demo_discrete.py`

To change demos, edit the relevant config file and change the type i.e. `Acrobot-v1`  

(Docker) 
To run:
1. First clone this repo
2. Build the dockerfile
3. run `docker-compose up`

## Todo

1. Add an argparser to change config params without having to edit the config directly.
2. Improve readme
