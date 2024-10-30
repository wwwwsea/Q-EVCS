# Quantum Reinforcement Learning for Real-time Optimization in EV Charging Systems(Q-EVCS)
This is an implementation code for our paper entitled: Quantum Reinforcement Learning for Real-time Optimization in EV Charging Systems.


# Requirements

see in requirements.txt

install packages with pip install -r requirements.txt

# Quick Start
`$ python Q_main.py`

Directly run the Q_main.py, the approach will run with the default settings.

# Code Structure

- `args_config.py`: Configuration file defining the default arguments and hyperparameters for QRL.
- `param_parser.py`: Configuration file defining the default arguments and hyperparameters for DQN and PPO.
- `env.py` : defines the EV charging scheduling environment.
- `Q_model.py` : functions to generate the Q-EVCS model, manage memory for experience replay, and perform Q-EVCS testing.
- `DRL_model.py` : defines deep reinforcement learning models including DQN, PPO, and baseline policies for EV charging scheduling, with functions for training, action selection, and loss tracking.
- `DPSO_model.py` : implements the Discrete Particle Swarm Optimization (DPSO) algorithm for task scheduling, including particle initialization, velocity and position updates, and fitness calculation to find the optimal solution. 
- `DRL_main.py` : test the performance of DQN and PPO..
- `DPSO_main.py` : test the performance of DPSO.
- `Q_main.py` : main function.
