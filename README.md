# Quantum Reinforcement Learning for Real-time Optimization in EV Charging Systems(Q-EVCS)
This is an implementation code for our paper entitled: Quantum Reinforcement Learning for Real-time Optimization in EV Charging Systems.


# Requirements

see in requirements.txt

install packages with pip install -r requirements.txt

# Quick Start
`$ python Q_main.py`

Directly run the Q_main.py, the approach will run with the default settings.

# Code Structure

- `args_config.py` : configuration file defining default arguments and hyperparameters.
- `Q_model.py` : functions to generate the Q-EVCS model, manage memory for experience replay, and perform Q-EVCS testing.
- `env.py` : defines the EV charging scheduling environment.
- `Q_main.py` : main function.
