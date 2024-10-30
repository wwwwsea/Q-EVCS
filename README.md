# Quantum Reinforcement Learning for Real-time Optimization in EV Charging Systems(Q-EVCS)
This is an implementation code for our paper entitled: Quantum Reinforcement Learning for Real-time Optimization in EV Charging Systems.


# Requirements

see in requirements.txt

install packages with pip install -r requirements.txt

# Quick Start
`$ python Q_main.py`

Directly run the Q_main.py, the approach will run with the default settings.

# Code Structure

- `args_config.py` : function for building agents.
- `Q_model.py` : function for building buffer, where some trained data would be saved.
- `env.py` : codes for creating a multi-agent hybrid cloud environment with VMs.
- `Q_main.py` : main function.
