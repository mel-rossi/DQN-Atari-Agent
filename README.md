# RL-Deep-Q-Net-Atari-Agent
Reinforcement Learning Deep Q-Network Agent Training for COMP 542 Group Project

## Group Members (Collaborators)
- [Mel Rossi Rodrigues](https://github.com/mel-rossi)
- [Maria Alexandra Lois Peralejo](https://github.com/MariaAlexandraPeralejo)
- [Angel Cortes Gildo](https://github.com/angcortes)
- [Colby Snook](https://github.com/colbysnook)

## Environment Setup

Inside project directory run the following commands:
  
```bash
# Create virtual enviroment (do this only once)
python3 -m venv venv

# Activate the virtual enviroment (every time you open a new terminal or its deactivated)
source venv/bin/activate

# Install requirements (if you haven't yet)
pip install -r requirements.txt

```
You can deactivate the enviroment by simply running `deactivate` in shell. 


## Check logs

Inside project directory run this command:

```bash
# Make sure you have activated venv
# and pip installed the requirements.txt
#check logs
tensorboard --logdir ./tensorboard_logs/ 

```
