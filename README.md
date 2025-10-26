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

## Download model and watch play

1. download the model you want from shared google drive
2. Make there is or create a models directory(folder) within the project folder 
3. Place the model within the folder (don't unzip)
4. Within watchPlay.py make sure the MODEL_PATH has the correct model
5. Inside project directory run this command:

```bash
  # Make sure you have activated venv
  # and pip installed the requirements.txt
  python3 src/watchPlay.py 
```

## Check logs

1. download the zipped tensorboard_logs from shared google drive folder
2. unzip and place the folder within our project directory
3. Inside project directory run this command:

```bash
# Make sure you have activated venv
# and pip installed the requirements.txt
#check logs
tensorboard --logdir ./tensorboard_logs/ 

```
4. Follow the local host link to see all the reports
