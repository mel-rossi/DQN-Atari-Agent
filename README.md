# DQN-Atari-Agent
Reinforcement Learning Deep Q-Network Agent Training for COMP 542 Group Project

## Group Members (Collaborators)
- [Mel Rossi Rodrigues](https://github.com/mel-rossi)
- [Maria Alexandra Lois Peralejo](https://github.com/MariaAlexandraPeralejo)
- [Angel Cortes Gildo](https://github.com/angcortes)
- [Colby Snook](https://github.com/colbysnook)

## Environment Setup

Inside your project directory run the following command:
  
```bash
bash setup.sh # Run this only once (auto activates virtual environment)
```
Run this instead if you're on zsh shell: `zsh setup.sh`

If you've already ran `setup.sh`, run this to activate the virtual environment whenever it is deactivated.
```bash
source venv/bin/activate
```

You can deactivate the enviroment by simply running `deactivate` in shell. 

Note: The virtual environment will be automatically deactivated if you close or refresh the terminal.

## Train & Watch the Model Play

Note: Make sure you've completed the environment set up and it is activated

1. Train the model
```
python src/train.py
```

2. Watch the model play
```
python src/play.py
```

## Check logs

Note: Make sure you've completed the environment set up and it is activated

1. Inside project directory run this command:

```bash
tensorboard --logdir ./logs/ 
```
2. Follow the local host link to see all the reports
