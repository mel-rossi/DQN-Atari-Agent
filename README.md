# DQN-Atari-Agent
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

You can check if the setup is correct by running:
```bash
python src/test.py
```
It should return: 
`['Pong-v0', 'Pong-v4', 'PongNoFrameskip-v0', 'PongNoFrameskip-v4', 'ALE/Pong-v5']`

## Train & Test 
⚠ Configuration Switching hasn't been implemented yet. 

Full training: 
```bash
python -m src.train
```

For easy configuration switching:
```bash
python train_with_config.py --config <mode>
```
- Where `<mode>` is the training mode: `default`, `test`, or `heavy`

Try Different Games : 
```bash
python train_with_config.py --config <mode> env ALE/<env_name>
```
- Where `<env_name>` is the name of the Atari Environment / Game

Test Trained Agent: 
```bash
python play.py
```
