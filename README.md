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

**Note:** The virtual environment will be automatically deactivate if you close or refresh the terminal.

## Train & Watch the Model Play

**Note:** Make sure you've completed the environment set up and it is activated

### Train the model

**Note:** The model in training will be automatically named `best_model.zip`. Once you've _finished_ training your model, rename it to something more specific.

Run the 
##### Command to train agent model:
```
python src/train.py
```

### Watch the model play
   
**Note:** A `best_model.zip` file must exist for this to work. 

#### Options
  1. If you're in the middle of training your file will already be named `best_model.zip`
  2. If you have multiple models / renamed your model after you finished training (highly recommened), set an alias (symlink) before running the command to watch the model play.
  3. Download a pre-trained model from the [models GitHub](https://github.com/mel-rossi/models) or clone the repository inside of the DQN-Atari-Agent repo directly
     - If you clone the `models` repository everytime you enter the folder you will be inside of that repository and git commands (`git push`, `git pull`, `git checkout`, etc) will be connected to that repo instead of this one.
     - If you decide to only download one model from there, make sure it's inside of a folder named models inside the project repository

If you chose **option 1**, run the **Command to watch the model play** directly. 
If you chose **option 2 or 3**, create an alias (symlink) for the model you want to watch play: 

**Note:** Replace <model_name> with the name of the model file you want to use (should be a zip file) 

##### Command to check if you already have a file under this alias
```
ls -l best_model.zip
```
  - If you do, something like this should show up: `best_model.zip -> <model_name>`
    - If <model_name> does not match the model you want to use, run this: `rm models/best_model.zip` before running the **Command to set alias to desired model file**
    - If it matches you're good to go, run the **Command to watch the model plat**

##### Command to set alias to desired model file
```
ln -s models/<model_name> models/best_model.zip
```

##### Command to watch the model play
```
python src/play.py
```

## Check logs

**Note:** Make sure you've completed the environment set up and it is activated. This will not work if you do not have a `logs` directory. 

1. Inside project directory run this command:

```bash
tensorboard --logdir ./logs/ 
```
2. Follow the local host link to see all the reports
