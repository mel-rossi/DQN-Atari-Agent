# Wraps the game enviroment to preprocess frames. 
# Takes raaw Atari frames and makes them suitable for neural network: 
# converts to grayscale, resizes to 84x84, stacks 4 frames together (motion),
# does frame skipping (repeats actions), and adds random no-ops at the start. 
# Makes the raw game playable by agent. 
