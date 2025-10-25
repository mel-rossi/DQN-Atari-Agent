# Storage for game experiences 
# Circular memory bank that stores the last 1 million game transitions (state, action, reward, next_state). The agent randomly samples from this to learn, which prevents it from only learning the recent experiences (important for stability). 
