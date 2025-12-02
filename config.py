# Copyright 2025 Melani Rossi Rodrigues, Angel Cortes Gildo, Maria Alexandra Lois Peralejo, and Colby Snook
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import torch

# File to store all project hyperparameters and settings

# Environment and Paths 
ENV_ID = "PongNoFrameskip-v4"
LOG_DIR = "./logs/"
MODEL_DIR = "./models/"
BEST_MODEL_PATH = f"{MODEL_DIR}/dqn_pong_best_model.zip"

# 2. Device 
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

#  3. Training Hyperparameters
TOTAL_TIMESTEPS = 2_000_000
LEARNING_RATE = 1e-4
BUFFER_SIZE = 1_000_000
LEARNING_STARTS = 100_000
BATCH_SIZE = 32
GAMMA = 0.99
TRAIN_FREQ = 4
GRADIENT_STEPS = 1
TARGET_UPDATE_INTERVAL = 1000

# 4. Evaluation 
EVAL_FREQ = 10_000
N_EVAL_EPISODES = 10
