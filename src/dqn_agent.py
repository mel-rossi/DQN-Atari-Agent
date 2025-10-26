import tensorflow as tf 
import numpy as np 
from dqn_network import create_dqn_network
from replay_buffer import ReplayBuffer

# Agent that implements Deep Q-Learning
class DQNAgent: 
    """
    DQN Agent implementation
    """

    # Sets up network, optimizer, replay buffer, and epsilon schedule
    def __init__(
        self, 
        num_actions, 
        observation_shape(84, 84, 4), 
        gamma = 0.99,
        update_horizon = 1, 
        min_replay_history = 50000, 
        update_period = 4, 
        target_update_period = 10000, 
        epsilon_fn = None, 
        epsilon_train = 0.01, 
        epsilon_eval = 0.001, 
        epsilon_decay_period = 1000000,
        replay_capacity = 1000000, 
        batch_size = 32, 
        learning_rate = 0.00025, 
        optimizer_epsilon = 0.01 / 32
    ): 
        """
        Initialize DQN agent. 

        Args: 
            num_actions: Number of actions the agent can take
            observation_shape: Shape of observations
            gamma: Discount factor
            min_replay_history: Minimum replay buffer size before training
            update_period: Period between DQN updates
            target_update_period: Period between target network updates
            epsilon_train: Epsilon value during training
            epsilon_eval: Epsilon value during evaluation
            epsilon_decay_period: Period over which epsilon is decayed
            replay_capacity: Maximum size of replay buffer
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
        """

        self.num_actions = num_actions # number of possible actions
        self.observation_shape = observation_shape # input state shape
        self.gamma = gamma # discount factor for future rewards
        self.update_horizon = update_horizon # how many steps into the future to consider
        self.min_replay_history = min_replay_history # min buffer size before training
        self.update_period = update_period # how often to update online newtork
        self.target_update_period = target_update_period # how often to sync target  network
        self.epsilon_train = epsilon_train # epsilon floor during training
        self.epsilon_eval = epsilon_eval # epsilon used during evalution
        self.epsilon_decay_period = epsilon_decay_period # steps over which epsilon decays
        self.batch_size = batch_size # training batch size

        # Training step counter 
        self.training_steps = 0 # counts environment steps
        self.eval_mode = False # flag for evaluation vs training

        # Create networks 
        self.online_network = create_dqn_network(num_actions) # main Q-network
        self.target_network = create_dqn_network(num_actions) # target Q-network

        # Initialize target network with online network weights 
        self.sync_target_network()

        # Optimizer - RMSprop like original DQN
        self.optimizer = tf.keras.optimizers.RMSprop(
            learning_rate = learning_rate,
            rho = 0.95, 
            momentum = 0.0, 
            epsilon = optimizer_epsilon, 
            centered = True
        )

        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity = replay_capacity)

        # For epsilon decay (schedule) 
        self._epsilon_schedule = self._create_epsilon_schedule()

        print(f"DQN Agent initialized with {num_actions} actions")
        print(f"Using TensorFlow {tf.__version__}")
        print(f"GPU Available: {len(tf.config.list_physical_devices('GPU')) > 0}")

    # Defines exploration decay
    def _create_epsilon_schedule(self): 
        """ Create epsilon decay schedule. """

        def epsilon_schedule(step): 
            if self.eval_mode: 
                return self.epsilon_eval 

            # Linear decay from 1.0 to epsilon_train over epsilon_decay_period 
            if step < self.min_replay_history: 
                return 1.0

            elif step >= self.min_replay_history + self.epsilon_decay_period: 
                return self.epsilon_train

            else: 
                steps_left = self_epsilon_decay_period + self.min_replay_history - step
                bonus = (1.0 - self.epsilon_train) * steps_left / self.epsilon_decay_period
                return self.epsilon_train + bonus 

        return epsilon_schedule

    # Keeps target network aligned
    def sync_target_network(self): 
        """ Sync target network weights with online network. """

        self.target_network.set_weights(self.online_network.get_weights())

    # Implements epsilon-greedy
    def select_action(self, state):
        """
        Select action using epsilon-greedy policy

        Args: 
            state: Current state (84, 84, 4)

        Returns: 
            Selected action
        """

        epsilon = self._epsilon_schedule(self.training_steps) # get current epsilon

        if np.random.random() < epsilon: # probability epsilon -> random action 
            return np.random.randit(0, self.num_actions)

        else: # forward pass through online network
            # Add batch dimension and get Q-values
            state_batch = np.expand_dims(state, axis = 0).astype(np.float34) / 255.0
            q_values = self.online_network(state_batch)
            
            return tf.argmax(q_values[0]).numpy()

    # Transition storage
    def store_transition(self, state, action, reward, next_state, terminal):
        """ Store transition in replay buffer. """

        self.replay_buffer.add(state, action, reward, next_state, terminal)

    # Coordinates replay sampling, training, and target updates
    def train_step(self):
        """
        Perform one training step. 

        Returns: 
            Loss value if training occured, None otherwise
        """

        # Check replay buffer readiness
        if not self.replay_buffer.is_ready(self.batch_size): 
            return None

        # Only train every update_period steps 
        if self.training_steps % self.update_period != 0: 
            return None

        # Sample batch from replay buffer
        states, actions, rewards, next_states, terminals = self.replay_buffer.sample(
            self.batch_size
        )

        # Normalize states to [0, 1]
        states = states / 255.0 
        next_states = next_states / 255.0

        # Compute loss and update network (train batch) 
        loss = self._train_batch(states, actions, rewards, next_states, terminals)

        # Update target network periodically 
        if self.training_steps % self.target_update_period == 0:
            self._sync_target_network()
            print(f"Target network updated at step {self.training_steps}")
        
        return loss

    @tf.function
    # computes TD targets, loss and applies gradients
    def _train_batch(self, states, actions, rewards, next_states, terminals): 
        """
        Train on a batch of transitions.
        
        Args:
            states: Batch of states
            actions: Batch of actions
            rewards: Batch of rewards
            next_states: Batch of next states
            terminals: Batch of terminal flags
            
        Returns:
            Loss value
        """

        # Compute target Q-values using target network 
        next_q_values = self.target_network(next_states)
        next_q_values_max = tf.reduce_max(next_q_values, axis=1)

        # Terminal states have 0 future value 
        next_q_values_max = next_q_values_max * (1.0 - terminals)

        # Compute target: r + gamma * max_a' Q(s', a')
        targets = rewards + self.gamma * next_q_values_max

        # Train online network 
        with tf.GradientTape() as tape: 
            # Get current Q-values
            q_values = self.online_network(states)

            # Select Q-values for actions taken 
            one_hot_actions = tf.one_hot(actions, self.num_actions)
            q_values_selected = tf.reduce_sum(q_values * one_hot_actions, axis=1)

            # Compute Huber loss 
            loss = tf.keras.losses.Huber()(targets, q_values_selected)

        # Compute gradients and update weights (backpropagation)
        gradients = tape.gradient(loss, self.online_network.trainable_variables)
        self.optimizer.apply_gradients(
            zip(gradients, self.online_network.trainable_variables)
        )

        return loss

    # Episode lifecycle hooks (integrate with environment)

    def begin_episode(self): 
        """ Called at the beginning of each episode. """

        pass # placeholder for episode start logic

    def step(self, reward, obsrvation): 
        """
        Called for each step in the environment. 
        
        Args:
            reward: Reward received
            observation: Current observation
            
        Returns:
            Selected action
        """

        self.training_steps += 1 # increment step counter 
        return self.select_action(observation) # choose action

    def end_episode(self, reward): 
        """ Called at the end of each epsiode. """

        pass # placeholder for episode end logic

    # Evaluation mode
    def set_eval_mode(self, eval_mode): 
        """ Set evaluation mode. """

        self.eval_mode = eval_mode

    # Handle checkpointing

    def save(self, checkpoint_dir, iteration):
        """ Save model checkpoint. """

        self.online_network.save_weights(
            f'{checkpoint_dir}/online_network_{iteration}.h5'
        )
        self.target_network.save_weights(
            f'{checkpoint_dir}/target_network_{iteration}.h5'
        )
        print(f"Model saved at iteration {iteration}")

    def load(self, checkpoint_dir, iteration):
        """ Load model checkpoint. """

        self.online_network.load_weights(
            f'{checkpoint_dir}/online_network_{iteration}.h5'
        )
        self.target_network.load_weights(
            f'{checkpoint_dir}/target_network_{iteration}.h5'
        )
        print(f"Model loaded from iteration {iteration}")


