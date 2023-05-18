'''
This is the module you'll submit to the autograder.

There are several function definitions, here, that raise RuntimeErrors.  You should replace
each "raise RuntimeError" line with a line that performs the function specified in the
function's docstring.
'''
import random
import numpy as np
import torch
import torch.nn as nn

class q_learner():
    def __init__(self, alpha, epsilon, gamma, nfirst, state_cardinality):
        '''
        Create a new q_learner object.
        Your q_learner object should store the provided values of alpha,
        epsilon, gamma, and nfirst.
        It should also create a Q table and an N table.
        Q[...state..., ...action...] = expected utility of state/action pair.
        N[...state..., ...action...] = # times state/action has been explored.
        Both are initialized to all zeros.
        Up to you: how will you encode the state and action in order to
        define these two lookup tables?  The state will be a list of 5 integers,
        such that 0 <= state[i] < state_cardinality[i] for 0 <= i < 5.
        The action will be either -1, 0, or 1.
        It is up to you to decide how to convert an input state and action
        into indices that you can use to access your stored Q and N tables.
        
        @params:
        alpha (scalar) - learning rate of the Q-learner
        epsilon (scalar) - probability of taking a random action
        gamma (scalar) - discount factor        
        nfirst (scalar) - exploring each state/action pair nfirst times before exploiting
        state_cardinality (list) - cardinality of each of the quantized state variables

        @return:
        None
        '''
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.nfirst = nfirst
        self.state_cardinality = state_cardinality

        # Calculate the total number of states
        total_states = 1
        for cardinality in state_cardinality:
            total_states *= cardinality

        # Initialize Q and N tables with 3 actions for each state (-1, 0, 1)
        self.Q = [[0 for _ in range(3)] for _ in range(total_states)]
        self.N = [[0 for _ in range(3)] for _ in range(total_states)]

    def state_action_to_index(self, state, action):
        '''
        Convert the input state and action into indices that can be used to access
        the stored Q and N tables.

        @params:
        state (list) - a list of integers representing the state
        action (int) - an integer representing the action (-1, 0, 1)

        @return:
        state_index (int) - the index of the state in Q and N tables
        action_index (int) - the index of the action in Q and N tables
        '''
        # Convert state to a single index using base conversion
        state_index = 0
        base = 1
        for i, cardinality in reversed(list(enumerate(self.state_cardinality))):
            state_index += state[i] * base
            base *= cardinality

        # Convert action to index
        action_index = action + 1  # Shifts the range of action from [-1, 1] to [0, 2]

        return state_index, action_index

    def report_exploration_counts(self, state):
        '''
        Check to see how many times each action has been explored in this state.
        @params:
        state (list of 5 ints): ball_x, ball_y, ball_vx, ball_vy, paddle_y.
          These are the (x,y) position of the ball, the (vx,vy) velocity of the ball,
          and the y-position of the paddle, all quantized.
          0 <= state[i] < state_cardinality[i], for all i in [0,4].

        @return:
        explored_count (array of 3 ints): 
          number of times that each action has been explored from this state.
          The mapping from actions to integers is up to you, but there must be three of them.
        '''
        explored_count = [0, 0, 0]

        # Iterate over the possible actions (-1, 0, 1)
        for action in range(-1, 2):
            # Convert the state and action into indices for the N table
            state_index, action_index = self.state_action_to_index(state, action)

            # Get the exploration count for the current action from the N table
            explored_count[action_index] = self.N[state_index][action_index]

        return explored_count

    def choose_unexplored_action(self, state):
        '''
        Choose an action that has been explored less than nfirst times.
        If many actions are underexplored, you should choose uniformly
        from among those actions; don't just choose the first one all
        the time.
        
        @params:
        state (list of 5 ints): ball_x, ball_y, ball_vx, ball_vy, paddle_y.
           These are the (x,y) position of the ball, the (vx,vy) velocity of the ball,
          and the y-position of the paddle, all quantized.
          0 <= state[i] < state_cardinality[i], for all i in [0,4].

        @return:
        action (scalar): either -1, or 0, or 1, or None
          If all actions have been explored at least n_explore times, return None.
          Otherwise, choose one uniformly at random from those w/count less than n_explore.
          When you choose an action, you should increment its count in your counter table.
        '''
        # Get the exploration counts for each action in the given state
        exploration_counts = self.report_exploration_counts(state)

        # Find the indices of underexplored actions (i.e., those with count less than nfirst)
        underexplored_actions = [i - 1 for i, count in enumerate(exploration_counts) if count < self.nfirst]

        if not underexplored_actions:
            # If all actions have been explored at least nfirst times, return None
            return None
        else:
            # Choose an underexplored action uniformly at random
            chosen_action = random.choice(underexplored_actions)

            # Increment the exploration count for the chosen action in the N table
            state_index, action_index = self.state_action_to_index(state, chosen_action)
            self.N[state_index][action_index] += 1

            return chosen_action

    def report_q(self, state):
        '''
        Report the current Q values for the given state.
        @params:
        state (list of 5 ints): ball_x, ball_y, ball_vx, ball_vy, paddle_y.
          These are the (x,y) position of the ball, the (vx,vy) velocity of the ball,
          and the y-position of the paddle, all quantized.
          0 <= state[i] < state_cardinality[i], for all i in [0,4].

        @return:
        Q (array of 3 floats): 
          reward plus expected future utility of each of the three actions. 
          The mapping from actions to integers is up to you, but there must be three of them.
        '''
        # Initialize an empty list to store the Q values for each action
        Q = [0.0, 0.0, 0.0]

        # Iterate over the possible actions (-1, 0, 1)
        for action in range(-1, 2):
            # Convert the state and action into indices for the Q table
            state_index, action_index = self.state_action_to_index(state, action)

            # Get the Q value for the current action from the Q table
            Q[action + 1] = self.Q[state_index][action_index]

        return Q

    def q_local(self, reward, newstate):
        '''
        The update to Q estimated from a single step of game play:
        reward plus gamma times the max of Q[newstate, ...].
        
        @param:
        reward (scalar float): the reward achieved from the current step of game play.
        newstate (list of 5 ints): ball_x, ball_y, ball_vx, ball_vy, paddle_y.
          These are the (x,y) position of the ball, the (vx,vy) velocity of the ball,
          and the y-position of the paddle, all quantized.
          0 <= state[i] < state_cardinality[i], for all i in [0,4].
        
        @return:
        Q_local (scalar float): the local value of Q
        '''
        # Get the Q values for the new state
        newstate_q_values = self.report_q(newstate)

        # Find the maximum Q value for the new state
        max_newstate_q_value = max(newstate_q_values)

        # Calculate the local Q value
        Q_local = reward + self.gamma * max_newstate_q_value

        return Q_local
    def learn(self, state, action, reward, newstate):
        '''
        Update the internal Q-table on the basis of an observed
        state, action, reward, newstate sequence.
        
        @params:
        state: a list of 5 numbers: ball_x, ball_y, ball_vx, ball_vy, paddle_y.
          These are the (x,y) position of the ball, the (vx,vy) velocity of the ball,
          and the y-position of the paddle.
        action: an integer, one of -1, 0, or +1
        reward: a reward; positive for hitting the ball, negative for losing a game
        newstate: a list of 5 numbers, in the same format as state
        
        @return:
        None
        '''
        # Convert the state and action into indices for the Q table
        state_index, action_index = self.state_action_to_index(state, action)

        # Get the current Q value for the state and action
        current_q_value = self.Q[state_index][action_index]

        # Calculate the local Q value
        local_q_value = self.q_local(reward, newstate)

        # Update the Q table using the formula
        self.Q[state_index][action_index] += self.alpha * (local_q_value - current_q_value)
    
    def save(self, filename):
        '''
        Save your Q and N tables to a file.
        This can save in any format you like, as long as your "load" 
        function uses the same file format.  We recommend numpy.savez,
        but you can use something else if you prefer.
        
        @params:
        filename (str) - filename to which it should be saved
        @return:
        None
        '''
        np.savez(filename, Q=self.Q, N=self.N)
        
    def load(self, filename):
        '''
        Load the Q and N tables from a file.
        This should load from whatever file format your save function
        used.  We recommend numpy.load, but you can use something
        else if you prefer.
        
        @params:
        filename (str) - filename from which it should be loaded
        @return:
        None
        '''
        data = np.load(filename)
        self.Q = data['Q']
        self.N = data['N']
        
    def exploit(self, state):
        '''
        Return the action that has the highest Q-value for the current state, and its Q-value.
        @params:
        state (list of 5 ints): ball_x, ball_y, ball_vx, ball_vy, paddle_y.
          These are the (x,y) position of the ball, the (vx,vy) velocity of the ball,
          and the y-position of the paddle, all quantized.
          0 <= state[i] < state_cardinality[i], for all i in [0,4].

        @return:
        action (scalar int): either -1, or 0, or 1.
          The action that has the highest Q-value.  Ties can be broken any way you want.
        Q (scalar float): 
          The Q-value of the selected action
        '''
        state_index, _ = self.state_action_to_index(state, 0)  # Get the state index (action is not used here)
        q_values = self.Q[state_index]
        max_q = np.max(q_values)
        action_index = np.argmax(q_values)
        action = action_index - 1  # Convert index to action (shifts the range from [0, 2] to [-1, 1])

        return action, max_q
    
    def act(self, state):
        '''
        Decide what action to take in the current state.
        If any action has been taken less than nfirst times, then choose one of those
        actions, uniformly at random.
        Otherwise, with probability epsilon, choose an action uniformly at random.
        Otherwise, choose the action with the best Q(state,action).
       Finally, update N(state,action) according to chosen action.
        
        @params: 
        state: a list of 5 integers: ball_x, ball_y, ball_vx, ball_vy, paddle_y.
          These are the (x,y) position of the ball, the (vx,vy) velocity of the ball,
          and the y-position of the paddle, all quantized.
          0 <= state[i] < state_cardinality[i], for all i in [0,4].
       
        @return:
        -1 if the paddle should move upward
        0 if the paddle should be stationary
        1 if the paddle should move downward
        '''
        exploration_counts = self.report_exploration_counts(state)
        underexplored_actions = [i - 1 for i, count in enumerate(exploration_counts) if count < self.nfirst]
        
        if underexplored_actions:  # If there are underexplored actions
            action = np.random.choice(underexplored_actions)
        else:
            if np.random.random() < self.epsilon:  # With probability epsilon, choose an action uniformly at random
                action = np.random.choice([-1, 0, 1])
            else:
                action, _ = self.exploit(state)  # Choose the action with the best Q(state, action)
         # Update exploration counts
        state_index, action_index = self.state_action_to_index(state, action)
        self.N[state_index][action_index] += 1
                        
        return action

    
import copy

class deep_q():
    def __init__(self, alpha, epsilon, gamma, nfirst):
        '''
        Create a new deep_q learner.
        Your q_learner object should store the provided values of alpha,
        epsilon, gamma, and nfirst.
        It should also create a deep learning model that will accept
        (state,action) as input, and estimate Q as the output.
        
        @params:
        alpha (scalar) - learning rate of the Q-learner
        epsilon (scalar) - probability of taking a random action
        gamma (scalar) - discount factor
        nfirst (scalar) - exploring each state/action pair nfirst times before exploiting

        @return:
        None
        '''
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.nfirst = nfirst
        
        #these parameters used for epsilon_gready when acting
        self.epsilon_initial = epsilon
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.0001
        self.epsilon = self.epsilon_initial
        
        #learning decay
        self.alpha_decay_steps = 1000  # decrease the learning rate every 1000 steps
        self.alpha_decay_rate = 0.95  # decrease the learning rate by 5% every time
        
        self.actor_model = self.create_actor_model()
        self.critic_model = self.create_critic_model()

        self.optimizer_actor = torch.optim.Adam(self.actor_model.parameters(), lr=self.alpha)
        self.optimizer_critic = torch.optim.Adam(self.critic_model.parameters(), lr=self.alpha)
        #learning decay
        self.scheduler_actor = torch.optim.lr_scheduler.StepLR(self.optimizer_actor, step_size=self.alpha_decay_steps, gamma=self.alpha_decay_rate)
        self.scheduler_critic = torch.optim.lr_scheduler.StepLR(self.optimizer_critic, step_size=self.alpha_decay_steps, gamma=self.alpha_decay_rate)
        
        self.loss_fn = nn.MSELoss()
        
        self.target_critic_model = copy.deepcopy(self.critic_model)
        
        self.batch_size = 64
        self.replay_buffer = self.ReplayBuffer()
        
    class ReplayBuffer():
        def __init__(self, max_size=1e6):
            self.storage = []
            self.max_size = max_size
            self.ptr = 0

        def add(self, data):
            if len(self.storage) == self.max_size:
                self.storage[int(self.ptr)] = data
                self.ptr = (self.ptr + 1) % self.max_size
            else:
                self.storage.append(data)

        def sample(self, batch_size):
            ind = np.random.randint(0, len(self.storage), size=batch_size)
            state, action, reward, newstate = [], [], [], []

            for i in ind: 
                s, a, r, ns = self.storage[i]
                state.append(np.array(s, copy=False))
                action.append(np.array(a, copy=False))
                reward.append(np.array(r, copy=False))
                newstate.append(np.array(ns, copy=False))

            return np.array(state), np.array(action), np.array(reward).reshape(-1,1), np.array(newstate)

    def update_target_model(self):
        '''
        Perform soft update (Polyak averaging) on target model parameters. 
        The target model is a separate network which has the same architecture as 
        the original 
        '''
        tau = 0.005  # a hyperparameter for how much to update the target model at each step
        for target_param, param in zip(self.target_critic_model.parameters(), self.critic_model.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    
    def create_actor_model(self):
        '''
        Create the actor model for the actor-critic algorithm.
        This model should accept the state as input and output the probability
        distribution over the actions.

        @return:
        model (nn.Sequential): A PyTorch model representing the actor network.
        '''
        model = nn.Sequential(
            nn.Linear(5, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
            nn.Softmax(dim=1)
        )
        return model

    def create_critic_model(self):
        '''
        Create the critic model for the actor-critic algorithm.
        This model should accept the state and action as input and estimate the
        Q-value for the state-action pair.

        @return:
        model (nn.Sequential): A PyTorch model representing the critic network.
        '''
        model = nn.Sequential(
            nn.Linear(5 + 1, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        return model
    
    def update_epsilon(self):
        """
        This function updates epsilon value based on the decay rate until it reaches the minimum epsilon value.
        """
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def act(self, state):
        '''
        Decide what action to take in the current state.
        You are free to determine your own exploration/exploitation policy -- 
        you don't need to use the epsilon and nfirst provided to you.

        @params: 
        state: a list of 5 floats: ball_x, ball_y, ball_vx, ball_vy, paddle_y.

        @return:
        -1 if the paddle should move upward
        0 if the paddle should be stationary
        1 if the paddle should move downward
        '''
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        # epsilon-greedy exploration strategy
        if np.random.rand() <= self.epsilon:
            action = np.random.choice([-1, 0, 1])
        else:
            action_probs = self.actor_model(state_tensor)
            action = torch.argmax(action_probs).item() - 1  # to adjust actions back to -1, 0, 1

        # decay epsilon
        self.update_epsilon()

        return action

    def _learn_from_replay_buffer(self):
        '''
        Perform one iteration of training on a deep-Q model using experiences from the replay buffer.

        @return:
        None
        '''
        state, action, reward, newstate = self.replay_buffer.sample(self.batch_size)

        state_tensor = torch.tensor(state, dtype=torch.float32)
        newstate_tensor = torch.tensor(newstate, dtype=torch.float32)
        action_tensor = torch.tensor(action, dtype=torch.float32) + 1  # to adjust actions as 0, 1, 2
        reward_tensor = torch.tensor(reward, dtype=torch.float32)
        
        action_tensor = action_tensor.unsqueeze(-1)  # add an extra dimension
        q_values = self.critic_model(torch.cat([state_tensor, action_tensor], dim=1))

        with torch.no_grad():
            max_q_values = []

            # iterate over all possible actions
            for action in [-1, 0, 1]:
                action_tensor = torch.tensor([action]*self.batch_size).float().unsqueeze(1)
                q_value = self.target_critic_model(torch.cat((newstate_tensor, action_tensor), dim=1))
                max_q_values.append(q_value)

            max_q_values = torch.stack(max_q_values, dim=-1)
            max_q_values, _ = torch.max(max_q_values, dim=-1)
            #Q_local=R(s) + gamma*max_{a}(Q_target(s,a))
            target_q_values = reward_tensor + self.gamma * max_q_values.unsqueeze(1)

        # Update critic
        critic_loss = self.loss_fn(q_values, target_q_values)
        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()
        self.scheduler_critic.step()  # Update learning rate with scheduler

        # Update actor
        with torch.no_grad():
            q_values = []
            for action in [-1, 0, 1]:
                action_tensor = torch.tensor([action]*self.batch_size).unsqueeze(1)
                q_value = self.critic_model(torch.cat((state_tensor, action_tensor), dim=1))
                q_values.append(q_value.squeeze(-1))

            q_values = torch.stack(q_values, dim=-1)

        action_probs = self.actor_model(state_tensor)
        actor_loss = -(action_probs * q_values).sum(dim=-1).mean()

        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()
        self.scheduler_actor.step()  # Update learning rate with scheduler

        # update the target model
        self.update_target_model()
    
    def learn(self, state, action, reward, newstate):
        '''
        Perform one iteration of training on a deep-Q model.

        @params:
        state: a list of 5 floats: ball_x, ball_y, ball_vx, ball_vy, paddle_y
        action: an integer, one of -1, 0, or +1
        reward: a reward; positive for hitting the ball, negative for losing a game
        newstate: a list of 5 floats, in the same format as state

        @return:
        None
        '''
        # Add the current experience to the replay buffer
        self.replay_buffer.add((state, action, reward, newstate))

        # Proceed with learning if there are enough experiences in the replay buffer
        if len(self.replay_buffer.storage) >= self.batch_size:
            self._learn_from_replay_buffer()


        
    def save(self, filename):
        '''
        Save your trained deep-Q model to a file.
        This can save in any format you like, as long as your "load" 
        function uses the same file format.
        
        @params:
        filename (str) - filename to which it should be saved
        @return:
        None
        '''
        torch.save(self.model.state_dict(), filename)
        
    def load(self, filename):
        '''
        Load your deep-Q model from a file.
        This should load from whatever file format your save function
        used.
        
        @params:
        filename (str) - filename from which it should be loaded
        @return:
        None
        '''
        self.model.load_state_dict(torch.load(filename))
        self.target_model.load_state_dict(torch.load(filename))
    def report_q(self, state):
        """
        Report the Q-values for a given state.

        @params:
        state (list of 5 floats): ball_x, ball_y, ball_vx, ball_vy, paddle_y.

        @return:
        q_values (list of floats): The Q-values for each action (-1, 0, 1) in the given state.
        """
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            actions = torch.tensor([-1.0, 0.0, 1.0]).unsqueeze(1)
            state_action_pairs = torch.cat([state_tensor.repeat((3, 1)), actions], dim=1)
            q_values = self.critic_model(state_action_pairs)
        return np.array(q_values.squeeze(0).tolist())


