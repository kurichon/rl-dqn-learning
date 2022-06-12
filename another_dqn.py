import sys
import gym
import pylab
import random
import numpy as np
import matplotlib.pyplot as plot
from collections import deque
from keras.models import Model
from keras.layers import Dense, Input
from keras.optimizers import Adam
from keras.models import Sequential
import tensorflow as tf

EPISODES = 300



class SumTree(object):
	
    def __init__(self, capacity):
         # Number of leaf nodes (final nodes) that contains experiences
         self.capacity = capacity#
         self.tree = np.zeros(2*capacity - 1)
         self.data = np.zeros(capacity, dtype=object)
         self.data_pointer = 0
         self.n_entries = 0
    def add(self, priority, data):
         #Look at what index we want to put the experience
         tree_index = self.data_pointer + self.capacity - 1
         self.data[self.data_pointer] = data # Update data frame
         self.update(tree_index, priority) # Update the leaf
         self.data_pointer += 1 # Add 1 to data_pointer
         if self.n_entries < self.capacity:
            self.n_entries += 1
         if self.data_pointer >= self.capacity: # If we’re above the capacity
              self.data_pointer = 0 # we go back to first index (overwrite)
			 
    def update(self, tree_index, priority):
    # Change = new priority score - former priority score
         change = priority - self.tree[tree_index]
         self.tree[tree_index] = priority
         while tree_index != 0:
		     # propagate changes through the tree
             tree_index = (tree_index - 1) // 2
             self.tree[tree_index] += change
    
    
    def get_leaf(self, v):
         parent_index = 0
         while True:
             left_child_index = 2*parent_index + 1
             right_child_index = left_child_index + 1
             # If we reach bottom, end the search
             if left_child_index >= len(self.tree):
                 leaf_index = parent_index
                 break
             else:
	       # downward search, always search for a higher priority node
                 if v <= self.tree[left_child_index]:
                     parent_index = left_child_index
                 else:
                     v -= self.tree[left_child_index]
                     parent_index = right_child_index
         data_index = leaf_index - self.capacity + 1
         return leaf_index, self.tree[leaf_index], self.data[data_index]
    @property
    def total_priority(self):
         return self.tree[0] #returns to root node
    def memory_length(self):
         return self.n_entries

class Memory(object):
    # stored as ( state, action, reward, next_state ) in SumTree
    PER_e = 0.01 #hyper-parameter
    PER_a = 0.4 #hyper-parameter
    PER_b = 0.0 #hyper-parameter
    PER_b_increment_per_sampling = 0.0001 #importance sampling
    absolute_error_upper = 1.# clipped abs error
    is_weight = [1.] * 64 #initial
    def __init__(self, capacity):
        self.tree = SumTree(capacity)# Making the tree
    def store(self, experience):
		#increment size
        # Find the max priority
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])
        if max_priority == 0:
           max_priority = self.absolute_error_upper
        self.tree.add(max_priority, experience)
           
    def sample(self, n):
        minibatch = []
        b_idx = []
        priorities = []
        #b_idx = np.empty((n,), dtype=np.int32)
        #priority_segment = self.tree.total_priority / n # priority segment
        priority_segment = self.tree.total_priority / n
        
        self.PER_b = np.min([self.absolute_error_upper,self.PER_b + self.PER_b_increment_per_sampling])
        
        for i in range(n):
            # A value is uniformly sample from each range
            a, b = priority_segment*i, priority_segment*(i + 1)
            value = np.random.uniform(a, b)# Experience that correspond to each value is retrieved
           # index, priority, data = self.tree.get_leaf(value)
           # b_idx[i]= index
            (index,priority,data) = self.tree.get_leaf(value)
            priorities.append(priority)
            b_idx.append(index)
            minibatch.append([data[0],data[1],data[2],data[3],data[4]])
        
        sampling_probabilities = priorities / self.tree.total_priority
        is_weight = np.power(self.tree.memory_length() * sampling_probabilities, -self.PER_b)
        is_weight /= is_weight.max()
        #print (sampling_probabilities)
        
        return b_idx, minibatch, is_weight
            
    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.PER_e # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.absolute_error_upper)
        ps = np.power(clipped_errors, self.PER_a)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)		
		
# DQN Agent for the Cartpole
# it uses Neural Network to approximate q function
# and replay memory & target q network
class DQNAgent:
    def __init__(self, state_size, action_size,per_state,ddqn_state,duel_state):
        # if you want to see Cartpole learning, then change to True
        self.render = False
        self.load_model = False
        self.mse = tf.keras.losses.MeanSquaredError()
        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size

		#enable PER
        self.PER_enable = per_state
		#enable Dueling
        self.duel_enable = duel_state
		#enable DDQN
        self.ddqn_enable = ddqn_state

        # These are hyper parameters for the DQN
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.batch_size = 64
        self.train_start = 1000
        # create replay memory
        if self.PER_enable == True:
              self.memory = Memory(2000)
              self.is_weight = self.memory.is_weight
        else:
              self.memory = deque(maxlen=2000)
        # create main model and target model
        self.model = self.build_model()
        self.target_model = self.build_model()

        # initialize target model
        self.update_target_model()

        if self.load_model:
            self.model.load_weights("./save_model/cartpole_dqn.h5")
            
    # approximate Q function using Neural Network
    # state is input and Q Value of each action is output of network
    def build_model(self):
        if self.duel_enable == False:
            model = Sequential()
            model.add(Dense(24, input_dim=self.state_size, activation='relu',
                        kernel_initializer='he_uniform'))
            model.add(Dense(24, activation='relu',
                        kernel_initializer='he_uniform'))
            model.add(Dense(self.action_size, activation='linear',
                        kernel_initializer='he_uniform'))
        else: #DUEL DQN Network
             input_value = Input(self.state_size)
             input_layer = Dense(24, input_shape= (self.state_size,), activation="relu")(input_value)
             hidden_layer = Dense(24, activation="relu")(input_layer)
             output_layer = Dense(2, activation="linear")(hidden_layer)
             state_value = Dense(1)(output_layer)
             action_advantage = Dense(self.action_size)(output_layer)	
             q_output = (state_value + (action_advantage - tf.math.reduce_mean(action_advantage, axis=1, keepdims=True)))
             model = Model(inputs = input_value, outputs = q_output)
             print("This is DUEL DQN")
        model.summary()
        
        if self.PER_enable == True:
			#PER-DQN
              model.compile(loss='mse', loss_weights=self.is_weight,optimizer=Adam(lr=self.learning_rate))
        else:	#DQN
              model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model
         
            
   
    def experience_replay(self):
        if self.memory.tree.memory_length() < self.train_start:
            return
		#’’’ Training on Mini-Batch with Prioritized Experience Replay  ’’’
		# create a minibatch through prioritized sampling
		
        tree_idx, mini_batch, self.is_weight = self.memory.sample(self.batch_size)
        current_state = np.zeros((self.batch_size, self.state_size))
        next_state = np.zeros((self.batch_size, self.state_size))
        qValues = np.zeros((self.batch_size, self.action_size)) 
        
        #action, reward, done = [], [], []
        action = np.zeros(self.batch_size, dtype=int)
        reward = np.zeros(self.batch_size)
        done = np.zeros(self.batch_size, dtype=bool)
        for i in range(self.batch_size):
            current_state[i] = mini_batch[i][0] # current_state
            action[i] = mini_batch[i][1]
            reward[i] = mini_batch[i][2]
            next_state[i] = mini_batch[i][3]# next_state
            done[i] = mini_batch[i][4]
            #current_state = update input , next_state = update_target
        qValues = self.model.predict(current_state) #target
        max_qvalue_ns = self.target_model.predict(next_state) #target_val
        
        if self.ddqn_enable == True:
            target_next = self.model.predict(next_state)
        
        for i in range(self.batch_size):
            if done[i]:
                 qValues[i][action[i]] = reward[i]
            else:
                if self.ddqn_enable == True: 
                # the key point of Double DQN
                # selection of action is from model
                # update is from target model  
                    a = np.argmax(target_next[i])
                    qValues[i][action[i]] = reward[i] + self.discount_factor * (max_qvalue_ns[i][a])
                else: #DQN
                    qValues[i][action[i]] = reward[i] + self.discount_factor*np.amax(max_qvalue_ns[i]) #max q value
        # update  priority in the replay memory
        target_old = np.array(self.model.predict(current_state))
        target = qValues
        indices = np.arange(self.batch_size, dtype=np.int32)
        absolute_errors = np.abs(target_old[indices,np.array(action)]- target[indices, np.array(action)])
        self.memory.batch_update(tree_idx, absolute_errors)
        
        # train the model
        self.model.fit(current_state, qValues,batch_size = self.batch_size,epochs=1, verbose=0)
        #self.update_epsilon()
    
    
   
    # after some time interval update the target model to be same with model
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # get action from model using epsilon-greedy policy
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(state)
            return np.argmax(q_value[0])

    # save sample <s,a,r,s'> to the replay memory
    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
    def add_experience(self, state, action, reward, next_state, done):
        experience = [state, action, reward, next_state, done]
        self.memory.store(experience)		
        
        # update epsilon with each training step
        if self.epsilon > self.epsilon_min:
           self.epsilon *= self.epsilon_decay
		

    # pick samples randomly from replay memory (with batch_size)
    def train_model(self):
        if len(self.memory) < self.train_start:
            return
        batch_size = min(self.batch_size, len(self.memory))
        mini_batch = random.sample(self.memory, batch_size)

        update_input = np.zeros((batch_size, self.state_size))
        update_target = np.zeros((batch_size, self.state_size))
        action, reward, done = [], [], []

        for i in range(self.batch_size):
            update_input[i] = mini_batch[i][0]
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])
            update_target[i] = mini_batch[i][3]
            done.append(mini_batch[i][4])

        target = self.model.predict(update_input)
        target_val = self.target_model.predict(update_target)

        if self.ddqn_enable == True:
            target_next = self.model.predict(update_target)


        for i in range(self.batch_size):
            # Q Learning: get maximum Q value at s' from target model
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
				#DDQN
                if self.ddqn_enable == True: 
					# the key point of Double DQN
                # selection of action is from model
                # update is from target model  
                    a = np.argmax(target_next[i])
                    target[i][action[i]] = reward[i] + self.discount_factor * (target_val[i][a])
                    #print("this is ddqn")
                else:
				#DQN
                    target[i][action[i]] = reward[i] + self.discount_factor * (np.amax(target_val[i]))
                    #print("this is dqn")

        # and do the model fit!
        self.model.fit(update_input, target, batch_size=self.batch_size,
                       epochs=1, verbose=0)
                       
                       
def run_DQN(agent,scores,episodes,ax):
    for e in range(EPISODES):
        done = False
        score = 0
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        while not done:
            if agent.render:
                env.render()

            # get action for the current state and go one step in environment
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            # if an action make the episode end, then gives penalty of -100
            reward = reward if not done or score == 499 else -100

            if agent.PER_enable == False:
           # save the sample <s, a, r, s'> to the replay memory
                agent.append_sample(state, action, reward, next_state, done)
           # every time step do the training
                agent.train_model()
            else:
			#PER
            # save the sample <s, a, r, s'> to the replay memory
                agent.add_experience(state, action, reward, next_state, done)
            # every time step do the training
                agent.experience_replay()
            score += reward
            state = next_state

            if done:
                # every episode update the target model to be same with model
                agent.update_target_model()

                # every episode, plot the play time
                score = score if score == 500 else score + 100
                scores.append(score)
                episodes.append(e)
                plot.figure(1)
                if agent.PER_enable == False and agent.ddqn_enable == False and agent.duel_enable == False:
                    ax[0,0].plot(episodes, scores, 'b',linewidth=1.5)
                    pylab.savefig("./save_graph/subplots.png")
                    plot.figure(2)
                    pylab.plot(episodes, scores, 'b',linewidth=1.5)
                    pylab.savefig("./save_graph/overall_output.png")
                    plot.figure(3)
                    pylab.plot(episodes,scores,'b',linewidth=1.5)
                    pylab.savefig("./save_graph/cartpole_dqn.png")
                    print("episode:", e, "  score:", score, "  memory length:", len(agent.memory), "  epsilon:", agent.epsilon)
                elif agent.PER_enable == True and agent.ddqn_enable == False and agent.duel_enable == False:
                    ax[0,1].plot(episodes, scores, 'g',linewidth=1.5)
                    pylab.savefig("./save_graph/subplots.png")
                    plot.figure(2)
                    pylab.plot(episodes, scores, 'g',linewidth=1.5)
                    pylab.savefig("./save_graph/overall_output.png")
                    plot.figure(3)
                    pylab.plot(episodes, scores, 'g',linewidth=1.5)
                    pylab.savefig("./save_graph/cartpole_per_dqn.png")
                    print("episode:", e, "  score:", score, "  memory length:",
                    agent.memory.tree.memory_length(), " loss:", np.mean(agent.is_weight), "  epsilon:", agent.epsilon, "   PER_b:", agent.memory.PER_b)
                elif agent.PER_enable == False and agent.ddqn_enable == True and agent.duel_enable == False:
                    ax[1,0].plot(episodes, scores, 'r',linewidth=1.5)
                    pylab.savefig("./save_graph/subplots.png")
                    plot.figure(2)
                    pylab.plot(episodes, scores, 'r',linewidth=1.5)
                    pylab.savefig("./save_graph/overall_output.png")
                    plot.figure(3)
                    pylab.plot(episodes, scores, 'r',linewidth=1.5)
                    pylab.savefig("./save_graph/cartpole_ddqn.png")
                    print("episode:", e, "  score:", score, "  memory length:", len(agent.memory), "  epsilon:", agent.epsilon)
                elif agent.PER_enable == True and agent.ddqn_enable == True and agent.duel_enable == False:
                    ax[1,1].plot(episodes, scores, 'c',linewidth=1.5)
                    pylab.savefig("./save_graph/subplots.png")
                    plot.figure(2)
                    pylab.plot(episodes, scores, 'c',linewidth=1.5)
                    pylab.savefig("./save_graph/overall_output.png")
                    plot.figure(3)
                    pylab.plot(episodes, scores, 'c',linewidth=1.5)
                    pylab.savefig("./save_graph/cartpole_per_ddqn.png")
                    print("episode:", e, "  score:", score, "  memory length:",
                    agent.memory.tree.memory_length(), " loss:", np.mean(agent.is_weight), "  epsilon:", agent.epsilon, "   PER_b:", agent.memory.PER_b)
                elif agent.PER_enable == False and agent.ddqn_enable == False and agent.duel_enable == True:
                    ax[2,0].plot(episodes, scores, 'm',label="Duel_DQN",linewidth=1.5)
                    pylab.savefig("./save_graph/subplots.png")
                    plot.figure(2)
                    pylab.plot(episodes, scores, 'm',linewidth=1.5)
                    pylab.savefig("./save_graph/overall_output.png")
                    plot.figure(3)
                    pylab.plot(episodes, scores, 'm',linewidth=1.5)
                    pylab.savefig("./save_graph/cartpole_duel_dqn.png")
                    print("episode:", e, "  score:", score, "  memory length:", len(agent.memory), "  epsilon:", agent.epsilon)
                elif agent.PER_enable == True and agent.ddqn_enable == False and agent.duel_enable == True:
                    ax[2,1].plot(episodes, scores, 'y',linewidth=1.5)
                    pylab.savefig("./save_graph/subplots.png")
                    plot.figure(2)
                    pylab.plot(episodes, scores, 'y',linewidth=1.5)
                    pylab.savefig("./save_graph/overall_output.png")
                    plot.figure(3)
                    pylab.plot(episodes, scores, 'y',linewidth=1.5)
                    pylab.savefig("./save_graph/cartpole_per_duel_dqn.png")
                    print("episode:", e, "  score:", score, "  memory length:",
                    agent.memory.tree.memory_length(), " loss:", np.mean(agent.is_weight), "  epsilon:", agent.epsilon, "   PER_b:", agent.memory.PER_b)
                # if the mean of scores of last 10 episode is bigger than 490
                # stop training
                
                if np.mean(scores[-min(10, len(scores)):]) > 490:
                    sys.exit()

        # save the model
        if e % 50 == 0:
            if agent.PER_enable == False and agent.ddqn_enable == False and agent.duel_enable == False:
                agent.model.save_weights("./save_model/cartpole_dqn.h5")
            elif agent.PER_enable == True and agent.ddqn_enable == False and agent.duel_enable == False:
                agent.model.save_weights("./save_model/cartpole_per_dqn.h5")
            elif agent.PER_enable == False and agent.ddqn_enable == True and agent.duel_enable == False:
                agent.model.save_weights("./save_model/cartpole_ddqn.h5")
            elif agent.PER_enable == True and agent.ddqn_enable == True and agent.duel_enable == False:
                agent.model.save_weights("./save_model/cartpole_per_ddqn.h5")
            elif agent.PER_enable == False and agent.ddqn_enable == False and agent.duel_enable == True:
                agent.model.save_weights("./save_model/cartpole_duel_dqn.h5")
            elif agent.PER_enable == True and agent.ddqn_enable == False and agent.duel_enable == True:
                agent.model.save_weights("./save_model/cartpole_per_duel_dqn.h5")



if __name__ == "__main__":
    # In case of CartPole-v1, maximum length of episode is 500
    env = gym.make('CartPole-v1')
    # get size of state and action from environment
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    
    #initialize graphs
    
    fig1, ax = plot.subplots(3, 2,figsize=(15,15))
    ax[0,0].set_title('DQN')
    ax[0,1].set_title('PER_DQN')
    ax[1,0].set_title('DDQN')
    ax[1,1].set_title('PER_DDQN')
    ax[2,0].set_title('Duel_DQN')
    ax[2,1].set_title('PER_Duel_DQN')
    
    plot.figure(2,figsize=(10,10))
    plot.figure(3)
    
    #per_state =  #implement Prioritized Experience Replay Memory
    #ddqn_state = #implement Dual DQN
    #duel_state = #implement Duel DQN
    
    #000 - dqn
    #010 - ddqn
    #001 - duel
    #100 - per dqn
    #110 - per_ddqn
    #111 - per_duel_dqn
    
   # agent = DQNAgent(state_size, action_size,per_state,ddqn_state,duel_state)
   
    #DQN
    print("This is DQN")
    agent = DQNAgent(state_size, action_size,False,False,False) #set parameters
    scores, episodes = [], []
    #run_DQN(agent,scores,episodes,ax)
    
    plot.figure(3)
    plot.clf()
    
    #PER_DQN
    print("This is PER_DQN")
    agent = DQNAgent(state_size, action_size,True,False,False) #set parameters
    scores, episodes = [], []
    #run_DQN(agent,scores,episodes,ax)
    
    
    plot.figure(3)
    plot.clf()
    
    #DDQN
    print("This is DDQN")
    agent = DQNAgent(state_size, action_size,False,True,False)
    scores, episodes = [], []
    #run_DQN(agent,scores,episodes,ax)
    
    plot.figure(3)
    plot.clf()
    
    #PER_DDQN
    print("This is PER_DDQN")
    agent = DQNAgent(state_size, action_size,True,True,False)
    scores, episodes = [], []
    run_DQN(agent,scores,episodes,ax)
    
    plot.figure(3)
    plot.clf()
    
    #Duel-DQN
    print("This is Duel_DQN")
    agent = DQNAgent(state_size, action_size,False,False,True)
    scores, episodes = [], []
    run_DQN(agent,scores,episodes, ax)
    
    plot.figure(3)
    plot.clf()
    
    #PER_Duel-DQN
    print("This is PER_Duel_DQN")
    agent = DQNAgent(state_size, action_size,True,False,True)
    scores, episodes = [], []
    run_DQN(agent,scores,episodes,ax)
    
