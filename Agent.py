from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import Adam
from Settings import Settings
from collections import deque
import numpy as np

class Agent:
    def __init__(self, settings):
        self.s = settings
        self.replay_memory = deque(max_len=s.replay_mem_size)

        #main model that gets trained and predicts optimal action
        self.model = self.create_model()

        #Secondary model used to predict future Q values
        #makes predicting future Q vals more stable
        #more stable bcs multiple predictions from same reference point
        #model / reference point updated to match main model on chosen interval
        self.stable_pred_model.create_model()
        self.stable_pred_model.weights = self.model.get_weights()

    #NOTE: replay memory must be updated from game loop
    #agent.replay_memory.append((state, action, new_state, reward, done))

    def create_model(self, load_model=None):
        if load_model:
            #TODO
            #load the model
            return #loaded model

        model = Sequential()
        model.add(Dense(64, input_shape=self.s.observation_shape)
        model.add(Activation('relu')
        model.add(Dropout(0.8)

        model.add(Dense(16)
        model.add(Activation('relu')
        model.add(Dropout(0.2))

        model.add(Dense(self.s.num_actions))
        model.add(Activation('softmax')) #try linear later

        model.compile(loss='mse', optimizer=Adam(lr=0.001), metrics='accuracy')
        return model

    def get_action(state):
        return np.argmax(self.model.predict(state))

    def train(done, step):
        #if just started to play & replay mem not long enough
        #then don't train yet, play more
        if len(self.replay_memory) < self.s.min_replay_len:
            return

        #build batch from replay_mem
        batch = random.sample(replay_memory, self.s.batch_size)

        #populate X and y with state (input (X), & q vals (output (y))
        #must alter q vals in accordance with Q learning algorith
        #network will train to fit to qvals
        #this will fit the network towards states with better rewards
        #   (taking into account future rewards as well)
        for (state, action, new_state, reward, done) in batch:
            #get output from network given state as input
            current_q_vals = self.model.predict(state)

            #update q vals for action taken from state appropiately
            #if finished playing (win or lose), theres no future reward
            if done:
                current_q_vals[action] = reward
            else:
                #get value network outputs for action taken from that state
                current_q_val = current_q_vals[action]

                #predict future q (using other network) with new state
                future_q_vals = self.stable_pred_model.predict(new_state)
                #chose best action in new state
                optimal_future_q = np.argmax(future_q_vals)

                #Q-learning! :)
                current_state[action] = reward + self.s.discount * future_q_val



