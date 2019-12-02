from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import Adam
from Settings import Settings
from collections import deque
import numpy as np
import random

class Agent:
    def __init__(self, num_actions, input_shape, settings, model_path=None):
        self.s = settings
        self.num_actions = num_actions
        self.input_shape = input_shape
        self.replay_memory = deque(maxlen=self.s.replay_mem_size)
        self.model_update_counter = 0

        #main model that gets trained and predicts optimal action
        self.model = self.create_model(model_path)

        #Secondary model used to predict future Q values
        #makes predicting future Q vals more stable
        #more stable bcs multiple predictions from same reference point
        #model / reference point updated to match main model on chosen interval
        self.stable_pred_model = self.create_model()
        self.stable_pred_model.set_weights(self.model.get_weights())

    def create_model(self, model_path=None):
        if model_path:
            return load_model(model_path)

        model = Sequential()
        model.add(Dense(512, input_shape=self.input_shape))
        model.add(Activation('relu'))
        #model.add(Dropout(0.2))

        model.add(Dense(256))
        model.add(Activation('relu'))
        #model.add(Dropout(0.2))

        model.add(Dense(self.num_actions))
        model.add(Activation('linear'))

        model.compile(loss='mse', optimizer=Adam(lr=0.001), metrics=['accuracy'])
        return model

    def get_action(self, state):
        return np.argmax(self.model.predict(state.reshape(-1,state.shape[0])))

    def train(self, env_info):
        #env info: (state, action, new_state, reward, done)
        #add to replay memory
        self.replay_memory.append(env_info)

        #if just started to play & replay mem not long enough
        #then don't train yet, play more
        if len(self.replay_memory) < self.s.min_replay_len:
            return

        """
        #if last x rewards  better than some margin, don't train
        #   (lets not overfit)
        if np.mean(self.replay_memory[-self.s.early_stop_count][3]) > self.s.early_stop_margin:
            return
        """

        #build batch from replay_mem
        batch = random.sample(self.replay_memory, self.s.batch_size)
        #get output from network given state as input
        states = np.array([elem[0] for elem in batch])
        current_q_vals = self.model.predict(states)
        #predict future q (using other network) with new state
        new_states = np.array([elem[2] for elem in batch])
        future_q_vals = self.stable_pred_model.predict(new_states)
        #NOTE: its better to predict on full batch of states at once
        #   predicting gets vectorized :)

        X, y = [], []
        #populate X and y with state (input (X), & q vals (output (y))
        #must alter q vals in accordance with Q learning algorith
        #network will train to fit to qvals
        #this will fit the network towards states with better rewards
        #   (taking into account future rewards while doing so)

        for i, (state, action, new_state, reward, done) in enumerate(batch):
            #update q vals for action taken from state appropiately
            #if finished playing (win or lose), theres no future reward
            if done:
                current_q_vals[i][action] = reward
            else:
                #chose best action in new state
                optimal_future_q = np.max(future_q_vals[i])

                #Q-learning! :)
                current_q_vals[i][action] = reward + self.s.discount * optimal_future_q


            X.append(state)
            y.append(current_q_vals[i])

        self.model.fit(np.array(X), np.array(y), batch_size=self.s.batch_size, shuffle=False, verbose=0)

        #check if time to update prediction model
        #env_info[4]: done
        if env_info[4] and self.model_update_counter > self.s.update_pred_model_period:
            self.stable_pred_model.set_weights(self.model.get_weights())
            self.model_update_counter = 0
        elif env_info[4]:
            self.model_update_counter += 1

