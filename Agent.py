from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import Adam
from Settings import Settings
from collections import deque
import numpy as np
import random

class Agent:
    def __init__(self, settings):
        self.s = settings
        self.replay_memory = deque(maxlen=self.s.replay_mem_size)
        self.model_update_counter = 1

        #main model that gets trained and predicts optimal action
        self.model = self.create_model()

        #Secondary model used to predict future Q values
        #makes predicting future Q vals more stable
        #more stable bcs multiple predictions from same reference point
        #model / reference point updated to match main model on chosen interval
        self.stable_pred_model = self.create_model()
        self.stable_pred_model.set_weights(self.model.get_weights())

    def create_model(self, load_model=None):
        if load_model:
            #TODO
            #load the model
            return #loaded model

        model = Sequential()
        model.add(Dense(64, input_shape=self.s.observation_shape))
        model.add(Activation('relu'))
        model.add(Dropout(0.8))

        model.add(Dense(16))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))

        model.add(Dense(self.s.num_actions))
        model.add(Activation('softmax')) #try linear later

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

        #build batch from replay_mem
        batch = random.sample(self.replay_memory, self.s.batch_size)

        X, y = [], []
        #populate X and y with state (input (X), & q vals (output (y))
        #must alter q vals in accordance with Q learning algorith
        #network will train to fit to qvals
        #this will fit the network towards states with better rewards
        #   (taking into account future rewards while doing so)
        for (state, action, new_state, reward, done) in batch:
            #get output from network given state as input
            current_q_vals = self.model.predict(state.reshape(-1, state.shape[0]))
            current_q_vals = current_q_vals.reshape(self.s.num_actions,)

            #update q vals for action taken from state appropiately
            #if finished playing (win or lose), theres no future reward
            if done:
                current_q_vals[action] = reward
            else:
                #predict future q (using other network) with new state
                future_q_vals = self.stable_pred_model.predict(new_state.reshape(-1, state.shape[0]))
                future_q_vals = future_q_vals.reshape(self.s.num_actions,)
                #chose best action in new state
                optimal_future_q = np.argmax(future_q_vals)

                #Q-learning! :)
                current_q_vals[action] = reward + self.s.discount * optimal_future_q

            #env[0]: current_state
            X.append(env_info[0])
            y.append(current_q_vals)

        self.model.fit(np.array(X), np.array(y), batch_size=self.s.batch_size, shuffle=False, verbose=0) #TODO callback


        #check if time to update prediction model
        #env_info[4]: done
        if env_info[4] and self.model_update_counter >= self.s.update_pred_model_period:
            self.stable_pred_model.set_weights(self.model.get_weights())
            self.model_update_counter = 1
        elif env_info[4]:
            self.model_update_counter += 1

