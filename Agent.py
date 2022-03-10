from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import Adam
from collections import deque
import numpy as np
import random
from time import time

class Agent:
    def __init__(self, num_actions, input_shape, model_name="model", model_path=None):
        ### model settings
        self.model_name = model_name
        self.num_actions = num_actions
        self.input_shape = input_shape

        # main model
        self.model = self.create_model(model_path)

        # secondary model used to predict future Q values
        self.prediction_model = self.create_model()
        self.prediction_model.set_weights(self.model.get_weights())

        ### training settings
        self.episodes = 15000
        self.batch_size = 64

        # secondary model update settings
        self.prediction_model_update_period = 5
        self.update_counter = 0

        # replay memory settings
        self.replay_mem_size = 50000
        self.min_replay_len = 1000
        self.replay_memory = deque(maxlen=self.replay_mem_size)

        # exploration settings
        self.epsilon = 1
        self.epsilon_decay = 0.99975
        self.min_epsilon = 0.01

        # anticipated future reward discount
        self.discount = 0.99

        # render settings
        self.render = True
        self.render_period = 100

        # success reward margin for stopping training
        self.success_margin = 200

        ### stats & saving settigns
        self.checkpoint_period = 50 # 50
        self.rolling_avg_min = 25 # 25
        self.autosave_period = 200 # 200
        self.save_location = f"./training_models/{self.model_name}/models/"

        # save model if best, avg, or worst model in a aggregation outpreform these thresholds
        self.save_thresholds = {"best": 200, "avg": 150, "worst": 50}

    def create_model(self, model_path=None):
        if model_path:
            return load_model(model_path)

        model = Sequential()
        model.add(Dense(512, input_shape=self.input_shape))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))

        model.add(Dense(256))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))

        model.add(Dense(self.num_actions))
        model.add(Activation('linear'))

        model.compile(loss='mse', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
        return model

    def get_action(self, state):
        # get index of highest q value
        return np.argmax(self.model.predict(state.reshape(-1,state.shape[0])))

    def train(self, env_info):
        # env info: (state, action, new_state, reward, done)

        # add env info to replay memory
        self.replay_memory.append(env_info)

        # don't start training until replay memory has some minimum training data
        if len(self.replay_memory) < self.min_replay_len:
            return

        # build batch from replay_mem
        batch = random.sample(self.replay_memory, self.batch_size)

        # get get q vals for each state in batch
        states = np.array([elem[0] for elem in batch])
        q_vals = self.model.predict(states)

        # use prediction model to predict next state's q values
        new_states = np.array([elem[2] for elem in batch])
        future_q_vals = self.prediction_model.predict(new_states)

        #populate X and y with state (input (X), & q vals (output (y))
        #must alter q vals in accordance with Q learning algorith
        #network will train to fit to qvals
        #this will fit the network towards states with better rewards
        #   (taking into account future rewards while doing so)

        # array of states
        X = []

        # array of target q vals for each state
        y = []

        # iterate over each state
        # use q learning formula to get target q vals for each state
        for i, (state, action, new_state, reward, done) in enumerate(batch):
            if done:
                # if game over, q value is final reward
                q_vals[i][action] = reward
            else:
                # get highest Q value for next state
                optimal_future_q = np.max(future_q_vals[i])

                # update Q values
                q_vals[i][action] = reward + self.discount * optimal_future_q


            # append state and updated (target) q values
            X.append(state)
            y.append(q_vals[i])

        # fit each state to the calculated target Q value
        self.model.fit(np.array(X), np.array(y), batch_size=self.batch_size, shuffle=False, verbose=0)


        # update prediction model (if appropriate) at the end of each game
        done = env_info[4]
        if done and self.update_counter > self.prediction_model_update_period:
            self.prediction_model.set_weights(self.model.get_weights())
            self.update_counter = 0
        elif done:
            self.update_counter += 1

    def save_with_stats(self, episode, aggregate_stats):
        # save model including the aggregate stats
        max_reward, average_reward, min_reward = aggregate_stats
        save_name = f"{self.model_name}_{episode}episode_{max_reward}max_"
        save_name += f"{min_reward}min_{avg_reward}avg_{int(time())}"
        save_location = f"{self.save_location}{save_name}.model"
        self.model.save(save_location)

    def save(self, episode):
        save_name = f"{self.model_name}_{episode}episode_{int(time())}"
        save_location = f"{self.save_location}{save_name}.model"
        self.model.save(save_location)

