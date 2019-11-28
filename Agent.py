from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import Adam
from Settings import Settings

class Agent:
    def __init__(self, settings):
        self.model = self.create_model()
        self.s = settings

    def create_model(self, load_model=None):
        if load_model:
            #load the model
            return

        model = Sequential()
        model.add(Dense(64, input_shape=self.s.observation_shape)
        model.add(Activation('relu')
        model.add(Dropout(0.2)

        model.add(Dense(64)
        model.add(Activation('relu')
        model.add(Dropout(0.2))

        model.add(Dense(self.s.num_actions))
        model.add(Activation('linear'))

        model.compile(loss='mse', optimizer=Adam(lr=0.001), metrics='accuracy')
        return model

