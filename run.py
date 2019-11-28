#!/usr/bin/env python3
#Jaime Danguillecourt
import gym
import random
from Settings import Settings

"""
Landing pad is always at coordinates (0,0).
Coordinates are the first two numbers in state vector.
Reward for moving from the top of the screen to landing pad and zero speed is about 100..140 points.
If lander moves away from landing pad it loses reward back.
Episode finishes if the lander crashes or comes to rest, receiving additional -100 or +100 points.
Each leg ground contact is +10. Firing main engine is -0.3 points each frame. Solved is 200 points.
Landing outside landing pad is possible.
Fuel is infinite, so an agent can learn to fly and then land on its first attempt.
Four discrete actions available: do nothing, fire left orientation engine, fire main engine, fire right orientation engine.
"""

def main():
    env = gym.make("LunarLander-v2")
    s = Settings(env)

    for i in range(s.episodes):
        done = False
        state = env.reset()

        while not done:
            action = random.randint(0, s.num_actions-1)

            new_state, reward, done, info = env.step(action)
            print(new_state)

            if s.render:
                env.render()

    env.close()
    return

if __name__ == "__main__":
    main()
