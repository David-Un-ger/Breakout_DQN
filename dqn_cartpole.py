import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import random
import gym

""" 
Cartpole:
The DQN trains and reaches a score of over 100 within the first 100 episodes and a a score of 498 (=maximum) within 200 episodes.
A grid search to find the optimal hyperparameter would be beneficial...
"""

class DQN:
    def __init__(self):

        self.scores = []
        self.env = gym.make('CartPole-v1') 
        # self.env = gym.make("Breakout-ram-v0")
        self.behavior_model = self.model()  # select actions based on behavior model
        self.target_model = self.model()  # assess value of a state based on target model
        self.target_model.set_weights(self.behavior_model.get_weights())
        self.memory = []
        self.BATCH_SIZE = 32
        self.EPISODES = 200
        self.GAMMA = 0.99
        self.epsilon = 1
        self.epsilon_stop = 0.2 
        self.epsilon_decay = (self.epsilon_stop / self.epsilon) ** (1 / (self.EPISODES-1))

    def model(self):
        input_size = self.env.observation_space.shape[0]
        
        output_size = self.env.action_space.n
        input_layer = keras.layers.Input(shape=input_size)
        x = keras.layers.Dense(8, activation="relu")(input_layer)
        #x = keras.layers.Dense(8, activation="relu")(x)
        #x = keras.layers.Dense(16, activation="relu")(x)
        output_layer = keras.layers.Dense(output_size, activation="linear")(x)  # outputs the value of taking an action 

        model = keras.Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer="adam", loss="mse", metrics=["accuracy"])
        return model


    def train(self):
        for epoch in range(self.EPISODES):
            state = self.env.reset()
            done = False
            score = 0
            ctr = 0
            while not done:

                # find action via epsilon greedy

                
                if random.random() > self.epsilon:  # perform greedy action
                    action = np.argmax(self.behavior_model.predict(state.reshape(-1, self.env.observation_space.shape[0])))
                else:  # perform random action
                    action = self.env.action_space.sample()
                
                next_state, reward, done, info = self.env.step(action)
                                
                lost = False  # issue: if we are done and won, is a different state than won and loose.
                if done and (score < 498):
                    lost = True
                elif done:
                    print(f"Won game after {epoch} episodes!!!")

                if lost:
                    reward = -10
                score += reward
                reward /= 10  # brings the negative reward to -1 and the positive one to +0.1

                # store experience in the memory
                self.memory.append([state, next_state, action, reward, lost])
                state = next_state

                # training part
                if len(self.memory) > self.BATCH_SIZE: 
                    # memory is long enough  -> start training

                    train_batch = np.array(random.sample(self.memory, self.BATCH_SIZE))
                    
                    # memory: s, s_, a, r, done
                    states = np.stack(train_batch[:, 0])
                    next_states = np.stack(train_batch[:, 1])
                    actions = train_batch[:, 2]
                    rewards = train_batch[:, 3]
                    losts = train_batch[:, 4]

                    # returns an array with action-values for the whole batch. Shape BatchSize x NumActions
                    # required, because we need a baseline for the backpropagation (y_true = y_pred, just for the selected action it will be different)
                    q_values = self.behavior_model.predict(states)  

                    next_rewards = np.max(self.target_model.predict(next_states), axis=1)
                    next_rewards[losts!=0] = 0  # if a game is done, the next reward is 0
                    # estimate value of next state

                    for i in range(self.BATCH_SIZE):
                        q_values[i, actions[i]] = rewards[i] + self.GAMMA * next_rewards[i]

                    # train model
                    self.behavior_model.fit(states, q_values, verbose=0)

                # if memory is too large, remove first entry
                if len(self.memory) > 10000:
                    self.memory.pop(0)

                ctr += 1

            
            # episode is done
            self.scores.append(score)
            print(f"Episode {epoch} finished. Epsilon: {self.epsilon:.2f} Score is: {score}")
            self.epsilon *= self.epsilon_decay
            
            
            # every 5 episodes: update target_model
            if epoch % 5 == 0:
                self.target_model.set_weights(self.behavior_model.get_weights())
                
                
dqn = DQN()
dqn.train()
