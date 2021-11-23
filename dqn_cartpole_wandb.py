import wandb
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import random
import gym
import os
from wandb.keras import WandbCallback

# use seeds for maximum reproducibility
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # warnings and errors are not printed (nvcuda.dll dlerror)
os.environ['WANDB_DIR'] = "C:\Scarlett\wandb\dqn_cartpole"

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

""" 
Cartpole:
Used a wandb sweep to find good parameters. Default parameters can win the game within 50 episodes.
"""

default_config={"env": 'CartPole-v1',  # "Breakout-ram-v0"  
                "loss_function": "mse",
                "episodes": 50,
                
                "batch_size": 16,
                "learning_rate": 0.001,
                "gamma": 0.9,
                "epsilon_start": 0.8,
                "epsilon_stop": 0.1,
                "memory_size": 5000,
                "update_target_model": 1,
                "epochs": 10,
                "dense_neurons_0": 64,
                "dense_neurons_1": 16,
                "dense_neurons_2": 0
                }

run = wandb.init(project="dqn_cartpole", 
                entity="davidu",
                config=default_config)

config = wandb.config    

class DQN:
    def __init__(self, config):
        self.config = config
        self.env = gym.make(config.env) 
        self.behavior_model = self.model()  # select actions based on behavior model
        self.target_model = self.model()  # assess value of a state based on target model
        self.target_model.set_weights(self.behavior_model.get_weights())
        self.memory = []
        self.epsilon = config.epsilon_start
        self.epsilon_decay = (config.epsilon_stop / config.epsilon_start) ** (1 / (config.episodes-1))

    def model(self):
        input_size = self.env.observation_space.shape[0]
        output_size = self.env.action_space.n
        input_layer = keras.layers.Input(shape=input_size)
        x = keras.layers.Dense(self.config.dense_neurons_0, activation="relu")(input_layer)
        if self.config.dense_neurons_1:
            x = keras.layers.Dense(self.config.dense_neurons_1, activation="relu")(x)
        if self.config.dense_neurons_2:
            x = keras.layers.Dense(self.config.dense_neurons_2, activation="relu")(x)
        output_layer = keras.layers.Dense(output_size, activation="linear")(x)  # outputs the value of taking an action 

        model = keras.Model(inputs=input_layer, outputs=output_layer)
        # lr_schedule = keras.optimizers.schedules.ExponentialDecay(self.config.learning_rate, decay_rate=0.95, decay_steps=100)
        opt = keras.optimizers.Adam(learning_rate=self.config.learning_rate)  # self.lr_schedule)
        model.compile(optimizer=opt, loss=self.config.loss_function, metrics=["accuracy"])
        return model


    def train(self):
        for episode in range(self.config.episodes):
            state = self.env.reset()
            done = False
            score = 0
            episode_memory = []
            # 1.1 run on episode using the current DQN
            while not done:
                # find action via epsilon greedy
                if random.random() > self.epsilon:  # perform greedy action
                    action = np.argmax(self.behavior_model.predict(state.reshape(-1, self.env.observation_space.shape[0])))
                else:  # perform random action
                    action = self.env.action_space.sample()
                
                next_state, reward, done, _ = self.env.step(action)
                                
                lost = False  # issue: if we are done and won, is a different state than won and loose.
                if done and (score < 498):
                    lost = True
                elif done:
                    print(f"Won game after {episode} episodes!!!")

                if lost:
                    reward = -10
                score += reward
                reward /= 10  # brings the negative reward to -1 and the positive one to +0.1

                # store experience in the memory
                episode_memory.append([state, next_state, action, reward, lost])
                state = next_state

            # 1.2 decide which memories you wand to store
            # just store the last 100 steps of the episode
            self.memory.extend(episode_memory[max(len(episode_memory)-100, 0):])
            
            # create a relatively small memory to forget some of the older experiences
            while len(self.memory) > self.config.memory_size:
                self.memory.pop(random.randint(0, int(self.config.memory_size)/2))
            

            # 1.3 episode is done
            print(f"Episode {episode} finished. Epsilon: {self.epsilon:.2f} ",
                f"Memory length {len(self.memory)} ",
                #f"Learning rate: {keras.backend.eval(self.behavior_model.optimizer.lr):.4f}",
                f"Score is: {score}")
            self.epsilon *= self.epsilon_decay

            # log to wandb
            wandb.log({
                "score": score,
                "memory": len(self.memory),
                "epsilon": self.epsilon,
                "episode": episode
            })

            # 2 train the DQN using the replay memory
            if len(self.memory) > self.config.batch_size: 
                # memory is long enough  -> start training
                # 1 convert memory list into memory array:
                memory = np.array(self.memory)

                # memory: s, s_, a, r, done
                states = np.stack(memory[:, 0])
                next_states = np.stack(memory[:, 1])
                actions = memory[:, 2]
                rewards = memory[:, 3]
                losts = memory[:, 4]

                # returns an array with action-values for the whole batch. Shape BatchSize x NumActions
                # required, because we need a baseline for the backpropagation (y_true = y_pred, just for the selected action it will be different)
                q_values = self.behavior_model.predict(states)  

                next_rewards = np.max(self.target_model.predict(next_states), axis=1)
                next_rewards[losts!=0] = 0  # if a game is done, the next reward is 0
                # estimate value of next state

                for i in range(len(self.memory)):
                    q_values[i, actions[i]] = rewards[i] + self.config.gamma * next_rewards[i]

                # train model
                self.behavior_model.fit(states, 
                                        q_values, 
                                        batch_size=self.config.batch_size, 
                                        epochs=self.config.epochs,
                                        verbose=0, 
                                        shuffle=True,
                                        callbacks=[WandbCallback()])
                
            # 2.2 update target_model every "update_target_model"th episode
            if episode % self.config.update_target_model == 0:
                self.target_model.set_weights(self.behavior_model.get_weights())
                     
dqn = DQN(config)
dqn.train()
