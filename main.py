import numpy as np
#import tensorflow
import time
import pygame
from game import Game
#from dqn import DQN

import gym

env = gym.make("Breakout-ram-v0")
#env = gym.make('BreakoutDeterministic-v4')

env.reset()

game = Game(env)
#dqn = DQN()


while not game.done:
    game.draw_main_menu()
    game.counter += 1

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            game.done = True

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                print("start a normal game")
                game.play_normal_game()

            if event.key == pygame.K_t:
                print("Training the DQN")
                dqn.train()

            if event.key == pygame.K_t:
                print("Test the DQN")
                dqn.play()

            if event.key == pygame.K_q:
                print("Quit game")
                game.done = True

pygame.quit()
game.env.close()

