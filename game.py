import pygame
import os
pygame.font.init()

FONT_LARGE = pygame.font.SysFont("comicsansms", 40)
FONT_MID = pygame.font.SysFont("comicsansms", 15)
FONT_SMALL = pygame.font.SysFont("comicsansms", 10)



class Game:

    def __init__(self, env):
        self.counter = 0
        self.imgs = {}
        self.env = env
        self.done = False
        self.MAIN_MENU_IMGS = [pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", str(i) + ".jpg"))) for i in range(26)]
        self.win = pygame.display.set_mode((320, 800))
        self.clock = pygame.time.Clock()
        self.highscore_human = 0
        self.highscore_dqn = 0


    def draw_main_menu(self):
        self.clock.tick(30)
        #self.win.fill((0,0,0))

        text = FONT_LARGE.render("Breakout DQN", True, (255, 255, 255))
        self.win.blit(text, (30, 30))

        self.win.blit(self.MAIN_MENU_IMGS[self.counter % 26], (0, 180))  # draw game image

        text = FONT_MID.render("Normal game - Press SPACE", True, (255, 255, 255))
        self.win.blit(text, (30, 600))

        text = FONT_MID.render("Train the DQN - Press T", True, (255, 255, 255))
        self.win.blit(text, (30, 650))

        text = FONT_MID.render("Test the DQN - Press P", True, (255, 255, 255))
        self.win.blit(text, (30, 700))

        text = FONT_MID.render("Quit Breakout - Press Q", True, (255, 255, 255))
        self.win.blit(text, (30, 750))

        pygame.display.update()


    def play_normal_game(self):
        done = False
        q = False
        self.env.reset()

        while not done:

            action = 0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    q = True  # end currend episode
                    self.done = True # end whole game/session

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        action = 2
                    elif event.key == pygame.K_RIGHT:
                        action = 3
                    elif event.key == pygame.K_SPACE:  # release ball
                        action = 1
                    elif event.key == pygame.K_q:
                        print("Quit game")
                        q = True # just end episode
            
            next_state, reward, done, info = self.env.step(action)
            print(next_state.shape, reward, done, info)
            img = self.env.render(mode='rgb_array')
            img = pygame.surfarray.make_surface(img)
            img = pygame.transform.rotate(img, 270)
            img = pygame.transform.scale2x(img)
            self.win.blit(img, (0, 180))
            self.clock.tick(15)
            pygame.display.update()
            if q:
                done = True