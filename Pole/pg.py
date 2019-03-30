from math import *
import random
import sys
from sklearn import GridSearchCV
import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import tensorflow
import keras
from keras import layers
from keras.models import Model
from keras import backend as K
from keras import utils as np_utils
from keras import optimizers
from keras import activations
from keras.models import load_model


# IMPORTANT CONSTANTS
dt = 500
gamma = 1.0
g = 1
Episodes = 30000

num_actions = 2

class Game(object):
    def __init__(self):
        self.acc = 0.000001
        self.gc = 0.0000015
        self.reset()

    def reset(self):
        self.pos = 0.0
        self.vel = 0.0
        self.gametime = 0

        S = np.array([self.pos, self.vel])
        return S

    def update(self, A, dt):
        R = -dt/1000
        end = False
        self.gametime += dt
        # for ascent or descent of the ball
        if A == 0:
            self.vel -= self.acc * dt
        if A == 1:
            self.vel += self.acc * dt       # Add control force

        self.vel -= self.gc*sin(self.pos) * dt  # Gravity
        self.vel -= self.vel * dt * 0.0001  # Friction
        self.pos += self.vel * dt           # Update position
        # updating the rewards based on the  result (towards the goal or against it)
        if self.pos >= pi:
            R = 10.0
            end = True
        if self.pos <= -pi:
            R = -10.0
            end = True
       # if self.gametime >= 10000:
            end = True

        if end:
            self.reset()

        S = np.array([self.pos, self.vel])

        return S, R, end

class Agent(object):
    def __init__(self):
        self.model = self.getModel()
        self.train = self.getTrain()

    def getModel(self):
        #Set up the Model
        X = layers.Input(shape=(2,))
        net = X
        # initial 32*32 ,  but trying to find the optimum params for the result and the  other models
       # parameters = {'N_neurons': [i from i in range(256)], 'N_layers':[i for i in range(9):] , get_param: [] }
        #grid_search = GridSearchCV(estimator=xgb_clf, param_grid=parameters, cv=10, n_jobs=-1) 
        # after many iterations , we get the main 
        net = layers.Dense(32)(net)
        net = layers.Activation("relu")(net)
        net = layers.Dense(32)(net)
        net = layers.Activation("relu")(net)
        net = layers.Dense(num_actions)(net)
        net = layers.Activation("softmax")(net)
        
        model = Model(inputs=X, outputs=net)

        return model

    def getTrain(self):
        prob_placeholder = self.model.output
        action_placeholder = K.placeholder(shape=(None, num_actions),
                                                  name="action_onehot")
        reward_placeholder = K.placeholder(shape=(None,),
                                                    name="discount_reward")

        action_prob = K.sum(prob_placeholder * action_placeholder, axis=1)
        log_prob = K.log(action_prob)

        loss = - log_prob * reward_placeholder
        loss = K.mean(loss)
        # 
        rms = optimizers.RMSprop()

        updates = rms.get_updates(params=self.model.trainable_weights,
                                   loss=loss)

        train = K.function(inputs=[self.model.input,
                                           action_placeholder,
                                           reward_placeholder],
                                   outputs=[],
                                   updates=updates)

        return train

    def getAction(self, S):
        S = np.expand_dims(S, axis=0)
        probs = np.squeeze(self.model.predict(S))
        sample = np.random.choice(np.arange(num_actions), p=probs)

        return sample

# Stats keeping track of agent performance
matplotlib.style.use('ggplot')
stats_scores = np.zeros(Episodes)
stats_lengths = np.zeros(Episodes)

#initialize display
pygame.init()
screen = pygame.display.set_mode((dx,dy))
clock = pygame.time.Clock()

# Initialize the game and the agents
g = Game()
agent = Agent()

for e in range(Episodes):
    s = g.reset()
    total_score = 0
    S = []
    A = []
    G = []

    for t in range(1,1000):
        dt = clock.tick(2)

        screen.fill(sky)
        pygame.draw.rect(screen, earth, (0,gLev,dx,dy-gLev), 0)
        pygame.draw.lines(screen, black, False, curve, 3)
        pygame.draw.ellipse(screen, star, (WIN, -dy/2 + gLev - 40 ,20,40), 0)
        pygame.draw.ellipse(screen, red, (LOSE, -dy/2 + gLev - 40 ,20,40), 0)

        # QUIT GAME
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()

        pygame.draw.circle(screen, green, (int(g.pos*dx/8 + dx/2), \
                                     int((cos(g.pos)-1)*dy/4 + gLev - rad)), rad, 0)
        pygame.display.update()

        a = agent.getAction(s)
        S.append(s)
        A.append(a)
        s, r, end = g.update(a, dt)
        G.append(r)
        total_score += r

        #print(S)

        if end:
            acc = 0
            for i in reversed(range(len(G))):
                acc = gamma * acc + G[i]
                G[i] = acc

            A = np_utils.to_categorical(A, num_classes=num_actions)

            agent.train([S,A,G])

            print("Game:", e, "completed in:", t, ", earning:", "%.2f"%total_score, "points.")
            stats_scores[e] = total_score
            stats_lengths[e] = t
            break

np.save('mountain-sarsa-scores', stats_scores)
np.save('mountain-sarsa-length', stats_lengths)

agent.model.save('pg-climb.h5')

window = 100
score_ave = np.convolve(stats_scores, np.ones((window,))/window, mode="valid")
t_ave = np.arange(score_ave.size)
plt.rcParams['figure.figsize'] = [15, 5]
plt.plot(t_ave, score_ave)
plt.show()

# working on changing the params of the  function.

