import gym
import math 
import random
import pygame
import numpy as np
from gym import utils
from gym import error, spaces
from gym.utils import seeding
from scipy.spatial.distance import euclidean

class RobotArmEnvV0(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.set_window_size([600,600])
        self.set_link_properties([100,100])
        self.set_increment_rate(0.01)
        self.target_pos = self.generate_random_pos()
        self.action = {0: "HOLD",
                       1: "INC_J1",
                       2: "DEC_J1",
                       3: "INC_J2",
                       4: "DEC_J2",
                       5: "INC_J1_J2",
                       6: "DEC_J1_J2"}
        
        self.observation_space = spaces.Box(np.finfo(np.float32).min, np.finfo(np.float32).max, shape=(4,), dtype=np.float32)
        self.action_space = spaces.Discrete(len(self.action))

        self.current_error = -math.inf
        self.seed()
        self.viewer = None

    def set_link_properties(self, links):
        self.links = links
        self.n_links = len(self.links)
        self.min_theta = math.radians(0)
        self.max_theta = math.radians(90)
        self.theta = self.generate_random_angle()
        self.max_length = sum(self.links)

    def set_increment_rate(self, rate):
        self.rate = rate

    def set_window_size(self, window_size):
        self.window_size = window_size
        self.centre_window = [window_size[0]//2, window_size[1]//2]

    def rotate_z(self, theta):
        rz = np.array([[np.cos(theta), - np.sin(theta), 0, 0],
                       [np.sin(theta), np.cos(theta), 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]])
        return rz

    def translate(self, dx, dy, dz):
        t = np.array([[1, 0, 0, dx],
                      [0, 1, 0, dy],
                      [0, 0, 1, dz],
                      [0, 0, 0, 1]])
        return t

    def forward_kinematics(self, theta):
        P = []
        P.append(np.eye(4))
        for i in range(0, self.n_links):
            R = self.rotate_z(theta[i])
            T = self.translate(self.links[i], 0, 0)
            P.append(P[-1].dot(R).dot(T))
        return P

    def inverse_theta(self, theta):
        new_theta = theta.copy()
        for i in range(theta.shape[0]):
            new_theta[i] = -1*theta[i]
        return new_theta

    def draw_arm(self, theta):
        LINK_COLOR = (255, 255, 255)
        JOINT_COLOR = (0, 0, 0)
        TIP_COLOR = (0, 0, 255)
        theta = self.inverse_theta(theta)
        P = self.forward_kinematics(theta)
        origin = np.eye(4)
        origin_to_base = self.translate(self.centre_window[0],self.centre_window[1],0)
        base = origin.dot(origin_to_base)
        F_prev = base.copy()
        for i in range(1, len(P)):
            F_next = base.dot(P[i])
            pygame.draw.line(self.screen, LINK_COLOR, (int(F_prev[0,3]), int(F_prev[1,3])), (int(F_next[0,3]), int(F_next[1,3])), 5)
            pygame.draw.circle(self.screen, JOINT_COLOR, (int(F_prev[0,3]), int(F_prev[1,3])), 10)
            F_prev = F_next.copy()
        pygame.draw.circle(self.screen, TIP_COLOR, (int(F_next[0,3]), int(F_next[1,3])), 8)

    def draw_target(self):
        TARGET_COLOR = (255,0,0)
        origin = np.eye(4)
        origin_to_base = self.translate(self.centre_window[0], self.centre_window[1], 0)
        base = origin.dot(origin_to_base)
        base_to_target = self.translate(self.target_pos[0], -self.target_pos[1], 0)
        target = base.dot(base_to_target)
        pygame.draw.circle(self.screen, TARGET_COLOR, (int(target[0,3]),int(target[1,3])), 12)
    
    def generate_random_angle(self):
        theta = np.zeros(self.n_links)
        theta[0] = random.uniform(self.min_theta, self.max_theta)
        theta[1] = random.uniform(self.min_theta, self.max_theta)
        return theta

    def generate_random_pos(self):
        theta = self.generate_random_angle()
        P = self.forward_kinematics(theta)
        pos = np.array([P[-1][0,3], P[-1][1,3]])
        return pos

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    @staticmethod
    def normalize_angle(angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    def step(self, action):
        if self.action[action] == "INC_J1":
            self.theta[0] += self.rate
        elif self.action[action] == "DEC_J1":
            self.theta[0] -= self.rate
        elif self.action[action] == "INC_J2":
            self.theta[1] += self.rate 
        elif self.action[action] == "DEC_J2":
            self.theta[1] -= self.rate
        elif self.action[action] == "INC_J1_J2":
            self.theta[0] += self.rate
            self.theta[1] += self.rate 
        elif self.action[action] == "DEC_J1_J2":
            self.theta[0] -= self.rate
            self.theta[1] -= self.rate

        self.theta[0] = np.clip(self.theta[0], self.min_theta, self.max_theta)
        self.theta[1] = np.clip(self.theta[1], self.min_theta, self.max_theta)
        self.theta[0] = self.normalize_angle(self.theta[0])
        self.theta[1] = self.normalize_angle(self.theta[1])
        # Calc reward
        P = self.forward_kinematics(self.theta)
        tip_pos = [P[-1][0,3], P[-1][1,3]]
        distance_error = euclidean(self.target_pos, tip_pos)

        reward = 0
        if distance_error >= self.current_error:
            reward = -1
        epsilon = 10
        if (distance_error > -epsilon and distance_error < epsilon):
            reward = 1

        self.current_error = distance_error
        self.current_score += reward

        if self.current_score == -10 or self.current_score == 10:
            done = True
        else:
            done = False

        observation = np.hstack((self.target_pos, self.theta))
        info = {
            'distance_error': distance_error,
            'target_position': self.target_pos,
            'current_position': tip_pos
        }
        return observation, reward, done, info

    def reset(self):
        self.target_pos = self.generate_random_pos()
        self.current_score = 0
        observation = np.hstack((self.target_pos, self.theta))
        return observation

    def render(self, mode='human'):
        SCREEN_COLOR = (50, 168, 52)
        if self.viewer == None:
            pygame.init()
            pygame.display.set_caption("RobotArm-Env")
            self.screen = pygame.display.set_mode(self.window_size)
            self.clock = pygame.time.Clock()
        self.screen.fill(SCREEN_COLOR)
        self.draw_target()
        self.draw_arm(self.theta)
        self.clock.tick(60)
        pygame.display.flip()

    def close(self):
        if self.viewer != None:
            pygame.quit()

class RobotArmEnvV1(RobotArmEnvV0):
    def __init__(self):
        super(RobotArmEnvV1, self).__init__()

        self.min_action = -1.0
        self.max_action = 1.0
        self.min_theta = np.radians(0)
        self.max_theta = np.radians(90)
        self.action_space = spaces.Box(low=self.min_action, high=self.max_action, shape=(2,), dtype=np.float32)

    def step(self, action):
        theta0 = np.interp(action[0], (self.min_action, self.max_action), (self.min_theta, self.max_theta))
        theta1 = np.interp(action[1], (self.min_action, self.max_action), (self.min_theta, self.max_theta))
        self.theta[0] = theta0
        self.theta[1] = theta1
        # Calc reward
        P = self.forward_kinematics(self.theta)
        tip_pos = [P[-1][0,3], P[-1][1,3]]
        distance_error = euclidean(self.target_pos, tip_pos)

        # Sharp reward
        reward = -distance_error / 100
        done = False
        epsilon = 5
        if (distance_error > -epsilon and distance_error < epsilon):
            done = True

        observation = np.hstack((self.target_pos, self.theta))
        info = {
            'distance_error': distance_error,
            'target_position': self.target_pos,
            'current_position': tip_pos
        }
        return observation, reward, done, info