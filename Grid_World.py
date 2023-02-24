import numpy as np
import random
import tensorflow as tf
import gymnasium as gym
from tqdm import tqdm
from time import sleep
from gymnasium import spaces
from gymnasium.spaces import Box, Sequence, Dict
import numpy as np
import math
from os.path import join
import glob

import pygame

from math import *
def cal_dis(x:np.array, y:np.array):
    if not (np.size(x) == 3 and np.size(y) == 3):
        raise ValueError('your input array must like[1,2,3]')
    sums = ((abs(x[0] - y[0])**2)) + ((abs(x[1] - y[1])**2)) + ((abs(x[2] - y[2])**2))
    res = sqrt(sums)
    return res
    
box_length = 400 #400mm

length_per_coordinate = box_length / 400.0
max_observation_distance = 60 * length_per_coordinate #果蝇最远可观察距离是6厘米
min_distance = 6 * length_per_coordinate #小于这个距离判定为撞上

import numpy as np
import gymnasium as gym
import stable_baselines3
from math import *

class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode = None, stop_render = False, seed = None):
        
        self.action_space = Box(
                        low = np.array([-200.0, -200.0, 1000.0]), 
                        high = np.array([200.0, 200.0, 1500.0]), 
                        dtype=np.float32
                    )

                    #'orientation': spaces.Box(-180,180),
                    #'good_trajectory': spaces.Discrete(2)
                    
        self.steps = 0
        self.reward = 0
        self.max_steps = 100
        self.start_coordi = self.get_random_coordinates()
        self.observation_space = Sequence(
                                        Box(
                                            low = np.array([-200.0, -200.0, 1000.0]), 
                                            high = np.array([200.0, 200.0, 1500.0]), 
                                            dtype=np.float32
                                            )
                                          )
    def get_random_coordinates(self):
        a = np.random.uniform(-200.0, 200.0)
        b = np.random.uniform(-200.0, 200.0)
        c = np.random.uniform(1000.0, 1500.0)
        array = np.array([a,b,c])
        return array
    
    def get_random_steps(self):
       
        a = 0.01 * np.random.uniform(-200.0, 200.0)
        b = 0.01 * np.random.uniform(-200.0, 200.0)
        c = 0.01 * np.random.uniform(-200.0, 200.0)
        array = np.array([a,b,c])
        return array
    # 根据当前坐标获得观测范围内其他果蝇的坐标
    # 输入参数是当前坐标，和当前进行到第几步了
    # 假设数据集中的果蝇不会受到代理的行为影响，是固定因素
    def get_other_coordinates(self, input_coordinates:np.array, timestep:np.uint8):
        
        other_coord = []
        for i in range(1407):
            if timestep >= len(data_all[i]):
                continue
            check_coordi = data_all[i][timestep]
            if cal_dis(input_coordinates, check_coordi) <= max_observation_distance:
                other_coord.append(check_coordi)
            
        other_coord = np.array(other_coord, dtype=np.float32)
        return other_coord
    
    def get_info(self):
        return {
            'distance': cal_dis(self.current_coordinates, self.start_coordi),
            'steps': self.steps
        }
    def compute_angle_rw(self):
        
        pass
    
    def compupte_speed_rw(self):
        
        pass
    def reset(self):
        # 初始坐标随机选择，但不超过容器上限
        new_coordinates = self.get_random_coordinates()
        self.current_coordinates = new_coordinates
        # 初始步数为 1
        self.steps = 1
        # 获得初始的观测值，即初始坐标周围的果蝇位置
        self.next_observation = self.get_other_coordinates(new_coordinates,1)
        # 设置初始坐标并保存到固定变量中
        self.start_coordi = self.current_coordinates
        self.reward = 0
        # 初始距离一般为 0
        info = self.get_info()
        return self.next_observation
    def cal_reward(self, term, **kwargs):
        rw = 0.0
        if term:
            return rw
        rw += self.compute_angle_rw()
        rw += self.compute_speed_rw()
        pass
    
    def step(self, action) -> Dict:
        # 使用 np.clip防止随机运动超过容器范围
        self.current_coordinates = np.clip(self.current_coordinates + self.get_random_steps(), [-200, -200, 1000], [200, 200, 1500])
        # 更新步数 + 1
        self.steps += 1
        # 根据下一个坐标更新观察范围
        self.obs = self.get_other_coordinates(self.current_coordinates, self.steps)
        # 判断撞到任何一个观察范围内的果蝇，则立即停止本次模拟
        terminate = False
        for coord in self.obs:
            if cal_dis(coord, self.current_coordinates) <= min_distance:
                terminate = True
                break
        
        self.reward += 1 if not terminate else 0
        # reward = dtw(traj) if not terminate else 0
        # 如果步数超过最大限制，则表示完成
        done = False
        if self.steps >= self.max_steps:
            done = True
        # 记录最远移动距离，作为特征值之一
        info = self.get_info()
        self.render(action, self.reward)
        # 返回值必须是一个字典形式
        return self.obs, self.reward, done, terminate, info, {}
        #raise NotImplementedError
    def render(self, action, rw):
        print(f"Steps:{self.steps}\nObservation:{self.obs}\nReward:{rw}")
        print(f"Current Coordinates:{action}\n")
        print("================================================")
