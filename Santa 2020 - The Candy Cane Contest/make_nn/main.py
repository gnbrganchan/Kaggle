
"""Greedy agent that chooses machine based on maximum expected payout

Uses a trained decision tree model to consider the other player's movements
in the expected payout.

See my other kernel for methodology for generating training data:
https://www.kaggle.com/lebroschar/generate-training-data

"""


import pickle
import base64
import random


import numpy as np
import pandas as pd
import sklearn.tree as skt
import sys
import os

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader,TensorDataset


SEED = 46

sys.path.append("/kaggle_simulations/agent")
working_dir = "/kaggle_simulations/agent"
# sys.path.append("/kaggle/working")
# working_dir = "/kaggle/working"

path_to_model1 = os.path.join(working_dir,"model1.sav")
path_to_model2 = os.path.join(working_dir,"model2.sav")
path_to_model3 = os.path.join(working_dir,"model3.sav")
path_to_model4 = os.path.join(working_dir,"model4.sav")
path_to_model5 = os.path.join(working_dir,"model5.sav")


# Parameters
FUDGE_FACTOR = 0.99
VERBOSE = False
TRAIN_FEATS = ['round_num', 'n_pulls_self', 'n_success_self', 'n_pulls_opp']
TARGET_COL = 'payout'

drop_rate = 0.3

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc4to256 = nn.Linear(4, 256)
        self.bn256 = nn.BatchNorm1d(256)
        self.fc256to64 = nn.Linear(256, 64)
        self.bn64 = nn.BatchNorm1d(64)
        self.fc64to16 = nn.Linear(64, 16)
        self.bn16 = nn.BatchNorm1d(16)
        self.fc16to4 = nn.Linear(16, 4)
        self.bn4 = nn.BatchNorm1d(4)
        self.fc4to1 = nn.Linear(4, 1)
        self.dropout = nn.Dropout(p=drop_rate)

    def forward(self, x):
        x = F.relu(self.fc4to256(x))
        # x = self.bn256(x)
        x = self.dropout(x)
        x = F.relu(self.fc256to64(x))
        # x = self.bn64(x)
        x = self.dropout(x)
        x = F.relu(self.fc64to16(x))
        # x = self.bn16(x)
        x = self.dropout(x)
        x = F.relu(self.fc16to4(x))
        # x = self.bn4(x)
        x = self.dropout(x)
        x = self.fc4to1(x)
        return x


class GreedyStrategy:
    """Implements strategy to maximize expected value

    - Tracks estimated likelihood of payout ratio for each machine
    - Tracks number of pulls on each machine
    - Chooses machine based on maximum expected value
    
    
    """
    def __init__(self, name, agent_num, n_machines):
        """Initialize and train decision tree model

        Args:
           name (str):   Name for the agent
           agent_num (int):   Assigned player number
           n_machines (int):   number of machines in the game
        
        """
        # Record inputs
        self.name = name
        self.agent_num = agent_num
        self.n_machines = n_machines
        
        # Initialize distributions for all machines
        self.n_pulls_self = np.array([0 for _ in range(n_machines)])
        self.n_success_self = np.array([0. for _ in range(n_machines)])
        self.n_pulls_opp = np.array([0 for _ in range(n_machines)])

        # Track other players moves
        self.opp_moves = []
        
        # Track winnings
        self.last_reward_count = 0

        # Create model to predict expected reward
        self.model1 = Net()
        self.model2 = Net()
        self.model3 = Net()
        self.model4 = Net()
        self.model5 = Net()
        self.model1.load_state_dict(torch.load(path_to_model1))
        self.model2.load_state_dict(torch.load(path_to_model2))
        self.model3.load_state_dict(torch.load(path_to_model3))
        self.model4.load_state_dict(torch.load(path_to_model4))
        self.model5.load_state_dict(torch.load(path_to_model5))

        self.model1.eval()
        self.model2.eval()
        self.model3.eval()
        self.model4.eval()
        self.model5.eval()
        
        # Predict expected reward
        features = np.zeros((self.n_machines, 4))
        features[:, 0] = len(self.opp_moves) / 2000
        features[:, 1] = self.n_pulls_self / 100
        features[:, 2] = self.n_success_self / 50
        features[:, 3] = self.n_pulls_opp / 100

        self.predicts = self.model1(Variable(torch.from_numpy(features).float())).detach().numpy().flatten() / 5
        self.predicts += self.model2(Variable(torch.from_numpy(features).float())).detach().numpy().flatten() / 5
        self.predicts += self.model3(Variable(torch.from_numpy(features).float())).detach().numpy().flatten() / 5
        self.predicts += self.model4(Variable(torch.from_numpy(features).float())).detach().numpy().flatten() / 5
        self.predicts += self.model5(Variable(torch.from_numpy(features).float())).detach().numpy().flatten() / 5

        

    def __call__(self):
        """Choose machine based on maximum expected payout

        Returns:
           <result> (int):  index of machine to pull
        
        """
        # Otherwise, use best available
        est_return = self.predicts
        max_return = np.max(est_return)
        result = np.random.choice(np.where(
            est_return >= FUDGE_FACTOR * max_return)[0])
        
        if VERBOSE:
            print('  - Chose machine %i with expected return of %3.2f' % (
                int(result), est_return[result]))

        return int(result)
    
        
    def updateDist(self, curr_total_reward, last_m_indices):
        """Updates estimated distribution of payouts"""
        # Compute last reward
        last_reward = curr_total_reward - self.last_reward_count
        self.last_reward_count = curr_total_reward
        if VERBOSE:
            print('Last reward: %i' % last_reward)

        if len(last_m_indices) == 2:
            # Update number of pulls for both machines
            m_index = last_m_indices[self.agent_num]
            opp_index = last_m_indices[(self.agent_num + 1) % 2]
            self.n_pulls_self[m_index] += 1
            self.n_pulls_opp[opp_index] += 1

            # Update number of successes
            self.n_success_self[m_index] += last_reward
            
            # Update opponent activity
            self.opp_moves.append(opp_index)

            # Update predictions for chosen machines
            self.predicts[[opp_index, m_index]] = self.model1(Variable(torch.from_numpy(np.array([
                [len(self.opp_moves)/2000,self.n_pulls_self[opp_index]/100,self.n_success_self[opp_index]/50,self.n_pulls_opp[opp_index]/100],
                [len(self.opp_moves)/2000,self.n_pulls_self[m_index]/100,self.n_success_self[m_index]/50,self.n_pulls_opp[m_index]/100]
            ])).float())).detach().numpy().flatten() / 5
            self.predicts[[opp_index, m_index]] += self.model2(Variable(torch.from_numpy(np.array([
                [len(self.opp_moves)/2000,self.n_pulls_self[opp_index]/100,self.n_success_self[opp_index]/50,self.n_pulls_opp[opp_index]/100],
                [len(self.opp_moves)/2000,self.n_pulls_self[m_index]/100,self.n_success_self[m_index]/50,self.n_pulls_opp[m_index]/100]
            ])).float())).detach().numpy().flatten() / 5
            self.predicts[[opp_index, m_index]] += self.model3(Variable(torch.from_numpy(np.array([
                [len(self.opp_moves)/2000,self.n_pulls_self[opp_index]/100,self.n_success_self[opp_index]/50,self.n_pulls_opp[opp_index]/100],
                [len(self.opp_moves)/2000,self.n_pulls_self[m_index]/100,self.n_success_self[m_index]/50,self.n_pulls_opp[m_index]/100]
            ])).float())).detach().numpy().flatten() / 5
            self.predicts[[opp_index, m_index]] += self.model4(Variable(torch.from_numpy(np.array([
                [len(self.opp_moves)/2000,self.n_pulls_self[opp_index]/100,self.n_success_self[opp_index]/50,self.n_pulls_opp[opp_index]/100],
                [len(self.opp_moves)/2000,self.n_pulls_self[m_index]/100,self.n_success_self[m_index]/50,self.n_pulls_opp[m_index]/100]
            ])).float())).detach().numpy().flatten() / 5
            self.predicts[[opp_index, m_index]] += self.model5(Variable(torch.from_numpy(np.array([
                [len(self.opp_moves)/2000,self.n_pulls_self[opp_index]/100,self.n_success_self[opp_index]/50,self.n_pulls_opp[opp_index]/100],
                [len(self.opp_moves)/2000,self.n_pulls_self[m_index]/100,self.n_success_self[m_index]/50,self.n_pulls_opp[m_index]/100]
            ])).float())).detach().numpy().flatten() / 5
            

def agent(observation, configuration):
    global curr_agent
    
    if observation.step == 0:
        # Initialize agent
        curr_agent = GreedyStrategy(
            'Mr. Agent %i' % observation['agentIndex'],
            observation['agentIndex'],
            configuration['banditCount'])
    
    # Update payout ratio distribution with:
    curr_agent.updateDist(observation['reward'], observation['lastActions'])

    return curr_agent()
