"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Author : James Arambam
Date   : 05 Mar 2018
Description :
Input : 
Output : 
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""
print "# ============================ START ============================ #"
# ================================ Imports ================================ #
import sys
import os
from pprint import pprint
import time
import torch
from torch.autograd import Variable
from parameters import INPUT_DIM, OUTPUT_DIM, HIDDEN_DIM, LEARNING_RATE, SEED


# =============================== Variables ================================== #

torch.manual_seed(SEED)


# ============================================================================ #

class Network(torch.nn.Module):
  def __init__(self, D_in, H, D_out):
    super(Network, self).__init__()
    self.linear1 = torch.nn.Linear(D_in, H)
    self.linear2 = torch.nn.Linear(H, D_out)


  def forward(self, x):
    h_relu = self.linear1(x).clamp(min=0)
    y_pred = self.linear2(h_relu)
    return y_pred


class RankingFn:

    def __init__(self, trainEp):

        self.model = Network(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
        self.lossFn = torch.nn.MSELoss(size_average=False)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=LEARNING_RATE)
        self.Train_Epoch = trainEp

    def train(self, x, y):

        for t in range(self.Train_Epoch):
            y_pred = self.model(x)
            loss = self.lossFn(y_pred, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def compare(self, x):
        # u : user_id
        # i1 : item 1
        # i2 ; item 2
        # x = concat(u, i1, i2)
        return self.model(x)


def main():
    print "Hello World"


# =============================================================================== #

if __name__ == '__main__':
    main()
    print "# ============================  END  ============================ #"