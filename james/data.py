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
from parameters import INPUT_DIM, OUTPUT_DIM, HIDDEN_DIM, LEARNING_RATE, TOTAL_USERS, TOTAL_ITEMS, USER_EMB_DIM, ITEM_EMB_DIM, SEED
import numpy as np

# =============================== Variables ================================== #

np.random.seed(SEED)
torch.manual_seed(SEED)

# ============================================================================ #

class syntheticData:

    def __init__(self):
        self.userEmbMat = Variable(torch.randn(USER_EMB_DIM, TOTAL_USERS), requires_grad=False)
        self.itemEmbMat = Variable(torch.randn(ITEM_EMB_DIM, TOTAL_ITEMS), requires_grad=False)
        arr = np.random.randint(0, 2, size=(TOTAL_USERS, TOTAL_ITEMS))
        self.RatingMat = torch.from_numpy(arr)

        tmp = [[0, 1, 3], [0, 2, 2], [0, 3, ]]



def main():
    print "Hello World"


# =============================================================================== #

if __name__ == '__main__':
    main()
    print "# ============================  END  ============================ #"