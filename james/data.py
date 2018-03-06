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
        self.trainData, self.testData = self.prepData()

    def prepData(self):

        self.binaryHash = {(0, 0):0, (0, 1):1, (1, 0):2, (1, 1):3}
        arr = np.random.randint(0, 2, size=(TOTAL_USERS, TOTAL_ITEMS))
        self.RatingMat = torch.from_numpy(arr)
        tmpTrain = []
        tmpTest = []
        for u in range(TOTAL_USERS):
            for i1 in range(TOTAL_ITEMS):
                for i2 in range(i1+1, TOTAL_ITEMS):
                    tmp2 = []
                    tmp2.append(u)
                    tmp2.append(i1)
                    tmp2.append(i2)
                    cls = (self.RatingMat[u][i1], self.RatingMat[u][i2])
                    tmp2.append(self.binaryHash[cls])
                    if self.binaryHash[cls] == 0:
                        tmpTest.append(tmp2)
                    else:
                        tmpTrain.append(tmp2)

        return torch.FloatTensor(tmpTrain), torch.FloatTensor(tmpTest)

def main():
    print "Hello World"


# =============================================================================== #

if __name__ == '__main__':
    main()
    print "# ============================  END  ============================ #"