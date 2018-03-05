"""
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Author : James Arambam
Date   : 04 Mar 2018
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
from parameters import INPUT_DIM, OUTPUT_DIM, HIDDEN_DIM, LEARNING_RATE, TOTAL_USERS, TOTAL_ITEMS, USER_EMB_DIM, ITEM_EMB_DIM, SEED
from torch.autograd import Variable
from rankingFn import RankingFn
from data import syntheticData
# =============================== Variables ================================== #
torch.manual_seed(SEED)
# ============================================================================ #

def main():

    data = syntheticData()
    print data.userEmbMat
    print data.itemEmbMat
    print data.RatingMat




    exit()

    N = 2
    x = Variable(torch.randn(N, INPUT_DIM))
    y = Variable(torch.randn(N, OUTPUT_DIM), requires_grad = False)

    print x
    exit()
    # ------- Ranking Function -------- #
    rf = RankingFn(20)
    rf.train(x, y)

    # ------- Ranking Inference ------ #
    # x2 = concat(user, item1, item2)
    x2 = Variable(torch.randn(1, INPUT_DIM))

    print rf.compare(x2)




# =============================================================================== #

if __name__ == '__main__':
    main()
    print "# ============================  END  ============================ #"