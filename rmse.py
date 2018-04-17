import os

from spotlight.datasets.synthetic import generate_sequential
import numpy as np
from spotlight.cross_validation import random_train_test_split
from spotlight.datasets.movielens import get_movielens_dataset
from spotlight.evaluation import rmse_score, mrr_score
# from spotlight.factorization.implicit import ImplicitFactorizationModel
from spotlight.factorization.implicitRanking import RankingModel
from auxLib import createDir

n_u = 200
n_i = 5000
sparsity = [0.05, 0.1, 0.15, 0.2, 0.25, 0.5]
n_interactions = [int(n_u*n_i*s) for s in sparsity]
rs = np.random.RandomState(500)


k_sample = [10, 20, 50, 100, 200]
sample = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
for k in k_sample:
	for i in sample:
		os.system("python ranking_rmse.py "+str(k)+" > rmse/log"+str(i)+"_"+str(k)+".txt")
