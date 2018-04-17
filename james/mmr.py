import os

from spotlight.datasets.synthetic import generate_sequential
import numpy as np
from spotlight.cross_validation import random_train_test_split
from spotlight.datasets.movielens import get_movielens_dataset
from spotlight.evaluation import rmse_score, mrr_score
# from spotlight.factorization.implicit import ImplicitFactorizationModel
from spotlight.factorization.implicitRanking import RankingModel
from auxLib import createDir
import pdb

k_sample = [10, 20, 50, 100, 200]
sample = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

dataset = get_movielens_dataset(variant='100K')
rs = np.random.RandomState(100)
train, test = random_train_test_split(dataset, random_state=rs)


for k in k_sample:
	dirName = "100kMovie" + "_k" + str(k)
	createDir("./log/mmr/real/", dirName)
	for i in sample:
		model = RankingModel(n_iter=10, batch_size=128, learning_rate=1e-4, k_sample=k, inputSample=train.ratings.shape[0])
		model.fit(train)
		mmr = mrr_score(model, test)
		with open("./log/mmr/real/" + dirName + "/" + str(i) + ".txt", 'w') as f:
			f.writelines(str(mmr))

