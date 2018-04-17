from spotlight.cross_validation import random_train_test_split
from spotlight.datasets.movielens import get_movielens_dataset
from spotlight.evaluation import rmse_score, mrr_score
# from spotlight.factorization.implicit import ImplicitFactorizationModel
from spotlight.factorization.implicitRanking import RankingModel
import numpy as np

import pdb
import rlcompleter

# --------------------------------------------------------------------- #



dataset = get_movielens_dataset(variant='100K')
train, test = random_train_test_split(dataset)
model = RankingModel(n_iter=50, batch_size=50, learning_rate=1e-4)
model.fit(train)
mrr = mrr_score(model, test)
print "------------------"
print "mrr :", mrr.mean()

# rmse = rmse_score(model, test)
# print rmse
