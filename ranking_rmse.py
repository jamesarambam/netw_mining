from spotlight.cross_validation import random_train_test_split
from spotlight.datasets.movielens import get_movielens_dataset
from spotlight.evaluation import rmse_score, mrr_score
# from spotlight.factorization.implicit import ImplicitFactorizationModel
from spotlight.factorization.implicitRanking import RankingModel
import numpy as np
<<<<<<< HEAD
import pdb
import rlcompleter
import sys

=======

import pdb
import rlcompleter
import sys
>>>>>>> ceef84d542ff6b80f4aa060d1c7d2ca79a4d94e4
# --------------------------------------------------------------------- #


k = int(sys.argv[1])
dataset = get_movielens_dataset(variant='100K')
<<<<<<< HEAD

=======
>>>>>>> ceef84d542ff6b80f4aa060d1c7d2ca79a4d94e4
train, test = random_train_test_split(dataset)
model = RankingModel(n_iter=50, batch_size=128, learning_rate=1e-4, k_sample=k)
model.fit(train)
rmse = rmse_score(model, test)
print "------------------"
print "rmse :", rmse
