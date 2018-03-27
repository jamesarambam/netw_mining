from spotlight.cross_validation import random_train_test_split
from spotlight.datasets.movielens import get_movielens_dataset
from spotlight.evaluation import rmse_score, mrr_score
# from spotlight.factorization.implicit import ImplicitFactorizationModel
from spotlight.factorization.implicitRanking import RankingModel
import numpy as np
# --------------------------------------------------------------------- #

dataset = get_movielens_dataset(variant='100K')
train, test = random_train_test_split(dataset)


user_ids = test.user_ids.astype(np.int64)
item_ids = test.item_ids.astype(np.int64)
model = RankingModel(n_iter=1, batch_size=5)
model.fit(train)

mrr = mrr_score(model, test)
print mrr
rmse = rmse_score(model, test)
print rmse
