from spotlight.cross_validation import random_train_test_split
from spotlight.datasets.movielens import get_movielens_dataset
from spotlight.evaluation import mrr_score
# from spotlight.factorization.implicit import ImplicitFactorizationModel
from spotlight.factorization.implicitRanking import RankingModel
import numpy as np
# --------------------------------------------------------------------- #

dataset = get_movielens_dataset(variant='100K')
train, test = random_train_test_split(dataset)

user_ids = test.user_ids.astype(np.int64)
item_ids = test.item_ids.astype(np.int64)
# user_ids = user_ids[0:10]
# item_ids = item_ids[0:10]


model = RankingModel(n_iter=1, batch_size=5)
model.fit(train)
print model.predict(user_ids, item_ids)
print "train"
exit()

mrr = mrr_score(model, test)
