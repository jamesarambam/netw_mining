from spotlight.cross_validation import random_train_test_split
from spotlight.datasets.movielens import get_movielens_dataset
from spotlight.evaluation import mrr_score
# from spotlight.factorization.implicit import ImplicitFactorizationModel
from spotlight.factorization.implicitRanking import ImplicitFactorizationModel


import numpy as np

dataset = get_movielens_dataset(variant='100K')

train, test = random_train_test_split(dataset)
model = ImplicitFactorizationModel(n_iter=1,
                                   loss='bce')



print test.user_ids.astype(np.int64)
print test.item_ids.astype(np.int64)

exit()

model.fit(train)

mrr = mrr_score(model, test)

