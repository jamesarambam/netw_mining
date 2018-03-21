from spotlight.cross_validation import random_train_test_split
from spotlight.datasets.movielens import get_movielens_dataset
from spotlight.evaluation import mrr_score, rmse_score
from spotlight.factorization.implicit import ImplicitFactorizationModel
import numpy as np

dataset = get_movielens_dataset(variant='100K')
train, test = random_train_test_split(dataset)

# user_ids = train.user_ids.astype(np.int64)
# item_ids = train.item_ids.astype(np.int64)
# print user_ids.shape, item_ids.shape
#
user_ids = test.user_ids.astype(np.int64)
item_ids = test.item_ids.astype(np.int64)
# print user_ids.shape, item_ids.shape
# exit()


model = ImplicitFactorizationModel(n_iter=1, loss='bpr', batch_size=2)
model.fit(train)

# mrr = mrr_score(model, test)
rmse = rmse_score(model, test)
# print mrr


# print model.predict(user_ids, item_ids)