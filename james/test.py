import sys

sys.path.append("../")

from spotlight.datasets.movielens import get_movielens_dataset


dataset = get_movielens_dataset(variant='100K')

print dataset.num_items
print dataset.num_users

print dataset.ratings.shape