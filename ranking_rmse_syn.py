from synthetic import generate_sequential
import numpy as np
from spotlight.cross_validation import random_train_test_split
from spotlight.datasets.movielens import get_movielens_dataset
from spotlight.evaluation import rmse_score, mrr_score
# from spotlight.factorization.implicit import ImplicitFactorizationModel
from spotlight.factorization.implicitRanking import RankingModel
from auxLib import createDir


def make_synthetic(sparsity = 0.05):
    n_u = 600
    n_i = 3000
    n = int(n_u * n_i * sparsity)
    rs = np.random.RandomState(100)
    test_split = float(20000.0 / n)
    dataset = generate_sequential(num_users=n_u, num_items=n_i, num_interactions=n, concentration_parameter=0.4, order = 3, random_state = rs)
    return test_split, dataset

sparsity = [0.05, 0.1, 0.15, 0.2, 0.25, 0.5]
sample = [1, 2, 3, 4, 5]

for s in sparsity:
    k = 100
    dirName = "s"+str(s)+"_k"+str(k)
    createDir("./log/rmse/synth/", dirName)
    for i in sample:
        rs = np.random.RandomState(100)
        split, dataset = make_synthetic(s)
        train, test = random_train_test_split(dataset, random_state=rs, test_percentage=split)
        model = RankingModel(n_iter=10, batch_size=128, learning_rate=1e-4, k_sample=k, inputSample=train.ratings.shape[0])
        model.fit(train)
        rmse = rmse_score(model, test)
        with open("./log/rmse/synth/"+dirName+"/"+str(i)+".txt", 'w') as f:
            f.writelines(str(rmse))
