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

sample = [1, 2, 3, 4, 5]

for n in n_interactions:
    k = 100
    dirName = "s"+str(n)+"_k"+str(k)
    createDir("./log/", dirName)
    for i in sample:
        rs = np.random.RandomState(np.random.randint(1, 100))
        dataset = generate_sequential(num_users=n_u, num_items=n_i, num_interactions=n, concentration_parameter=0.01,
                                      order=1, random_state=rs)
        train, test = random_train_test_split(dataset)

        model = RankingModel(n_iter=50, batch_size=128, learning_rate=1e-4, k_sample=k)
        model.fit(train)
        rmse = rmse_score(model, test)
        with open("./log/"+dirName+"/"+str(i)+".txt", 'w') as f:
            f.writelines(str(rmse))
