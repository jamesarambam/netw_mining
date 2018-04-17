from spotlight.cross_validation import random_train_test_split
from spotlight.datasets.movielens import get_movielens_dataset
from spotlight.datasets.amazon import get_amazon_dataset
from spotlight.datasets.goodbooks import get_goodbooks_dataset
from evaluation import mrr_score, rmse_score
from implicit import ImplicitFactorizationModel
from representations import *
from spotlight.factorization.explicit import ExplicitFactorizationModel
import numpy as np
import argparse
from synthetic import generate_sequential
import numpy as np

def make_synthetic(sparsity = 0.05):
    n_u = 600
    n_i = 3000
    n = int(n_u * n_i * sparsity)
    rs = np.random.RandomState(100)
    test_split = float(20000.0 / n)
    dataset = generate_sequential(num_users=n_u, num_items=n_i, num_interactions=n, concentration_parameter=0.4, order = 3, random_state = rs)
    return test_split, dataset
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Baseline')
    parser.add_argument('--net', type = str, default =  'DeepNet')
    parser.add_argument('--model', type = str, default = 'implicit')
    parser.add_argument('--variant', type = str, default = '100K')
    parser.add_argument('--n_epoch', type = int, default = '20')
    parser.add_argument('--loss', type = str, default = 'bpr')
    parser.add_argument('--lr', type =float, default = 1e-4)
    parser.add_argument('--sparsity', type = float, default = 0.05)
    parser.add_argument('--data', type = str, default = 'synthetic')
    args = parser.parse_args()
    if str(args.data) == 'synthetic':
        split, dataset= make_synthetic(args.sparsity)
    elif str(args.data).lower() == 'movielens':
        print('MovieLens')
        dataset = get_movielens_dataset(variant=args.variant)
        split = 0.2
    elif str(args.data).lower() == 'amazon':
        print('Amazon')
        dataset = get_amazon_dataset()
        split = 0.2
    else:
        print('GoodBook')
        dataset = get_goodbooks_dataset()
        split = 0.2
    rmses = []
    mrrs = []
    rs = np.random.RandomState(100)
    pdb.set_trace()
    for i in range(5):
        print('Split - {} , Run {}'.format(split, i))
        train, test = random_train_test_split(dataset, random_state = rs, test_percentage = split)
        if args.model == 'implicit':
            model = ImplicitFactorizationModel(n_iter=args.n_epoch, loss=args.loss, use_cuda = True, learning_rate = args.lr,
                    representation = args.net)
        elif args.model == 'explicit':
            model = ExplicitFactorizationModel(n_iter=args.n_epoch, loss=args.loss, use_cuda = True, learning_rate = args.lr)
        model.fit(train, verbose = 0)

        rmse = rmse_score(model, test)
        rmses.append(rmse)
        mrr = mrr_score(model, test)
        mrrs.append(np.mean(mrr))

rmses = np.array(rmses)
mrrs = np.array(mrrs)
print('RMSE: {} +- {}'.format(np.mean(rmses),np.var(rmses)))
print('MRR: {} +- {}'.format(np.mean(mrrs), np.var(mrrs)))
