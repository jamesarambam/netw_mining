CUDA_VISIBLE_DEVICES=7 python baseline.py --model implicit --loss bpr --net DeepNet --lr 1e-4 --n_epoch 20 --data MovieLens
CUDA_VISIBLE_DEVICES=7 python baseline.py --model implicit --loss bpr --lr 1e-3 --net BilinearNet --n_epoch 3 --data MovieLens
#CUDA_VISIBLE_DEVICES=7 python baseline.py --model explicit --loss logistic --data MovieLens
