# netw_mining
## Exploring Deep Learning Approaches for Personalized Item Recommendation

We provide our implementation of the ranking network, deep BPR and the scripts to compare with several baselines, i.e., MF, BPR.

### Sample Code
To run BPR
```sh
python baseline.py --model implicit --loss bpr --net DeepNet --lr 1e-4 --n_epoch 20 --data MovieLens
```
To run MF
```sh
python baseline.py --model explicit --loss logistic --data MovieLens
```
To run DBPR
```sh
python baseline.py --model implicit --loss bpr --lr 1e-3 --net BilinearNet --n_epoch 3 --data MovieLens
```
To run Ranking Network
```sh
python ranking_rmse.py
```
### Scripts
Scripts to replicate our experiments are provided in folder ```scripts/```
