import os


k_sample = [10, 20, 50, 100, 200]
sample = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
for k in k_sample:
	for i in sample:
		os.system("python ranking_rmse.py "+str(k)+" > rmse/log"+str(i)+"_"+str(k)+".txt")