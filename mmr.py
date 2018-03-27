import os



for i in range(1, 11):
	os.system("python ranking_mmr.py > mmr/log"+str(i)+".txt")