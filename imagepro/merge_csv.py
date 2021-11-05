import pandas as pd
import csv

f1 = pd.read_csv("output1.csv") 
f2 = pd.read_csv("processed.csv") 

with open("merged.csv", mode='w') as out_file:
	out_writer = csv.writer(out_file, delimiter=',')
	out_writer.writerow(['id', 'artist', 'country', 'h', 'w', 't', 'price', 'dominant', 'brightness', 'unique_color', 'cornerP', 'edgeP'])
	for i in range(0, f1.shape[0]):
		for j in range(0, f2.shape[0]):
			if(f1['id'][i] == f2['id'][j]):
				out_writer.writerow([f1['id'][i], f1['artist'][i], f1['country'][i], f1['h'][i][:-2], f1['w'][i][:-2], f1['t'][i][:-2], f1['price'][i], f2['dominant'][j], f2['brightness'][j], f2['unique_color'][j], f2['cornerP'][j], f2['edgeP'][j]])
				print(i)
				continue