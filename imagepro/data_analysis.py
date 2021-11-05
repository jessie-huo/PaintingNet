import pandas as pd
import csv
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing

f1 = pd.read_csv("data.csv")
f2 = pd.read_csv("data.csv")

h = f1['h'].values
w = f1['w'].values
t = f1['t'].values
price = f1['price'].values
brightness = f1['brightness'].values
unique_color = f1['unique_color'].values
cornerP = f1['cornerP'].values
edgeP = f1['edgeP'].values

le = preprocessing.LabelEncoder()
f1 = f1.apply(preprocessing.LabelEncoder().fit_transform)

j = 0
for i in range(len(price)):
	price[i] = price[i].replace(',','')
	if(price[i] != "unknown"):
		j += float(price[i])


average = j / i


h = [float(i) for i in h]
w = [float(i) for i in w]
t = [float(i) for i in t]
for i in range(len(price)):
	if(price[i] == "unknown"):
		price[i] = average
price = [float(i) for i in price]
brightness = [float(i) for i in brightness]
unique_color = [float(i) for i in unique_color]
cornerP = [float(i) for i in cornerP]
edgeP = [float(i) for i in edgeP]

artist = np.zeros(max(f1['artist'])+1)
country = np.zeros(max(f1['country'])+1)

for i in range(len(f1['artist'])):
	artist[f1['artist'][i]] += price[i]
for i in range(len(f1['country'])):
	country[f1['country'][i]] += price[i]


a = 0
for i in range(len(f1['country'])):
	if(f1['country'][i] == np.argmax(country)):
		a = i
		break
print("The max sum of prices of paintings (countries): %s %f" %(f2['country'][a], max(country)))
a = 0
for i in range(len(f1['country'])):
	if(f1['country'][i] == np.argmin(country)):
		a = i
		break
print("The min sum of prices of paintings (countries): %s %f" %(f2['country'][a], min(country)))

a = 0
for i in range(len(f1['artist'])):
	if(f1['artist'][i] == np.argmax(artist)):
		a = i
		break
print("The max sum of prices of paintings (artists): %s %f" %(f2['artist'][a], max(artist)))
a = 0
for i in range(len(f1['artist'])):
	if(f1['artist'][i] == np.argmin(artist)):
		a = i
		break
print("The min sum of prices of paintings (artists): %s %f" %(f2['artist'][a], min(artist)))

plt.title("h to Price")
plt.plot(h, price, 'ro')
plt.xlabel("h")
plt.ylabel("Price")
plt.show()


plt.title("w to Price")
plt.plot(w, price, 'ro')
plt.xlabel("w")
plt.ylabel("Price")
plt.show()

plt.title("t to Price")
plt.plot(t, price, 'ro')
plt.xlabel("t")
plt.ylabel("Price")
plt.show()

plt.title("brightness to Price")
plt.plot(brightness, price, 'ro')
plt.xlabel("brightness")
plt.ylabel("Price")
plt.show()

plt.title("unique_color to Price")
plt.plot(unique_color, price, 'ro')
plt.xlabel("unique_color")
plt.ylabel("Price")
plt.show()

plt.title("cornerP to Price")
plt.plot(cornerP, price, 'ro')
plt.xlabel("cornerP")
plt.ylabel("Price")
plt.show()

plt.title("edgeP to Price")
plt.plot(edgeP, price, 'ro')
plt.xlabel("edgeP")
plt.ylabel("Price")
plt.show()