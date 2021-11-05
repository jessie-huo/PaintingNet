import pandas as pd
import csv

f1 = pd.read_csv("merged.csv")

f1.drop_duplicates(subset = 'id', keep = 'last', inplace = True)

f1.to_csv('data.csv', index = False)
