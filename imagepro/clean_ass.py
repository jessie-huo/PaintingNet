import pandas as pd
f1 = pd.read_csv("data.csv")
f1['price'] = f1['price'].replace("unknown", "2817")
f1.to_csv("data_output.csv")