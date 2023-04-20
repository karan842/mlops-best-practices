import pandas as pd

df = pd.read_csv('raw_data\data.csv')
num_features = df.select_dtypes(exclude='object').columns
cat_features = df.select_dtypes(include='object').columns

print(num_features)
print(cat_features)