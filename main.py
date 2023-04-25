# import pandas as pd
# import numpy as np
# from sklearn.feature_selection import mutual_info_classif
# from src.utils import class_imbalance, handling_class_imbalance
# from imblearn.over_sampling import SMOTE
 

# train_data = pd.read_csv("raw_data/data.csv")

# # X = train_data.drop(['Exited'],axis=1)
# # y = train_data['Exited']

# # # X, y = handling_class_imbalance(X,y,thresh=60)

# # # print(X.shape,y.shape)
    
# # smote = SMOTE()
# # X, y = smote.fit_resample(X,y)
# # print(X.shape,y.shape)

# min, max = train_data['Age'].min(), train_data['Age'].max()

# print(min,max)

