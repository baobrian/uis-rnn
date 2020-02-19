# coding=utf-8

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# data={'A':[1,2,3,4,5],'B':[2,3,4,5,6],'C':[3,4,5,6,7],'D':[4,5,6,7,8]}
# df=pd.DataFrame(data)
# target=pd.DataFrame([10,9,8,7,6],columns=['target'])
# X_train, X_test, Y_train, Y_test=train_test_split(df,target,test_size=0.25)

X, y = np.arange(10).reshape((5, 2)), range(5)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
pass