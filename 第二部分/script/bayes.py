# Naive Bayes Classification
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix

# 导入数据
dataset = pd.read_csv('iris.csv')

#得到X, y
X = dataset.iloc[:,:4].values
y = dataset['species'].values

# 得到train和test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 82)

# Fitting Naive Bayes Classification to the Training set with linear kernel

nvclassifier = GaussianNB()
nvclassifier.fit(X_train, y_train)

# 预测结果
y_pred = nvclassifier.predict(X_test)

# 混淆矩阵
cm = confusion_matrix(y_test, y_pred)
print(cm)