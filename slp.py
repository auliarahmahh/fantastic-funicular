# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 13:36:45 2019

@author: LENOVO
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math as math

    
data=pd.read_csv(r"C:\Users\LENOVO\Documents\SEMESTER 6\MACHINE LEARNING\irisdata.csv")
print("Describing the data: ",data.describe())
print("Info of the data:",data.info())

dataset=data.iloc[0:100].values
A=np.where(dataset[4]=='Iris-sentosa',0,1)
array=data.iloc[0:100,[0,1,2,3]].values


y = data.iloc[0:100, 4].values
y =np.where(y == 'Iris-setosa', 0, 1)


weight = [0.1 , 0.1 , 0.1 , 0.1]

#activation
def activation(row, weight,target):
    result=0
    activation=0
    bias = [0.1]
    for i in range(len(row)):
        result += weight[i] * row[i]
        result=result+bias
        
    activation = 1/(1+math.exp(-result))
    delta(row,activation,target,bias)
    return activation

#weight
def weight(row,d_weight, alfa):
    alfa=0.1
    w=row[-1]-(alfa*d_weight)
    
#new weight
def delta(row,activation,y,bias):
    for i in range(len(row)):
        d_weight=2*row[i]*(activation-y)*(1-activation)*activation

        new_weight=weight[i]
        new_weight[i]=weight-0.1*d_weight

#prediction
def predict(row,activation):
    return 1.0 if activation >= 0.5 else 0.0

    output = activation(row, weight,y)
    prediction=predict(output)
    error=math.pow((y-activation),2)
    print("Predicted=%d, Error=%s" % (prediction,error))
    
    
#k-fold cross validation
from sklearn.model_selection import KFold
x=data.describe   
y=data.info
kf=KFold(n_splits=5, shuffle=False)
print(kf) 
i=1        
for train_index, test_index in kf.split(x):
    print("Fold ", i)
    print("TRAIN :", train_index, "TEST :", test_index)
    x_train=x[train_index]
    x_test=x[test_index]
    y_train=y[train_index]
    y_test=y[test_index]
    i+=1

print("shape x_train :", x_train.shape)
print("shape x_test :", x_test.shape)

#grafik loss function
pn = (0.1, 300)
pn.fit(array, dataset)
plt.plot(range(1, len(pn.errors) + 1), pn.errors, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.show()

#grafik accurancy
