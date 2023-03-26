# -- coding: utf-8 --
"""
Created on Wed Sep 14 20:56:44 2022

@author: Atilla
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

from mpl_toolkits.mplot3d import Axes3D

## data

data = pd.read_csv("data.csv")

print(data.head())

#data.drop(["başlık1", "başlık2"], axis =1, inplace=True) gereksiz sutunları çıkarma


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


KULLEME = data[data.result == 'kulleme']
print(KULLEME)
OTLAR = data[data.result == 'otlar']
SAP = data[data.result == 'sap']
TOPRAK = data[data.result == 'toprak']
YAPRAK = data[data.result == 'yaprak']

ax.scatter(KULLEME.r, KULLEME.g, KULLEME.b, c='r', marker='o')
ax.scatter(OTLAR.r, OTLAR.g, OTLAR.b, color='g',label='Weeds',alpha=0.3)
ax.scatter(SAP.r, SAP.g, SAP.b, color='black',label='Stem',alpha=0.3)
ax.scatter(TOPRAK.r, TOPRAK.g, TOPRAK.b, color='r',label='Soil',alpha=0.3)
ax.scatter(YAPRAK.r,YAPRAK.g,YAPRAK.b,color='brown',label='Healthy leaf',alpha=0.3)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()


# plt.scatter(KULLEME.r, KULLEME.g, KULLEME.b, color='b',label='Gallery',alpha=0.3)
# # plt.legend()
# # plt.show()

# plt.scatter(OTLAR.r, OTLAR.g, OTLAR.b, color='g',label='Weeds',alpha=0.3)
# # plt.legend()
# # plt.show()

# plt.scatter(SAP.r, SAP.g, SAP.b, color='black',label='Stem',alpha=0.3)
# # plt.legend()
# # plt.show()

# plt.scatter(TOPRAK.r, TOPRAK.g, TOPRAK.b, color='r',label='Soil',alpha=0.3)
# # plt.legend()
# # plt.show()

# plt.scatter(YAPRAK.r,YAPRAK.g,YAPRAK.b,color='brown',label='Healthy leaf',alpha=0.3)
# plt.legend()
# plt.show()

#data.result = [1 if each == "dirt" else 0 for each in data.result]

# for i in range(0,data.result.size,1):
#     if(data.result[i]=="leaf"):
#         data.result[i]=1
#     elif(data.result[i]=="dirt"):
#         data.result[i]=0
#     else:
#         data.result[i]=2 #infected


x_data = data.drop(['result'],axis=1)
y = data.result.values

## narmalization

#x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data))

## train test

    
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_data, y, test_size=0.1, random_state=1)

## model

from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()

## train

dt.fit(x_train,y_train)

test = [[]]

image = cv2.imread("Resim2.jpeg")

frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

y_pred = dt.predict([[0,0,0]])

yaprak = 0
kulleme = 0

# KULLEME = data[data.result == 'kulleme']
# OTLAR = data[data.result == 'otlar']
# SAP = data[data.result == 'sap']
# TOPRAK = data[data.result == 'toprak']
# YAPRAK = data[data.result == 'yaprak']

for i in range(0,image.shape[0],1):
    for j in range(0,image.shape[1],1):
        y_pred=dt.predict([[image[i,j,0],image[i,j,1],image[i,j,2]]])
        if(y_pred=='kulleme'):
            frame[i,j]=0
            kulleme = kulleme + 1
        elif(y_pred=='otlar'):
            frame[i,j]=120

        elif(y_pred=='sap'):
            frame[i,j]=240 

        elif(y_pred=='toprak'):
            frame[i,j]=90
 
        else:
            frame[i,j]=180 
            yaprak = yaprak + 1 


cv2.namedWindow("result",cv2.WINDOW_NORMAL)
cv2.imshow("result",frame)

cv2.imwrite("copy2.jpg", frame) #RESİM kopyasını KAYDET
cv2.waitKey(0)
cv2.destroyAllwindows()

print("Hastalık oranı : %" + str(100*((kulleme)/(kulleme + yaprak))))

# score

# from sklearn.metrics import confusion_matrix
# cm = confusion_matrix(y_test, y_pred)
# print (cm)

# from sklearn.metrics import classification_report
# print(classification_report(y_test,y_pred))




