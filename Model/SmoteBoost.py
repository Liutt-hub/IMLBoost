# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 15:39:24 2020

@author: 2019020600
"""

import math
import xgboost as xgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

#数据读取和数据处理
data=pd.read_csv(r"1diabetes.csv",encoding='gbk')
X = data.iloc[:, 0:-1]
y = data.iloc[:, -1]
'''
标准化分布,归一化
'''
data['Pregnancies']=np.log(data['Pregnancies'].values+1)
data['Insulin']=np.log(data['Insulin'].values+1)
data['DiabetesPedigreeFunction']=np.log(data['DiabetesPedigreeFunction'].values+1)
data['Age']=np.log(data['Age'].values+1) 
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=26)

'''
SmoteBoost,改成XGB+采样
'''
from sklearn.ensemble import AdaBoostClassifier
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
def xboost(X_train, test_X, train_y):
    model = xgb.XGBClassifier()
    model.fit(X_train, train_y)
    y_pred = model.predict(test_X)
    return y_pred
'''
求概率试试
'''
def xboostprob(X_train, test_X, train_y):
    model = xgb.XGBClassifier()
    model.fit(X_train, train_y)
    y_pred = model.predict_proba(test_X)
    return y_pred

'''#SMOTE'''
sm = SMOTE(random_state=42)
train_X_sm, train_y_sm = sm.fit_sample(train_X, train_y)
y_smote = xboost(train_X_sm, test_X, train_y_sm)
print('SMOTE AdaBoost')
print(confusion_matrix(test_y, y_smote))
print("SMOTEBoost的F1_score指标：",metrics.f1_score(test_y,y_smote,average='weighted'))
c = confusion_matrix(test_y, y_smote)
re = metrics.recall_score(test_y, y_smote)
sp = c[1, 1]/(c[1, 0]+c[1, 1])
print("SMOTEBoost的G-mean指标：", math.sqrt(re*sp))


sm = SMOTE(random_state=42)
train_X_sm, train_y_sm = sm.fit_sample(train_X, train_y)
y_smote = xboostprob(train_X_sm, test_X, train_y_sm)
def softmax(x):                           
    e = np.exp(x)
    return e / np.sum(e)
y_pred=np.array([softmax(y_smote[i]) for i in range(len(y_smote))])[:,1]

'''#ROC'''
from matplotlib import pyplot
from sklearn.metrics import roc_curve, auc
fpr,tpr,threshold = roc_curve(test_y,y_pred) ##XGB
roc_auc= auc(fpr,tpr)
pyplot.plot(fpr, tpr, color='cyan',
         lw=1, label=' SMOTEBoost {:.2%}'''.format(roc_auc))
pyplot.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
pyplot.xlim([0.0, 1.0]);pyplot.ylim([0.6, 1.05])
pyplot.xlabel('False Positive Rate');pyplot.ylabel('True Positive Rate')
pyplot.title(r'$\bf{Heart Failure}$',x=0.05,y=1.05)
pyplot.legend(loc="lower right")
pyplot.savefig('SMOTEBoost.png',dpi=500,bbox_inches = 'tight')

