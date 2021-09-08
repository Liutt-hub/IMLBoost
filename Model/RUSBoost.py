# -*- coding: utf-8 -*-
"""
Created on Sep 

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
from sklearn.ensemble import AdaBoostClassifier,ExtraTreesClassifier,DecisionTreeClassifier
from sklearn.ensemble.weight_boosting import _samme_proba
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号



'''数据读取和数据处理'''
data=pd.read_csv(r"1diabetes.csv",encoding='gbk')
X = data.iloc[:, 0:-1]
y = data.iloc[:, -1]


'''标准化分布,归一化'''
data['Pregnancies']=np.log(data['Pregnancies'].values+1)
data['Insulin']=np.log(data['Insulin'].values+1)
data['DiabetesPedigreeFunction']=np.log(data['DiabetesPedigreeFunction'].values+1)
data['Age']=np.log(data['Age'].values+1) 
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=26)

'''预测相关'''
def xboost(X_train, test_X, train_y):
    model = xgb.XGBClassifier()
    model.fit(X_train, train_y)
    y_pred = model.predict(test_X)
    return y_pred
def xboostprob(X_train, test_X, train_y):
    model = xgb.XGBClassifier()
    model.fit(X_train, train_y)
    y_pred = model.predict_proba(test_X)
    return y_pred

'''RUSBoost'''
X_full = data.copy()
X_maj = X_full[X_full.Outcome== 0]
X_min = X_full[X_full.Outcome == 1]
X_maj_rus = resample(X_maj, replace=False, n_samples=len(X_min)+60, random_state=50)
X_rus = pd.concat([X_maj_rus, X_min])
train_X_rus = X_rus.drop(['Outcome'], axis=1)
train_y_rus = X_rus.iloc[:,-1]
scaler = MinMaxScaler()
train_X_rus= scaler.fit_transform(train_X_rus)
'''#RUSBoost'''
y_rus = xboost(train_X_rus, test_X, train_y_rus)                #全预测为一类
print('RUSBoost')
print(confusion_matrix(test_y,y_rus))
print("RUSBoost的F1_score指标：",metrics.f1_score(test_y,y_rus,average='weighted'))
c = confusion_matrix(test_y,y_rus)
re = c[0,0]/(c[0,0]+c[0,1])
sp = c[1, 1]/(c[1, 0]+c[1, 1])
print("RUSBoost的G-mean指标：", math.sqrt(re*sp))
''''''
y_rus = xboostprob(train_X_rus, test_X, train_y_rus) 
def softmax(x):                           
    e = np.exp(x)
    return e / np.sum(e)
y_pred=np.array([softmax(y_rus[i]) for i in range(len(y_rus))])[:,1]
#ROC
from matplotlib import pyplot
from sklearn.metrics import roc_curve, auc
fpr,tpr,threshold = roc_curve(test_y,y_pred) ##XGB
roc_auc= auc(fpr,tpr)
pyplot.plot(fpr, tpr, color='g',
         lw=1, label=' RUSBoost {:.2%}'''.format(roc_auc))
pyplot.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
pyplot.xlim([0.0, 1.0]);pyplot.ylim([0.6, 1.05])
pyplot.xlabel('False Positive Rate');pyplot.ylabel('True Positive Rate')
pyplot.title(r'$\bf{Heart Failure}$',x=0.05,y=1.05)
pyplot.legend(loc="lower right")
pyplot.savefig('RUSBoost.png',dpi=500,bbox_inches = 'tight')