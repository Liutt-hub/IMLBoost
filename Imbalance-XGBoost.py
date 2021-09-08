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

'''
HF+XGB
'''
m1=xgb.DMatrix(train_X, train_y)
m2=xgb.DMatrix(test_X, test_y)

def softmax(x):                           
    e = np.exp(x)
    return e / np.sum(e)

def weif(a_sample_prob,target_label):                 #focal losss的导数
    gamma=4 ;alpha=0.1;                                             #输入训练输出和实际输出标签     ，输出一阶二阶导数
    target = target_label;    p = a_sample_prob  ;    pt=p[target]
    kClasses=len(a_sample_prob)
    assert target >= 0 or target <= kClasses    #判断合理性
    grad = np.zeros(kClasses, dtype=float)
    hess = np.zeros(kClasses, dtype=float)   
    for c in range(kClasses):
        pc=p[c]
        if c == target:                                                                                            
            g=0.5*(alpha*(gamma * np.power(1-pt,gamma-1) * pt * np.log(pt) - np.power(1-pt,gamma) ) * (1 - pc) )-0.5*(pc-pt)
            h = alpha*(-4*(1-pt)*pt*np.log(pt)+np.power(1-pt,2)*(2*np.log(pt)+5))*pt*(1-pt)
        else:
            g = 0.5*(alpha*(gamma * np.power(1-pt,gamma-1) * pt * np.log(pt) - np.power(1-pt,gamma) ) * (0 - pc)) +0.5* (1-pt)*pt
            h = alpha*(pt*np.power(pc,2)*(-2*pt*np.log(pt)+2*(1-pt)*np.log(pt) + 4*(1-pt)) - pc*(1-pc)*(1-pt)*(2*pt*np.log(pt) - (1-pt)))
        grad[c] = g
        hess[c] = h
    return grad,hess 

def weif_loss(y_pred,  dtrain: xgb.DMatrix):
    y_true = dtrain.get_label()
    krows,kclass=y_pred.shape           
    grad = np.zeros((krows, kclass), dtype=float)
    hess = np.zeros((krows, kclass), dtype=float)
    for r in range(krows):
        target=int(y_true[r])
        p=softmax(y_pred[r,:])
        grad_r,hess_r = weif(p,target)   
        grad[r] = grad_r
        hess[r] = hess_r 
    grad = grad.reshape((krows * kclass, 1))
    hess = hess.reshape((krows * kclass, 1))
    return grad,hess

modelwf= xgb.train({'booster':'gbtree','eta':0.1,'num_class':2},#,'objective':'binary:logistic','num_class':2
                  m1,num_boost_round=100
                  ,obj=weif_loss)

predt_custom = modelwf.predict(m2)  #类别
predt_raw = modelwf.predict(m2, output_margin=True)#每一类概率
y_pred5=np.array([softmax(predt_raw[i]) for i in range(len(predt_raw))])[:,1]
#y_pred5 = modelhfl.predict(m2)

#做ROC
from matplotlib import pyplot
from sklearn.metrics import roc_curve, auc

fpr,tpr,threshold = roc_curve(test_y,y_pred5) ##XGB
roc_auc= auc(fpr,tpr)
pyplot.plot(fpr, tpr, color='r',
         lw=1.5, label=' HFXGBoost {:.2%}'''.format(roc_auc))


pyplot.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
pyplot.xlim([0.0, 1.0]);pyplot.ylim([0.2, 1.05])
pyplot.xlabel('False Positive Rate');pyplot.ylabel('True Positive Rate')
pyplot.title(r'$\bf{Heart Failure}$',x=0.05,y=1.05)
pyplot.legend(loc="lower right")
pyplot.savefig('HFXGBoost.png',dpi=500,bbox_inches = 'tight')
#print(classification_report(test_y, predt_custom, labels=[0, 1]))
print(confusion_matrix(test_y, predt_custom))
print("HFXGBoost的F1_score指标：",metrics.f1_score(test_y,predt_custom,average='weighted'))
c = confusion_matrix(test_y, predt_custom)
re = metrics.recall_score(test_y, predt_custom)
sp = c[1, 1]/(c[1, 0]+c[1, 1])
print("HFXGBoost的G-mean指标：", math.sqrt(re*sp))