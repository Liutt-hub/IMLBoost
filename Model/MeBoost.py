# -*- coding: utf-8 -*-
"""
Created 2020

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
from sklearn.ensemble import AdaBoostClassifier,ExtraTreeClassifier,DecisionTreeClassifier
from sklearn.ensemble.weight_boosting import _samme_proba
from sklearn.metrics import roc_auc_score
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号



#数据读取和数据处理
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


'''MeBoost'''
class MEAdaBoost(AdaBoostClassifier):
    def __init__(self, n_estimators, depth, split, neighbours):
        self.M = n_estimators
        self.depth = depth
        self.split = split

    def customSampler(self, X, Y):
        #neighbours = 3
        #sampler = RandomUnderSampler(return_indices=True,replacement=False,ratio=0.9, random_state=0)#
        index = []
        for i in range(len(X)):
            index.append(i)
        return X, Y, index

    def fit(self, X, Y):
        self.models = []
        self.alphas = []
        best_alpha = []
        best_tree = []
        top_score = 0
        N, _ = X.shape
        W = np.ones(N) / N
        for m in range(self.M):
            if m %2 == 0:
                tree = DecisionTreeClassifier(max_depth=self.depth, min_samples_split=self.split)
            else:
                tree = ExtraTreeClassifier(max_depth=self.depth, min_samples_split=self.split)
            X_undersampled, y_undersampled, chosen_indices = self.customSampler(X, Y)
            tree.fit(X_undersampled, y_undersampled, sample_weight=W[chosen_indices])
            P = tree.predict(X)
            err = np.sum(W[P != Y])
            if err > 0.5:
                m = m - 1
            if err <= 0:
                err = 0.00000001
            else:
                try:
                    if (np.log(1 - err) - np.log(err)) == 0:
                        alpha = 0
                    else:
                        alpha = 0.5 * (np.log(1 - err) - np.log(err))
                    W = W * np.exp(-alpha * Y * P)  # vectorized form
                    W = W / W.sum()  # normalize so it sums to 1
                except:
                    alpha = 0
                    W = W * np.exp(-alpha * Y * P)  # vectorized form
                    W = W / W.sum()  # normalize so it sums to 1

                self.models.append(tree)
                self.alphas.append(alpha)

                FX = np.zeros(N)
                for alpha, tree in zip(self.alphas, self.models):
                    FX += alpha * tree.predict(X)
                FX = np.sign(FX)
                score = roc_auc_score(Y, FX)
                if top_score < score:
                    top_score = score
                    best_alpha = self.alphas
                    best_tree = self.models


        self.alphas = best_alpha
        self.models = best_tree


    def predict(self, X):
        N, _ = X.shape
        FX = np.zeros(N)
        for alpha, tree in zip(self.alphas, self.models):
            FX +=  alpha* tree.predict(X)
        return np.sign(FX), FX
    def predict_proba(self, X):
        proba = sum(tree.predict_proba(X) * alpha for tree , alpha in zip(self.models,self.alphas) )
        proba = np.array(proba)
        proba = proba / sum(self.alphas)
        proba = np.exp((1. / (2 - 1)) * proba)
        normalizer = proba.sum(axis=1)[:, np.newaxis]
        normalizer[normalizer == 0.0] = 1.0
        proba = proba /  normalizer
        return proba
    def predict_proba2(self, X):
        proba = sum(_samme_proba(est , 2 ,X) for est in self.models )
        proba = np.array(proba)
        proba = proba / sum(self.alphas)
        proba = np.exp((1. / (2 - 1)) * proba)
        normalizer = proba.sum(axis=1)[:, np.newaxis]
        normalizer[normalizer == 0.0] = 1.0
        proba = proba / normalizer
        return proba.astype(float)
    
clf = MEAdaBoost(n_estimators=2, depth=2, split=2,neighbours=2)
clf.fit(X,y)
y_cost=clf.predict(test_X)[0]
print("MEBoost的F1_score指标：",metrics.f1_score(test_y,y_cost,average='weighted'))
c = confusion_matrix(test_y,y_cost)
re = metrics.recall_score(test_y,y_cost)
sp = c[1, 1]/(c[1, 0]+c[1, 1])
print("MEBoost的G-mean指标：", math.sqrt(re*sp))
print(c)


def softmax(x):                           
    e = np.exp(x)
    return e / np.sum(e)
y_pre=clf.predict_proba(test_X)
y_pred=np.array([softmax(y_pre[i]) for i in range(len(y_pre))])[:,1]

#ROC
from matplotlib import pyplot
from sklearn.metrics import roc_curve, auc
fpr,tpr,threshold = roc_curve(test_y,y_pred) ##XGB
roc_auc= auc(fpr,tpr)
pyplot.plot(fpr, tpr, color='fuchsia',
         lw=1, label=' MEBoost {:.2%}'''.format(roc_auc))
pyplot.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
pyplot.xlim([0.0, 1.0]);pyplot.ylim([0.0, 1.05])
pyplot.xlabel('False Positive Rate');pyplot.ylabel('True Positive Rate')
pyplot.title(r'$\bf{Heart Failure}$',x=0.05,y=1.05)
pyplot.legend(loc="lower right")
pyplot.savefig('MEBoost.png',dpi=500,bbox_inches = 'tight')
