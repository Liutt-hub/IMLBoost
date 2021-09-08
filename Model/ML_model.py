# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 18:07:29 2020

@author: 2019020600
"""
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score,classification_report, confusion_matrix
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

# 随机拆分训练集与测试集
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=26)


'''决策树'''
from sklearn.tree import DecisionTreeClassifier
model_tree = DecisionTreeClassifier(min_samples_leaf=3)#如果生长过盛就是使得每个叶子都是一个类，类别就确定了。
#model_tree = DecisionTreeClassifier(max_depth=5,min_impurity_decrease=0.08,min_samples_leaf=2)
model_tree.fit(train_X, train_y)
y_pred1 = model_tree.predict_proba(test_X)[:,1]  
#y_pred=np.where(y_prob > 0.5, 1, 0)
y_pre1 = model_tree.predict(test_X)
print(confusion_matrix(test_y, y_pre1))
print("决策树的F1_score指标：",metrics.f1_score(test_y,y_pre1,average='weighted'))
c = confusion_matrix(test_y, y_pre1)
re = metrics.recall_score(test_y, y_pre1)
sp = c[1, 1]/(c[1, 0]+c[1, 1])
print("决策树的G-mean指标：", math.sqrt(re*sp))

'''SVM'''
from sklearn.svm import SVC 
sv=SVC(kernel='linear',probability=True)
sv.fit(train_X, train_y)
y_pred2 = sv.predict_proba(test_X)[:,1] 
y_pre2 = sv.predict(test_X)
print(confusion_matrix(test_y, y_pre2))
print("SVM的F1_score指标：",metrics.f1_score(test_y,y_pre2,average='weighted'))
c = confusion_matrix(test_y, y_pre2)
re = metrics.recall_score(test_y, y_pre2)
sp = c[1, 1]/(c[1, 0]+c[1, 1])
print("SVM的G-mean指标：", math.sqrt(re*sp))


'''原始XGBoost'''
import xgboost as xgb
clf = xgb.XGBClassifier()
clf.fit(train_X, train_y)
y_pred3=clf.predict_proba(test_X)[:,1] 
y_pre3 = clf.predict(test_X)
print(confusion_matrix(test_y, y_pre3))
print("XGBoost的F1_score指标：",metrics.f1_score(test_y,y_pre3,average='weighted'))
c = confusion_matrix(test_y, y_pre3)
re = metrics.recall_score(test_y, y_pre3)
sp = c[1, 1]/(c[1, 0]+c[1, 1])
print("XGBoost的G-mean指标：", math.sqrt(re*sp))


'''F+XGB'''
def softmax(x):                           
    e = np.exp(x)
    return e / np.sum(e)

def focal_logloss_derivative_gamma2(a_sample_prob,target_label):                 #focal losss的导数
    gamma=4 ;alpha=0.1                                                  #输入训练输出和实际输出标签     ，输出一阶二阶导数
    target = target_label;    p = a_sample_prob  ;    pt=p[target]
    kClasses=len(a_sample_prob)
    assert target >= 0 or target <= kClasses    #判断合理性
    grad = np.zeros(kClasses, dtype=float)
    hess = np.zeros(kClasses, dtype=float)   
    for c in range(kClasses):
        pc=p[c]
        if c == target:
            g=alpha*(gamma * np.power(1-pt,gamma-1) * pt * np.log(pt) - np.power(1-pt,gamma) ) * (1 - pc)
            h = alpha*(-4*(1-pt)*pt*np.log(pt)+np.power(1-pt,2)*(2*np.log(pt)+5))*pt*(1-pt)
        else:
            g = alpha*(gamma * np.power(1-pt,gamma-1) * pt * np.log(pt) - np.power(1-pt,gamma) ) * (0 - pc)
            h = alpha*(pt*np.power(pc,2)*(-2*pt*np.log(pt)+2*(1-pt)*np.log(pt) + 4*(1-pt)) - pc*(1-pc)*(1-pt)*(2*pt*np.log(pt) - (1-pt)))
        grad[c] = g
        hess[c] = h
    return grad,hess 

def focal_loss(y_pred,  dtrain: xgb.DMatrix):
    y_true = dtrain.get_label()
    krows,kclass=y_pred.shape        
    grad = np.zeros((krows, kclass), dtype=float)
    hess = np.zeros((krows, kclass), dtype=float)
    for r in range(krows):
        target=int(y_true[r])
        p=softmax(y_pred[r,:])
        grad_r,hess_r = focal_logloss_derivative_gamma2(p,target)   
        grad[r] = grad_r
        hess[r] = hess_r 
    grad = grad.reshape((krows * kclass, 1))
    hess = hess.reshape((krows * kclass, 1))
    return grad,hess


m1 = xgb.DMatrix(train_X, train_y)
m2= xgb.DMatrix(test_X, test_y)  
modelfl=xgb.train({'eta':0.1,'booster': 'gbtree','num_class':2,
                    },#,'objective':'binary:logistic','num_class':2,'objective': 'binary:logistic'
                  m1,num_boost_round=100
                  #,obj=logistic_obj
                  ,obj=focal_loss )
predt_custom1 = modelfl.predict(m2)  #类别
predt_raw1 = modelfl.predict(m2, output_margin=True)#每一类概率
y_pred4=np.array([softmax(predt_raw1[i]) for i in range(len(predt_raw1))])[:,1]
#predt_score=[softmax(predt_raw[i]) for i in range(len(predt_raw))]#取最大的
#y_pred4=[softmax(predt_raw[i]) for i in range(len(predt_raw))]#取最大的
#y_pred4=modelfl.predict(m2)
y_pre4 = modelfl.predict(m2)
print(confusion_matrix(test_y, y_pre4))
print("FXGBoost的F1_score指标：",metrics.f1_score(test_y,y_pre4,average='weighted'))
c = confusion_matrix(test_y, y_pre4)
re = metrics.recall_score(test_y, y_pre4)
sp = c[1, 1]/(c[1, 0]+c[1, 1])
print("FXGBoost的G-mean指标：", math.sqrt(re*sp))


'''Hinge+XGB'''
from keras.utils.np_utils import to_categorical
def Hinge_loss(preds: np.ndarray, dtrain: xgb.DMatrix):
    label=dtrain.get_label()
    label=to_categorical(label,num_classes=2) 
    grad=2*(1-label*preds) #梯度
    hess=2*label #2阶导**2
    grad = grad.reshape((614*2, 1))
    hess = hess.reshape((614*2, 1))
    return grad,hess
booster_custom = xgb.train({'eta':0.01,'booster': 'gbtree','num_class':2},
                           m1,num_boost_round=50,  #迭代次数
                           obj=Hinge_loss,  #自定义的函数                        
                          )
predt_custom1 = booster_custom.predict(m2)  #类别
predt_raw1 = booster_custom.predict(m2, output_margin=True)#每一类概率
y_pred6=np.array([softmax(predt_raw1[i]) for i in range(len(predt_raw1))])[:,1]#每一类概率

y_pre6 = booster_custom.predict(m2)
print(confusion_matrix(test_y, y_pre6))
print("HXGBoost的F1_score指标：",metrics.f1_score(test_y,y_pre6,average='weighted'))
c = confusion_matrix(test_y, y_pre6)
re = metrics.recall_score(test_y, y_pre6)
sp = c[1, 1]/(c[1, 0]+c[1, 1])
print("HXGBoost的G-mean指标：", math.sqrt(re*sp))


'''HF+XGB'''
def hfocal_logloss_derivative_gamma2(a_sample_prob,target_label):                 #focal losss的导数
    gamma=4 ;alpha=0.1;w1=0.99                                                #输入训练输出和实际输出标签     ，输出一阶二阶导数
    target = target_label;    p = a_sample_prob  ;    pt=p[target]
    kClasses=len(a_sample_prob)
    assert target >= 0 or target <= kClasses    #判断合理性
    grad = np.zeros(kClasses, dtype=float)
    hess = np.zeros(kClasses, dtype=float)   
    for c in range(kClasses):
        pc=p[c]
        if c == target:                                                                                            #创新点体现，求一阶导后面加了一块其他函数
            g=w1*( alpha*(gamma * np.power(1-pt,gamma-1) * pt * np.log(pt) - np.power(1-pt,gamma) ) * (1 - pc) )+(1-w1)*(pt-pc)
            h = alpha*(-4*(1-pt)*pt*np.log(pt)+np.power(1-pt,2)*(2*np.log(pt)+5))*pt*(1-pt)
        else:
            g = w1*( alpha*(gamma * np.power(1-pt,gamma-1) * pt * np.log(pt) - np.power(1-pt,gamma) ) * (0 - pc) )+(1-w1)*(pt-pc)
            h = alpha*(pt*np.power(pc,2)*(-2*pt*np.log(pt)+2*(1-pt)*np.log(pt) + 4*(1-pt)) - pc*(1-pc)*(1-pt)*(2*pt*np.log(pt) - (1-pt)))
        grad[c] = g
        hess[c] = h
    return grad,hess 

def hfocal_loss(y_pred,  dtrain: xgb.DMatrix):
    y_true = dtrain.get_label()
    krows,kclass=y_pred.shape           
    grad = np.zeros((krows, kclass), dtype=float)
    hess = np.zeros((krows, kclass), dtype=float)
    for r in range(krows):
        target=int(y_true[r])
        p=softmax(y_pred[r,:])
        grad_r,hess_r = hfocal_logloss_derivative_gamma2(p,target)   
        grad[r] = grad_r
        hess[r] = hess_r 
    grad = grad.reshape((krows * kclass, 1))
    hess = hess.reshape((krows * kclass, 1))
    return grad,hess



modelhfl= xgb.train({'booster':'gbtree','eta':0.01,'num_class':2,},#,'objective':'binary:logistic','num_class':2
                  m1,num_boost_round=200
                  ,obj=hfocal_loss)
predt_custom = modelhfl.predict(m2)  #类别
predt_raw = modelhfl.predict(m2, output_margin=True)#每一类概率
y_pred5=np.array([softmax(predt_raw[i]) for i in range(len(predt_raw))])[:,1]
#y_pred5 = modelhfl.predict(m2)
y_pre5 = modelhfl.predict(m2)
print(confusion_matrix(test_y, y_pre5))
print("HFXGBoost的F1_score指标：",metrics.f1_score(test_y,y_pre5,average='weighted'))
c = confusion_matrix(test_y, y_pre5)
re = metrics.recall_score(test_y, y_pre5)
sp = c[1, 1]/(c[1, 0]+c[1, 1])
print("HFXGBoost的G-mean指标：", math.sqrt(re*sp))


#做ROC
from matplotlib import pyplot
from sklearn.metrics import roc_curve, auc
fpr,tpr,threshold = roc_curve(test_y,y_pred1) ##决策树
roc_auc= auc(fpr,tpr) ###计算auc的值
lw = 1
pyplot.figure(1)
pyplot.plot(fpr, tpr, color='cyan',
         lw=lw, label=' DTree {:.2%}'''.format(roc_auc)) ###假正率为横坐标，真正率为纵坐标做曲线


fpr,tpr,threshold = roc_curve(test_y,y_pred2) ##SVM
roc_auc= auc(fpr,tpr)
pyplot.plot(fpr, tpr, color='g',
         lw=lw, label=' SVM {:.2%}'''.format(roc_auc)) 



fpr,tpr,threshold = roc_curve(test_y,y_pred3) ##XGB
roc_auc= auc(fpr,tpr)
pyplot.plot(fpr, tpr, color='orange',
         lw=lw, label=' XGBoost {:.2%}'''.format(roc_auc))
''''''
fpr,tpr,threshold = roc_curve(test_y,y_pred6) ##XGB
roc_auc= auc(fpr,tpr)
pyplot.plot(fpr, tpr, color='b',
         lw=lw, label=' SHXGBoost {:.2%}'''.format(roc_auc))

fpr,tpr,threshold = roc_curve(test_y,y_pred4) ##XGB
roc_auc= auc(fpr,tpr)
pyplot.plot(fpr, tpr, color='pink',
         lw=lw, label=' IFXGBoost {:.2%}'''.format(roc_auc))

fpr,tpr,threshold = roc_curve(test_y,y_pred5) ##XGB
roc_auc= auc(fpr,tpr)
pyplot.plot(fpr, tpr, color='r',
         lw=1.5, label=' IMBoost {:.2%}'''.format(roc_auc))


pyplot.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
pyplot.xlim([0.0, 1.0]);pyplot.ylim([0.4, 1.05])
pyplot.xlabel('False Positive Rate');pyplot.ylabel('True Positive Rate')
pyplot.title(r'$\bf{Heart Failure}$',x=0.05,y=1.05)
pyplot.legend(loc="lower right")
pyplot.savefig('对比实验曲线.png',dpi=500,bbox_inches = 'tight')