# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 17:32:01 2020

@author: lenovo
"""

import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import MinMaxScaler
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt


def DataProcessing(filepath):
    data = pd.read_csv(open(filepath))
    #data.info()
    data = data.drop(data[(data['LOCATION_ID'] == 'LOHARU') | (data['LOCATION_ID'] == 'NUH') | (data['LOCATION_ID'] == 'SAFIDON')].index)
    #处理数据
    x = data.iloc[:,:22]
    y = data['Risk']
    data = pd.read_csv(open(filepath))
    x = Imputer().fit_transform(x)
    scaler = MinMaxScaler()
    result = scaler.fit_transform(x)
    X = pd.DataFrame(result)
    return X, y


if __name__ == '__main__':
    filepath = r'C:\Users\lenovo\Desktop\audit_data\audit_risk.csv'
    X, y = DataProcessing(filepath)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
    clf_tree = tree.DecisionTreeClassifier(max_depth = 4)
    clf_tree = clf_tree.fit(x_train, y_train)
    y_predict = clf_tree.predict(x_test)
    confusion_matrix = confusion_matrix(y_test, y_predict)
    print(confusion_matrix)
    y_probs = clf_tree.predict_proba(x_test) #模型的预测得分
    y_probs = y_probs[:, 1]
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)  #auc为Roc曲线下的面积
    #开始画ROC曲线
    plt.plot(fpr, tpr, 'b',label='AUC = %0.2f'% roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([-0.1,1.1])
    plt.ylim([-0.1,1.1])
    plt.xlabel('False Positive Rate') #横坐标是fpr
    plt.ylabel('True Positive Rate')  #纵坐标是tpr
    plt.title('AUC_Tree')
    plt.show()
    