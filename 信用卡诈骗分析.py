#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn

from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import f1_score,accuracy_score,precision_score,recall_score

import warnings
warnings.filterwarnings('ignore')

#导入数据
data = pd.read_csv('creditcard.csv')

#数据探索
print(data.shape)
print(data.describe())
print(data.head())

#时间Time
data['Time'].value_counts().sort_values()

#目标值Class
data['Class'].value_counts(normalize = True)
#负例占比过于小，使用F1值或者查准查全好于准确率

#数据清理
plt.figure(figsize=(31,31))
corr = data.corr()
seaborn.heatmap(corr,annot=True)
#说实话，虽然原数据是处理后的数据，但与目标值的相关性都太低了，对这个数据的合理性难免生疑。

#模型创建-模型训练-模型评估
#分类模型
classifier = [
    SVC(random_state=1),
    DecisionTreeClassifier(random_state=1),
    RandomForestClassifier(random_state=1),
    LogisticRegression(random_state=1),
    KNeighborsClassifier(),
    AdaBoostClassifier(random_state=1)
]

#模型名称
classifier_name = [
    'svc',
    'decisiontreeclassifier',
    'randomforestclassifier',
    'logisticregression',
    'kneighborsclassifier'
    'adaboostclassifier'
]

#模型参数
classifier_grid_param = [
    {'svc__C':[0.5,1,1.5],'svc__kernel':['rbf','linear','poly']},
    {'decisiontreeclassifier__max_depth':[4,6,8,10,12],'decisiontreeclassifier__criterion':['gini','entropy'],'decisiontreeclassifier__min_samples_leaf':[2,4,6]},
    {'randomforestclassifier__max_depth':[4,6,8,10,12],'randomforestclassifier__criterion':['gini','entropy'],'randomforestclassifier__n_estimators':[20,50,100,120,150]},
    {'logisticregression__penalty':['l1','l2'],'logisticregression__solver':['sag','saga','lbfgs','liblinear']},
    {'kneighborsclassifier__n_neighbors':[4,8,12,16],'kneighborsclassifier__algorithm':['ball_tree','kd_tree','brute']},
    {'adaboostclassifier__n_estimators':[10,20,50,100],'adaboostclassifier__learning_rate':[0.01,0.05,0.1,0.5,1,1.5]}
]

#模型创建
def GridSearchCV_work(pipeline,X,y,param_grid,cv,score='f1'):
    gridsearch = GridSearchCV(estimator=pipeline , param_grid = param_grid , scoring=score , cv = cv)
    search = gridsearch.fit(X,y)
    print('最佳参数',search.best_params_)
    print('最优分数',search.best_score_)
    
for model,model_name,model_param_grid in zip(classifier,classifier_name,classifier_grid_param):
    pipeline = Pipeline([
        ('scaler',StandardScaler()),
        (model_name,model)
    ])
    GridSearchCV_work(pipeline,data,y,model_param_grid,5,score='f1')

