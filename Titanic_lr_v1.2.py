# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 17:15:33 2018

@author: hejipei
"""
#PS:

'''
调参：penalty C 以及随机分配的随机种子random_state 
贝叶斯优化参数auc得分 0.869
无调参数auc得分 0.84
'''

import pandas as pd
import numpy as np
from model.Utils import split_data,model_metrics,ROC_charts,cm_plot,split_data_train
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,f1_score,accuracy_score
from sklearn.model_selection import cross_val_score
import gc
from sklearn.model_selection import KFold
from sklearn.cross_validation import StratifiedKFold
#获取训练数据
X_train_id,X_test_id,X_train, X_test,y_train, y_test = split_data_train()

'''LR'''


#----------------------------------------------------------------------------------------------- 
#利用Hyperopt进行参数优化 -- 贝叶斯优化
def Hyperopt_get_best_parameters(Metrics = 'roc_auc' ,evals_num =30):
    from hyperopt import fmin, tpe, hp, STATUS_OK, Trials,partial
    penalty_list= ['l1', 'l2']
    parameter_space = {
        'C': hp.uniform('C', 0, 1),
        'penalty': hp.choice('penalty', penalty_list),
        }
    def hyperopt_train_test(params):
        clf = LogisticRegression(**params,random_state=123)
        auc = cross_val_score(clf,X_train,y_train,cv=5,scoring= Metrics).mean()  # replace 2
        return auc

    count = 0
    def function(params):
        auc = hyperopt_train_test(params)
        global count
        count = count + 1
        print({'loss': auc, 'status': STATUS_OK,'count':count})
        return -auc
    
    count = 0
    def fuction_model(params):
    #    print(params)
        folds = KFold(n_splits=5, shuffle=True, random_state=546789)
        train_preds = np.zeros(X_train.shape[0])
        train_class = np.zeros(X_train.shape[0])
        feats = [f for f in X_train.columns if f not in ['Survived','PassengerId']]  # 注意用户编号也要去掉
        for n_fold, (trn_idx, val_idx) in enumerate(folds.split(X_train)):
            trn_x, trn_y = X_train[feats].iloc[trn_idx], y_train.iloc[trn_idx]
            val_x, val_y = X_train[feats].iloc[val_idx], y_train.iloc[val_idx]
            clf = LogisticRegression(**params,random_state=123)
            clf.fit(trn_x,trn_y)
            train_preds[val_idx] = clf.predict_proba(val_x)[:, 1]
            train_class[val_idx] = clf.predict(val_x)
            
            del clf, trn_x, trn_y, val_x, val_y
            gc.collect()
        global count
        count = count + 1
        if   Metrics =='roc_auc': 
            score = roc_auc_score(y_train, train_preds)  
        elif Metrics == 'accuracy':
            score = accuracy_score(y_train, train_class)  
        elif Metrics == 'f1':
            score = f1_score(y_train, train_class)
        print("第%s次，%s score为：%f" % (str(count),Metrics,score))
        return -score
    
    
    algo = partial(tpe.suggest,n_startup_jobs=20)
    trials = Trials()
    #max_evals  -- 寻找最优参数的迭代的次数 
    best = fmin(fuction_model, parameter_space, algo= algo, max_evals= evals_num, trials=trials)
    
    #best["parameter"]返回的是数组下标，因此需要把它还原回来
    best["penalty"]  = penalty_list[best['penalty']]
    print ('best:\n', best)

    
    clf = LogisticRegression(**best,random_state=123)
    phsorce = cross_val_score(clf,X_train,y_train,cv=5,scoring=Metrics).mean() # replace 4 roc_auc f1 accuracy 
    print('贝叶斯优化参数得分：',phsorce)
    
    clf = LogisticRegression( random_state=123)
    nosorce = cross_val_score(clf,X_train,y_train,cv=5,scoring=Metrics).mean() # replace 5
    print('自己调参数得分：',nosorce)

    return best

# best = Hyperopt_get_best_parameters(Metrics = 'roc_auc',evals_num = 30 )

#------------------------------------------------------------------------------

'''模型融合技术stack'''
def train_model(X_train, X_test, y_train,Metrics ='roc_auc', evals_num =30):
    best = Hyperopt_get_best_parameters(Metrics = Metrics,evals_num = evals_num )
    path = r'..\best_score_log.txt'
    fold_scores = []
    folds = KFold(n_splits=5, shuffle=True, random_state=546789)
    train_preds = np.zeros(X_train.shape[0])
    train_class = np.zeros(X_train.shape[0])
    test_preds = np.zeros(X_test.shape[0])

    feats = [f for f in X_train.columns if f not in ['Survived','PassengerId']]  # 注意用户编号也要去掉
    
    for n_fold, (trn_idx, val_idx) in enumerate(folds.split(X_train)):
        trn_x, trn_y = X_train[feats].iloc[trn_idx], y_train.iloc[trn_idx]
        val_x, val_y = X_train[feats].iloc[val_idx], y_train.iloc[val_idx]
        
        clf = LogisticRegression(**best,random_state=123)
        clf.fit(trn_x,trn_y)
        train_preds[val_idx] = clf.predict_proba(val_x)[:, 1] # 概率值
        train_class[val_idx] = clf.predict(val_x) # 分类值
        test_preds += clf.predict_proba(X_test[feats])[:, 1] / folds.n_splits

        if   Metrics =='roc_auc': 
            print('Fold %2d roc_auc score : %.6f' % (n_fold + 1, roc_auc_score(val_y, train_preds[val_idx])))
        elif Metrics == 'accuracy':
            print('Fold %2d accuracy score : %.6f' % (n_fold + 1, accuracy_score(val_y, train_class[val_idx])))
        elif Metrics == 'f1':
            print('Fold %2d f1 score : %.6f' % (n_fold + 1, f1_score(val_y, train_class[val_idx])))
        fold_scores.append(roc_auc_score(val_y, train_preds[val_idx]))
        del clf, trn_x, trn_y, val_x, val_y
        gc.collect()
    # 输出5次综合的评估指标 
    print(fold_scores)
    if   Metrics =='roc_auc': 
        print('Full  roc_auc score %.6f' % roc_auc_score(y_train, train_preds)) 
    elif Metrics == 'accuracy':
        print('Full  accuracy score %.6f' % accuracy_score(y_train, train_class))  
    elif Metrics == 'f1':
        print('Full  f1 score %.6f' % f1_score(y_train, train_class)) 
    with open(path,'a') as f:
        f.write( Metrics+' ' +str(roc_auc_score(y_train, train_preds))+' '+str(best) +'\n')
    # 测试数据预测概率值
    Test_pred = pd.DataFrame()
    Test_pred['PassengerId'] =  X_test_id['PassengerId']
    Test_pred['TARGET'] = test_preds
    # 训练数据预测概率值
    Train_pred = pd.DataFrame()
    Train_pred['PassengerId'] = X_train_id['PassengerId']
    Train_pred['TARGET'] = train_preds
    
    return Train_pred,Test_pred, 





# 网格搜索最优参数
from sklearn.linear_model.logistic import LogisticRegression  
from sklearn.model_selection import GridSearchCV    # 网格搜索系数
from sklearn.pipeline import Pipeline  
from sklearn.preprocessing import StandardScaler
#from sklearn.decomposition import PCA
pipline = Pipeline([('sc', StandardScaler()),
                    ('clf', LogisticRegression(random_state=1))
                    ])
parameters = {  
            'clf__penalty': ('l1', 'l2'),  
            'clf__C': (0.01, 0.1, 1, 10),  
            }  
grid_search = GridSearchCV(pipline, 
                           parameters,
                           n_jobs =-1,
                           verbose =1 ,
                           scoring="roc_auc", #  f1 / precision / recall/ roc_auc
                           cv = 3)  

grid_search.fit(X_train,y_train)

print ('最佳效果：%0.3f'%grid_search.best_score_)
print (grid_search.grid_scores_)  #单独输出
print ('最优参数组合：')
best_parameters=grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print('\t%s:%r'%(param_name,best_parameters[param_name]))



if __name__ == '__main__':
    
    Metrics='accuracy'
    Test_pred,Test_pred = train_model(X_train, X_test, y_train,Metrics =Metrics)

    Metrics='f1'
    Test_pred,Test_pred = train_model(X_train, X_test, y_train,Metrics =Metrics)

    Metrics='roc_auc'
    Test_pred,Test_pred = train_model(X_train, X_test, y_train,Metrics =Metrics)
 