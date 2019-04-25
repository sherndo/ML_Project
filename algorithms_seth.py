import pandas as pd
import os, sys, collections
from math import log2 as log
import pickle,time,numpy
import matplotlib.pyplot as plt

from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier
from sklearn.model_selection import cross_validate, GridSearchCV
from xgboost import XGBClassifier, plot_importance
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import log_loss

def load_data(file_name):
    # read csv into panda dataframe using headers and found delimiter
    tourney_data = pd.read_csv(file_name, delimiter=',')

    all_vals = ['Year','T1 TeamID','T1 Seed','T1 SoS','T1 OE','T1 AdjOE','T1 DE','T1 AdjDE','T1 WinDev','T1 AdjWin','T2 TeamID','T2 Seed','T2 SoS','T2 OE','T2 AdjOE','T2 DE','T2 AdjDE','T2 WinDev','T2 AdjWin','Winner']
    select_vals = ['T1 Seed','T1 SoS','T1 OE','T1 AdjOE','T1 DE','T1 AdjDE','T1 WinDev','T1 AdjWin','T2 Seed','T2 SoS','T2 OE','T2 AdjOE','T2 DE','T2 AdjDE','T2 WinDev','T2 AdjWin','Winner']

    tourney_data = tourney_data.loc[:, select_vals]

    return tourney_data.iloc[:,:-1], tourney_data.iloc[:,-1]

def trainXGBoost(data, classes, k_fold, scoring):
    avg_acc = []
    avg_roc = []
    avg_loss = []

    # params = {
    # "learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
    # "max_depth"        : [10, 12, 15],
    # "min_child_weight" : [ 1, 3, 5, 7 ],
    # "gamma"            : [ 0.0, 0.2 , 0.4 ],
    # "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ],
    # }

    gbc = XGBClassifier(max_depth=10, learning_rate=0.15, gamma=0.1, colsample_bytree=0.3, min_child_weight=3)

    # glf = GridSearchCV(gbc, params,cv=k_fold)
    # glf.fit(data,classes)
    # print(glf.best_params_)
    # plt.figure()
    # plt.bar(range(len(glf.cv_results_['mean_test_score'])),glf.cv_results_['mean_test_score'])
    # plt.title("Grid Search Results for XGBoost")
    # plt.savefig('/home/sherndo/Documents/Classes/MachineLearning/ML_Project/xg_param_search.png')
    
    gbc_scores = cross_validate(gbc, data, classes, cv=k_fold, scoring=scoring, return_train_score=False, return_estimator=True)

    for i in range(len(gbc_scores['estimator'])):
        gbc_i = gbc_scores['estimator'][i]
        plot_importance(gbc_i, importance_type='cover')
        plt.savefig('/home/sherndo/Documents/Classes/MachineLearning/ML_Project/xg_'+str(i)+'.png')

    return gbc_scores

def trainLogisticRegression(data, classes, k_fold, scoring):
    lrc = LogisticRegression(penalty='l1', tol=0.00001, C=100, solver='liblinear')#, max_iter=100)

    # params = {
    # 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    # 'tol': [0.001, 0.0001, 0.00001, 0.000001, 0.0000001],
    # 'solver': ['liblinear', 'lbfgs', 'saga'],
    # 'max_iter': [10,100,1000,10000]
    # 'penalty': ['l2','l1']
    # }

    # clf = GridSearchCV(lrc, params,cv=k_fold)
    # clf.fit(data,classes)
    # print(clf.best_params_)
    # plt.figure()
    # plt.bar(range(len(clf.cv_results_['mean_test_score'])),clf.cv_results_['mean_test_score'])
    # plt.title("Grid Search Results for Logistic Regression")
    # plt.savefig('/home/sherndo/Documents/Classes/MachineLearning/ML_Project/lr_param_search.png')

    lrc_scores = cross_validate(lrc, data, classes, cv=k_fold, scoring=scoring, return_train_score=False, return_estimator=True)
    print(lrc_scores)

    return lrc_scores

def trainRandomTrees(data, classes, k_fold, scoring):
    rtc = ExtraTreesClassifier(n_estimators=100, criterion='gini')

    # params = {
    # 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    # 'tol': [0.001, 0.0001, 0.00001, 0.000001, 0.0000001],
    # 'solver': ['liblinear', 'lbfgs', 'saga'],
    # 'max_iter': [10,100,1000,10000]
    # # 'penalty': ['l2','l1']
    # }

    # rtf = GridSearchCV(rtc, params,cv=k_fold)
    # rtf.fit(data,classes)
    # print(rtf.best_params_)
    # plt.figure()
    # plt.bar(range(len(rtf.cv_results_['mean_test_score'])),rtf.cv_results_['mean_test_score'])
    # plt.title("Grid Search Results for Logistic Regression")
    # plt.savefig('/home/sherndo/Documents/Classes/MachineLearning/ML_Project/lr_param_search.png')

    rtc_scores = cross_validate(rtc, data, classes, cv=k_fold, scoring=scoring, return_train_score=False, return_estimator=True)
    print(rtc_scores)

    return rtc_scores


def trainMultiPerceptron(data, classes, k_fold, scoring):
    mlc = MLPClassifier(hidden_layer_sizes=(50,200,260,500,1000,500,260,200,50,), solver='adam',
    alpha=0.0001, batch_size='auto', learning_rate='adaptive', learning_rate_init=0.01,
    max_iter=100000, shuffle=True, random_state=None, tol=0.0001,
    early_stopping=True, validation_fraction=0.1, n_iter_no_change=10)

    # params = {
    # 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    # 'tol': [0.001, 0.0001, 0.00001, 0.000001, 0.0000001],
    # 'solver': ['liblinear', 'lbfgs', 'saga'],
    # 'max_iter': [10,100,1000,10000]
    # # 'penalty': ['l2','l1']
    # }

    # mlf = GridSearchCV(mlc, params,cv=k_fold)
    # mlf.fit(data,classes)
    # print(mlf.best_params_)
    # plt.figure()
    # plt.bar(range(len(mlf.cv_results_['mean_test_score'])),mlf.cv_results_['mean_test_score'])
    # plt.title("Grid Search Results for Logistic Regression")
    # plt.savefig('/home/sherndo/Documents/Classes/MachineLearning/ML_Project/lr_param_search.png')

    mlc_scores = cross_validate(mlc, data, classes, cv=k_fold, scoring=scoring, return_train_score=False, return_estimator=True)
    print(mlc_scores)

    return mlc_scores

def trainSGD(data, classes, k_fold, scoring):
    sgc = SGDClassifier(loss='modified_huber', penalty='l2', max_iter=None,
    tol=0.0001, shuffle=True, learning_rate='optimal',
    early_stopping=True, validation_fraction=0.1,
    n_iter_no_change=10, n_iter=None)

    # params = {
    # 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    # 'tol': [0.001, 0.0001, 0.00001, 0.000001, 0.0000001],
    # 'solver': ['liblinear', 'lbfgs', 'saga'],
    # 'max_iter': [10,100,1000,10000]
    # # 'penalty': ['l2','l1']
    # }

    # sgf = GridSearchCV(sgc, params,cv=k_fold)
    # sgf.fit(data,classes)
    # print(sgf.best_params_)
    # plt.figure()
    # plt.bar(range(len(sgf.cv_results_['mean_test_score'])),sgf.cv_results_['mean_test_score'])
    # plt.title("Grid Search Results for Logistic Regression")
    # plt.savefig('/home/sherndo/Documents/Classes/MachineLearning/ML_Project/lr_param_search.png')

    sgc_scores = cross_validate(sgc, data, classes, cv=k_fold, scoring=scoring, return_train_score=False, return_estimator=True)
    print(sgc_scores)

    return sgc_scores


def trainKNN(data, classes, k_fold, scoring):
    knc = KNeighborsClassifier(n_neighbors=10) #weights='distance')

    # params = {
    # 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    # 'tol': [0.001, 0.0001, 0.00001, 0.000001, 0.0000001],
    # 'solver': ['liblinear', 'lbfgs', 'saga'],
    # 'max_iter': [10,100,1000,10000]
    # # 'penalty': ['l2','l1']
    # }

    # knf = GridSearchCV(knc, params,cv=k_fold)
    # knf.fit(data,classes)
    # print(knf.best_params_)
    # plt.figure()
    # plt.bar(range(len(knf.cv_results_['mean_test_score'])),knf.cv_results_['mean_test_score'])
    # plt.title("Grid Search Results for Logistic Regression")
    # plt.savefig('/home/sherndo/Documents/Classes/MachineLearning/ML_Project/lr_param_search.png')

    knc_scores = cross_validate(knc, data, classes, cv=k_fold, scoring=scoring, return_train_score=False, return_estimator=True)
    print(knc_scores)

    return knc_scores

def trainRandomForest(data, classes, k_fold, scoring):
    rfc = RandomForestClassifier(n_estimators=100,max_features='auto',bootstrap=True)

    # params = {
    # 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    # 'tol': [0.001, 0.0001, 0.00001, 0.000001, 0.0000001],
    # 'solver': ['liblinear', 'lbfgs', 'saga'],
    # 'max_iter': [10,100,1000,10000]
    # # 'penalty': ['l2','l1']
    # }

    # rff = GridSearchCV(rfc, params,cv=k_fold)
    # rff.fit(data,classes)
    # print(rff.best_params_)
    # plt.figure()
    # plt.bar(range(len(rff.cv_results_['mean_test_score'])),rff.cv_results_['mean_test_score'])
    # plt.title("Grid Search Results for Logistic Regression")
    # plt.savefig('/home/sherndo/Documents/Classes/MachineLearning/ML_Project/lr_param_search.png')

    rfc_scores = cross_validate(rfc, data, classes, cv=k_fold, scoring=scoring, return_train_score=False, return_estimator=True)
    print(rfc_scores)

    return rfc_scores

def trainNaiveBayes(data, classes, k_fold, scoring):
    nbc = GaussianNB()

    nbc_scores = cross_validate(nbc, data, classes, cv=k_fold, scoring=scoring, return_train_score=False, return_estimator=True)

    return nbc_scores

if __name__=="__main__":
    file_name = "/".join(__file__.split("/")[:-1])
    print(file_name)
    if file_name == "":
        file_name = "tourney_matchup_results_wmirrors.csv"
    else:
        file_name += "/tourney_matchup_results_wmirrors.csv"

    scoring = {
        'accuracy': 'accuracy',
        'loss': 'neg_log_loss',
        'roc_auc': 'roc_auc'
    }

    data,classes = load_data(file_name)
    # classes[classes=='T1'] = 0
    # classes[classes=='T2'] = 1
    # classes = numpy.array(classes,dtype='int')
    k_fold = StratifiedKFold(16)
    # k_fold.get_n_splits(data, classes)
    # for train_index, test_index in k_fold.split(data, classes):
    #     print(train_index, test_index)

    lr = trainLogisticRegression(data, classes, k_fold, scoring)
    xg = trainXGBoost(data, classes, k_fold, scoring)
    nb = trainNaiveBayes(data, classes, k_fold, scoring)
    rf = trainRandomForest(data, classes, k_fold, scoring)
    knn = trainKNN(data, classes, k_fold, scoring)
    mlp = trainMultiPerceptron(data, classes, k_fold, scoring)
    # sgd = trainSGD(data, classes, k_fold, scoring)

    x = 1